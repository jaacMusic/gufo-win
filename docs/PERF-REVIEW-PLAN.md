# Performance Review Plan

Status: active plan, 2026-10-23.

How to find real performance issues in this codebase. This complements
`docs/PERFORMANCE.md` (measurement rules) — nothing here overrides it.

## Principles

1. Measure first. A change without a before/after number on a named workload
   is not a performance change, it is a guess.
2. Production target only: Linux x86-64 Strix Halo `gfx1151`. Windows-host
   numbers are directional, never promotional.
3. One mechanism per change, one commit, revert line in the message.
4. Static reading finds candidates; profiling convicts them. Most static
   "wins" in this repo have regressed when measured (see
   `docs/LINUX-PERF-PORT.md` history).

## Ordered workflow

### Step 1 — Pick one named workload

Choose from the bench suite (`gufo bench`, serving benchmark) and freeze it:
same model artifact, quant, context, prompt, seed, and token counts. Record
it in the change description. Candidates:

- pp2048 prefill (closed: 3.3–9× llama.cpp, `models/qwen3.8-flash-next/BENCHMARKS.md`)
- d32768 decode, greedy and sampled+MTP variants
- model load time (I/O bound; verify before optimizing)

### Step 2 — Profile the workload, not the code

- `tools/prof/prof.py` — pipeline/wall-time phases
- `tools/prof/isa_mix.py` — instruction mix for kernel suspects
- `tools/bench/gfx1151_peak.hip` — hardware ceiling to classify the limiter
  (bandwidth / compute / launch / sync / scheduling)

Take the top cost center only. Stop if it is <10% of wall time.

### Step 3 — Classify before touching

| Limiter | Typical fix location | Usually NOT worth it |
|---|---|---|
| Bandwidth | kernel data layout, copies, precision | instruction micro-tuning |
| Compute | kernel math, occupancy | CPU-side refactors |
| Launch | batching, graphs | single-kernel speed |
| Sync | executor polling (`executor.cpp`), event waits | more threads |
| Scheduling | runner pool, request packing (`src/cli/serve`) | kernel work |

CPU-side hot paths worth knowing: `src/core/sampling.cpp` (full-vocab
softmax per verify row), `speculative_verifier.cpp` verify loop,
tokenizers (decode path). GPU kernels live under
`src/models/*/kernels/rocm/` and are gated on the `optimize-kernel` skill.

### Step 4 — Focused reproducer + oracle

Standalone HIP experiment in `tools/bench/build.sh` harness or a CPU
microbench. Correctness oracle: existing parity/unit tests plus bit-exact
output comparison on a fixed seed. Reject on any mismatch.

### Step 5 — Gate and land

- Focused test for the touched path (e.g. `logit_sampler_test` for sampling)
- Full workload re-run vs baseline; keep only measured wins
- `tools/ci/check-format.py` before commit
- Conventional Commit, one line; revert instruction in the body
- End-to-end numbers recorded in `docs/models/<model>/BENCHMARKS.md`,
  not the experiment diary

## Anti-checklist (already ruled out by review, 2026-10)

- Static sweeps of HIP kernels without a profile — regressions historically
- `hipGetDeviceProperties`/malloc/event sites — all init-time
- GGUF reader I/O — mmap-based, load-time only
- Executor `hipStreamSynchronize` polls — deliberate graph-completion design
- Micro container swaps (hash map vs sorted vector) in sampling — measured
  noise-level; the cost is streaming the vectors, not the lookups

## Current state (2026-10-23)

- Committed: `e5c561f` `SampleResidual` skips sorted rebuild (less work per
  rejected draft; end-to-end delta unverified — needs Step 1/5 run).
- Open target: pp2048 prefill gap vs upstream.

## Static review: `src/core/hip` + `qwen38_flash_next` (2026-10-25)

Read every file in both trees without a profile, looking for work that is
provably redundant rather than merely unmeasured. Two changes were justified
as strictly-less-work with bit-identical output; both are CPU-side, so neither
needs a GPU parity run, only the existing unit/parity tests.

Landed:

- `ngram.cpp` `HashNgramRows` — the per-order XOR mix was re-summed from
  `ctx[0]` for every n-gram order, costing `n(n-1)/2` multiplies per token.
  The mix for order `k` is order `k-1`'s mix plus one term, so it now grows
  incrementally: `n-1` multiplies, identical terms in identical order. Runs
  once per token of every prefill chunk and every decode step.
- `engine.cpp` `DecodeBatchImpl` — the drafting round loop declared
  `heads`/`bodies` inside the loop body, so each of up to ~7 rounds per decode
  cycle heap-allocated two vectors. Both now live outside and are cleared per
  round.

Examined and deliberately left alone (with the reason, so this is not
re-litigated):

- `prompt_lookup.hpp` `Find` — the `at(i)` branch per backward comparison
  costs ~0.1 µs per proposal and splitting the tail/committed runs would add an
  invariant to a correctness-critical path.
- `executor.cpp` `Run` double lookup (`warmed_` + `graphs_`) — required, a key
  can be warmed eagerly and captured later.
- `executor.cpp` `WaitFlag` `steady_clock::now()` per poll — CPU burn, not
  latency; polling is the deliberate graph-completion design.
- `executor.cpp` `std::function` layer bodies and `CheckCancellation` — sub-µs
  against a ~25 ms decode cycle.
- `DecodeBatchImpl` `std::any_of` over ≤8 requests — 64 pointer compares.
- `cpu_ops.cpp` `Rope` recomputing `powf` per head — reference/oracle path.
- `kernels.hip.cpp` RoPE `powf` sites — ~0.5 µs/token; `mmq/` is vendor code.
- `mtp_policy.hpp`, `mtp_sampling.hpp`, `draft_vocab.hpp`,
  `platform/tuning.hpp`, `core/hip/detail/dispatch_telemetry.hpp` — already
  cached/precomputed or single-pass.

Conclusion: inside this scope the remaining static candidates are below
measurement noise. The real gap (pp2048 prefill) needs Step 2 profiling, not
more reading.

## Review follow-up: fix, bench recipe, host workflow (2026-10-25)

Commits from this pass: `c6f8c070` (n-gram incremental mix), `8b8ac115`
(drafting vector reuse), `9b00ab72` (this document), `24d220e1`
(SmallGemm materialize fix, see below). All verified: 14/14
`qwen38_flash_next.*` ctest green, `check-format.py` pass.

### SmallGemm width instability (found by the suite, fixed)

`projection_ops` failed "small projection changes with batch width": a row's
low bits differed between the `tokens=1` and `tokens=2` instantiations of
`SmallGemmKernel`. Root cause: the inner `dot += w*x` chain let the
compiler's FMA contraction differ **between template instantiations**; the
per-`acc` `Materialize` barrier was not enough. Fix: materialize every
product in the chain (`24d220e1`), matching the existing pattern in the
sibling kernels. Lesson for any future width-templated kernel: if output must
be bit-identical across instantiations, `Materialize` (or equivalent) every
accumulation step, not just the loop-carried sum. Diagnosis recipe that
worked: temporarily print `n`, `type`, `rows`, `cols` and the first
differing element from the throw site, rebuild only that test target.

### Quick bench recipe (Windows host, directional only)

Same target as the diary tables: unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`
shards + `MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` sidecar, both under
`%USERPROFILE%\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\<hash>\`.

```
gufo bench -m <shard-1-of-4> --speculative mtp --mtp-model <shared-Q8_0>
           -p 2048 -n 16 -d 0,32768 -r 3
```

Run from `build\gpu-test` (DLLs are deployed there). Result of this pass vs
the 2026-10-25 diary row (after #463): pp2048 d0 1612.87±14 vs 1627.12
(neutral, within noise), pp2048 d32K 1404.50 vs 1487.96 (−5.6%, different
session/thermals; re-run an A/B before believing it), tg16 d0 59.85 vs 28.32
and d32K 77.05 vs 25.82 (large gain, attributable to the prompt-lookup
default flip at `561c532e`, which postdates the diary row, not to this
review's changes). Windows numbers stay directional per the rules above.

A/B settled it (2026-10-25, worktree at `561c532e` vs `489e42b1`, same
workload): all four metrics identical within noise (pp2048 1620.33 vs
1619.64–1630.59; tg16 59.96 vs 59.41–60.24; pp2048 d32K **1406.37 ± 0.72**
vs 1405.60–1408.38; tg16 d32K 77.39 vs 77.36–77.47). The review commits and
the SmallGemm fix are performance-neutral. The pp d32K dip vs the diary row
is present in both binaries and predates them — it rides with `561c532e`
(prompt-lookup default on), the same change that took tg16 from 28 to 60
and 26 to 77: a ~5% deep-context prefill cost buying ~2.5x decode.

### Windows host workflow gotchas

- Test executables live in `build\gpu-test\tests\...`, three levels below
  where `build.ps1` deploys the ROCm/vcpkg DLLs. Copy `build\gpu-test\*.dll`
  (and the `hipblaslt`/`rocblas` subdirs) next to a test exe before running
  it directly, or run through `ctest` with TheRock on `PATH`.
- TheRock is not on the system PATH:
  `$env:PATH = "C:\TheRock\build\bin;C:\TheRock\build\lib\llvm\bin;$env:PATH"`.
- `python3` does not exist on this host; use `python` (3.13). Its Scripts
  dir carries the `clang-format.exe` that `tools/ci/check-format.py` uses.
- `gufo.exe diagnose` verifies the whole GPU stack before any bench.
- `qwen38_flash_next_session_test` needs a local model artifact via argv
  (argv[4] = MTP path) and is not ctest-registered; it is the only
  automated-style coverage for `DecodeBatchImpl` changes.

## MTP decode follow-up (2026-10-28)

Step 2 still needs the Linux gfx1151 host: `C:\TheRock` carries neither
`rocprofv3` nor `amd-smi`, so no profile can be captured here. What the Windows
host answered instead:

- Prefill is monotone with no anomaly — 14.3 t/s at 1 new token, 576 at 128,
  1199 at 512, 1544 at 2048, 1665 at 8192, i.e. a 0.57–0.59 ms/token asymptote.
  The only expensive region is 64 tokens and below, which is MoE expert-weight
  bandwidth, not a kernel defect.
- Decode on chat prose: MTP beats AR by 26–28% greedy and 31–37% sampled at
  temperature 1.0, at 49–67% acceptance (tables in
  `models/qwen3.8-flash-next/BENCHMARKS.md`). The `-8.2%` repetitive cell at
  depth 12,288 is a prompt-modelling artifact, not an engine regression.
- `548c5177` makes `serve` enable MTP from `--mtp-model` alone, mirroring the
  existing DSpark promotion. `prompt` and `bench` keep the explicit flag so a
  recorded benchmark never switches mode silently.
- Stage 3 (side-stream overlap) stays rejected: upstream measured +3.3% pp512
  and nothing at pp2048, and `alt_stream_` in `executor.cpp` is a
  queue-freshness ping-pong whose concurrent use would break graph capture and
  the NPU scratch sharing.

## Phase-timeline pass (2026-10-28, Windows)

Step 2 without `rocprofv3`: the in-engine timeline (`GUFO_TRACE=1`,
`kernels/rocm/trace.hpp`) stamps the GPU wall clock at each phase boundary, so
the decode loop splits on this host. Prefill stays one `verify` phase, so it is
characterised by prompt scaling instead. Reproduce from `build\release`:

```powershell
gufo bench -m <shard1> -p 512,1024,2048,4096,8192 -n 8 -d 0 -r 2 -v
$env:GUFO_TRACE="1"
gufo bench -m <shard1> -p 2048 -n 1024 -d 0 -r 1 -v
gufo bench -m <shard1> --speculative mtp --mtp-model <shared Q8_0> `
  -p 2048 -n 1024 -d 0 -r 1 -v
gufo bench -m <shard1> --speculative mtp --mtp-model <shared Q8_0> `
  -p 2048 -n 512 -d 32768 -r 1 -v
```

- Prefill (AR): 1,197.5 / 1,455.6 / 1,556.4 / 1,635.4 / 1,661.5 tok/s at
  512–8,192 tokens, marginal 0.537 / 0.599 / 0.580 / 0.592 ms/token. Flat, no
  quadratic term: attention is not a measurable share of prefill up to 8K and
  the pipeline is linear.
- AR decode at depth 2,048: 26.93 ms verify-body plus 0.68 ms prefix per token,
  GPU busy 96.1–97.1%, idle 0.83–1.13 ms. The 1.8–2.2 ms host submit stays
  hidden behind the previous body.
- MTP decode at depth 0: verify-body 37.65 ms at w3 (119 cycles) against 26.93
  at w1 (32 cycles), so each extra verified row costs 5.4 ms; draft 1.75–3.29
  ms, rollback 0.21 ms, idle 1.64 ms (3.8%). Deep context does not degrade it:
  100.19 tok/s tg512 at depth 32,768 with 447/447 drafts and 438/438
  prompt-lookup accepted. The depth-32K trace window's 4.6 s gaps are prefill
  chunks, not a load stall: the PLE gather is hidden behind them (131,072 rows
  per chunk, 344 distinct, 78 ms of reads on the first chunk and none after;
  the trace's GPU-busy figure counts decode-cycle work only), so the load path
  is not the limit.
- Decisions from the data: WMMA flash attention for prefill is rejected (no
  quadratic term to attack, and 32K decode runs at 100 tok/s, so the arithmetic
  change cannot pay for the greedy-replay gate); F32 chunked DeltaNet stays
  rejected (`f3645979` prices the remaining 2.2% pp2048 at the same gate);
  MTP acceptance is already 100% on the repetitive probe, and the only line
  item of size left is the 5.4 ms verify-row marginal cost (13% of a w1
  cycle), which needs per-kernel timing — the Linux host or new gated
  instrumentation, a separate proposal. The 96–97% busy loop leaves nothing
  for the Windows dispatch path.
- Follow-up 2026-10-30, `GUFO_TRACE_LAYERS=1` (commit `f2919f39`) answered
  the verify-row item on the Windows host with per-layer GPU stamps: the
  w2→w8 marginal 28.18 ms splits as 27.58 ms spread uniformly across all 47
  layers (~0.10 ms/row/layer; dense layers 00–02 highest at 0.81–1.29 ms per
  6 rows) and only 0.60 ms in the head (norm + LM head + logits). No single
  hot kernel exists: the marginal row cost is the whole per-layer body (mixer
  state update + GEMM tier) scaling with verify width. Per-kernel rocprofv3
  work on Linux would refine inside one layer (~0.10 ms/row), but no
  layer-local target of ≥1 ms/row exists, so a width-batched or merged-row
  attack across the shared layer path is the only lever that can pay.

## Prefill step-2 pass (2026-10-30, Windows, GUFO_TRACE_LAYERS)

Per-layer prefill marks (`w255`) at pp2048 d0 on the warm APU close the open
"pp2048 gap needs Step 2 profiling" item:

- Baseline pp2048 d0: 1613-1631 tok/s plain, 1620 traced; GPU busy 99.8% with
  0.063 ms inter-pass idle. Dispatch and synchronization keep the GPU fed.
- All 47 layer buckets land flat at 24.5-27.2 ms/pass. Outliers: layer01
  37.8 ms (also 1.3 vs 0.5 ms at decode, so structural) and layer43 32.4 ms,
  together ~1.3% of the pass; the head bucket is 68.7 ms (last layer + LM
  head + logits download).
- Body ~= wall: 47 x ~26 ms + head ~= 1.29 s against the 1.264 s bench wall.
  At 111 GB of touched weights that is ~88 GB/s effective, 71% of the
  124 GB/s in-model streaming figure; upstream 1,801 tok/s is ~98 GB/s (79%).
- A tempting "350 ms host-side prefill overhead" reading of the trace is a
  counting artifact: the `verify-body w16` average mixes the slower warm pass
  with the 16-token prefix pass. Wall time equals GPU time; there is no
  host-side prefill stall to fix.
- hipBLASLt plan-cache tuning is a non-lever for this model: the plan cache is
  off here and hipBLASLt only serves the unquantized projections, not the
  MoE-bound mass (STRATA review row); attention, KV
  initialization and launch overhead have no measurable share at 2K.
- What is left of the pp2048 gap therefore lives inside the MoE/quantized GEMM
  tier (vendor `mmq/`) at wide shapes, i.e. upstream items #708 and #779, and
  carries the same gate as every prior prefill-kernel proposal: bit-identical
  outputs plus greedy-replay pricing before vendor code moves.
- Correction (2026-10-30, `tools/bench/routed_stream_probe.hip`): the 124 GB/s
  "in-model streaming ceiling" used above is stale. A replay of the exact
  RoutedF16GEMM Q4_K fetch pattern sustains 198 GB/s (213 GB/s at 64 B/lane,
  so fetch width caps the gain at +8%). The routed experts move ~73 GB in a
  traced 388 ms = 188 GB/s, i.e. ~95% of what the pattern can sustain:
  RoutedF16GEMM does not need fixing. The gap to upstream instead lives in
  the ~676 ms non-routed tier (dense F16 GEMMs, HC combines, DeltaNet, W8A8,
  attention; ~38 GB moved = ~56 GB/s equivalent, 3.5x off stream rate), which
  needs a fresh per-kernel trace (rocprofv3, Linux host) before any next move.
  The 71%-of-ceiling and 88 GB/s figures above should be read with this in
  mind.

