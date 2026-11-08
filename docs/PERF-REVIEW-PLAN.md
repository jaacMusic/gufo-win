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
- Follow-up 2026-10-30 (same probe, `HcCombineStream`): the HC-combine access
  pattern (per token a 4 x hidden F32 residual read/write-back, a broadcast
  block_out read and a narrowed F16 norm write, 2048 tokens) streams at
  178.3 GB/s while the routed fetch in the same run sustains 193.7 GB/s, i.e.
  ~92% of the achievable rate. Combine-class traffic (~0.23 GB/call, 61 ms
  over 47 layers) is therefore not pattern-limited; the ~143 ms 10.5% bucket's
  remainder is the mix/inject epilogue and MoE-fused variants, and the rest of
  the tier is matrix-bound dense GEMM plus the GDN recurrence. No bandwidth
  lever remains among these without the per-kernel trace.

## Prefill phase split (2026-10-31, Windows, intra-layer marks)

`GUFO_TRACE_LAYERS=1` now stamps phase boundaries inside the layer body
(`Timeline::kPhaseMark`, `Executor::MarkPhase`), so the "fresh per-kernel
trace" the step-2 pass asked for is available on the Windows host without
rocprofv3. Reproduction (warm APU, `build/release`):

```powershell
$env:GUFO_TRACE='1'; $env:GUFO_TRACE_LAYERS='1'
.\gufo.exe bench --model <shard1> --n-prompt 2048 --n-gen 384 --n-depth 0 --repetitions 1
```

pp2048 at the same settings: 1635-1668 tok/s (traced / plain, one binary);
GPU busy 99.8%, idle 0.06 ms per pass. Per pass (1.25 s wall, 3 passes
traced) the buckets are:

| bucket | ms/pass | share | measured rate | headroom |
| --- | --- | --- | --- | --- |
| routed experts (48 calls) | 435 | 35% | 9.07 ms/layer, ~73 GB = 188 GB/s | 198 GB/s probe, 95% |
| GDN in-projection (36) | 177 | 14% | 4.90 ms/layer | ~35 TFLOP/s F16 |
| HC combines (attn+ffn) | 135 | 11% | 0.95 + 1.86 ms/layer | 92% of 193.7 GB/s |
| HC mixers (attn+ffn) | 114 | 9% | 1.38 + 0.99 ms/layer | norm + inject GEMMs |
| GDN scan (36) | 98 | 8% | 2.71 ms/layer | sequential recurrence |
| full attention, 11 layers + head | 88 | 7% | 4.10 proj + 3.25 core | ~34 TFLOP/s |
| GDN out-projection (36) | 73 | 6% | 2.04 ms/layer | ~32 TFLOP/s |
| MoE shared expert (48) | 55 | 4% | 1.15 ms/layer | W8A8 GEMM |
| LM head | 66 | 5% | one 2048-row pass | ~24 TFLOP/s |
| MoE router (48) | 18 | 1% | 0.38 ms/layer | |

Sum 1259 ms against the 1253 ms wall, so the map is complete: no hidden
host-side stall, no unattributed tier. Everything that moves weights is
within 5-8% of the rate the probes measured for its own access pattern, and
every GEMM in the pass runs at 24-35 TFLOP/s (the F16 WMMA rate the fused
attention projection proves the device sustains). This closes the tier: no
bucket above the 10% cost-centre rule has a remaining lever larger than the
measurement noise, so the prefill item is settled until weight traffic itself
drops (#708 device-resident weights, smaller quant), which is a format
decision rather than a kernel change. The GLU round trip (#31188) is ~0.6%
of the pass; #779 is masked-width only.

## Width scaling and the fixed per-pass cost (2026-10-31, Windows)

`gufo bench --n-prompt 512,1024,2048,4096,8192 --n-gen 16 --n-depth 0
--repetitions 2` with no other engine resident: 1195.30 / 1462.87 / 1572.98 /
1637.08 / 1663.61 tok/s. A straight fit gives 1708 tok/s marginal and **~129 ms
fixed per pass**, i.e. pp512 loses 30% to a cost that does not scale with
width.

That cost is routing, not scheduling. At 512 tokens and top-4 the router makes
2048 expert hits over 128 experts, so every expert is read regardless of width:
the traced `head` bucket costs 64.1 ms at 512 rows against 66.0 ms at 2048
rows, and per-layer MoE drops only 9.07 -> 5.62 ms. Below ~2K tokens the ~73 GB
expert fetch is therefore width-invariant, and the same effect explains why the
serve prefill rate climbs with prompt width rather than staying flat.

## Serve path: TTFT, chunking, cache (2026-10-31, Windows)

Measured through `/v1/chat/completions` on the production command line
(`%TEMP%\opencode\ttft-matrix.ps1`; the stats object is `usage.gufo`):

| prompt tokens | cold TTFT | cold prefill tok/s | chunks | cached TTFT |
| --- | --- | --- | --- | --- |
| 2105 | 1704 ms | 1268 | 3 | 22.7 ms |
| 8210 | 5426 ms | 1559 | 3 | 44.8 ms |
| 32594 | 21672 ms | 1526 | 6 | 121.3 ms |

Snapshot write costs 73-374 ms per request and restore 18 ms. Chunking only
starts above the 8192 `--prefill-batch` cap (992 / 1942 / 3816 / 7367 tokens
each planned one data chunk plus a prefix chunk; 14641 planned three), so short
prompts are not being clipped. There is no long-context degradation: 8K and 32K
cold prefill both hold 1526-1559 tok/s.

## Batching is the remaining lever (2026-10-31, Windows)

`gufo serve -j/--sessions N` is the only concurrency knob (default 1). Four
distinct ~2100-token prompts, sequential wall time against four parallel curl
processes (`%TEMP%\opencode\conc-test.ps1`):

| server | sequential | 4 concurrent | plan | speedup |
| --- | --- | --- | --- | --- |
| `-j 4`, ctx 32768 | 6702 ms | 3295 ms | `batched-w4` | 2.0x |
| `-j 2`, ctx 131072 | 6371 ms | 3210 ms | `batched-w2` | 1.98x |
| `-j 4`, ctx 262144 | fails to load: `request-state claim exceeds state capacity` | | | |
| `-j 3`, ctx 262144 | fails to load: same | | | |
| `-j 2`, ctx 262144 | loads and serves (matrix not rerun) | | | |

Two sessions already capture the whole gain, because one pass pays the fixed
129 ms plus the full expert fetch no matter how many requests it carries.
Memory is the constraint: `-j 2` at 128K uses 94.6 GiB of 113 GiB, and the cap
that fails is total request state, not per-session context. Only one engine
process fits at a time (see `docs/WINDOWS-DEV.md`), so batch width trades
against context length, never against a second server.

### State-capacity wall, validated (2026-10-31)

The 262144-context failure is arithmetic, not a leak. At claim time the flash
next runner sees ~25.5 GiB free (113 GiB total minus ~73 GiB resident weights
plus OS/vision overhead) and needs ~10.3 GiB per session of dense fp16 KV, so
`capacity / per_request` rounds down to 2 sessions:

| ctx | per session | max sessions at 262144 weights |
| --- | --- | --- |
| 32768 | ~1.3 GiB | 4+ (scratch-limited, not state) |
| 131072 | ~5.2 GiB | 4 |
| 262144 | ~10.3 GiB | 2 (3 x 10.3 > 25.5) |

Measured: `-j 3` @ 262144 fails, `-j 2` @ 262144 loads (101.3 GiB device used,
20.2 GiB still free after load). The claim now derives capacity from
`total - resident weights` (bounded by measured free) and the error message
reports session count, per-session bytes and free bytes, so the next failure
names its own numbers. Halving per-session bytes (fp8/Q8 KV, ~5.2 GiB at
262144) is the only path to 4 sessions at 262K; everything else (paged or
shared cross-session KV, growable arena) is upstream work (#740).

## Kernel-level re-review (2026-11-01, static, engine resident — no benches run)

Second static pass over `qwen38_flash_next` kernels + the trace corpus, hunting
outside the previously reviewed set. Findings with per-phase numbers are from
`bench-trace.log` (MTP decode bench; w255 rows there are pp2048-scale passes,
w16 rows are warm/16-token prefix passes per the 2026-10-30 note).

### Decode AR (26.93 ms/token, weight-traffic-bound, GPU 99.8 % busy)

Per-token phase totals (w1, 247 cycles): routed MoE 7.7 ms, GDN proj 4.3,
head 4.63, HC mix 4.3, attn proj 1.3, shared 1.4, router 1.0, scan 1.1,
combines 1.0, attn core 2.9.

- **LM-head Q4_1 decode copy already exists** (commit `d6c667e4`): class
  `lm_head` in `device_model.cpp` (`wants("lm_head")` → `add(m->output_, ...)`),
  off in the default class set because logits numerics change. **Measured
  2026-11-08: +4.9 % decode** (34.99 → 36.70 t/s, −1.33 ms/token) for +0.37 GiB;
  greedy replay diverges once at a paraphrase and reconverges. Table and replay
  detail in `LINUX-PERF-PORT.md`. The default is unchanged — flipping it is a
  quality call, not a measurement one.
- **`q4` copy classes are live but not default**: `GUFO_DECODE_COPY=q4` adds
  `attn_out` + `hc` (zero code). HC mix + combines are ~5.3 ms/token still on
  Q8_0. **Measured 2026-11-08: no gain** (35.42 vs 34.99 t/s is drift) for
  +0.48 GiB, so they stay opt-in.
- **MTP per-row cost is expert-set growth, closed**: w1 hits 4 expert slices,
  w3 ~10 distinct experts (+~3.3 ms/row of the +5.4); `SmallGemmKernel` and
  `GdnKernel` already batch rows on one weight read and rollback snapshots use
  rank-1 factors for t>0. No kernel lever; only draft selection overlap helps.
  This explains the earlier "uniform 0.10 ms/row/layer" note.

### Prefill (pp2048 ≈ 1259 ms; >10 % buckets confirmed at probe ceilings)

- **LM head: closed 2026-11-08, no tile to sweep.** The earlier "24 TFLOP/s vs
  ~35 for sibling GEMMs, 66 ms/pass, sweep 151936×2560×2048 tiles" figure is a
  2048-logit-row shape; production prefill passes `n_logits = 1` per chunk
  (`engine.cpp:892`, chunk = up to 8192 tokens), so the head is the 1-row Q8_0
  GEMV: 1.98 ms/pass at 209 GB/s, already at the host stream ceiling (measured
  with `tests/models/qwen38_flash_next/lm_head_bench.cpp`). `DenseF16Gemm` at
  255 rows is 6.38 ms / 31 TFLOP/s but the head never reaches it. See
  `LINUX-PERF-PORT.md`, "the LM-head tile sweep has no shape to win".
- **`head` bucket unresolved at prefill width**: decode head with ~1–2 logit
  rows is 4.63 ms, but `head w255` is 65.5 ms per pp2048-scale pass while
  `Dense(output)` is only ~2 ms of it. The bucket's split (last-layer HC tail vs
  `Dense(output)` vs download) has no marks. First move: two extra `MarkPhase`
  splits around `Dense(output, ...)` in `executor.cpp` prefill-logits block and
  around the `logits_host_` copy, then re-trace idle.
- **Router chain: 18 ms/pass (~3 TF/s, 14 launches/layer, m=128 rows)**.
  Fused logits+top-k could reclaim ~10 ms (+1 %); lowest priority.

### Checked and already tight (do not re-derive)

`SmallGemmKernel` batches tokens on one weight read; `GdnKernel` loads state
once/token-loop/write-once; `DenseF16GEMMKernel` tiles were swept per shape with
measured ms comments (LM head is the exception, above); HC combines at 92 % BW;
routed tier at 95 % of the 198 GB/s gather probe. The "35 → 55 TFLOP/s" WMMA gap
is sustained clock/power sharing, not a kernel defect.

Lesson recorded: `lm_head` decode-copy class was committed (d6c667e4) but absent
from every doc — a `Select-String` over `src/models/qwen38_flash_next` for the
class name would have prevented the duplicate implementation attempt.


