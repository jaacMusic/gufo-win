# NPU / perf brief

Handoff note so a fresh session can work on the next item without re-deriving
context. Everything here was verified against the tree at `a43391cd`; anything
that is not in section 1 was not confirmed.

## 1. Verified state

- Upstream source: `haloforge-labs/bubo`. Code tip `51d1495` is what we synced
  in `83dd5da3`; the later `1d744ad` (v0.1.1) touches docs only.
  `src/models/qwen38_flash_next/npu` (54 files) vs bubo `src/npu` (34): shared
  files are identical apart from namespace/include rewrites. Ours adds
  guard, chip power, fault telemetry, runtime status, platform info, chunk
  retry, mem trace, and rewrites `invalid_bits` as a bitset and
  `QualifiedChunk` extraction.
- NPU prefill is strictly opt-in: `GUFO_NPU_PREFILL=1` (`engine.cpp:242-270`).
  Upstream serves it by default. Knob defaults match upstream (`SLICES=6`,
  `GATE=0`, `TIMEOUT=8000`, keepalive 1 s) except the shared-expert lane, which
  is off here because it costs 7 % prefill on this host (section 3).
- Executor overlap is preserved: `BeginLinearZ` (`executor.cpp:2095`) → GPU
  prefix `DenseF16SsmGemm` (2113) → `FinishLinearZ` (2164); the MoE
  shared-expert wait lands after routed prep.
- Guard (`guard_config.hpp`, `guard.cpp`, `staging.cpp`) is ours only. It is an
  *envelope* guard: 82 W whole-chip RAPL mean over 3 consecutive chunks (or 98 %
  memory commit) trips a 600 s NPU hold; held chunks simply run on the GPU. It
  never throttles the GPU. `GUFO_NPU_GUARD=on|off|log`, default `log`: the
  sampler reads whole-chip power, which a GPU-only prefill already exceeds on a
  desk host, so a hold can only ever switch off a lane that now wins.
- Whole-chip RAPL cannot attribute power to the NPU. That is documented and
  benign on this host (see section 3); proving anything better needs a
  power-capped host.

## 2. False leads — do not re-investigate

Each of these was checked and is wrong:

- `GUFO_NPU_GUARD` is read and honoured; there is no `GUFO_POWER_LIMIT_W` and no
  `guard_active()`.
- There is no `NPU_PREFILL_DEFAULT` constant; NPU prefill is opt-in.
- There is no NPU busy/duty sampling anywhere in the npu tree.
- `tools/bench/npu_sweep.sh` does not exist and never existed.
- The 4.6 s idle gaps in the depth-32K trace are GPU prefill chunks. The PLE
  gather is hidden behind them: 131,072 rows per chunk, 344 distinct, 78 ms of
  reads on the first chunk and zero after. The trace's GPU-busy percentage
  counts decode-cycle work only, so it reads 0.2 % during a 32K prefill.

## 3. Measurements

pp8192, UD-Q4_K_XL, 32 GB BOSGAME host (`-d 0 -n 128 -r 2`, 2026-10-28, current
defaults: 8192-token chunks, `SLICES=6`, shared-expert lane off, guard off):

| build                                   | tok/s        |
| --------------------------------------- | ------------ |
| GPU only                                | 1655.49      |
| NPU                                     | 1689.96      |
| NPU, slices 12                          | 1426.85      |
| NPU, shared-expert lane on              | 1566.54      |
| NPU, guard on (holds after 3 chunks)    | 1658.92      |

The NPU path is +2.1 % here. The earlier "6-8 % slower" verdict came from two
defects, both fixed on 2026-10-28: the shared-expert lane was on by default (it
dispatches GEMMs of at most 1024 rows, so one dispatch costs more than the GPU
work it removes, and the stream waits on it 47 times per chunk: -7 %), and the
engine capped prefill chunks at 4096 tokens whenever the NPU was on, giving up
the 8192-token batching the GPU path gets (+2 %). Slice count is a flat plateau
at 5-6 and a cliff at 12, where the NPU becomes the critical path
(`stall_ms[linear]` 616-851 ms per chunk).

The guard trips as designed on this host (`chunk_means_w=[112.0, 85.9, 83.6] >
82.0`, hold 600 s, the rest of the process on the GPU), which is why it now
defaults to `log`: with the lane winning, a hold only costs the 2.1 %.

The ceiling is arithmetic, not tuning: the lane computes 4.64 TFLOP per
4096-token chunk (6 slices x 36 layers x 4096x2560x1024) in about 2.4 s, i.e.
about 1.9 TFLOP/s effective, against roughly 100 TFLOP/s for the GPU's prefill
on this model. Per chunk it issues 216 XRT runs of 1-3 ms dispatch each; larger
runs (n > 1024) need a rebuilt xclbin, and the AIE sources are not in this tree.

Older numbers (1608.09 / 1563.61 / 1527.68, 4096-token chunks) in
`docs/LINUX-PERF-PORT.md` are superseded.

Two runtime traps on this host:

- `build/release/gufo.exe` looks for artifacts in `<exe>/../tools/npu`, which the
  CMake tree does not have, so the NPU silently falls back to the GPU with
  `artifact open failed`. Pass `GUFO_NPU_DIR=<repo>/tools/npu` for dev builds.
  The default is correct for the installed layout (`result/bin/gufo`).
- `GUFO_NPU_PREFILL=1` alone is not enough to see NPU numbers; it also needs the
  XRT artifacts above.

Upstream's claimed +5-7 % at 8K and +3.5-5.5 % at 16K was measured on a 70 W
128 GB laptop. Here a 4096-token NPU slice costs about 2.6 s against a 2.55 s
GPU chunk, so there is no overlap win to collect on an uncapped host.

## 4. Work list

1. Closed 2026-10-28: recalibration after the chunk change, the shared-expert
   default and the guard default (section 3).
2. Open, needs the AIE toolchain: rebuild `final_4096x2560x1024_128x64x128`
   with a wider `n` than 1024 (or fewer, larger runs) to cut the 216 dispatches
   per chunk. `tools/npu` has only the xclbin and its instruction stream; bubo
   names `mm_bfp_mixed.cc` and `n32_core_placed.py` but neither ships in our
   tree, and bubo records no compiler version for the pinned binary.
3. Open, unverified: Windows XRT host-pointer BOs. The adapter always bounce-
   copies A (11.25 MiB) and C (6 x 4.5 MiB per chunk group) with `host_only`
   BOs. bubo's fork imports userptrs on Linux; whether the Windows runtime
   allows it is untested, and a copy is worth at most a few ms per layer.

Closed as false leads: the PLE gather (see section 2) and the MTP verify-row
marginal. Width is not saturated with depth, and the default already runs at
the top of the curve:

| draft tokens | tg128 @ d0 | tg128 @ d32768 |
| ------------ | ---------- | -------------- |
| 1            | 41.5       | –              |
| 2            | 57.5       | 57.2           |
| 3            | 57.4       | 69.7           |
| 4            | 59.3       | 77.3           |
| 5            | 57.9       | 83.8           |
| 6            | 57.6       | –              |
| 7 (default)  | –          | 100.2          |

At depth the extra row pays for itself; at depth 0 it does not, so the flat part
of the curve is correct behaviour, not a scheduling bug. Logits are sampled on
device per row (`SupportsDeviceResidentSampling`), so there is no per-row D2H to
remove.

## 5. Checklist before claiming progress

- Re-read the code at the cited lines; line numbers drift.
- Measure with `result/bin/gufo` or `build/release/gufo`, one trace command,
  same model revision.
- Do not treat a skipped model as a pass.
