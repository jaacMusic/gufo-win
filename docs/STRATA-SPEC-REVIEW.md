# Strata speculative-decoding review (2026-10-26, second pass 2026-10-27)

Review of `github.com/niko1221/strata` @ `4b9e05a` (2026-10-25) for
speculative techniques portable to Gufo's qwen3.8-flash-next decode path.
The 2026-10-27 pass sweeps the 0.2.7 work landed after that commit.
Strata's speculative stack: `src/spec/` (controller, draft policy, suffix
drafter), `src/ngram/` (PLE table reader), docs in `docs/A770_SPEED_SPEC.md`
and `docs/STRIX_HALO.md`.

## Ported

| Item | Commit | Notes |
| --- | --- | --- |
| Online accept-probability table | `df097b30` | `MtpAcceptTable`: Beta hit/miss counts per (depth-class, top-prob bin), seeded with the 2026-09-25 prior at weight 32, refined from every verified cycle. Replaces the frozen `kMtpAcceptByTop` lookup feeding the survival trim. Output-invariant (draft-side only; `output_sha256` identical in A/B). |
| Stall clamp on cycle-cost samples | `df097b30` | `MtpBatchController::Observe` clamps any cycle-ms sample >4x the bin's running estimate before the 0.25-EMA, so one preemption/page-fault spike cannot close a width for an occupancy/context bin. Strata's measured constant. |
| 8192-token prefill chunk (`e7e48be2`) | Raising `kPrefillChunkTokens` 4096 -> 8192 (our Linux chunk was already 4096, not their 1024) is +2.2% at pp8192 and +2.6% at pp32768, -1.6% at pp4096, decode unchanged (tg64 55.3/55.5 at d0, 95.4/95.5 at d16K, MTP shared-Q8_0, depth 0, 2 reps). Runs at d32768 without OOM (pp4096 1543.76, tg8 27.10). |

A/B (`tools/windows/bench-flash-next-context.ps1`, 1 rep, deterministic
seeds; report: `artifacts/bench-context/spec-ab-report.html`, commit
`6e8399df`): tg128 MTP 50.17 -> 51.13 t/s @ 8K, 93.88 -> 93.54 @ 32K; both
within single-run noise (untouched AR control moved +4.2%). The runs are
too short to fill the table (~32 observations per bin); throughput-neutral,
output-identical.

Extended check (2026-10-26): tg512 @ 8K (126 cycles, 48.82 vs 48.79 t/s,
`output_sha256` identical) and the four `speculative-corpus` chat prompts
at 128 tokens (marginal acceptance 57-74%, the trim's regime). All eight
runs: identical SHA-256 and identical `cycles/drafted/accepted` counters —
the table changed no decision — tps within ±1.1%. The table only diverges
from the prior after ~32 observations per bin, which needs long generations
(~1k tokens); there it is the mechanism that protects against a prior that
mismatches the running quant/domain. Production value, not benchmark value.

## Evaluated and rejected

| Item | Reason |
| --- | --- |
| `--mtp-window 8192` draft-KV cap | Our draft layer already attends through the sparse indexer (`indexer_top_k = 2048` < 8192); nothing to cap. |
| hipBLASLt per-shape tuning table (gfx1151) | Strata's prefill is dense-GEMM bound; ours is MoE. `blaslt_->Gemm` serves only the unquantized projections (router, alpha/beta, indexer q/k) and the hot indexer shapes already run our own kernels (`OwnF16`/`OwnBf16`). FLOP estimate at pp4096: dense GEMMs <0.5% of prefill. Not measured with rocprof (unavailable on this Windows host). |
| Suffix-automaton drafter (~1.1k LoC) | Fallback for models without an MTP head; all our targets ship one and the fused prompt-lookup covers copy-heavy text. |
| PLE n-gram table (26.8 GiB streamed) | DeepSeek-V4-specific artifact; no equivalent tensor in our models. |
| Draft-source contention FSM | Solves prompt-lookup vs MTP as separate sources; our single fused chain encodes the same policy. |
| Probe-budget decay (5%, 0.995/round) | Our `probe_depth`/`probe_delay` mechanism is bounded differently; nothing measurable to fix. |
| Strix scheduling asymmetry (`DS4_ASYNC_T=4`, `63b8a8d3`): verify window synchronous instead of mailbox/async | Their win is dropping a cross-stream mailbox; our decode never overlaps two streams (one submission per pass, the alternate stream is picked only after `WaitFlag` returns), so there is no handoff to remove. The closest switch, `flag_waits`, is *faster* than `hipStreamSynchronize` on gfx1151; measured (UD-Q4_K_XL + shared-Q8_0 MTP, `gufo bench --n-prompt 9 --n-gen 64 --n-depth 0,16384`, 2 interleaved rounds, tg64 t/s): all-on 55.48/55.48 and 95.49/95.23, `-flag_waits` 54.65/54.76 and 93.82/93.81, `-copy_kernels` 55.22/55.21 and 94.76/95.07, `none` 54.50/54.47 and 93.13/93.48. Every Windows switch helps; the package is worth +1.4% at depth 0 and +2.5% at 16K, the opposite sign of Strata's result. Not re-tested on Linux (no gfx1151 Linux host here), where the same switches are off by default. |
| Long-context indexer select (`STRATA_SELECT_LONG`, `b434a8aa`): one CTA per 2048 blocks of top-k capacity, shuffle-free per-head block scorer | Our `SelectScoreKernel` already builds the head sum in registers with an in-block butterfly and the score sweep is already spread over CTAs; the multi-CTA top-k only pays off at 135K-300K score cells, while our select chunk caps at 512 keys x 32-block rows (32768 cells). Measured per call: ~15 us at 32K context, ~25 us at 128K; 12 select layers make select ~1% of a decode step and their change only touches the top-k half, so the ceiling is 0.4-0.6% for a bit-exact two-level rewrite. |
| Head at 8 columns per weight pass (`e4328457`) | Already satisfied: `MMVQ_MAX_BATCH_SIZE = 8` gives Q8_0 heads one weight read for up to 8 columns, K-quant heads go through the 64-row `get_mmq_y_host` MMA tile, and the verify head is a single `Dense(..., n_tokens)` call; only the drafter's per-step 1-row head is left, which has no rows to batch. |

## Third pass (2026-10-30): the borrow direction flipped

Checked `origin/main` up to `v0.2.8` (4,730 commits past `4b9e05a`).
`src/spec/` and `src/ngram/` have **zero** commits since v0.2.7 — nothing new
in their speculative stack, and our `df097b30` + #882 already cover it. The
v0.2.8 work is SYCL/Arc, Metal, CUDA-5090 and app-level: not portable to the
gfx1151 flash-next path.

Their `STRATA_PF_F16` prompt-expert path (`src/prefill/moe_f16_kq.cu`)
carries the header "Adapted from gufo 97b8645f … `RoutedF16GEMMKernel`" — it
is a re-import of our `RoutedF16Gemm`, which our prefill already runs by
default (`executor.cpp` `wmma_experts`: `RoutedGatedF16Gemm` + 64-token down
tiles, landed `f36da345` at pp2048 1040 → 1298 tok/s). Their 85.3 → 53.5
ms/layer at 16K tokens is the f16-vs-*their-int8* delta; against our tree the
win is already banked. The `WmmaExperts` predicate leaves the narrower
`mmq/` `dp4a` MoE paths to decode, verify and prompt widths below its floor. Their `--prefill-chunk 8192` Windows default matches ours,
and their shared-expert second stream measures 6.9% *slower* on this GPU,
independently confirming the #758 rejection. The pp2048 gap (1,613–1,631 vs
upstream 1,801) therefore lives in our own `RoutedF16Gemm` wide-shape
efficiency (~71% of the 124 GB/s in-model ceiling), i.e. upstream #708/#779
territory, not in anything strata has.

Borrowable method, not code: their rounding-change acceptance gate is
first-token KL across 12 prompts of 1.5K–24K tokens with top-1 match 12/12 —
the right shape for pricing any future "rounds differently" kernel if the
bit-identical rule is ever relaxed for prefill experts.

## Strata findings worth remembering

- Overlapping drafter and verify on separate streams (`--spec-split`)
  *hurts* on their Strix/A770 measurements (9.9-10.3 vs 11.7-12.6 tok/s).
  Do not propose stream overlap for the draft head.
- Spec width beyond 4 was worse and `spec-min-p ~0.5` optimal in their
  sweeps; both match our controller's converged behavior.
- Their conclusion for prefill/decode headroom: per-layer handoff latency
  and CPU-pool/GPU overlap, not kernel rewrites.
