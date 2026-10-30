# Qwen3.8 Flash-Next experiments

## Confidence-gated MTP chains (October 29, 2026)

Upstream `main` 34ebb5fe against the branch, Strix Halo (ROCm 7.2.4), UD-Q4_K_XL + shared Q8_0 MTP, `gufo serve --sessions 1 --context 65536`, greedy, thinking off, five fixed prompts (CPU caches, a Python red-black tree, TCP congestion control, Czech railway history, a C++ ISO 8601 parser) of 384 generated tokens; one server process per arm, means of the five decode rates. The first process after another workload reads cold PLE rows and runs 2-4% slower, so every hold started with a discarded warm-up and main ran before and after the candidates.

| Experiment | Outcome |
| --- | --- |
| End a chain after its first draft below the head's probability 0.85 (sampled: q below 0.95); unplanned cycles draft to `--draft-tokens`; planned greedy batches keep their width and drop a final proposal below 0.6 | Retained. Replies byte-identical. C1 MTP 50.3 (main, three runs) → 55.8/55.9 tok/s (+11%; every prompt faster, +4% to +16%). On a four-session server: C1 49.7 → 53.3, C4 wall throughput 76.2/76.9 → 85.4 tok/s (+11%). 16-token replies (four runs per arm), decode rate after the first token, main against the branch: prompts 1 and 2 41.5-42.0 → 37.5-37.7 and 52.7-54.0 → 47.5-48.2 tok/s (about −10%, +35 to +65 ms per reply), prompt 3 unchanged, prompt 4 24.3 → 25.3-25.6, prompt 5 within its run-to-run spread (44.9-63.1 against 59.9-62.0). This short-reply cost is not fixed; keeping the controller's first-cycle width until acceptance is observed did not change it. Two 1.5-1.8 s first-token outliers on the branch (and 0.35-0.37 s ones on main) are prefill time, before any drafting. |
| Downstream sweeps (Q4_K/Q3_K-class recipe) | Greedy gate 0.7-0.95: 0.85 best on prose and four/eight users; sampled 0.9-0.98 within noise. Acceptance of verified C8 drafts by head probability: 0.1-0.2 22%, 0.4-0.5 53%, 0.6-0.7 71%, 0.8-0.9 86%, ≥ 0.9 97%; batch drop thresholds 0.35/0.5/0.65/0.8 gave 347.9/345.9/345.8/349.3 s against 352.6 s on an eight-way 28-question slice. Rejected: continuing while the product of acceptance estimates stays above 0.55/0.45; C1 gates 0.7/0.5 (−3.0%/−7.1%). |

## Chunked DeltaNet removed (October 27, 2026)

The release-tier functional `tool-reasoning` suite failed in both MTP and AR
modes at `literal_protocol_raw_xml`: asked to echo a short XML literal with a
tool declared (a 299-token prompt), Flash-Next answered "Hello! I'm here to
help you with any questions or tasks you might have." at temperature 0 and
presence penalty 0.

- Bisect with the `rocm10` package of each commit (ROCm 10.1): `4a4a47c`
  echoes the literal, `538e2fd` (#758) greets.
- Main with the chunked scan compiled out echoes it, with and without MTP.
- llama.cpp b11382 on Gufo's rendered prompt ranks `<` first (log
  probability −0.715) and `Hello` third (−2.12): the scan flips a 1.4-nat
  greedy decision. The same request without tools (48 tokens, below the
  scan's 256-token threshold) echoes on every build.
- The scan's F16 Q/K/T/P operands put its layer error 500 times above the
  F32 recurrence (2.6e-4 against 5.1e-7 relative RMS), and its chat prompt
  NLL rose by a significant 0.0145 nats (table below).

The scan, its prep, the fused norm and their operator test are removed: about
4 % of prefill does not justify flipping greedy decisions. Both side streams
stay; they are lossless, and beside the row-split recurrence on ROCm 10.1
they still shorten short passes. `gufo bench -n 0 -r 3`, two alternated
rounds, 55 °C cooldown before each process:

| Build | pp512 | pp2048 | pp4096 |
| --- | ---: | ---: | ---: |
| main (chunked scan, both side streams) | 1,348/1,346 | 1,936/1,932 | 2,010/1,993 |
| row split, no side streams | 1,286/1,288 | 1,858/1,825 | 1,909/1,897 |
| row split, both side streams (retained) | 1,331/1,329 | 1,855/1,853 | 1,916/1,908 |

## Sparse attention and routed repacks from the updated source branch (October 27, 2026)

Ideas from the October 25–27 revision of halo-box/strix-llama.cpp's
Flash-Next prefill work, after rebasing onto main `c0e0be58`. All runs use
the `rocm10` shell, whose library path loads the CWSR-patched ROCr; the
binaries' own RPATH points at the unpatched runtime. Every process waits for
the GPU to cool to 55 °C, and builds alternate round-robin. Both experiments
are lossless (bit-identical by construction or by output hash). Neither is
retained.

| Experiment | Outcome |
| --- | --- |
| Depth profile before porting | Diagnostic: a pp2048 pass takes 1,056 ms at d0 and 1,153 ms at d50K (`prof.py`, timed pass only). Of the +97 ms, attention adds 63 and block scoring 39. The source branch's own d0 → d50K cost is +162 ms, so Gufo already degrades less with depth; its d50K pp2048 is 1,766–1,772 tok/s. |
| Sparse-window attention with K and V read into WMMA registers (8 waves split the head dimension, S partials summed through LDS, no K/V staging) | Rejected. The operator test passes with the same error as the current kernel. Synthetic `attn_bench` at 16K/50K/100K: 6.98/8.06/9.46 ms now, 7.22/7.78/8.92 ms reading V rows directly. A per-call block-packed V copy was slower again, about 1 ms at 100K. Deriving each step's key mask from the union membership bits was required, without it the kernel ran 2.6 ms slower. End to end at d50K pp2048: 1,772/1,763 now against 1,766/1,740 tok/s. The kernel stays at 210 VGPRs, and `amdgpu_waves_per_eu(8)` does not reduce them. |
| Expert gate/up rows interleaved per 8-row tile (one 128-byte line per 16-byte piece of the eight rows) | Not retained. Bit-identical in `routed_gemm_bench`: 5.45 → 5.26 ms at 2,048 tokens, 8.55 → 8.06 ms at 4,096. It needs the decode, MMQ and unpaired readers rewritten for the new layout. |
| High-nibble Q4_K codes decoded as F16 subnormals (one `pk_fma` with the scale × 2^20) | Not retained: within noise, 0 to −2 %. Low nibbles cannot use the trick without a shift: at mantissa bits 8–11 they overlap the exponent. The scale must be widened in F32, because fast-math folds the two F16 doublings into one overflowing constant. The largest sampled F16(d × sc) is 0.054, under the 0.0625 limit. |
| Q5_1 down experts read into registers from 16-row tiles, then re-encoded so a 5-bit code is an F16 subnormal (one `pk_fma` per two weights) | Not retained. Bit-identical in `routed_gemm_bench`: 3.08/3.89/5.47 → 2.87/3.66/5.06 ms at 512/2,048/4,096 tokens. Prefill-only end to end, with the layout applied at load: pp512 1,356/1,373 → 1,393/1,389, pp2048 1,983/1,976 → 1,988/1,995, pp4096 2,051/2,035 → 2,065/2,066 tok/s (+0.5 to +1.8 %). The repack would replace the only copy of the weights, so decode, MTP verification and the MMQ tier would each need a reader for it. Probes that run short passes fail until those readers exist. |

### ROCm 7.2 (the default package)

Clang 22 spills `GdnChunkScanKernel` to 1,172–1,400 bytes of scratch per
lane (240 VGPRs, 307–383 spilled; ROCm 10's Clang 24 fits it in 237/185
VGPRs). With the spill, the head was slower than main on ROCm 7.2:
pp512/2048/4096 1,283/1,849/1,936 → 1,246/1,734/1,824 tok/s. Lifting the
launch bound to 256 VGPRs, scheduling barriers per state tile and a later V
prefetch did not remove it. Clang 22 builds therefore keep the row-split
recurrence and do not emit the scan.

Beside the row split, both side streams also cost ROCm 7.2 prefill: pp2048
1,851/1,838 for main, 1,813/1,685 with both, 1,829/1,701 without the Z
overlap, 1,823/1,794 without the shared-expert overlap and 1,852/1,853 with
neither. The overlaps were tuned beside the chunked scan, so they now follow
it. On ROCm 7.2 the branch then matches main: wide-prefill logits are
byte-identical on the four fixture prompts. Speed is equal within noise:
pp512 1,288/1,285 → 1,291/1,290, pp2048 1,853/1,853 → 1,851/1,852, pp4096
1,924/1,919 → 1,925/1,920, d10K 1,747/1,747 → 1,746/1,746 tok/s. ROCm 10
logits and speed are unchanged. All 20 Flash-Next CTest targets pass on
ROCm 7.2.

### Lossy changes in combination

Builds on main `c0e0be58`, each with the fixture probe commit: lossless
(L: side streams and mixer down tiles), + chunked DeltaNet and fused norm
(G), + BF16 residual streams (B), and both (A, the head before the BF16
residual streams were removed). G merged in #758; its chunked DeltaNet and
fused norm were removed on October 27, which leaves L.
`gufo bench -n 0 -r 3`, two round-robin rounds (pp512 from two dedicated
rounds), tok/s as the mean of the rounds. NLL intervals are 95 % paired
bootstraps from `f32_reference.py compare --against`; G's overall change is
within noise, but its chat prompt rises by a significant +0.0145 (+0.0049
to +0.0241).

| Build | pp512 | pp2048 | pp4096 | pp2048 at d10K | Next-token NLL vs main (95 %) | Largest per-prompt NLL change |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| main | 1,286 | 1,854 | 1,909 | 1,732 | — | — |
| L | 1,332 (+3.6 %) | 1,858 (+0.2 %) | 1,909 (0.0 %) | 1,740 (+0.5 %) | byte-identical | — |
| G | 1,349 (+4.9 %) | 1,930 (+4.1 %) | 1,992 (+4.3 %) | 1,801 (+4.0 %) | +0.0021 (−0.0128 to +0.0162) | chat +0.0145 (+0.0049 to +0.0241) |
| B | 1,349 (+4.9 %) | 1,879 (+1.4 %) | 1,948 (+2.0 %) | 1,764 (+1.8 %) | +0.0104 (−0.0040 to +0.0245) | tools +0.0192 (−0.0259 to +0.0661) |
| A | 1,369 (+6.4 %) | 1,961 (+5.8 %) | 2,029 (+6.3 %) | 1,824 (+5.3 %) | +0.0175 (+0.0030 to +0.0318) | tools +0.0444 (0.0000 to +0.0913) |

## Prefill overlap and chunked DeltaNet on ROCm 10.1 (October 24, 2026)

Ideas adapted from halo-box/strix-llama.cpp's Flash-Next prefill work.
`gufo bench -n 0 -r 3` on the `rocm10` toolchain, main `8b52b42c` as
the control; every run waits for the GPU to cool to 55 °C. Main measures
pp512/2048/4096 1,265–1,284 / 1,811–1,860 / 1,880–1,924 tok/s across
these sessions. Native pp2048 at d0 spends about 1,116 ms per pass: routed
experts 388 (DRAM floor about 305), dense F16 GEMMs about 350, HC combines
133, DeltaNet 97, W8A8 62, attention 34. At pp512 the routed experts
(200 ms of 404) already stream their weights at the measured DRAM peak.

| Experiment | Outcome |
| --- | --- |
| Shared expert on a side stream beside the routed experts | Retained: byte-identical logits. The two streams' kernels slow each other (routed gate/up 223 → 251 ms per pass), but the pass shortens: with chunked DeltaNet, disabling only this overlap gives 1,309/1,896/1,965 against 1,335/1,907/1,976 tok/s. Kept after the chunked scan's [removal](#chunked-deltanet-removed-october-27-2026). |
| GDN Z rows on a side stream after the QKV tiles | Retained: byte-identical. Beside the row-split recurrence (96 blocks) it was neutral at pp2048: the recurrence slowed 62 → 115 ms per pass. Starting Z together with QKV was also neutral. With chunked DeltaNet, disabling only this overlap gives 1,320/1,898/1,964 tok/s. Kept after the chunked scan's [removal](#chunked-deltanet-removed-october-27-2026). |
| Chunked WMMA DeltaNet for row-split prefills of 256+ tokens | Rejected (lossy), removed on October 27 after merging in #758: on a 299-token tool prompt it turns a greedy literal echo into an unrelated greeting ([removal](#chunked-deltanet-removed-october-27-2026)). 32-token chunks: parallel prep (L2 norms, log-decay prefix sums, `(I + A)^-1` by forward substitution, F16 Q/K/T/P), then one block per value head scanning the chunks with an F32 state in WMMA accumulators. V is read in F32 from the convolution output. The gates are computed in prep. A checkpoint inside a chunk does not change the pass: the scan saves the state at the chunk boundary before it, and the row split carries that copy over the remaining tokens (at most 31). With both overlaps, against main: 1,281/1,850/1,906 → 1,333/1,888/1,964 tok/s. `--validate-prefill 2048` against the sequential path: rmse 0.252 → 0.266, cosine 0.99694 → 0.99662. |
| Gated norm inside the chunked scan | Rejected: removed with the chunked scan on October 27. It removed the epilogue pass and the raw F32 rows (22 ms per pp2048 pass). 1,335/1,907/1,976 → 1,344/1,928/1,993 tok/s. Last 64 rows of a 2,228-token prompt against main: KL 0.0098 → 0.0079 mean. |
| 32-row, rows-first W8A8 tiles for the mixer down up to 1,024 tokens | Retained: byte-identical. At pp512 the call falls from 187 to 165 µs (20 → 40 blocks); end to end it is within noise. At 2,048 tokens it is slower (649 vs 444 µs), so wider batches keep 64-row tiles. |
| Residual streams kept as BF16 between two normed wide layers | Rejected (lossy), removed on October 27: on top of chunked DeltaNet it adds only 1.5–2 % (combination table above) for a measurable NLL cost, worst on tool text. On the F32 fixtures it raises the next-token NLL by 0.0175 nats (interval +0.003 to +0.033), mostly on chat and tool text. Only the two fused combines read the residual there. They hold BF16 (round to nearest even) at the front of each F32 row, convert at the PLE layer, the last layer and pass boundaries, and save 126 MB per layer at pp2048. Same-session A/B: 1,335/1,925/1,989 → 1,367/1,945/2,014 tok/s (repeat 1,286/1,933/1,983 → 1,367/1,954/2,026). F32 oracle in the table below. |
| The same 32-row tiles for the shared expert | Rejected: the copies on the side stream only stretch beside the routed experts; the pass span is unchanged (393 → 394 ms at pp512). |
| 128 × 64 dense F16 tiles for wide-tile grids under 160 blocks | Rejected: byte-identical, but at pp512 1,302–1,307 against 1,333 tok/s (same binary, toggled); pp1024 flat. |
| F16 Z rows stored by the SSM projection for the chunked norm | Rejected: +0.9 % at pp512 only, flat at pp2048/4096; rmse rises to 0.274. |
| Benchmark tokens drawn like llama-bench (seeded random vocabulary) | Diagnostic: 1,335/1,925/1,997 against 1,345/1,934/2,008 tok/s for Gufo's sequential ids, so expert coverage does not explain the gap to the reference numbers. |

Accuracy against the F32 CPU reference (`reference_probe --dump`, run alone),
on a 661-token prompt prefilled as 512 + 149 rows. The last 64 positions are
compared. These are single-prompt figures: differences near 0.001 KL are
within this sample's spread.

| Build | KL mean | KL median | KL max | Top-1 agreement | Logit RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| main `8b52b42c` | 0.00637 | 0.00190 | 0.0947 | 59/64 | 0.217 |
| + overlap, chunked DeltaNet, fused norm, mixer down tiles | 0.00796 | 0.00379 | 0.1148 | 60/64 | 0.249 |
| + BF16 residual streams | 0.00627 | 0.00246 | 0.0853 | 58/64 | 0.242 |

The reference took 858 s for 661 tokens on an otherwise idle host (3.8 s
for the first token). Running it inside `gpu_probe --reference` while the
GPU model is resident starves the page cache and costs 46 s per token.

F32 reference fixtures (`tests/fixtures/qwen38_flash_next/f32_reference`,
2,479 positions: the last 256 rows of every prefill pass over documentation,
C++ source, a four-turn chat and a tool-calling session). Paired
differences against main, 95 % bootstrap intervals (`f32_reference.py
compare --against`):

| Build | KL to F32, all prompts | NLL of the next prompt token | Largest per-prompt NLL change |
| --- | ---: | ---: | --- |
| main | 0.0341 | 1.1112 | — |
| without BF16 residual streams | 0.0301 (−0.0040, −0.0107 to +0.0029) | +0.0021 (−0.0128 to +0.0162) | chat +0.0145 (+0.0049 to +0.0241) |
| with BF16 residual streams | 0.0358 (+0.0017, −0.0039 to +0.0071) | +0.0175 (+0.0030 to +0.0318) | tools +0.0444 (0.0000 to +0.0913) |

The F32 reference's own NLL on these positions is 1.1079. Main is already far
from it on the tool-calling prompt (KL 0.0997, p99 2.02); the other prompts
stay below 0.011.

Final alternated A/B (main → head with the BF16 residual streams, two
rounds, 55 °C cooldown before each process): pp512 1,260/1,280 → 1,362/1,374 (+7.6 %), pp2048 1,845/1,861 →
1,965/1,970 (+6.3 %), pp4096 1,918/1,913 → 2,038/2,034 (+6.3 %), pp2048 at
d10K 1,736/1,719 → 1,794/1,819 (+4.2 %). At d100K the untimed 100K-token
preparation heats the GPU to 100 °C, so multi-rep runs varied by up to
±276 tok/s on both builds. With one rep per process from 50 °C, the three
alternated rounds give 1,125/1,168/1,195 → 1,210/1,257/1,417 tok/s.

Perplexity over 116,213 positions of repository documentation and Flash-Next
sources (`gpu_probe --nll`, 2,048-token passes): main 4.5669, without the
BF16 residual streams 4.5643, with them 4.5685. Paired block bootstrap (512-token
blocks) of the per-token NLL difference against main: −0.055 % (95 %
interval −0.16 to +0.06 %) and +0.037 % (−0.064 to +0.16 %). Neither change
is distinguishable from main at this resolution. Individual positions do
move: about 23 % differ by more than 0.1 nats in either direction.

Batching several sequences into one prefill pass is not ported: Gufo
prefills sessions one at a time.

## Selector scheduling on newer LLVM (October 23, 2026)

| Experiment | Outcome |
| --- | --- |
| Keep one selector query head live on Clang 23 and newer | Retained: full head unrolling causes 304 scalar-register spills with ROCm 10.1's Clang 24; the bounded loop reduces this to 18 and VGPR use from 169 to 164. Neither 10.1 variant uses private scratch; the original Clang 23 build does. The F32 queries, FMA sequence, reduction tree and selected blocks stay unchanged. On Clang 24, the isolated 64-query × 8,192-block scorer falls from 109.3 to 76.3 µs, with identical output bytes. Clang 22 retains its original unrolling. |
| Reproduce older activation rounding on newer compilers | Rejected: first-layer Q8 codes differ at 37 of 327,680 values; the newer compiler agrees more often with independent FP32 and FP64 calculations. Restoring the old fused remainder matches the first mixer but does not recover sampled thinking throughput. Production rounding and sampling remain unchanged. |
| Price the actual C1 sampled draft head | Retained with the compact target path: fixed-history measurements replace full-head estimates only for eligible sampled C1 cycles. The first cycle and short tails keep their established costs; the focused short-reply control passes all 84 phases. Full target-row verification also keeps the original curve, removing a thinking/tool pause increase (111 → 120 ms, then 113 ms with the guard; all 63 focused phases pass). Full-head, greedy and concurrent curves stay unchanged. See [migration qualification and timing limits](ROCM.md). |

The scorer measurement is the median of five warmed calls with production
compiler flags. Full-model and request-level qualification are recorded in
[quality](QUALITY.md#llvm-selector-scheduling-october-23-2026); the microbenchmark
percentage is not an end-to-end speedup.

An independent `tools/prof/prof.py` capture of the final pp2048 pass at d32K
retains all 1,574 launches, including 45 scorer calls. Scoring falls from
41.25 to 25.86 ms; total GPU time falls from 1203.93 to 1189.16 ms and the
pass span from 1207.74 to 1192.95 ms. These are diagnostic profiled timings;
the HTTP table uses separate unprofiled production runs.

## Exact sampled decode (October 21, 2026)

| Experiment | Outcome |
| --- | --- |
| Skip a redundant neural proposal after an established sampled copy | Retained only with an accepted source, a full 24-token match and at least three available copies. New sources and short resume matches still use the neural proposal. Target rejection/correction, private history and restored RNG remain authoritative. |
| Compact sampled target verification | Retained for unconstrained top-k/min-keep up to 64 and bounded penalty history. Transfer the exact top 64 unpenalized logits plus every penalized logit, then apply the existing FP64 target sampler. Grammar activation, forced reasoning closure and unsupported settings use full rows. Independent CPU sorting and full-row distributions agree bit for bit, including correction RNG. Complete transfer/selection/sampling costs for 1/3/7 rows: 94.60/273.17/594.92 → 75.84/156.63/323.45 µs. Reconstructing full CPU rows first was exact but slower and was removed. |
| KV-only predictor catch-up when no proposal is needed | Retained: scalar and uniform cache-only cohorts stop after K/V and indexer updates; mixed cohorts preserve the original row arithmetic. No discarded attention/output/FFN residual is produced in the uniform path. Independent predictor checks and full-logit/RNG/cache replay pass. |
| Prepare image-capable slots' stable RoPE descriptor before text graphs | Retained after a cancellation/resume control exposed a repeatable 5.8 ms first-token regression. Admission profiling locates 5.61 ms in image-context setup; the first image descriptor invalidated previously captured text graphs. Preparing the empty descriptor at session creation keeps its pointer stable. The same resumed request falls from 9.45–9.46 to 3.71 ms TTFT (main 3.62–3.78), with identical output and faster decode. Other Qwen models retain their existing lazy initialization. |
| Recalibrate deterministic sampled draft lengths | Rejected after auditing depths 0/4096/32768, C1/2/4/6/8 and widths 1–8 with the actual small sampled head. New-code generation improved 75.27 → 81.57 tok/s, but thinking fell 51.90 → 49.18 and the focused repeat reproduced 51.67 → 49.04. That global replacement was not retained. |
| Pack Q8 output waves and vary Q4 expert rows per wave | Rejected. Q8 variants were flat/slower after matching production allocation and rotating weights beyond MALL. Q4 one-row waves changed result bits for a sub-1% gain; exact four/eight-row variants were slower than the existing two-row path. No production GEMV changes retained. |
| Wider lookup blocks, lookup without a sidecar and Flash-Next DFlash2 | Not retained. Wider blocks need independently qualified decode arithmetic and recurrent rollback beyond eight rows; the AR-only path currently has no speculative reserve. The available DFlash2 fork still needs an audit of its exported proposal probabilities and restore/batch fallback before a lossless port. These are separate implementation projects, not enabled features. |

Matched serving gains are modest: about 1–2.5% for copying and flat ordinary
coding/thinking. See [the focused comparison and quality limits](QUALITY.md#exact-sampled-decode-october-21-2026).

## Mid-size prefill passes (October 21, 2026)

| Experiment | Outcome |
| --- | --- |
| Wide prefill routes for every prefill pass above eight rows | Retained: the fused SSM projection/convolution, fused attention projection, half-only HC mix, last-layer query skip and paired 128-row gate/up with 64-row down tiles started at 1,024 rows, so pp1023 ran at 1,432 tok/s against 1,624 for pp1024. Native AR pp, before → after: pp9 93.5 → 98.3, pp33 229.8 → 243.7, pp128 541.7 → 594.1, pp256 802.5 → 899.3, pp470 1,054.4 → 1,225.5, pp768 1,216.7 → 1,446.9, pp1023 1,432.2 → 1,578.7; pp1024+ unchanged. The paired experts supply most of the gain (pp470 1,082 → 1,225). Every row's prefill logits are byte-identical to main at 30–845 tokens and for 2,518 tokens split at 2,048; AR/MTP tg64 after 33/470/900-token prefixes keeps output hashes and draft counts. Decode and verification batches keep their routes. HTTP, pi system prompt on a new prefix (2,048 + 477-token passes), alternating arms: TTFT 1,745–1,756 → 1,684–1,696 ms; a 2,968-token single pass is unchanged. |

## Component-cache admission (October 20, 2026)

| Experiment | Outcome |
| --- | --- |
| One rank per eligible slot, checkpoint-presence queries and one cold reset | Retained: the 128-checkpoint CPU fixture measures 107.05 µs for a full lookup and 31.49 µs for presence. All six HTTP profiles pass correctness, with exact responses and cache/prefill work. Native AR/MTP replay and d0/C1–C8 controls retain quality. Per-request timings remain partly inconclusive; see [qualification](QUALITY.md). |
| Share completed full-prompt retry captures across peers | Rejected under the existing phase gate. The identical cohort avoids three 133,129,584-byte copies, and sampled MTP replies replay exactly, but larger physical decode batches repeatedly add 7–9 ms to some decode phases. Total request time improves; eight phase measurements fail. The default fix preserves capture behavior. |

## Host compaction and cache transfers (October 18, 2026)

KFD SMI events and HIP traces distinguish queue suspension from slower model
kernels. Host page migration can suspend every GPU queue in the process,
including text work unrelated to the allocation that moved. Controls against
main `bff0eec6` also exposed smaller delays from preparing cache backing while
capturing a checkpoint.

| Experiment | Outcome |
| --- | --- |
| Nonpaged gfx1151 queue-save areas, launch/copy arenas and loader staging | Retained in the bundled ROCr patch. HIP can obtain kernel arguments from the coarse pool, bypassing ROCr's dedicated kernarg pool; traces also locate runtime-owned code-object upload buffers. Application SVM and ordinary fine-grained host allocations retain their policies. |
| Long-term pins for owned HIP host buffers | Retained for uploads, Flash-Next execution and disk staging. Exact GPU/CPU visibility and restricted-policy fallback tests pass. |
| Vision device weights and bounded pinned uploads | Retained with existing heap headroom and managed fallback. All 29 observed vision stages remain byte-identical. |
| Bounded cache growth outside captures | Retained: approximately 128 MiB segments, bounded page population and pinning ranges, a 2 GiB reserve before readiness/substantial prefill chunks, and smaller reserves for short prefill chunks. Preparation follows any pending fallback capture. Existing budgets remain authoritative. |
| Prepare cold HC, Q8 and argmax functions | Retained: resolve actual function variants before readiness, without trial launches or activation changes. |
| Aligned snapshot upload kernel | Retained: use the existing exact byte-copy kernel for registered host-to-device restores, preserving unaligned, pageable and borrowed-device fallbacks. |
| Default `HSA_USERPTR_FOR_PAGED_MEM=0` | Rejected: removed eviction events but slowed short prefill by 5–6%. The narrower runtime fix preserves normal prefill speed. |
| Extra priority transfer queues | Rejected: two short C4 decodes rose from approximately 46/45 ms to 104/58 ms. Keeping the original shared queue pool removes this regression. |
| Grow backing during each allocation or every prompt capture | Rejected: background population/registration delayed captures and the following decode. Request-release refill also delayed the next admission. Preparation now follows pending fallback captures, immediately before actual prefill work. Fully cached decode does not start it. |
| Disable all ROCr SVM | Rejected: faults when a kernel writes application-managed storage. The independent memory test covers this failure. |

See [quality and timing controls](QUALITY.md) for measured scope and remaining
limitations. Raw requests, profiles and rejected controls remain outside Git.

| Experiment | Decision / evidence |
| --- | --- |
| Full-width MTP RMSNorm and split projection | Retained after independent CPU stage audit; one 10240-wide normalization, embedding projection shared across HC branches. |
| Full Q8 vocabulary head | Retained; private Q4 shortlist removed. Sampled top-64 proposals use exact target verification. |
| Per-request MTP draft vocabulary with the existing length controller | Retained: C1 sampling, including both real default presets, scores 98,304 prefix rows plus a 1,024-row table with vocabulary fallback. Greedy C1 and jointly planned greedy batches without penalties/constraints also use the subset. Every request retains its existing length controller and AR backoff. Standard d0/d32K greedy MTP gains 5–12%; default Python/story gains 3–10%, and two 35K default Python continuations gain 5–8%. Target logits are byte-exact; sampled wording can change. A non-default translation slowdown and cache timing outliers remain in the [review evidence](artifacts/draft-vocabulary-review.json). The extra 2,793,472 bytes/speculative session are reserved; tables rebuild from committed history after restore. |
| Confidence-based draft stopping | Rejected in review: an exact-output C4 code request regressed from 5.37 s to 6.14 s, and maximum token gaps increased. C1 sampled changes shifted slowdowns between requests. Even limiting greedy confidence to the existing budget slowed a 16-token request from 286 ms to 322 ms; removing that gate while retaining the small head takes 260 ms with identical output and draft acceptance. The new sampled calibration also broke native snapshot replay because its state was not persisted. Retain the established length policy for both greedy and sampled requests. |
| Batched MTP transformer and heads | Retained; independent body/head comparisons, private KV/recurrent/rollback/RNG state. |
| Batched decode mixers, residual epilogues and MTP norms | Retained; each request keeps its scalar reduction and private state; exact C2/C4/C6/C8 logits, acceptance and RNG. |
| HC Q8 weight prefetch | Retained for 1–8 rows of the 320×10240 projection; exact original products/FMA order, no extra allocation. |
| Q4 shared-expert weight reuse | Retained with a separate compact kernel for single-request experts; faster repetitive C1/C4/C8, no mixed-work regression, exact projection/model replay. |
| Q5 high-bit expansion | Retained; exact integer multiply/mask replaces repeated shifts, with unchanged dot products and faster serving. |
| Short convolution/history fusion | Retained for 1–8 tokens; exact output, rolling state and rollback snapshots, fewer launches and no additional allocation. |
| Batched GDN recurrence | Retained; private ragged state/rollback rows, cancellation isolation and exact session replay. Helps shallow batches most; end-to-end gains are modest. |
| Batched small projections and MoE preparation | Retained; group independent rows, quantize activations once in existing scratch, and batch router/shared-expert work. Exact scalar/batch outputs and sampled state; C1 unchanged. |
| Q4 expert grouping across the full batch | Retained; bounded groups share weights across request boundaries, with original scalar arithmetic and exact session replay. Q5 down grouping/reordering was slower on mixed routing and was rejected. |
| Expert-ordered prefill intermediates | Rejected: exact gate/up → down results with captured routing at 1024/2048 tokens, but no useful projection-chain gain for Q4_K/Q5_1 or Q5_K/Q8_0. |
| Wave64 batched Q4 gate/up | Retained above eight rows; independent 32-lane reductions preserve exact outputs and sampled state. Repetitive serving improves, C1 remains stable. Four output rows, 16-input groups and replacing the slot scan did not improve performance. |
| Register-cached vision softmax | Retained for bounded row sizes; unchanged reduction order, byte-identical Flash-Next/Qwen27B embeddings, no additional allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical GEMMs/embeddings, lower latency and 24 MiB less attention scratch. Other shapes keep their original tiles. |
| Partitioned BF16 WMMA vision value projection | Rejected: failed the full-encoder reference gate despite lower isolated FP64 error. |
| Integer WMMA Q8 verification | Retained through 48 input rows for wide target projections/heads; preserves K8 partials/FMA order and reduces each result once, with exact session replay. Small batches retain vector kernels. |
| Q8 activation reuse across output rows | Rejected: no repeatable gain on the real projection shapes. |
| Wider Q8 decode and Q5 expert tiles | Rejected: 56–64 dense rows, phased/packed token tiles, Q5 wave64 and 16/32 Q5 expert rows were slower despite exact output. |
| Compact expert launch groups | Rejected: improved shared routing but negligible mixed-routing gain. |
| Sparse attention register/cache retuning | Rejected: exact d128K output, but register scheduling/occupancy gave no material gain and reloading queries was slower. |
| Shared attention selection lists | Rejected: exact outputs, but roughly 1% isolated gain did not justify another buffer and setup kernel. |
| Skip unused attention Q8 staging | Rejected: exact chunk/full-logit checks passed, but no clear end-to-end prefill gain justified the extra dispatch logic. |
| FP32 selector load scheduling | Retained; bounded scheduling removes scalar-register spills, preserves every score bit and lowers d32K selection time to 22.4 ms per pp2048. Matched d128K AR prefill improves about 1.5%. |
| Integer WMMA value transpose and paired FP32 selector lanes | Rejected: bit-preserving transpose and exact selector scores, but both were slower on deep-context inputs. |
| Packed Q8 prefill staging | Rejected: exact output, but extra decode/register/transpose costs outweighed reduced LDS use. |
| Persistent packed Q8 SSM weights | Rejected: small prefill/1–4-row gains would cost roughly 14% on eight-row decoding; two-step prefetch did not recover it. No second weight copy or private format retained. |
| Transient F16 SSM weights and parallel HC branches | Rejected: F16 staging was exact but slower overall; parallel HC branches changed quantization ties. |
| Smaller-LDS SSM projection, unrolled HC expert sum and wave64 HC combine | Rejected: exact output but no useful prefill speed gain. |
| HC quantized-output store layouts | Rejected: row-major staging plus transpose is slower; cooperative stores within one kernel save only about 2% in isolation. Residuals, normalized activations and Q8 bytes remain exact, including ragged rows. |
| Transient key transpose and mixed expert tiles | Rejected: exact outputs; key transpose slows deep selection, mixed tile sizes provide no useful prefill gain. |
| Query sharing, paired-key FP32 scoring, query LDS caching, MoE prefetch barriers/unrolling | Rejected: exact outputs, but no useful speed gain. Four-wave Q8 matrix reduction also lost to eight waves. |
| Approximate selector screening followed by exact rescoring | Rejected: GPU thresholding and compaction erased the isolated gain, before accounting for runtime error bounds. Remains a synthetic experiment; no approximate production selector added. |
| Ratio-four predictor QSA, FP32 ranking queries | Retained with sparse state, rewind and deep selector checks. |
| Greedy batch cost controller | Retained only for all-greedy C>1; separate occupancy/context bins, stable plain controls, no transition timings. Sampled replay uses fixed curves calibrated from median warmed cycles on 2026-09-20. |
| One-row MTP prefill lag | Retained; 320 KiB kept hidden state/session, avoids replaying a final prefill chunk. |
| Final-tile MTP catch-up | Retained; preserve every KV/indexer row, rank only the final attention tile, then restrict output projection, mixers and shared experts to the final aligned tile(s), with one routed Q8 expert row. Full predictor/catch-up stages, candidates and recursive carry match exactly at 224/257/2047/2048 rows. |
| Batched MTP catch-up tail | Retained; after writing every attention cache row, compact final request rows into existing scratch and skip unused output projections/FFNs. Preserve the scalar arithmetic of one-row tails; full-head candidates and recursive carry match independent execution. |
| Routed Q8 accumulation order | Retained; explicit rounded product/FMA prevents identical rows changing with column placement. Independent FP64 dot and predictor carry checks pass. |
| Isolated MMQ quantizer rounding change | Deferred: it changes half-integer tie behavior shared with W8A8 and fails their existing agreement gate. No quantizer change retained. |
| SSM tile, wave and compiler scheduling variants | Rejected: four/sixteen-wave groups, wave64, wider token tiles and iterative ILP scheduling retained exact output but did not beat the existing eight-wave projection. |
| SSM fragment lifetime and prefetch pipeline | Rejected: shorter-lived K16 fragments reduce register use without a useful gain; delayed prefetch and double-buffered LDS are slower. Projection/convolution outputs remain exact at 2048/2049 tokens. |
| Captured shared stages in batched target decoding | Rejected: exact C2/C4/C6/C8 logits, sampled state and cancellation checks, but no serving or warmed-cycle gain justified graph metadata/capture overhead. |
| Compact recurrent rollback | Retained; one full state plus exact FP32 update operands, with the original product/FMA order. All rollback prefixes match fresh execution. Depth grows on demand; seven-draft cap about 147 MiB/session, released on reset. |
| Live final frontier and async prompt snapshots | Retained; immutable branch snapshot, worker capture, bounded persistence outside the lookup lock. |
| Thread-local decode graph capture | Retained; independent snapshot workers no longer invalidate a peer's capture. Snapshot bytes and captured/replayed logits are exact; no request serialization added. |
| Chunk-equivalent projections/attention | Retained with exact continued-image/cache/full-logit gates; one-token tails keep prefill arithmetic. |
| More Q8 vocabulary rows/block | Rejected: no C1 improvement. |
| Private shared-expert F16 rows | Retained; the shared expert's SwiGLU rows for its F16 down projection get their own buffer instead of overwriting the routed experts' narrowed token rows, which removes one full-batch narrowing pass per layer. Every GEMM reads identical bytes; greedy output, chunk-boundary logits and the C2–C8 batch checks are exact. Interleaved Nix A/B at d0: 1559.6 → 1568.1 tok/s (+0.6%); the profiled kernel time fell 1362.8 → 1333.1 ms. |
| 256-row routed down-projection blocks | Rejected: exact output, Q5_1 unchanged and Q8_0 slower. The routed kernels already stream expert weights at roughly two thirds of DRAM bandwidth, so per-block prologue/epilogue and activation restaging were not the bound. |
| 64-token HC down-projection tiles | Rejected: exact output, 23% slower than the 128-token tile despite twice the resident blocks. |
| One-tile-ahead LDS fragment prefetch (dense and routed WMMA) | Rejected: exact output, but the explicit double buffer raised register use (dense 221 → 253 VGPRs; the 128-token pair reached 256 with spills) and every GEMM slowed 3–14%. |
| hipBLASLt F16 dense projections | Rejected: the pinned library reaches 19–26 TFLOPS on the 2048-token dense shapes against 30–34 TFLOPS for the Q8→F16 WMMA kernels; `tools/qwen-flash/dense_blaslt_sweep.hip` reproduces the sweep. |
| Side-stream inject/shared-expert overlap | Rejected: exact output and real kernel overlap in the trace, but the co-running kernels slowed each other and interleaved wall-clock runs were 0.5–0.9% slower. |
| Q4_1 decode weight copies (`GUFO_DECODE_COPY`, default `balanced`) | Retained from the mikealanni fork: large Q8_0 dense projections get a least-squares-refit Q4_1 copy (1.58 GiB here) that serves the 1–8 row decode GEMV, so decode reads 20 instead of 34 bytes per 32 weights; prefill and wide batches keep the Q8_0 WMMA path and the Q8_0 weights stay resident. Interleaved A/B at d0: tg128 30.18/30.24 → 35.64/35.69 tok/s (+18%), pp2048 1636/1644 → 1650/1651, greedy text identical, load unchanged. Decode numerics change; `GUFO_DECODE_COPY=off` restores the exact Q8_0 path. Qualifier at depth: on the native bench's degenerate repetition the changed numerics flip the greedy trajectory at d16384, which starves prompt lookup (28 proposals instead of 102, 56.93 vs 78.40 tok/s against upstream 0.24.0); `off` reaches 89.05 there but costs 6.8% at d32768 on token-identical output, so the default stays on. |

Separate d32K pp2048 profiling attributes 29.1% of kernel time to MoE, 34.9%
to dense projections and 12.6% to attention/indexing. Final-tile catch-up
takes 20.9 ms of MTP kernel time; the target takes 1450.2 ms.

A d0 pp2048 profile (2026-09-21) is GPU-bound: 1362.8 ms of kernel time in a
1380 ms span. Routed expert GEMMs take 35%, dense F16 projections 29%,
hyper-connection combines 10.5%, the GDN recurrence 9.5%, HC down/inject/shared
projections 7% and attention 5.5%. The dense projections run at 30–34 TFLOPS
against the measured 59 TFLOPS matrix ceiling and the combines at about
200 GB/s; the routed gate/up and down kernels stream 0.9–1.1 GB of expert
weights per layer at 160–190 GB/s.

A C4 mixed MTP trace with 32 output tokens per request is 84.9% GPU-busy
during inference, excluding model loading. Batched GDN updates take 40.9 ms,
rollback replay 14.5 ms and lazy allocations 31.8 ms. The scheduling thread
spends 5.4% of the window outside HIP API calls, including model-side CPU
work; this is not pure scheduler overhead. These are profile observations,
not unprofiled throughput measurements.

Next: improve prefill at depth and target/draft batch projection reuse while
preserving [quality](QUALITY.md). The 1700 tok/s PP and flat d0–d128K
objectives remain unmet; see [current benchmarks](BENCHMARKS.md).
