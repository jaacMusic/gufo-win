# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP (direct `gufo bench` for
the long-context sweep), greedy, thinking off.
Gufo single-user MTP, concurrency, loading and llama.cpp MTP at 64K/128K:
September 23, 2026; other results: September 22; direct 8K-256K sweep:
October 28, 2026; bubo comparison: October 21, 2026.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1628.52 | 489.59 | +232.6% | 26.04 | 22.20 | +17.3% |
| 4,096 | 1523.39 | 455.24 | +234.6% | 25.98 | 21.21 | +22.5% |
| 8,192 | 1499.13 | 428.85 | +249.6% | 25.91 | 20.40 | +27.0% |
| 12,288 | 1477.98 | 400.21 | +269.3% | 25.46 | 19.64 | +29.6% |
| 16,384 | 1457.16 | 375.89 | +287.7% | 25.20 | 18.95 | +33.0% |
| 32,768 | 1421.93 | 301.31 | +371.9% | 24.33 | 16.54 | +47.1% |
| 65,536 | 1304.01 | 221.94 | +487.6% | 23.21 | 11.62 | +99.7% |
| 131,072 | 1292.02 | 144.78 | +792.4% | 21.93 | 7.98 | +174.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1602.82 | 468.76 | +241.9% | 32.18 | 31.73 | +1.4% | 59.41 | 48.12 | +23.5% |
| 4,096 | 1506.32 | 423.79 | +255.4% | 34.57 | 34.48 | +0.3% | 47.92 | 45.71 | +4.8% |
| 8,192 | 1492.95 | 394.98 | +278.0% | 34.97 | 32.20 | +8.6% | 50.02 | 44.31 | +12.9% |
| 12,288 | 1488.03 | 365.91 | +306.7% | 35.18 | 32.32 | +8.8% | 39.95 | 43.53 | -8.2% |
| 16,384 | 1471.81 | 341.98 | +330.4% | 33.84 | 32.07 | +5.5% | 50.55 | 43.66 | +15.8% |
| 32,768 | 1449.56 | 277.61 | +422.2% | 30.67 | 25.41 | +20.7% | 46.81 | 39.28 | +19.2% |
| 65,536 | 1316.70 | 208.04 | +532.9% | 31.78 | 19.65 | +61.7% | 45.12 | 27.75 | +62.6% |
| 131,072 | 1335.91 | 135.42 | +886.5% | 28.12 | 14.63 | +92.2% | 42.32 | 20.96 | +101.9% |
<!-- /bench -->

The repetitive 12,288 cell is a prompt-modelling effect, not an engine
regression: the bench logs report "ended its turn after 12 of 128 tokens", so
the remaining tokens are post-turn free text that drafts cannot predict, and
the autoregressive rate stays flat across the same depths (33.08/32.94/32.50
tok/s at 8,192/12,288/16,384).

![Single user, MTP](artifacts/charts/single-mtp.svg)

### Chat-prose decode

The mixed column above comes from an external prompt suite that is not in this
repository, so it cannot be regenerated here. `gufo bench` has no
mixed/repetitive switch, and its own corpus is semi-repetitive, so its drafts
are accepted far more often than chat text would be. One ~400-word prose prompt,
single session, `-n 256`, Windows release build:

| Configuration | tok/s | vs AR | Accepted/drafted |
| --- | ---: | ---: | ---: |
| AR, greedy | 35.22 | — | — |
| MTP, greedy | 44.5–45.2 | +26 to +28% | 102/152 (67%) |
| AR, sampled `-t 1.0` | 32.08 | — | — |
| MTP, sampled `-t 1.0` | 41.98 | +30.9% | 132/219 (60%) |
| MTP, `--top-k 20 --top-p 0.8` | 44.00 | +37.2% | 122/247 (49%) |

`--draft-tokens` only caps the proposals: `-d 3`, `-d 5` and `-d 7` produce
identical cycles and counters, because the length controller picks the width
from its acceptance estimates and the measured cycle costs in `mtp_costs.hpp`.
At 60–67% acceptance it settles on one or two drafts per cycle. Sidecars:
`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` does not load (`blk.48.hc_attn_down.weight`
has no Q4_K support in the sidecar loader); shared-Q8_0 and full-Q8_0 measure
within 1%, so the shared head is the right default.

## Long-context sweep, direct

`gufo bench`, greedy, one process at a time (the 96 GB carve holds one Q4 model
and its state). Unsloth `UD-Q4_K_XL` Dynamic v3.1 (revision `d9620119`, local
repack directory) + shared-Q8_0 MTP. Two timed repetitions after one
untimed pass; full sweep re-measured November 8, 2026 (within single-run
noise of the October 28 `38bb39ee` values, as expected for a repack with
identical quantization bytes). Prefill and AR decode use an autoregressive session; MTP decode
uses adaptive MTP (7 draft tokens, minimum 1).

The 256K row prefill is 262,143 tokens and decode depth is 262,015: the
262,144-token native context needs the extra slot for the next token and 128
generated tokens. Other rows use the exact context.

<!-- bench:context-sweep -->
| Flash-Next Q4<br>Context (tokens) | Prefill pp (tok/s) | Decode AR (tok/s) | Decode MTP (tok/s) | MTP accepted |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1695.47 ± 9.80 | 32.57 ± 2.11 | 51.58 ± 1.80 | 66% (81/122) |
| 32,768 | 1642.09 ± 2.41 | 32.22 ± 0.03 | 99.39 ± 2.42 | 100% (112/112) |
| 131,072 | 1579.40 ± 0.60 | 29.54 ± 0.05 | 94.59 ± 1.97 | 100% (112/112) |
| 262,144 | 1508.00 ± 0.60 | 27.35 ± 0.09 | 89.40 ± 1.52 | 100% (112/112) |
<!-- /bench -->

The bench prompt repeats a 20-token pattern. From 32K the model continues that
pattern verbatim, so MTP accepts almost every draft and the MTP column is an
upper bound for repeated text, not chat or code. The mixed-prompt MTP table
above (28-35 tok/s at depth) is the realistic decode range; the AR column is
the no-speculation reference.

`start.cmd` additionally enables survival stopping, the Latin draft vocabulary
and prompt lookup, and samples at temperature 1.0. `gufo bench` cannot set those
switches, so served rates differ; measure those over HTTP with
`tools/serving/gufo-serving-bench.py`.

```powershell
# Prefill C new tokens, then decode at depth C (AR), then with MTP.
# C = 8192, 32768, 131072, 262144; use C-1 and C-129 at 256K.
gufo bench -m $model -p C      -n 128 -d 0 -r 2 --temperature 0
gufo bench -m $model -p 1      -n 128 -d C -r 2 --temperature 0
gufo bench -m $model -p 1      -n 128 -d C -r 2 --temperature 0 `
  --speculative mtp --mtp-model $mtp
```

`tools/windows/bench-flash-next-context.ps1` runs the whole sweep sequentially
and writes one log per run to `artifacts/bench-context/`. One-page report:
[bench-context.html](bench-context.html).

Run notes for future sweeps:

- Run one benchmark process at a time. The 96 GB GPU carve leaves no room for
  two Q4 instances. Model loads observed here took 47-156 s, so a full sweep
  costs tens of minutes even though the timed runs are short.
- `-p` is new prefill work; `-d` is cached depth. Bench sizes the context as
  `max(4096, depth + prompt + 1, max(depth, 16) + generation + 1)`. At the
  native limit use prompt 262,143 and depth 262,015 with `-n 128`.
- Flash-Next benches only support concurrency 1.
- The checkpoint declares 262,144 context tokens; `start.cmd` serves 131,072 by
  default. The sweep raises the server context per run.
- About 26.8 GiB of the 103.7 GiB checkpoint is the PLE n-gram table, streamed
  from disk. Roughly 76.9 GiB stays resident, which is why 256K state fits the
  96 GB carve.
- The script picks the newest Hugging Face snapshot that has both the first
  `UD-Q4_K_XL` shard and the shared-Q8_0 MTP sidecar; override with `-Model`,
  `-Mtp`, `-Sizes` or `-Repetitions`. Use `-Exe` for another engine binary, for
  example `-Exe C:\dev\bubo\bin\bubo.exe -OutDir
  artifacts\bench-context-bubo`.
- 8K MTP acceptance (64%) is the only row below 90%. From 32K the fixed bench
  pattern is repeated verbatim, so the MTP column is a repeated-text upper
  bound, not chat or code throughput.

### Bubo comparison

Bubo `0.1.2` (revision `53e4c3e4b`, engine gufo 0.26.0) against Gufo-win at
`5377e35e`, both measured 2026-11-08/09 on the same bench command: greedy, one
process at a time, two timed repetitions. Bubo 0.1.2's stock loader rejects the
Dynamic v3.1 local repack snapshot (`d9620119`, "Malformed or truncated GGUF"),
so its rows use the `38bb39ee` snapshot — identical quantization bytes, within
single-run noise of the repack. The 256K row uses the same 262,143 prompt and
262,015 depth. Bubo 0.1.2's `bench` runs MTP with prompt lookup and the auto
draft vocabulary (the serve defaults), so the MTP column matches Gufo's bench
defaults.

<!-- bench:bubo-context-sweep -->
| Flash-Next Q4<br>Context (tokens) | Bubo pp | Gufo pp | Bubo AR | Gufo AR | Bubo MTP | Gufo MTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1657.78 ± 15.50 | 1695.47 ± 9.80 | 28.64 ± 0.31 | 32.57 ± 2.11 | 44.37 ± 0.81 | 51.58 ± 1.80 |
| 32,768 | 1588.13 ± 3.38 | 1642.09 ± 2.41 | 28.61 ± 0.06 | 32.22 ± 0.03 | 95.68 ± 2.48 | 99.39 ± 2.42 |
| 131,072 | 1519.72 ± 0.53 | 1579.40 ± 0.60 | 27.82 ± 0.01 | 29.54 ± 0.05 | 91.53 ± 2.41 | 94.59 ± 1.97 |
| 262,144 | 1451.21 ± 0.41 | 1508.00 ± 0.60 | 26.94 ± 0.01 | 27.35 ± 0.09 | 86.72 ± 1.82 | 89.40 ± 1.52 |
<!-- /bench -->

All values are tok/s. Against Bubo 0.1.2, Gufo prefill is 2.3-3.9% higher, AR
decode is 1.5-13.7% higher (13.7% at 8K; treat that row loosely — Gufo's 8K AR
ends the turn after 28 tokens), and MTP decode is 3.1-16.3% higher. Bubo 0.1.2's
auto draft vocabulary lifted its 8K MTP from 34.97 (0.1.1) to 44.37, closing
part of the 8K gap; every other cell moved within ~1% of 0.1.1 (pp 1661.34 →
1657.78, 96.36 → 95.68 at 32K, 86.84 → 86.72 at 256K). Bubo's serving launcher
also enables survival and vision; these direct bench numbers do not. Logs:
`artifacts\bench-context-bubo-012` and `artifacts\bench-context` (git-ignored).
The 0.1.0 comparison of 2026-10-21 showed the same ranking with weaker Bubo
numbers (Bubo pp 1293.86-1540.60, MTP 33.96-67.36).

### Bubo 0.1.1 cached first-token claim

Bubo's release notes claim a follow-up request's first token arrives in 527 ms
at 32K context and 515 ms at 100K, with queue wait falling to 7-8 ms. Measured
here with both engines on the same server path (`/v1/chat/completions`,
sessions 1, context 131072, greedy, 16-token replies, same filler prompt and
snapshot; one cold pass then three cached repeats). Token counts differ slightly
between engines (32542 vs 32584, 99259 vs 99301) because the tokenizers differ;
output was byte-identical cold vs cached on both. Gufo rows are from the
2026-11-08 build (tokenization memoization, resident-row reuse and the pinned
payload pool); the Bubo rows are the 2026-11-01 measurement.

| prompt chars | Gufo cold | Gufo cached TTFT | Bubo cold | Bubo cached TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 168,800 (~32K) | 20,449 ms | 9-12 ms | 20,572 ms | 8.6-9.0 ms |
| 515,000 (~100K) | 65,026 ms | 11.6-11.8 ms | 46,542 ms | 102-113 ms |

Gufo leads the cached first token at 100K by about 9x and matches Bubo at 32K
(9-12 ms against 8.6-9.0 ms). Bubo 0.1.1's keep-alive path serves a resident
session directly (queue 7-18 ms, matching its claim), so its cached turn never
touches a snapshot. Gufo restores per request; for an exact repeat the request
returns to the session that wrote the checkpoint, so only the ~240 MB of mutable
recurrent state is copied and the resident K/V rows stay in place
(`Executor::RestoreSnapshot` with `reuse_resident_rows`, gated on session
identity and epoch). That took 32K from 57-66 ms to 21-23 ms and 100K from
152-170 ms to 27-30 ms; the earlier 113-132 and 363-373 ms figures were
pre-memoization and, at 100K, a second-repeat outlier. Holding the snapshot
payloads in a pinned-host pool (2026-11-08) then took 32K to 9-12 ms and 100K to
11.6-11.8 ms. One cached repeat out of three still pays a 70-280 ms queue
transition at both sizes (80.4 and 203.1 ms above in the 2026-11-08 runs); that
time is outside `cache_restore_ms` and is not measured yet. Harness:
`%TEMP%\opencode\ttft-claim.py`.

Warm-turn anatomy (2026-11-02, `--sessions 1`, MTP on, same harness with
`cache_snapshot_ms`/`cache_shared_prefix_ms` read out): a follow-up turn that
appends 47 tokens spends almost all its wait on checkpoint capture, not on
prefill or restore. At 32K the turn measured 615 ms TTFT = 17.7 queue + 17.0
restore + 253.8 own-boundary snapshot (2.03 GB D2H) + 133.5 shared-prefix
checkpoint (1.01 GB D2H) + ~30 ms of actual delta prefill; at 100K it was
1,164 ms = 26.8 + 25.8 + 650.3 (own) + 399.0 (shared) + ~60 ms prefill. The
captures are the fixed per-turn cost that earlier delta sweeps measured as
~770 ms at 32K; they scale with payload bytes, not delta tokens. Running with
`--speculative off` leaves the prefill wait unchanged (479 ms for the same
47-token delta), so MTP catch-up is not a factor. Snapshot payloads now come
from a 3 GiB pinned-host pool (`hipHostMalloc` in `SessionSnapshot`, falling
back to pageable memory when the pool is full or the allocation fails), so the
restore's host-to-device copy reads resident pages instead of faulting 4 KiB at
a time: restore fell to 8.5-11.8 ms and the exact-repeat TTFT to 9-12 ms at 32K
(2026-11-08). An unbounded pinning attempt had cut the 32K restore to 10.5 ms
earlier, but a 100K request failed with a 500 once ~5.9 GB of pinned snapshots
were retained against 7.4 GB of available host memory; the cap keeps a 100K
payload (5.7 GB, plus its 4.0 GB shared prefix) pageable, and 100K restores in
10.7-11.4 ms. When the pool is already full -- for example after a 100K session
is retained -- a later 32K payload stays pageable and restores in ~17 ms. The
next real win is skipping the shared-prefix capture when this request's own
stable-boundary snapshot already retains a prefix within
`shared_prefix_min_tokens` (128) of the divergence point: today both copies of
nearly the same 32.5K-token prefix are written every turn.

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.85 | 22.34 | +15.7% |
| 2 | 45.70 | 37.08 | +23.2% |
| 4 | 76.29 | 54.82 | +39.2% |
| 6 | 95.66 | 65.34 | +46.4% |
| 8 | 108.67 | 68.79 | +58.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.09 | 31.77 | +1.0% | 59.20 | 46.93 | +26.1% |
| 2 | 51.68 | 42.79 | +20.8% | 93.16 | 52.57 | +77.2% |
| 4 | 75.92 | 48.78 | +55.6% | 127.91 | 48.23 | +165.2% |
| 6 | 89.18 | 52.59 | +69.6% | 142.14 | 48.51 | +193.0% |
| 8 | 106.47 | 61.92 | +71.9% | 157.22 | 58.13 | +170.5% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.45 | 117.83 | +662.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Routed gated GEMV rows per block (November 22, 2026)

mul_mat_vec_moe_gated at 	okens == 1 ran mul_mat_vec_q_moe<type, 2, true>
(two rows per 64-thread block). Widening the tile to four rows halves the block
count (3200 to 1600 per layer) and doubles the dot-product ILP per warp, which
frees bandwidth the 16K KV reads compete for. outed_wmma_ops passes unchanged.
pp2048, --temperature 0, single repetitions.

<!-- bench:moe-vec-rows -->
| Tile | tg128 d0 (tok/s) | tg128 d16384 (tok/s) |
| ---: | ---: | ---: |
| 2 rows (before) | 36.01 | 30.38 |
| 4 rows | 36.17 | 33.50 +/- 0.06 (2 reps) |
| 8 rows | 35.87 | 33.30 |
<!-- /bench -->

Four rows is kept: +0.4% at depth 0 (noise), +10.3% at depth 16384, and the
8-row tile gives the gain back.

## Q2_K decode copies (November 22, 2026)

`GUFO_DECODE_COPY=q2` requantizes the plain dense decode projections
(ssm_in, ssm_out, attn_qkv, attn_out) to Q2_K instead of the default Q4_1
balanced copies: 0.82 GiB vs ~1.6 GiB of copies, read once per token.
Same command as above, Qwen3.8-Flash-Next-UD-Q4_K_XL, temperature 0.

| mode | tg128 @ d0 | tg128 @ d16384 |
|------|-----------|----------------|
| balanced (Q4_1) | 36.17 | 33.50 |
| q2 (Q2_K) | 40.56 | 36.44 |

+12.1% / +8.8% tokens/s. Greedy generations remain coherent; the Q2_K
row-batched GEMV keeps llama.cpp dot-product grouping.

Follow-up: `GUFO_DECODE_COPY=q2` +
`GUFO_DECODE_COPY_CLASSES=lm_head` keeps the LM head on a Q4_1 copy
(1.19 GiB total copies) — logits stay 4-bit — and stacks with the Q2_K
dense stack:

| mode | tg128 @ d0 | tg128 @ d16384 |
|------|-----------|----------------|
| q2 + lm_head Q4_1 | 42.45 | 37.83 |

+17.9% / +13.3% over balanced. This combined setting is what the
engine_console launchers now set.

Quality follow-up (same day): the full four-class Q2_K set (attn_out
included) eventually collapses greedy decode into repetition loops — the
2-bit noise is additive across projections and crosses a cliff only with
all four. Dropping `attn_out` (it falls back to its original Q8_0 weights;
`GUFO_Q2_CLASSES=ssm_in,ssm_out,attn_qkv` is the new default) removes the
loops on repeated 400-token greedy probes while keeping most of the speed
(1.13 GiB copies):

| q2 class set | tg128 @ d0 | tg128 @ d16384 | greedy quality |
|------|-----------|----------------|----------------|
| ssm_in,ssm_out,attn_qkv,attn_out | 42.45 | 37.83 | loops |
| ssm_in,ssm_out,attn_qkv (default) | 41.71 | 36.64 | coherent |
| ssm_in,ssm_out only | 39.49 | 33.40 | coherent |

## MTP cost recalibration (November 22, 2026)

Fresh `--cost-audit 0` sweep (default decode-copy settings, 120 rows) shows
`kMtpCycleMilliseconds` from 2026-09-20 is now ~2-3 ms/cycle conservative
(C1@0 width1: 38.4 -> 28.5 ms) after this session's kernel wins. Swapping in
the fresh table, however, made the controller draft longer chains whose
marginal acceptance fell (98/155 = 63% vs 66%), netting 44.6-45.3 tok/s on
the greedy chat probe vs 46.8 with the stale table. The conservative table is
kept deliberately; a note in `mtp_costs.hpp` records this.

Also measured: Q2_K decode copies hurt MTP (~45.4 vs 46.8 tok/s) - q2 noise
lowers draft acceptance and MTP verify is multi-token (Q8_0 path), so q2 only
helps width-1 slices. Keep `GUFO_DECODE_COPY=q2` for AR-only engines, default
(balanced) for MTP-serving engines.

## MTP cost table scale sweep (November 22, 2026)

Swept kMtpCycleMilliseconds scales on the greedy chat probe (256 tokens,
default decode-copy settings):

| Table | tok/s | cycles | drafted | accepted |
|-------|-------|--------|---------|----------|
| stale 2026-09-20 (x1.0) | 46.8 | - | - | 66% |
| x0.90 | 53.2 | 111 | 211 | 144 |
| x0.80 | 53.9 / 53.8 (2 reps) | 111 | 211 | 144 |
| raw fresh 2026-11-22 (~x0.74) | 44.6 / 45.3 | 80 | 155 | 98 |

The raw re-measurement over-drafts (marginal acceptance falls to 63%); the
stale table under-drafts and forfeits the faster kernels. Shipped the x0.80
scale (+15% MTP tok/s over stale); comment in mtp_costs.hpp records the
rationale and probe command.

## V Q8 KV cache (November 22, 2026) - dead end

Per-(token, kv_head) scaled-int8 V cache (K stays fp16), dequantized in the
AttentionKernel PV load and the WMMA tile load, stores fused into
PrepareAttentionKernel plus a StoreVQ8 path; the fused F16 prefill projection
was bypassed (it writes fp16 V directly). Bench (bench 2048/128, d0 and d16k,
temperature 0, same-process A/B):

| V | tg128 d0 | tg128 @16k | pp2048 d0 | pp2048 @16k |
|---|----------|-----------|-----------|-------------|
| fp16 | 35.84 | 33.28 | 1676.9 | 1482.3 |
| Q8 | 35.89 | 33.32 | 1666.8 | 1449.4 |

Decode is noise-level: only 11 of 47 layers are full attention (36 GDN, no KV
cache), so V read traffic is too small a slice of decode for halving it to
show. Prefill regresses (unfused projection plus quantize work). Greedy output
stays coherent. Reverted; memory-constrained builds can revisit the same
design for footprint, not speed.

## Default lm_head decode copy

`GUFO_DECODE_COPY_CLASSES` default now includes `lm_head` (Q4_1 copy; env
still overrides). Previously launcher-only. Measured 2026-11-22, bench
2048/128, temperature 0, one repetition, defaults elsewhere:

| config | MTP tg128 d0 | AR tg128 d0 |
|---|---|---|
| lm_head off (old default) | 48.8 | 36.1 |
| lm_head on (new default) | 53.6 | 37.5 |

Greedy probes coherent on story and engine prompts; MTP acceptance unharmed
(lm_head is a one-shot read, not recurrent). Logits numerics do change; opt
out with `GUFO_DECODE_COPY_CLASSES=ssm_in,ssm_out,attn_qkv,shexp`.

## Attention decode splits 8 -> 16

`kAttnSplits` (narrow-batch key-tile split blocks per attention row) raised
from 8 to 16; 32 measured worse. Bench 2048/128, temperature 0, defaults:

| splits | tg128 d0 | tg128 @16k | MTP tg128 d0 |
|---|---|---|---|
| 8 | 37.2 | 34.0 | 53.6 |
| 16 | 37.9 | 34.9 | 53.5 |
| 32 | 37.7 | 34.6 | - |

Bitwise-identical merge math (same kernel, more partial blocks); prefill
and MTP unaffected.

## MTP cost table rescale 0.80 -> 0.72

The lm_head Q4_1 copy and 16-way attention split made draft cycles cheaper,
so the controller's cost table was rescaled by another 0.90. Bench
2048/128 MTP, temperature 0:

| scale | tg128 d0 | tg128 @16k |
|---|---|---|
| 0.80 | 53.5 | 43.4* |
| 0.90 (of prior, = 0.72 raw) | 54.5 | 44.5 |
| 0.95 | 54.8 | - |
| 0.85 | 54.3 | 44.0 |

*interpolated neighbor. d0 optimum is flat 0.90-0.95; 0.90 wins at depth.
Greedy engine-explanation probe coherent.

## Draft-head Q2_K proxy built with the canonical requant kernel

The screening proxy was quantized by a one-off kernel (former
`draft_head.hip.cpp`) that fit each group by min/max without refit; the
canonical `RequantQ8_0ToQ2_K` (same byte layout) refits the group scale by
least squares, as llama.cpp `make_q2_k_quants` does. The duplicate kernel is
deleted; `Executor::Create` now calls the canonical one (one-shot at load).

Better proxy ranking -> the exact rescore sees more of the rows the Q8 head
would have won with -> higher MTP acceptance. `tools/serving/agent-session.py`
4 turns x 64 tokens, port 8888 production command, 2 reps each:

| proxy quantizer | median tok/s (rep1/rep2) | weakest turn |
|---|---|---|
| one-off min/max | 49.5 / 50.1 | 35.4 / 35.9 |
| canonical LS refit | 52.3 / 53.5 | 40.9 / 39.2 |

~+6 % median, +14 % on the lowest-acceptance turn. Screened picks are still
rescored exactly, so output values are unchanged; only which rows get scored.
Depth check, greedy MTP `-p 1 -n 128 -d 16384 -r 2`: 95.75 ± 3.36 tok/s with
the canonical proxy vs 96.89 ± 2.25 with the one-off kernel, i.e. no depth
regression from the better fit.

## Screened logits sealed below the worst exact rescore (November 22, 2026)

After `Executor::ScreenHead` scattered the 64 exact rescoring results,
`Argmax` and the second `MtpTopCandidates` pass still read ~130k rows holding
raw 2-bit proxy values; a row the screen missed could outrank every exact
score and win the argmax or enter the candidate list with a proxy logit.
`ScreenCap`/`CapLogits` now cap all unscreened rows one below the smallest
rescored value before the scatter, so only exactly-scored rows can be picked.

Correctness fix, no measurable cost (two ~0.5 us kernels per draft position):
same harness, seal on vs off: 51.0 / 53.3 vs 52.3 / 53.5 median tok/s (within
run-to-run noise). Covered by `qwen38_flash_next_mtp_ops_test` CheckScreenSeal.

## Screen rescore 64 -> 128 rows (November 22, 2026) - no keep

Rescoring 128 proxy-ranked rows instead of 64 costs ~1 MB per draft position
and can only add rows a 2-bit proxy misranks. Measured flat: 52.8 / 52.0 vs
51.0 / 53.3 / 52.2 median tok/s, and a cold first rep read 41.4. The canonical
requant refit already puts the true best rows inside the proxy top 64, so the
extra rows recover nothing. Reverted; the screen keeps 64.

## --mtp-relaxed 0.1 acceptance loosening (November 22, 2026) - no keep

Sampled acceptance kept a draft the exact test would reject when the target
gives it >= 0.1 probability. Same harness, relaxed vs exact:
52.1 / 53.6 vs 51.0 / 53.3 median tok/s - within run-to-run noise. The
survival policy already trims the chain where acceptance decays, so loosening
the per-token test buys almost nothing here. Default stays 0 (exact).

## Survival draft gate 0.40 -> 0.25 (November 22, 2026) - no keep

The sampled survival gate stops drafting when the cumulative acceptance
estimate falls below `kMtpSurvivalFloor`. A verified row costs ~5 ms against
~17 ms for a decode step, so a lower gate looked profitable on paper. Same
harness, 0.25 vs 0.40, three reps each: 52.1 / 53.5 / 52.3 vs 51.0 / 53.3 /
52.2 median tok/s - flat. The extra proposals below 0.40 are mostly rejected,
so they only add verifier rows. Gate stays 0.40.
