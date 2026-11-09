Prefill is 1-4% faster at every context depth (register-resident routed prefill experts, +3.8% on pp16384 in a quiet-GPU A/B) and MTP decode gains another 2-3%. Cached follow-up turns now answer in **9-12 ms** at 32K-100K context (down from 126-365 ms; ~9x faster than Bubo at 100K). Long prefills no longer block other sessions: prompt passes are split by layers, so decoding sessions keep running while a prompt fills. This release is also where the fork takes its own name: **Gufo-win**.

**Measured vs v0.5.0** (same machine, direct `gufo bench`, greedy, one process at a time, UD-Q4_K_XL + shared-Q8_0 MTP; baseline is the v0.4.0/v0.5.0 model path - v0.5.0 changed serving only - re-measured November 8, 2026 on the Dynamic v3.1 repack):

| Context | Prefill | AR decode | MTP decode | MTP accepted |
| ---: | --- | --- | --- | ---: |
| 8,192 | 1,678 -> 1,695 (**+1.0%**) | 32.15 -> 32.57 | 50.08 -> 51.58 (**+3.0%**) | 64% -> 66% |
| 32,768 | 1,636 -> 1,642 (**+0.4%**) | 32.15 -> 32.22 | 97.15 -> 99.39 (**+2.3%**) | 100% |
| 131,072 | 1,570 -> 1,579 (**+0.6%**) | 29.44 -> 29.54 | 92.44 -> 94.59 (**+2.3%**) | 100% |
| 262,144 | 1,503 -> 1,508 (**+0.3%**) | 27.35 -> 27.35 | 87.94 -> 89.40 (**+1.7%**) | 100% |

a The Dynamic v3.1 repack has identical quantization bytes; its numbers are within single-run noise of the October 28 run, so the deltas above are code, not weights.

**Against Bubo 0.1.2** (released Nov 4; measured 2026-11-08/09, same machine, greedy, one process at a time, two repetitions. Bubo rows use the byte-identical `38bb39ee` snapshot: its stock loader rejects the Dynamic v3.1 local repack):

| Context | Bubo pp | Gufo pp | Bubo AR | Gufo AR | Bubo MTP | Gufo MTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1,657.8 | **1,695.5** | 28.64 | **32.57** | 44.37 | **51.58** |
| 32,768 | 1,588.1 | **1,642.1** | 28.61 | **32.22** | 95.68 | **99.39** |
| 131,072 | 1,519.7 | **1,579.4** | 27.82 | **29.54** | 91.53 | **94.59** |
| 262,144 | 1,451.2 | **1,508.0** | 26.94 | **27.35** | 86.72 | **89.40** |

All values tok/s; Gufo-win leads every cell (prefill +2.3-4.2%, AR +1.5-13.7%, MTP +3.1-16.3%). Bubo 0.1.2's auto draft vocabulary lifted its 8K MTP from 34.97 to 44.37 tok/s; every other cell moved within ~1% of 0.1.1.

**Cached time-to-first token** (follow-up request in the same session, `/v1/chat/completions`, sessions 1, context 131,072, greedy, 16-token replies, one cold pass then three cached repeats, 2026-11-08 Gufo-win build; v0.5.0 column is the pre-restore-reuse path measured on the same harness):

| Prompt | Gufo v0.5.0 | Gufo v0.6.0 | Bubo 0.1.1 |
| ---: | ---: | ---: | ---: |
| ~32K (168,800 chars) | 126 ms | **9-12 ms** | 8.6-9.0 ms |
| ~100K (515,000 chars) | ~365 ms | **11.6-11.8 ms** | 102-113 ms |

The win comes from restoring a checkpoint by copying only the ~240 MB of mutable recurrent state when the request returns to the session that wrote it (resident K/V rows stay in place), keeping snapshot payloads in a bounded pinned-host pool so the restore copy reads resident memory, and memoizing tokenization. Cold TTFT on the same runs: 20,449 ms at ~32K, 65,026 ms at ~100K; output was byte-identical cold vs cached on both engines.

**What changed**

*Features*
- **Identity: the engine is now Gufo-win** (jaacMusic fork); README tables and prose relabelled accordingly.
- **Prompt passes split by layers**: a long prefill yields between layers so other sessions keep decoding; both streams are drained around layer-hook yields with failure causes reported. `--prefill-chunk` is deprecated (prompts beside decoders are split by layers now).
- **serve: `/tokenize` endpoint** (upstream #1082), and a fix for tool-call values that start with `</parameter>` (upstream #1102).
- **Growable MMQ arena and prompt-state admission budget** (upstream #740, #1083).
- **Weights re-pinned**: UD-Q4_K_XL to Unsloth Dynamic v3.1 (`d9620119`), which also reads Q8_0 engram tables.
- **Windows**: the Flash-Next launcher defaults to the disk continuation cache.

*Performance*
- **Register-resident routed prefill experts** (upstream #890): pp16384 **+3.8%** in a quiet-GPU A/B, with bit-identical prefill as the quality gate.
- **Fused decode HC combine + Q8_0 mixer** in one launch (upstream #894); dense sweep of pre-budget rows in budget-crossing attention batches (upstream #876, A/B benched).
- Wide-row verification and routed GEMM rework; shared-expert and Z-row overlap on a side stream (wide rows only); profitable greedy batch drafts clear a stale MTP AR backoff.
- **serve: cached follow-up turns in 9-12 ms**: checkpoint restore reuses resident K/V rows for the writing session (copies only ~240 MB of mutable recurrent state), snapshot payloads live in a bounded pinned-host pool, and tokenization is memoized — cached TTFT 126 → 9-12 ms at 32K, ~365 → ~12 ms at 100K.
- **serve**: a queued fan-out request waits until the resident publishes its prefix instead of re-prefilling it.
- Tried and rolled back on 2026-11-08 rather than kept: batched target-FFN HC-pair fusion, MTP backoff recovery and PLE tracing (measured losses or quality risk). Records in `docs/LINUX-PERF-PORT.md`.

*Fixes*
- DSpark greedy output is bit-equal to plain decoding (upstream #1105).
- Metadata-only GGUF shards without alignment padding load again.
- Pinned XRT runtime re-certified; stale ready-Q8 rows dropped from the batched routes; multi-pass prefill steps aligned and the draft hidden row counted.

The DSpark equality fix and the v3.1 repack can change generated output where the old behaviour was wrong; text may differ from v0.5.0.

Unpack, run start.cmd, point it at the model downloads in README.txt. Requires a Strix Halo machine and a current AMD driver.
