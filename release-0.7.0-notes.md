A warm turn at 124K tokens now restores the full 3.2 GB boundary from disk in **0.77 GiB of RSS and 3.25 s**, where v0.6.0 restores a 101K-token grid, re-prefills the gap, and costs 3.54 GiB and 14.10 s. This needs the staging limit to hold the boundary (3.5 GiB at 124K, 8 GiB at 262K); with the automatic 768 MiB limit both releases keep the grid and land at 13.7 s. What the code changes is that a boundary that fits staging is kept on disk and no grid checkpoint is captured, so the grid can no longer shadow the cheaper restore. The snapshot budget also stops being a fixed pool — it may take the room below 75% of total RAM, with everything the process already holds counted against it — tens of GiB on a 128 GB Strix Halo box, but on a 32 GB box with the 27B weights loaded the remaining room can be smaller than the old 6 GiB pool. Most server flags can change without a restart, and a replayed tool call keeps the numbers the model wrote.

**Measured vs v0.6.0** (same machine, one process at a time, greedy; v0.6.0 built from tag `fef0d347`, this release from `18ef48ea`; Unsloth UD-Q4_K_XL `38bb39ee` + shared-Q8_0 MTP, `gufo bench`, two timed repetitions per cell):

| Context (tok/s) | v0.6.0 pp | v0.7.0 pp | v0.6.0 AR | v0.7.0 AR | v0.6.0 MTP | v0.7.0 MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1709.6 | 1705.7 | 34.27 | 34.33 | 52.62 | 52.16 |
| 32,768 | 1651.9 | 1654.2 | 32.40 | 32.48 | 100.08 | 99.85 |
| 131,072 | 1581.4 | 1580.2 | 29.56 | 29.55 | 94.81 | 94.91 |
| 262,144 | 1513.8 | 1510.6 | 27.55 | 27.54 | 90.30 | 90.37 |

Every cell is within ±0.9% (pp ±0.2%), and the greedy MTP output is byte-identical between the two binaries at all four depths (same SHA-256, same 81/122 accepted drafts at 8K and 112/112 deeper). The 8K AR turn stops at 28 of 128 tokens on both.

| Warm turn, 120,859-token prompt, 3 GiB tier | Host RSS | Turn latency |
| --- | ---: | ---: |
| v0.6.0, 3.5 GiB staging, second conversation | 3.54 GiB | 14.10 s |
| v0.7.0, 3.5 GiB staging, second conversation | **0.77 GiB** | **3.25 s** |
| v0.6.0, 3.5 GiB staging, cold start | 0.70 GiB | 3.24 s |
| v0.7.0, 3.5 GiB staging, cold start | 0.70 GiB | 3.33 s |
| v0.6.0 and v0.7.0, 768 MiB staging (automatic), second conversation | 3.54 GiB | 13.7 s |

All rows hard-set the 3 GiB tier, and both binaries write the same 3.44 GB payload to the disk store. In-process, v0.6.0's snapshot lookup hits its 101,227-token grid first and never consults the disk copy, so it prefills 19,632 tokens; v0.7.0 captures no grid and restores the boundary (2.45 s of the 3.25 s is the read). After a restart the grid is gone from RAM and both binaries restore from disk at the same speed. With automatic staging — 768 MiB on this host, capped at 1 GiB — the boundary is refused and both releases keep the 2.9 GB grid. The budget change is not in that table: a 262K-token boundary is estimated at about 7 GB (120K tokens measures 3.2 GB), which never fit the old 6 GiB pool, while the new ceiling leaves room for it on a 128 GB host — not re-measured, so treat it as headroom, not a number.

**Draft head**: proposals read a Q4_K copy of the head prefix instead of the shared Q8_0 block (upstream #1209). The A/B above does not reproduce the port's claimed +1.5% tg128 / +1.3% depth: MTP decode is 52.16 vs 52.62 at 8K and 90.37 vs 90.30 at 256K, with byte-identical drafts, so count it as neutral here. An earlier attempt measured no gain (tg128 49.9 vs 49.6, tg128@d16384 -3%) and was reverted; the 2026-11-14 fall-through fix came after the landing.

**What changed**

*Features*
- **serve: runtime configuration** — `GET /admin/config` lists the flags a running server can change, `POST /admin/config` applies a JSON object and answers per flag (`applied` / `rejected` with a reason; a rejected flag keeps its value and the rest still apply). Live: `served-model-name`, `max-tokens`, the sampling and reasoning defaults, the scheduler limits (the buffered-output ceiling now updates in place), `max-connections`, `max-request-bytes`, `api-key` (write-only; reads say whether one is set), `cache-ram-gb` and the disk-cache budgets. Model artifacts, `--context`, `--sessions`, speculative decoding, `--prefill-batch`, `--host`, `--port` and the `--cache-disk` directory stay load-time and are rejected, so the client restarts.
- **Snapshot budget is a whole-process ceiling**: automatic default is `available - 25% of total RAM` (the engine's footprint stays inside 75%), sampled once so the pool cannot evict itself as it fills; `--cache-ram-gb` / `GUFO_CACHE_RAM_GB` still hard-set it (bounded by available memory), and `0` restores the automatic value. The 6 GiB default and 7 GiB cap are gone, and the cache re-reads the budget at each snapshot admission, so a lowered `cache-ram-gb` trims retained checkpoints as traffic continues instead of waiting for a restart. The automatic disk-staging limit is a quarter of the retained budget instead of an eighth of available RAM.
- **Disk-staged boundary skips the grid checkpoint**: if the boundary fits staging with 1 MiB to spare, no intermediate checkpoint is captured — restoring the full boundary from disk beats restoring a 90% grid and prefilling the gap, and because a snapshot hit short-circuits the disk lookup, dropping the grid is also what makes the full boundary reachable while the process is live.
- **`gufo eval --max-completion-tokens N`**: the completion budget was pinned at 16000, so accuracy could only be read at one budget; the report already carries `execution.transport_ms` and `usage.completion_tokens`, so accuracy at matched wall-clock comes out of the existing JSON.
- **Qwen3.8-27B verify GEMM**: IQ4_XS shapes swept on the TurboFCFusion merge GGUF — +13-48% over the generic dispatch for attn_gate, ffn_down, ssm_alpha/out, packed ffn_gate+up, attn_k and attn_q at 2-8 rows, +6-53% for Q5_K attn_v and ssm_qkv, each bit-exact against the generic path. That GGUF still is not servable here (chat template SHA not pinned).
- **Docs**: command-line flag reference for every subcommand in `README.md`, a release procedure with the single-zip asset convention, and a NCM review recording which of its ideas transfer (the `--max-completion-tokens` sweep) and which do not.

*Performance*
- **AVX-512 sampling, IQ4_NL/bf16 dequant and n-gram hashing** (upstream #1127), 1.17-25x on the isolated CPU paths and neutral end to end — CPU margin, not tok/s. Windows detects AVX-512 with CPUID + XCR0 because clang-cl does not link `__cpu_model`.
- **Snapshot admission charges the pinned budget per live allocation**, so host RAM no longer grows with every snapshot size; a refused admission keeps the restored frontier instead of dropping back to the older prefix.
- **Decode shared-expert overlap is calibrated, not assumed**: both layouts are timed inside `Create`, the faster one wins (serial here by ~40% per layer pass), and `GUFO_TRACE=1` records both so a host or runtime flip is visible later. Steady-state decode is unchanged by construction.
- Tried and rolled back on 2026-11-09/10 rather than kept: a fixed 32 MiB hipBLAS workspace (llama.cpp `#31838`; no cell outside noise, decode -1%, 32 MiB per handle), longest-first WMMA prompt dispatch (+0.1-0.3%, inside r=3 spread), and shrinking the pool on memory pressure (the pinned budget already prices GPU pressure). Records in `docs/LINUX-PERF-PORT.md`.

*Fixes*
- **Tool-call numbers replay as the model wrote them** (upstream #1146): `1.0` no longer becomes `1` and `1e5` no longer becomes `100000` in the rendered prompt, so a replayed call still matches its checkpoint.
- **Prompt state is charged once** (upstream #1164): the claim and the per-token prompt budget both billed the same K/V rows, which could refuse prompts that fit.
- **Compacted MTP inject rows stay out of the norm buffer** (upstream #1230): `tail.inject` no longer aliases `mtp_h`, which the FFN combine writes while the inject partials are still read.
- **The Qwen K/V cache no longer clamps the context to 8,192 tokens**, and the default context is 32,768 instead of 4,096, so a caller that omits `--context` no longer silently runs the 27B path at a fraction of its window.
- **`--max-pending 0` means "reject when every session is busy"** (upstream #1206/#1207): admission now counts the pool's free sessions instead of a scheduler-side table that could disagree with the leases, and a full queue answers 429 instead of being deferred.
- Windows debug-suffixed vcpkg imports documented (`libcurl-d.dll` shim), and the stale `wide_verification_test` target that never compiled is gone; `session_test` covers wide verification against the real model.

The automatic budget changes RAM use in both directions: on a 128 GB host it is now tens of GiB, on a 16-32 GB host the room below 75% can be **smaller** than v0.6.0's 6 GiB pool because the weights already count against it — pin `--cache-ram-gb` there. Tool-call rescoring changes prompts that carry floats or exponents. MTP proposals read a Q4_K head prefix; greedy replies are byte-identical.

Known limitation, documented rather than fixed: the batched and single-row decode pipelines are **not bit-identical** on this host (they agree to ~1e-8 through layers 0-1, cross a quantization rounding boundary at layer 2-3, and amplify like any 48-layer quantized network). Same-state and same-pipeline checks stay exact; the two retained GPU suites' cross-pipeline equality assertions fail deterministically and are recorded as unreachable without making the batch path call the serial kernels per row.

Unpack, run start.cmd, point it at the model downloads in README.txt. Requires a Strix Halo machine and a current AMD driver.
