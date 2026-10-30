# Coding-agent session: Gufo vs Bubo (26 Oct 2026)

Same machine, same day, one 10-turn coding-agent session per engine. Gufo was
run twice; Bubo once. These are the per-turn numbers, transcribed from the run
tables, plus a review of Bubo's serving stack against ours.

Transcription is verbatim (units and percentages as reported). Arithmetic added
below is marked as derived.

## Per-turn latency and throughput

| Area | jaacMusic run 1 / run 2 | Bubo 0.1.1 |
| --- | --- | --- |
| Agent session: total wait for first tokens over 10 turns | 45.1 s / 44.9 s | 9.2 s |
| Agent session: first token on turn 10 | 5.2 s / 5.2 s | 1.7 s |
| Agent session: decode, median per turn | 52.8 / 56.3 tok/s | 64.0 tok/s (+14–21 %) |
| Agent session: slowest turn | 24.0 / 24.4 tok/s | 49.0 tok/s |
| Rename edit | 99–101 / 100 tok/s | 97 tok/s |
| Comment edit | 90–91 / 90 tok/s | 86–87 tok/s |
| Prose | 40–41 / 40–41 tok/s | 41–42 tok/s |
| Reading a 2K-token prompt | 1,243 / 1,221 tok/s | 1,269 tok/s |

## Same-day ratio summary

| Task (same-day, single runs) | jaacMusic ÷ Bubo |
| --- | --- |
| Six-prompt mix | 0.85 (Bubo +18 %) |
| C++ | 0.90 (Bubo +11 %) |
| Python | 0.96 (Bubo +4 %) |
| Prose | 1.01 (level) |
| Edits | 1.04 / 1.05 (jaacMusic +4–5 %) |
| Agent session decode | 0.77 (47.4 vs 61.6 tok/s) |
| Agent session, tokens re-read | 52,186 vs 6,761 (7.7x more) |
| Agent session, total wait for first tokens | 45 s vs 9.2 s |

## Values to beat, and where we stand

The two tables above are the record: table 1 is one 10-turn session per engine
on the same machine, table 2 is the same-day per-task ratio. Every number in
them is a target. Where we have re-measured on UD-Q4_K_XL (gfx1151, `-c 8192`,
`-j 1`) the current value is listed; a blank means only the image value exists.

| Cell | Bubo | Ours now | Standing |
| --- | --- | --- | --- |
| 10-turn first-token wait | 9.2 s | 5.0 s | ahead, on shorter prompts |
| First token, turn 10 | 1.7 s | 0.50 s | ahead, on shorter prompts |
| Median session decode | 64.0 tok/s | 35.1; 50.7 with MTP latin + survival | behind 21 % |
| Slowest turn | 49.0 tok/s | 34.6 (not a slow turn: flat) | behind |
| Re-read share of prompt tokens | 13 % (6,761 of 52,186) | 14.4 % (1,651 of 11,432) | close, smaller session |
| 2K-token prompt read | 1,269 tok/s | 1,221–1,243 | behind 2–4 % (image value) |
| Rename edit | 97 tok/s | 99–101 | ahead (image value) |
| Comment edit | 86–87 tok/s | 90–91 | ahead (image value) |
| Prose | 41–42 tok/s | 40–41 | behind 1–2 % (image value) |
| Six-prompt mix | 1.00 | 0.85 | behind 18 % (image value) |

"Ours now" is not apples to apples with Bubo's session: our prompts reached
1,606 tokens instead of tens of thousands, so the wait totals are small in both
engines and prove nothing on their own. Closing the two real gaps means
re-running this table on a transcript long enough for Bubo's advantage to
show — the re-read share then drives the wait, which is what the 52k vs 6,761
row is about. The single-turn cells (prose, edits, 2K read) have not been
re-measured since the cache fix and still hold their pre-fix image values.

**MTP acceptance sets session decode; no flag moved it.** A four-turn
`tools/serving/agent-session.py` replay (16k context, `--speculative mtp
--mtp-draft-vocab latin`, 64 tokens per turn) reached a median 50.7 tok/s with
accepted/proposed counts of 53/62, 30/51, 38/71 and 11/26 — 85 % on the cold
turn, 54 % mid-session. Repeating the run without `--mtp-policy survival` gave
the *same* four counts, and proposals per step stay at `accepted + 1` because
the chain stops at the first rejection by design, so tokens per step is
`1 + accepted` and acceptance is the only lever. Beating Bubo's 64 tok/s needs a
draft tree or a wider draft model, not a switch.

## What the numbers say

- **The gap is prompt reuse, not kernels.** A 10-turn session re-read 52,186
  prompt tokens against Bubo's 6,761. If turn *N* reuses turn *N-1*'s frontier,
  the sum over 10 turns is roughly the first prompt plus the appended deltas;
  52k is the sum of whole prompts, i.e. near-zero cross-turn reuse.
- **First-token wait follows the same number.** 45.1 s of prefill wait against
  9.2 s, and 5.2 s vs 1.7 s on turn 10, scales with re-prefilled tokens, not
  with raw prefill speed: reading a 2K prompt is level (1,243 vs 1,269 tok/s).
- **Single-turn generation is level or ahead.** Prose is level and edits are
  4–5 % faster (a claim the second table's 0.77 session decode does not
  contradict: it is a mean over turns, not a per-token kernel rate).
- **In-session decode, measured here:** 35.1 tok/s median without
  speculation, 45.7 with `--speculative mtp --mtp-draft-vocab latin`, which is
  Bubo's `subagents` profile. So part of the 0.77 (and the 24.0 slowest turn)
  is speculation being off, not a kernel difference, and the remaining gap to
  64.0 needs a same-machine comparison before it is called a kernel problem.
  Turn-timed windows also include serving work (checkpoint capture, the extra
  prefill pass), which is why per-turn decode is below the direct long-context
  AR sweep
  ([Flash-Next benchmarks](models/qwen3.8-flash-next/BENCHMARKS.md#bubo-comparison)).

## Bubo's serving stack against ours

Reviewed at `haloforge-labs/bubo` `d922839` (engine 0.10.0, the version already
pinned in [the Flash-Next tables](models/qwen3.8-flash-next/BENCHMARKS.md)).
Path names are shared, so each row is a diff in our tree, not a missing file.

| Mechanism | Ours | Bubo |
| --- | --- | --- |
| Retained checkpoint records | 128 (`kSnapshotRecordCapacity`), independent of `--sessions`, which now bounds only active requests | 128 records, independent of `--sessions` |
| Positions retained per request | stable boundary before assistant framing + complete prompt in RAM, on a cold request and again on a warm hit; the disk tier adds shared-prefix boundaries (≥128 tokens apart, at most four per request, 2,048-token minimum step) | same two in RAM, plus up to four intermediate checkpoints on a 2,048-token grid (skips grid points within 2,048 of a warm frontier and 128 of the end) |
| Divergence points | learned in RAM from any retained state: prefill stops at the longest shared prefixes (≥128 tokens, at most four per request) and retains a checkpoint there, so the third sibling conversation restores it; the disk tier persists the same positions. No 512-token gain gate, no purpose-ranked eviction | learns the point where two conversations of one input identity diverge, keeps it once the gain exceeds 512 tokens, restores it from the third conversation; RAM does this without disk |
| Concurrent requests sharing a prefix | no waiting mechanism: a later request takes the nearest retained checkpoint or re-prefills the shared tokens itself | the later request parks with no session; the resident one publishes a checkpoint at the shared position (≥512 tokens ahead of what the waiter could restore) and the waiter restores it. Measured 13.4 s → 4.6 s for four 3.8k shared-prefix requests |
| Stable boundary capture | prefill stops at the boundary, then a separate pass | captured inside the final prefill pass when it lands within 8 tokens of the batch end; the batch may carry up to 128 tokens past a 4,096 chunk |
| Tool-turn replay | arguments parsed from JSON and re-rendered by the template | typed argument values rendered the way the reference templates' `tojson` does (and Responses namespaces flattened) so a replayed call reproduces the generated tokens and its checkpoint still matches |
| Eviction policy | byte budget plus entry count | purpose ranking (retry / history / continuation / branch point); retry copies may not displace the last useful checkpoint of a prefix family; a shared prefix two conversations diverge after is never treated as covered |
| Operator control | `GUFO_CACHE_RAM_GB` / `--cache-ram-gb` override (6 GiB default, 7 GiB cap), fitted intermediate checkpoint | `--cache-ram-bytes`, `GUFO_CACHE_HOST_FLOOR_MB`, `GUFO_CACHE_REMOVAL_LOG`; a release profile with separate `subagents` values (`--sessions 4`, `--context 131072`, `--prefill-chunk 512`, `--mtp-draft-vocab latin`) |

Snapshot payload is not the constraint to solve first: per Bubo's own
measurements a Flash-Next checkpoint is about 0.5 GB at 5k tokens, so 128
records only make sense behind a byte budget, which their defaults do
(automatic RAM budget, 8 GiB disk).

## Where to start, cheapest first

1. **Done: reproduced with our own diagnostics.**
   `python tools/serving/agent-session.py --url http://127.0.0.1:5815` replays a
   ten-turn tool-calling session and prints prompt, cached, re-read tokens,
   first-token time and decode rate per turn, plus `cache_miss_reason` and the
   common-prefix/nearest-checkpoint split whenever a turn missed
   ([server](SERVER.md)); `--json` keeps it as an artifact. Run against
   UD-Q4_K_XL on gfx1151 it reproduced the gap: cached tokens stayed pinned at
   689 while prompts grew 694 → 1606, so 39.7 % of prompt tokens were re-read
   (Bubo re-reads about 13 %), and every turn after the first missed with
   `prefix_changed common <prompt-1>`. That ruled out replay rendering (the
   prefix diverged at its last token, not inside a replayed tool call) and led
   to item 3.
2. **Done: checkpoint records are decoupled from `--sessions`.** The pool now
   keeps 128 records (`kSnapshotRecordCapacity` in `text_model_runner.cpp`)
   instead of `state_count * 2`, which cost nothing because records hold
   snapshot payloads only and the byte budget still bounds them. Two records
   already suffice for one linear conversation, so this alone does not explain
   the 7.7x; it covers rewritten tails (retries, dropped reasoning, branches)
   and keeps a shared system prompt from being the only survivor. Also relevant
   to the re-run: the snapshot reservation charges about 36 KB per prompt token
   (12 full-attention layers, 2 KV heads, 256-wide keys and values in fp16,
   plus the indexer), so 4.8 GB at 131k tokens and 9.7 GB at the native 262k,
   against a RAM budget of half of `MemAvailable` measured after the weights
   load. Read the snapshot events before blaming the hit rate on matching.
3. **Done: a warm turn checkpoints its own stable boundary.** Only the
   pre-framing boundary was ever reusable, for a tokenizer reason: a checkpoint
   taken at the end of a prompt can never be a prefix of the next turn, because
   BPE merges the closing newline of the assistant framing with the first
   character the client appends, so the ids diverge at the final position.
   Warm turns meanwhile re-checkpointed the frontier they had just reused
   (`count = prefill_offset`), which is already retained and equally
   unextendable. `src/cli/serve/text_model_runner.cpp` now checkpoints at
   `cache_prefix_tokens`, the position the prompt layer guarantees the next
   turn shares, and resumes after it. Ten-turn re-read on UD-Q4_K_XL fell from
   39.7 % to 14.4 % of prompt tokens with decode unchanged at 35 tok/s; the
   leftover is the mutable tail plus the new tool result, plus one extra
   prefill pass per turn (item 7). At these prompt sizes (694 → 1606 tokens)
   that pass costs about what the saved re-read saves, so first-token wait is
   level with the pre-fix run (5.0 s over ten turns); the win grows with
   transcript length, which is where Bubo's 5.2 s vs 1.7 s shows.
4. **Replay rendering parity** was the suspected cause and is not:
   `AppendToolCalls` in `src/models/qwen/chat_template.cpp` already renders
   typed argument values the way the reference `tojson` does. Keep this only as
   a check when a future miss shows `common_prefix_tokens` diverging *inside* a
   replayed call rather than at its end.
5. **Intermediate grid checkpoints** are implemented: when the stable boundary
   exceeds the budget, prefill stops once at the largest prefix below 90 % of it
   and publishes that checkpoint before continuing. The budget defaults to
   6 GiB (`GUFO_CACHE_RAM_GB`, capped at 7 GiB or the available memory). On
   UD-Q4_K_XL with a 126,024-token prompt, the default retained the
   126,017-token boundary (3.58 GB); the next turn re-read 30 tokens and reached
   its first token in 0.69 s instead of re-reading 126,047 tokens in 85.5 s. At
   a 3 GiB budget the 126K boundary needs 3.58 GB and is refused; the fitted
   101,227-token checkpoint (2.70 GiB) let the next turn re-read 24,820 instead
   of 126,047 tokens and reach its first token in 18.1 s instead of 85.7 s.
   **Learned divergence points** are implemented for the RAM tier: when an
   incoming prompt diverges from a retained conversation after 128 or more
   tokens, prefill stops once at that position and retains a checkpoint there
   (`continuation_cache.cpp::SharedPrefixBoundaries` feeds the existing
   `CaptureSharedPrefix` path, which no longer needs a disk store), so a shared
   system prompt is restored from the third conversation onward.
   `tests/cli/text_model_runner_test.cpp` covers the learn/restore loop with the
   fake runner and `tests/cli/continuation_cache_test.cpp` the boundary query;
   both run on CPU. Still open: a GPU replay of two sibling
   `tools/serving/agent-session.py` runs to quantify the saved re-read, and
   Bubo's 512-token gain gate plus purpose-ranked eviction, which this keeps
   simpler by using the byte budget alone. Upstream's narrower grid
   (`1b8ad798`, #773) was reverted the same day, so keep the fake-runner test
   with any change here.
6. **Leader/follower waiting** for fan-out: subagents started from one system
   prompt currently each prefill the shared prefix, and prefill is serial, so
   the copies cost their sum. Concurrent requests still re-prefill the shared
   part; only the sequential sibling case above is served by RAM checkpoints.
7. **In-pass boundary capture** last: it is a runner change (per-model state
   borrowing) for roughly one prefill chunk per turn, not a scheduling win.

## Correction to the earlier Bubo verdict

The 2026-10-26 survey closed Bubo with "no new commits since 2026-10-13; V2K
stays rejected, NPU offload stays the only big-ticket deferred item". That
verdict covers kernels. The measured gap here is in the serving cache, which
was never compared: no kernel A/B explains 7.7x re-read tokens or a 5x
first-token wait while 2K-prompt prefill stays level.
