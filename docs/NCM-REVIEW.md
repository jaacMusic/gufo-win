# Next-Concept Modeling review (2026-11-12)

Review of two sources for portable ideas:

- arXiv `2610.14162v1` (2026-10-09), *NCM — Next-Concept Modeling: Fast,
  Parallel Coarse-to-Fine Text Generation*.
- Hazy Research blog, 2026-10-12, same title.
- Reference code: `github.com/HazyResearch/ncm`.

NCM is a **training** framework: a multiscale residual VQ-VAE tokenizer
(stage 1) plus a transformer that generates one level of discrete codes at a
time (stage 2). Both stages learn new weights, so nothing here is a decode-path
change we can adopt. What transfers is their *evaluation* argument and their
warning about error sensitivity. This note records both decisions so the next
pass does not re-derive them.

## What NCM does

Stage 1 encodes text to `K` levels of residual codes, level `k` having `l_k`
codes with `l_1 < ... < l_K = L`, each level with its own codebook; nested
dropout forces the coarse levels to carry meaning, and the decoder emits token
distributions from any cumulative reconstruction. Stage 2 freezes the tokenizer
and trains a transformer to predict all `l_k` codes of a level in parallel from
the cumulative reconstruction of earlier levels, one forward pass per level at
inference (`K` steps, not `L`). Conditioning codes are corrupted during
training so the generator tolerates residual-quantization errors.

Reported results are ~100M-parameter models on one H100: 10-31x faster than
diffusion baselines at comparable quality, 4-5x fewer training tokens.

## Ported

| Item | Where | Notes |
| --- | --- | --- |
| `gufo eval --max-completion-tokens N` | `src/cli/eval/eval.cpp` | The completion budget was pinned at 16000, so accuracy could only be measured at one budget. The report already records `execution.transport_ms` and `usage.completion_tokens` per case, so with a budget sweep accuracy-vs-seconds is computed from the existing JSON and needs no further code. Default and report metadata unchanged (`run.max_completion_tokens` now echoes the flag). |

The measurement rule behind that change is the one real borrow: NCM's authors
reject NFE as an efficiency metric because it ignores per-call cost. The same
hole exists in speculative decoding — cycles/drafted/accepted count calls, not
time, and a fast-but-narrower draft is invisible in both t/s and NFE. Quality
at a fixed wall-clock budget is the honest comparison.

## Evaluated and rejected

| Item | Reason |
| --- | --- |
| Multiscale residual VQ tokenizer | Stage-1 training plus a new artifact format (codebooks per level, decoder, reconstruction operators). No text-VQ path exists in the engine and no checkpoint needs one. |
| NCM generator (block-diagonal attention over levels, `K` parallel steps) | No public checkpoints, and the paper's scale (25-120M) is two orders below our targets (27B-700B). A runtime would have no model to load. |
| Coarse-to-fine hierarchy for global planning (their Sudoku result) | Same training dependency; the premise (level structure) is baked into the weights, not into the sampler. |
| Entropy next to accuracy in the eval summary | The pinned DS4 suite is `mcq`/`linespec`/`integer`; answers are one to three tokens, so a unigram-entropy number would be noise, not a diversity guardrail. Worth revisiting only with a free-form subset. |
| Structured-output integrity as a fixture test | The risk is real (see below) but a fixture HTTP server cannot model draft acceptance; the check needs weights and belongs in a benchmark run, not CI. Kept as a runbook item. |
| MBR / best-of-N test-time scaling | Needs a task metric (they use ROUGE) and a `n>1` path in `serve`. Neither exists; MBR over chat answers has no scorer. |
| Adaptive computation / early exit | Already present in another form: `enable_adaptive_draft_length` and the draft-length controller choose speculation depth per step. |
| Block-diffusion drafter | Already shipped: `src/models/qwen36_a3b/dflash.cpp` (DFlash-2) and the DeepSeek DSpark sampler. NCM cites DFlash as related work, not as a source. |

## Follow-ups

Each has a trigger; none is scheduled.

- **NCM runtime** — revisit if an open NCM-family checkpoint of useful size
  appears (target-scale weights plus a documented codebook/decoder layout).
  Work then: a VQ decode path and `K`-step sampler; the existing GGUF metadata
  reader has no place for per-level codebooks.
- **Diversity metric** — revisit when `gufo eval` grows a free-form/prose
  subset. Then add mean per-sample unigram entropy and a repeated-pair rate
  next to the pass count, and read them as a degeneration alarm only, not as a
  quality score (`docs/models/qwen3.8-flash-next/QUALITY.md` already states the
  same limit for the current report).
- **Structured-output integrity under speculation** — run before the next
  change to draft proposals (the reduced draft vocabulary in #603 changed which
  tokens can be proposed). Procedure below.
- **Best-of-N in `serve`** — only if a request for `n>1` appears; without a
  verifier it is not a quality feature.

## Runbook: quality at matched wall-clock

`gufo eval` drives one already-running single-model server, so one server per
speculative mode:

```sh
# per mode: AR, then the MTP or DFlash-2 server
for budget in 256 512 1024 16000; do
  gufo eval --questions 75 --max-completion-tokens $budget \
    --output eval-$mode-$budget.json
done
```

`cases[].execution.transport_ms` is the per-case wall time;
`summary.passed / abs(completed_cases)` is accuracy at that budget. Plot
accuracy against cumulative seconds and interpolate at the budget under test.
A mode that wins on tokens/s but loses accuracy at equal seconds is speculation
buying speed with answer quality; a mode that holds accuracy has bought speed.
Keep `--greedy` fixed across modes or the sampler, not the speculation, decides
the comparison.

## Runbook: structured integrity under speculation

NCM's tokenizer result is 100% exact reconstruction on Sudoku because a single
wrong code invalidates the solution. Decode has the same shape: a draft token
accepted in a near-tie silently rewrites a digit, a bracket or a key. For each
mode (AR, MTP, DFlash-2), on the same prompt set:

1. Ask for strict JSON output and `json.loads` it; ask for a small program and
   run its tests; ask for a grid task with one verifiable answer.
2. Record acceptance rate alongside the pass count. A mode with higher t/s and
   a lower pass count on these three is a regression regardless of throughput.
3. Repeat with the model's default sampling, not only greedy: penalties and
   constraints change which proposals survive verification.

## Caveats

- NCM's speedups (10-31x) and token savings (4-5x) are measured on ~100M models
  on an H100 against diffusion baselines. They say nothing about gfx1151 and
  must not appear in a Gufo benchmark table.
- The paper's evaluation protocol (generative perplexity under GPT-2-large,
  mean unigram entropy) is standard in diffusion-LM work and absent from ours;
  importing the numbers would compare different tasks, not different engines.
- A missing-model skip is not a quality pass (`AGENTS.md`). Both runbooks need
  a real GGUF and a real server.
