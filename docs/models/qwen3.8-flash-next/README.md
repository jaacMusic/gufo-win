# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

**UD-IQ4_XS** also loads (unqualified): its IQ3_S/IQ4_XS gate/up and IQ4_NL
down experts run on the vendored MMQ/MMVQ kernels, and its Q6_K embedding and
output head are re-encoded to Q8_0 at load (Q8_0's step is about a quarter of
Q6_K's, so the added rounding is small, but it is not bit-identical).
`qwen38_flash_next_iq_experts_ops_test` checks those expert paths against a
double-precision reference.

UD-Q4_K_XL at revision `d9620119` (Dynamic v3.1) keeps
`per_layer_token_embd.weight` in its own shard (`-00004-of-00004`), so a table
from the repo's `engram/` folder can be swapped in without downloading the
weights again. BF16, Q8_0 and IQ4_NL tables are read directly; Q5_1 tables are
not. Stamp the replacement for the quant's slot with `engram/set_split.py`,
rename it to that slot's file name, and start the model from shard 1 as usual.
The bundled IQ4_NL table is 28.8 GB; the Q8_0 (54.4 GB) and BF16 (102.4 GB)
alternatives cut its relative RMSE from 0.076 to 0.005 and 0.

v3.1 is a repack of the `38bb39ee` weights, not a requant: greedy 256-token
runs on both revisions are byte-identical, and same-binary pp2048/tg128
benches at depths 0 and 16384 agree within 1%. Its shard 1 is metadata-only
and ends at the GGUF header without alignment padding, which the reader only
accepts since 2026-11-07 (`gguf_reader_test` covers it).

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision d96201194f103c906994f283f504209dc8d2a10c \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --mtp-model "$MTP" \
  --sessions 2 --context 32768
```

On Windows, `tools\windows\fetch-hf-shards.ps1` pulls the same files in
resumable chunks when `hf download` cannot finish the 100 GB pull; see
[WINDOWS-DEV.md](../../WINDOWS-DEV.md#model-downloads).

The loader discovers the remaining shards. `serve` selects MTP whenever
`--mtp-model` is given, as it already does for DSpark with `--dspark-model`;
pass `--speculative off` to load the head without drafting. `prompt` and `bench`
still require the explicit `--speculative mtp`, so a benchmark never changes
mode silently.

Omit the speculative options for AR; AR sessions allocate no predictor state
even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Eligible
requests draft over the first 98,304 token IDs plus a 1,024-row table containing
special tokens and other tokens already used by the request. The full head
remains the fallback when the anchor or two of the last 32 committed tokens
fall outside this vocabulary.

Single-session sampling, including the default thinking-on and thinking-off
presets, can use this smaller head with the existing acceptance/length
controller and AR backoff. Greedy requests without penalties or constraints
can use it in C1 and jointly planned batches. Greedy penalties/constraints and
unplanned C>1 requests retain the full head.

The existing length controller and AR backoff apply to every request. The
target always verifies with its full vocabulary, including
exact sampled acceptance and residual correction. Sampled text can differ
across builds when proposals change. All-greedy C>1 batches retain their shared
planner and may use measured cycle costs. Each request keeps private caches,
rollback and RNG; snapshots preserve the learned acceptance history.
See [MTP qualification](QUALITY.md).

Three opt-in MTP options change only what the draft side proposes; verification
is unchanged, so sampled outputs keep the target distribution and greedy text is
byte-identical. They apply to single-session decoding (measured on Windows,
sampled, thinking on, 3 seeds per prompt, against the default):

| Option | What it does | Measured |
| --- | --- | --- |
| `--mtp-draft-vocab latin` | The draft head scores only special and ASCII/Latin-script text tokens (130K of 248K), reading half of the output head per draft step; ids map back on the GPU | +4.9% decode, identical texts and acceptance |
| `--mtp-policy survival` | Sampled drafting stops when the product of per-proposal acceptance estimates (from the draft head's top probability, by depth) falls below 0.40, instead of the per-cycle length controller | With the draft vocabulary: +8.7% over the default (prose +7%, code +7%, reasoning +13%) |
| `--prompt-lookup` | After each kept MTP draft, a match of 12+ tokens in the conversation proposes the tokens that followed it (point-mass proposals, exact under rejection sampling). Shorter matches propose nothing: an ungated 3-token rule lost 6-8% at temperature 1 | File-editing replies +10% sampled; log quoting +23% greedy; prose unchanged. Timings report `lookup_n` / `lookup_n_accepted` |

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
