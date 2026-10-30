Prefill is ~3% faster at every context depth and decode is unchanged. `serve` now auto-enables MTP drafting when a draft sidecar is present: on real chat prose that is **+26–31%** over autoregressive. Everything else in this release is output correctness.

**Measured vs v0.3.0** (same machine and weights, direct `gufo bench`, greedy, one process at a time, UD-Q4_K_XL + MTP Q8_0; v0.3.0 run Oct 26, dev run Oct 28):

| Context | Prefill | AR decode | MTP decode | MTP accepted |
| ---: | --- | --- | --- | ---: |
| 8,192 | 1,637 → 1,678 (**+2.5%**) | 33.52 → 32.15 | 57.16 → 50.08 ᵃ | 69% → 64% |
| 32,768 | 1,588 → 1,636 (**+3.0%**) | 32.08 → 32.15 | 96.79 → 97.15 | 100% |
| 131,072 | 1,527 → 1,570 (**+2.8%**) | 29.29 → 29.44 | 92.47 → 92.44 | 100% |
| 262,144 | 1,464 → 1,503 (**+2.6%**) | 27.35 → 27.35 | 87.71 → 87.94 | 100% |

ᵃ The 8K MTP cell is accounting, not a regression: the bench now flags end-of-turn, so post-turn free text no longer counts as generated. MTP rows from 32K are a repeated-text upper bound. Realistic chat-prose decode (~400-word prompt, `-n 256`): AR 35.22 / 32.08 (sampled) → MTP **44.5–45.2 / 41.98 (+26–31%)**, 49–67% acceptance.

**What changed**

*Features*
- **serve: `--cache-ram-gb`** sets the snapshot-cache budget; the largest checkpoint that fits is retained.
- **serve: MTP auto-enables with a draft sidecar** — no flag needed — plus an online accept-probability table (Strata MtpAcceptTable port). The length controller picks draft width from acceptance, so `--draft-tokens` is only a cap. The shared-Q8_0 sidecar is the right default (full-Q8_0 measures within 1%).
- **NPU (opt-in)**: module synced to Bubo 51d14955 — L-REL-2 chunk retry, fault guards, guard/monitor/trace knobs honoured; XRT batch-phase logs per prefill chunk.

*Performance*
- Prefill runs in **8,192-token chunks**: the +2.5–3% pp in the table above.
- Bit-identical kernel work: SelectScore register-bloat cut, RoPE sincos/tables hoisted to shared memory, narrow mixer-down W8A8 tile at ≤1,024 tokens.
- Tried and removed rather than kept: chunked DeltaNet prefill (upstream claims +12%; locally −1–2% *and* it changed greedy output) and a SelectScore warp rewrite (slower). Records in `docs/LINUX-PERF-PORT.md`.

*Fixes*
- Q8_0 inject and alpha/beta weights now run on their F32 kernels (#788) — they were processed by the wrong quantized kernel.
- Standalone RopeKernel rounds identically to the fused preparation path.
- Block selection and draft-ID remap are bounded by their buffers (#849).
- Loader retries a failed unbuffered weight read through the buffered file (#858).
- serve: warm-turn checkpoints are taken at the stable prefix boundary.
- build: `GUFO_NPU` is propagated to consumers of the flash-next headers (fixes mismatched downstream builds).

The first two fixes change generated output where the old behaviour was wrong; text may differ from v0.3.0.

Unpack, run start.cmd, point it at the model downloads in README.txt. Requires a Strix Halo machine and a current AMD driver.