# Gufo-win: the Strix Halo inference engine

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

Gufo-win is jaacMusic's Windows port of
[Gufo](https://github.com/gufo-org/gufo), a vertical local inference engine
specifically built and optimized for the AMD Strix Halo hardware: Ryzen AI
MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified
memory. This port builds and runs it natively on Windows 11 x64 against AMD's
TheRock ROCm — no WSL, no containers. The port details are in
[docs/WINDOWS.md](docs/WINDOWS.md).

## Headline results

Qwen3.8-Flash-Next Q4 (`UD-Q4_K_XL` + MTP Q8_0), 128 GB unified memory with a
96 GB dedicated carve; direct `gufo bench`, greedy, two timed repetitions.
Long-context sweep, 8K to 256K (October 28, 2026):

| Context | Prefill (tok/s) | AR decode (tok/s) | MTP decode (tok/s) | MTP accepted |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 1,678.35 | 32.15 | 50.08 | 64% |
| 32,768 | 1,635.77 | 32.15 | 97.15 | 100% |
| 131,072 | 1,570.13 | 29.44 | 92.44 | 100% |
| 262,144 | 1,502.72 | 27.35 | 87.94 | 100% |

The 256K context holds the ~89 GiB model and state fully inside the 96 GB
carve, so it decodes at full speed. The MTP column is an upper bound: the
bench prompt repeats a 20-token pattern and from 32K the model repeats it
verbatim (drafts accepted 100%). Realistic mixed-prompt MTP decode at depth is
**28–35 tok/s**; the AR column is the no-speculation reference. Full table
with error bars and settings:
[context sweep](docs/models/qwen3.8-flash-next/bench-context.html).

Against Bubo 0.1.1 (haloforge-labs release binary, same model snapshot, same
`bench` calls through `tools/windows/bench-flash-next-context.ps1`, greedy,
two repetitions, 2026-11-01):

| Context | Gufo pp | Bubo pp | Δ | Gufo AR | Bubo AR | Δ | Gufo MTP | Bubo MTP | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | **1,688.9** | 1,661.3 | **+1.7%** | **33.6** | 28.1 | **+19.8%** | **52.0** | 35.0 | **+48.6%** |
| 32,768 | **1,642.6** | 1,592.5 | **+3.1%** | **32.0** | 28.7 | **+11.4%** | **99.4** | 96.4 | **+3.1%** |
| 131,072 | **1,582.4** | 1,523.6 | **+3.9%** | **29.3** | 27.9 | **+5.1%** | **94.4** | 92.1 | **+2.5%** |
| 262,144 | **1,516.2** | 1,454.7 | **+4.2%** | **27.4** | 27.0 | **+1.3%** | **90.1** | 86.8 | **+3.8%** |

Gufo leads every cell (bold; Δ = Gufo ÷ Bubo − 1). The 8K AR cell ends the
turn after 28 generated tokens. Bubo is faster on the cached first token of a
32K conversation turn: its serving stack keeps session KV resident and answers
in 8.6–9.0 ms, against Gufo's 21–23 ms (27–30 ms at 100K, where Bubo takes
102–113 ms). Gufo restores a checkpoint by copying only the ~240 MB of mutable
recurrent state when the request returns to the session that wrote it
(2026-11-02); that replaced a full 2.0–5.7 GB re-upload and, together with
tokenization memoization (2026-11-01), took the cached turn from 126 ms to
21–23 ms at 32K and from ~365 ms to 27–30 ms at 100K. The 32K remainder is that
copy crossing pageable host memory at ~11 GB/s
([TTFT comparison](docs/models/qwen3.8-flash-next/BENCHMARKS.md)).

Against upstream 0.24.0 (the stock `gufo` release binary, same files, same
direct `gufo bench` calls, greedy, 32-token budget, three repetitions, one
process per mode, NPU off, 2026-10-27; AR cells remeasured 2026-10-31 with
two repetitions):

| Depth | Mode | Gufo pp (tok/s) | upstream pp | Gufo tg (tok/s) | upstream tg | Gain tg |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | AR | 1,669.74 | 1,663.39 | 35.54 | 27.55 | **+29.0%** |
| 16,384 | AR | 1,500.88 | 1,451.50 | 31.86 | 27.90 | **+14.2%** |
| 32,768 | AR | 1,457.91 | 1,431.52 | 32.28 | 27.79 | **+16.2%** |
| 0 | MTP | 1,655.92 | 1,618.86 | 51.48 | 30.67 | **+67.9%** |
| 16,384 | MTP | 1,466.45 | 1,406.66 | 87.29 | 85.01 | **+2.7%** |
| 32,768 | MTP | 1,437.28 | 1,380.76 | 85.70 | 85.12 | **+0.7%** |

Prefill runs 0.4-3.4% ahead and model loading stays within 1%. Decode gains
14-29% without speculation and 68% at depth 0 with MTP; from 16K both binaries
copy the repeated prompt at 85-87 tok/s, so those two cells measure that copy
speed rather than drafting. The short budget also keeps both binaries on the
same greedy prefix: at 16K they emit identical tokens for the first 32 steps,
while a longer budget lets the chunked prefill's last-bit numerics move a
near-tie and the two sequences drift onto different text.

## Models and benchmarks

**Confirmed on this machine:** Qwen3.8-Flash-Next (adaptive MTP, images) —
that is the model measured below. [docs/models](docs/models/README.md) and
[docs/WINDOWS.md](docs/WINDOWS.md) also list Qwen3.8-27B (DFlash2) and
Qwen3.6-35B-A3B as servable; they may be available in other builds or repos,
but we cannot confirm that with measurements of our own. DeepSeek V4 Flash
and the audio/video/image pipelines are documented but not validated on
Windows.

### Qwen3.8 Flash-Next

Full guides: [model guide](docs/models/qwen3.8-flash-next/README.md) ·
[benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) ·
[quality](docs/models/qwen3.8-flash-next/QUALITY.md).

| Model | Inference modes | Hugging Face weights |
| --- | --- | --- |
| Qwen3.8 Flash-Next | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) |

A hybrid recurrent/QSA mixture-of-experts text/image model. Point the server
at the first `UD-Q4_K_XL` shard and the loader discovers the rest; the MTP
sidecar enables adaptive speculative decoding (default on, up to 7 draft
tokens), `mmproj-BF16.gguf` adds image input. Native context is 262,144
(YaRN extension unsupported); `start.cmd` serves 131,072 by default and sizes
the context to your carve. The server is ready when the log shows
`event=load_completed`.

Measured on this machine: Windows 11, AMD Strix Halo `gfx1151`, 128 GB unified
memory, 96 GB dedicated carve; unsloth `UD-Q4_K_XL` + shared-Q8_0 MTP; direct
`gufo bench`, greedy, two timed repetitions after one untimed pass.

October 2026 upstream sweep (v0.2.0): per-stream scratch arenas for the mmq
kernels, K-order BF16 indexer projections, and occupancy-priced sampled MTP
drafts landed with greedy output byte-identical to the previous release.
Interleaved A/B versus v0.1.0 (`gufo bench -p 2048 -n 128 -r 1`, MTP):

| metric | v0.1.0 | v0.2.0 | delta |
| --- | ---: | ---: | ---: |
| MTP tg128, depth 0 | 34.76 tok/s | 38.95 | **+12%** |
| MTP tg128, depth 32768 | 75.62 tok/s | 79.27 | **+4.8%** |
| MTP pp2048, depth 0 | 1588.89 tok/s | 1598.69 | +0.6% |
| AR pp2048 / tg128 | 1619–1632 / 30.1–30.5 | 1636–1645 / 30.4–30.5 | neutral |
| Sampled serving, 2 users, temp 1.0 | 34.53 tok/s | 36.48 | **+5.6%** |

Draft acceptance was identical (63%) in the sampled A/B, so the decode gains
are the ported fixes, not sampling luck.

October 26, 2026 (v0.3.0): Q4_1 decode weight copies (large dense Q8_0
projections decode from a least-squares-refit Q4_1 copy) and MTP prompt lookup
on by default. Long-context sweep, same protocol as above:

| metric | v0.2.0 | v0.3.0 | delta |
| --- | ---: | ---: | ---: |
| MTP tg128, 8K context | 45.66 tok/s | 57.16 | **+25%** |
| MTP tg128, 32K context | 78.81 tok/s | 96.79 | **+23%** |
| MTP tg128, 131K context | 75.03 tok/s | 92.47 | **+23%** |
| MTP tg128, 256K context | 72.10 tok/s | 87.71 | **+22%** |
| AR tg128, 8K / 256K context | 28.51 / 24.21 tok/s | 33.52 / 27.35 | +18% / +13% |
| pp2048, 8K context | 1,626.37 tok/s | 1,637.48 | +0.7% |

The Q4_1 copies change decode numerics; `GUFO_DECODE_COPY=off` restores the
exact Q8_0 path.

Windows decode switches stay high-speed and bit-identical: with every Windows
switch on, logit dumps are bit-identical at all 4,418 positions and seeded
decode is byte-identical to the Linux path (sampled probes):

| Prompt | Windows (tok/s) | Linux (tok/s) |
| --- | ---: | ---: |
| prose | 34.5 | 32.0 |
| code | 37.2 | 34.3 |
| reasoning | 48.3 | 44.1 |

Loading: the 103.7 GiB checkpoint keeps ~76.9 GiB resident and streams its
~26.8 GiB n-gram table from disk; with a 96 GB carve loads in ~30 s. A 64 GB
carve works but loads ~10× slower and reads weights beyond the carve 6–22%
slower.

Measured sampling options (single-session decoding, sampled, thinking on,
3 seeds per prompt, Windows):

| Option | What it does | Measured |
| --- | --- | --- |
| `--mtp-draft-vocab latin` | Draft head scores only Latin-script tokens (130K of 248K), half the output head per draft step | **+4.9% decode**, identical texts and acceptance; the only opt-in (`start.cmd` asks) |
| `--mtp-policy survival` | Drafting stops when the per-proposal acceptance product falls below 0.40 instead of the length controller | **+8.7%** with the draft vocabulary (prose +7%, code +7%, reasoning +13%); on by default in `start.cmd` |
| `--prompt-lookup` | A 12+-token conversation match proposes the tokens that followed it | File-editing replies **+10%** sampled; log quoting **+23%** greedy; prose unchanged; on by default |

Reproduce the context sweep yourself (one benchmark process at a time; the
96 GB carve fits a single Q4 instance):

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\bench-flash-next-context.ps1
```

**Not validated on Windows** — runs in the codebase but is untested or
unqualified here:

- DeepSeek V4 Flash and the ASR / TTS / video / image-generation pipelines are
  documented under [docs/models](docs/models/README.md) but not validated on
  Windows.
- The `UD-IQ4_XS` quantization loads, but its expert paths are unqualified on
  Windows (known small re-rounding of the Q6_K heads).
- Upstream unquantized-model / GGUF-conversion parity remains unqualified
  (inherited from the model guide).
- Test status: every Flash-Next operator suite passes and 86/92 CPU-labelled
  tests pass; Linux-only Python dev tools are not ported, and newer TheRock
  clang versions can round fused kernels differently
  ([docs/WINDOWS.md](docs/WINDOWS.md): port internals, memory and test notes).

Served rates differ from the direct numbers: `start.cmd` additionally enables
survival stopping, the Latin draft vocabulary, prompt lookup and temperature
1.0, which direct `gufo bench` cannot set. Measure served rates over HTTP with
`tools/serving/gufo-serving-bench.py`.

## Goals

What this fork is about, in one list:

- **Native Windows on Strix Halo.** Build and run Gufo directly on Windows 11
  against AMD's TheRock ROCm — no WSL, no containers — and publish prebuilt
  release zips so a working server is unpack-and-run from a fresh Strix Halo
  machine.
- **Stay close to upstream.** Port upstream gufo's fixes and optimizations as
  they land, and keep the Windows path behaviorally identical to it: bit-exact
  logit checks and seeded-decode byte-identity are the gate for accepting any
  Windows-specific change.
- **Only publish what is measured.** Anything in these docs carries numbers
  from this machine, on the exact weights named; everything not confirmed here
  is marked unvalidated instead of inherited from upstream.
- **Depth over breadth.** Keep the model list short and the flagship model
  (Qwen3.8-Flash-Next) fully documented: benchmarks, quality notes, memory
  sizing, and the settings that reproduce them.
- **The one-stop shop stays the destination.** Text, audio, image and video
  models for Strix Halo, added as they can be validated on this hardware the
  same way — smaller memory configurations still work and keep their speed for
  what fits.

## Quickstart

1. Reserve dedicated GPU memory for the model you want (AMD Software >
   Performance > Tuning > Variable Graphics Memory): **96 GB** runs
   Qwen3.8-Flash-Next at the full 256K context and decode speed.
2. Download `gufo-windows-0.1.0.zip` from
   [Releases](https://github.com/jaacMusic/gufo-win/releases/tag/v0.1.0) and
   unpack it. `gufo.exe` carries its ROCm, vcpkg and VC++ runtimes; it needs
   only a Strix Halo machine, a current AMD driver and the models. Its
   `README.txt` lists the downloads. Prefer building yourself? See
   [Build from source](#build-from-source).
3. Download the qualified Flash-Next files with the Hugging Face CLI
   (`pip install -U huggingface_hub`):

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
run start.cmd
```

`start.cmd` lists the servable models it finds with their draft and vision
sidecars, number keys toggle features (speculative decoding, prompt lookup,
survival stopping, the Latin draft vocabulary, images, thinking, context,
network access) and Enter starts the server on
`http://127.0.0.1:8080/v1`; it serves OpenAI-compatible Chat Completions
(text, images, tools, streaming) and Responses (text, streaming). The scriptable
form is `tools\windows\run-flash-next.ps1`, with the common variations
`-Think off`, `-Draft mtp3|off`, `-Context N` and `-Mode bench` (pp/tg at
depths 0..128K).

Then, from another terminal, ask it something:

```powershell
curl.exe http://127.0.0.1:8080/v1/models   # the model id to use below
curl.exe http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\": \"<id from /v1/models>\", \"messages\": [{\"role\": \"user\", \"content\": \"Say something\"}]}'
```

For Open WebUI, VS Code and OpenAI SDK clients, set the API base URL to
`http://127.0.0.1:8080/v1`. See the [API contract](docs/SERVER.md).

`start.cmd` sizes the context to your dedicated carve — 131,072 by default on
a 96 GB carve; a direct `gufo serve llm --model ...` uses the model's native
context unless you pass `--context N`, and generates until EOS or the context
is full. `--max-tokens N` sets a default response limit that clients can
override. Reasoning tokens count toward that response limit.

## Build from source

Windows 11 x64 on AMD Strix Halo (`gfx1151`), with test statuses, memory notes
and port internals in [docs/WINDOWS.md](docs/WINDOWS.md).

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.0.0 (Windows, gfx1151) | `C:\TheRock\build` | [`therock-dist-windows-gfx1151-10.0.0.tar.gz`](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz); extract so that `C:\TheRock\build\bin` exists. Its clang compiles C, C++ and HIP |
| vcpkg | `C:\vcpkg` | `vcpkg install icu curl openssl libpng libjpeg-turbo --triplet x64-windows` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only |
| CMake 3.21+ and Ninja | `PATH` | |

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, what fits, how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\gufo.exe
```

`build.ps1` imports the MSVC environment, configures with TheRock clang and
the vcpkg toolchain, builds, and copies the ROCm and vcpkg runtime DLLs plus
the hipBLASLt/rocBLAS kernel libraries next to `gufo.exe`, so `build\release`
runs as it is. `check.ps1` covers the whole setup: the toolchain, the build,
memory, every model it finds with a fit estimate, and runtime traps.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
Model weights are not bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
