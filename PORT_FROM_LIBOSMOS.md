# What Can Be Ported from libosmos to gufo-win

## Overview
`libosmos` is a deterministic inference engine for DeepSeek-V4.1-Flash (768B MoE) on AMD Strix Halo (gfx1151, 128 GB). Both projects target the same hardware, making many libosmos components portable to gufo-win.

## Highest Priority (Most Impact, Least Effort)

### 1. `kernels/lane.h` — MoE Expert Arithmetic (CRITICAL)
**Single header, ~200 lines.** Contains **bit-for-bit identical GPU/CPU arithmetic** essential for MoE correctness:
- `lane_exp` / `lane_log2e` / `lane_ln2_*` — math functions with exact reproducibility
- `lane_swiglu` — Swish-GLU activation (the most critical function for MoE experts)
- `lane_q8_*` — Q8 quantization for expert inputs
- **Why port**: gufo-win lacks this exact arithmetic. Without it, MoE models like DeepSeek-V4.1-Flash will have bit drift between GPU and CPU runs. This is the single most valuable port.

**Effort**: Add one `.h` file, `#include` it in the appropriate kernel compilation. No build system changes needed if placed in `src/core/hip/`.

### 2. `kernels/quant.h` — Advanced Quantization Decoders
**~400 lines.** Adds decoder kernels for many formats already recognized by gufo-win's GGUF reader but without GPU kernels:
- `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `MXFP8`
- `Q5_KR` (DSpark repacked tiles)
- `decode_h` helpers producing f16 output with exact rounding
- **Why port**: gufo-win's CPU dequant supports these formats, but GPU paths are limited. Porting these would enable GPU decoding of widely-used quantization formats.

**Effort**: Medium. Requ adapting the decode kernels to gufo-win's HIP naming conventions and block layouts. The `quant_grids.h` and `round.h` dependencies are small.

### 3. `kernels/rope.h` — RoPE with YaRN Interpolation
**~50 lines.** `rope_cs` function supports both:
- Plain rotation (theta_i as running product, like ggml's rope cache)
- YaRN interpolation (linear scaling + ext_factor mix, like llama.cpp GPU rope)
- **Why port**: gufo-win likely has plain RoPE but not the YaRN variant, which is important for long-context models.

**Effort**: Low. Add the header and the `rope_cs` extern "C" function.

## Medium Priority

### 4. `kernels/ops.h` — Attention Kernel Primitives
**~600 lines.** Provides gfx1151-optimized implementations of:
- `compress_kernel` — KV cache compression
- `put_rows_kernel` — putting rows at positions
- `index_score_kernel` — attention score computation (uses kvload, has staging)
- `select_kernel` — top-k selection (radix sort based)
- **Why port**: gufo-win already has similar kernels in `src/models/deepseek_v4_flash/kernels/rocm/`, but libosmos versions are verified against the PyTorch oracle and tuned for gfx1151.

**Effort**: High. These are large kernels with complex shared memory patterns. Could be referenced for optimization but not directly dropped in.

### 5. `kernels/attn.hip` — Pre-built Attention Kernel
A single HIP attention kernel file. Could be compiled and linked into gufo-win's model pipeline.

**Effort**: Medium. Would need to adapt the kernel entry points to gufo-win's naming conventions.

## Lower Priority (Structural, Higher Effort)

### 6. `libevt` — Event Scheduler
Manages multiple concurrent conversations on one engine. gufo-win has its own session/model management.

### 7. `tools/api/` — OpenAI-Compatibile Server
Full C server with tokenizer, chat encoding, streaming, tool calls. Could be adapted as an alternative server for gufo-win but is a substantial codebase.

### 8. `tools/install/` — Installer/Pack System
Pack manifest and installation scripts. Less relevant to the inference engine itself.

## Summary: What to Port First

| Priority | Component | Lines | Impact | Effort |
|----------|-----------|-------|--------|--------|
| 1 | `lane.h` | ~200 | Critical for MoE correctness | Trivial |
| 2 | `quant.h` | ~400 | Adds GPU decode for many GGUF formats | Medium |
| 3 | `rope.h` | ~50 | YaRN RoPE support | Trivial |
| 4 | `ops.h` | ~600 | Optimized attention kernels | High |
| 5 | `attn.hip` | Unknown | Pre-built attention kernel | Medium |

**Start with `lane.h`**. It's a single header, the functions are isolated and well-documented, and it directly enables running MoE models bit-for-bit reproducibly on gfx1151.