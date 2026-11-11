# Performance Ranking: What to Port from libosmos for Maximum Gain

## Ranking by Expected Token/sec Improvement

### Tier 1: Highest Impact (Try First)

#### 1. `kernels/quant.h` — GPU Quantization Decoders
- **What**: GPU decode kernels for IQ1_M, IQ2_XXS, MXFP4, MXFP8, Q5_KR, and other formats
- **Why port**: gufo-win currently decodes these on CPU. Moving decode to GPU:
  - Reduces host-device transfer overhead
  - Frees CPU for other work (sampling, prompt processing)
  - Enables larger batch sizes by offloading per-token work
- **Expected gain**: 15-30% throughput increase for quantized models (the most common case)
- **Effort**: Medium (adapt ~400 lines of decode kernels to gufo-win's HIP naming)
- **Risk**: Low — these are element-wise operations, no shared memory contention

#### 2. `kernels/ops.h` — Attention Kernel Primitives
- **What**: `compress_kernel` (KV cache compression), `index_score_kernel` (attention scores), `select_kernel` (top-k)
- **Why port**: Attention is the dominant computation in LLM inference. These are:
  - gfx1151-optimized (targeted at Strix Halo)
  - Verified against PyTorch oracle numerics
  - Already have the complex shared memory patterns figured out
- **Expected gain**: 20-40% throughput increase if porting `index_score_kernel` or `compress_kernel` into the active codepath
- **Effort**: High (large kernels, complex staging, shared memory)
- **Risk**: Medium — must preserve exact numerics; the oracle verification helps

### Tier 2: Enabling + Indirect Gains

#### 3. `kernels/lane.h` — MoE Expert Arithmetic
- **What**: `lane_exp`, `lane_swiglu`, `lane_q8_*` — bit-for-bit GPU/CPU math
- **Why port**: **Required** for MoE models (DeepSeek-V4.1-Flash, etc.) to run at all without bit drift. Without this, MoE experts produce different results on GPU vs CPU.
- **Expected gain**: Enables MoE execution (would otherwise fail/crash). No direct speed increase, but makes MoE workloads possible.
- **Effort**: Trivial (single header, ~200 lines)
- **Risk**: Very low — explicitly documents IEEE contraction order

#### 4. `kernels/rope.h` — RoPE with YaRN
- **What**: `rope_cs` with plain rotation + YaRN interpolation
- **Why port**: YaRN (RoPE + YaRN) is important for long-context quality but doesn't affect throughput.
- **Expected gain**: None (quality improvement only)
- **Effort**: Trivial (add header + one function)
- **Risk**: Very low

### Tier 3: Lowest Priority for Performance

#### 5. `kernels/attn.hip` — Pre-built Attention Kernel
- **What**: Single HIP attention kernel file
- **Why port**: Could be compiled and linked, but less tailored than the `ops.h` primitives
- **Expected gain**: Moderate, depends on integration
- **Effort**: Medium
- **Risk**: Medium — integration points may not match

---

## Summary: Optimal Porting Order for Performance

1. **`kernels/quant.h`** — Immediate throughput gain for quantized models
2. **`kernels/ops.h`** — Largest potential gain, but higher effort; port select kernels first
3. **`kernels/lane.h`** — Essential for MoE correctness; trivial add
4. **`kernels/rope.h`** — Quality fix; add if long-context is needed
5. **`kernels/attn.hip`** — Only if the pre-built kernel fits the use case

**Start with `kernels/quant.h`** for the fastest performance win with lowest risk.