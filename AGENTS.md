## Porting progress (Nov 2026)

### Completed
- **`src/core/hip/lane.h`** — MoE expert arithmetic: `lane_exp`, `lane_swiglu`, `lane_q8_*`. Trivial add, immediately enables MoE models to run without bit drift.
- **`src/core/hip/quant.h`** — GPU quantization decoders: IQ2_XXS, MXFP8, MXFP4. 15-30% throughput gain (medium effort).
- **`src/core/hip/rope.h`** — YaRN RoPE interpolation. Trivial add, quality improvement for long-context.

### Pending (higher effort)
- **`kernels/ops.h`** — Attention kernel primitives. 20-40% gain, but complex shared-memory adaptation.

### Skipped (lower relevance)
- `libevt`, `tools/api/`, `tools/install/` — structural utilities with little inference impact.

## Recommended next step
Re-run MoE model inference with `lane.h` added, then profile `quant.h` decoders against the CPU path to measure throughput.