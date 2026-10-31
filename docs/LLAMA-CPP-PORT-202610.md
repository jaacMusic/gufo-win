# llama.cpp Tier A triage (2026-10-30)

Scan: `ggml-org/llama.cpp` master `8956c769e` (b12017), merges 2026-08-01 →
2026-10-30. Outcome of the Tier A "small, direct to our HIP path" list. Gufo
already vendors llama.cpp quant/attention HIP kernels
(`src/models/qwen38_flash_next/kernels/rocm/mmq/VENDOR.md`); three fixes were
imported, the rest needed no change.

| PR | Upstream change | Verdict here |
| --- | --- | --- |
| #28195 | RDNA3.5 64-row MMQ tiles for some quants | Already covered: `QFN_WMMA_MMQ_Y = 64` for every type and J (`mmq.hpp:134`, `get_mmq_y_host:143`, `get_mmq_y_device:152`). Only gap is upstream's J=96 carve-out, unmeasured on gfx1151. |
| #28935 | Broaden RDNA3.5 MoE tile heuristic | Already covered: `qfn_mmq_routed_tile_cols_for_counts` (`qfn_mmq.hip.cpp`) selects 16–80 columns without a capability gate. |
| #29478 | `__vsubss4` → `__vsub4` for packed dequant | **Ported**: 12 call sites in `mmq.hpp` and `vecdotq.hpp`. |
| #27962 | SWAR `__vsub4`/`__vcmpne4` in `vendors/hip.h` | **Ported**: both helpers replaced with the SWAR forms. |
| #30885 | bf16 WMMA operand vector type on gfx11/12 | **Ported**: `bf16x8_t`/`bf16x16_t` now `short` in Flash-Next and DeepSeek-V4 `mma` headers. |
| #29927, #30915 | `__builtin_amdgcn_perm` q1_0 unpack + fallback | N/A: `vec_dot_q1_0_q8_1` is a scalar loop; no q1_0 path is served. |
| #30103, #28576 | fattn race fix, MFMA fp32 accumulation | N/A: no `flash_attn`/`fattn` code in Gufo. |
| #31073 | KV rotation constants on GPU | N/A: RoPE is fused into Gufo's own kernels. |
| #29720 | rms_norm float4/register variants | N/A: Gufo norm kernels are custom and already have register variants. |
| #31321, #29749 | Loader staging/direct-io copies | N/A: Gufo has its own O_DIRECT loader (`core/hip/weight_upload.cpp`). |
| #27466 | ROCm radix TOP_K | N/A: Gufo already uses MSD radix top-k. |

Validation of the three imports: SWAR results checked against a per-byte
reference (200,036 input pairs, no mismatches); Flash-Next routed-WMMA,
projection and MoE-id operator tests pass on gfx1151; DeepSeek-V4 MMQ target
builds; format check passes. AR/MTP session replay was not run (needs the full
89 GiB model resident).
