#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_

#include <cmath>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

struct Q2KBlock {
  std::uint8_t scales[16];
  std::uint8_t qs[64];
  __half d;
  __half dmin;
};
static_assert(sizeof(Q2KBlock) == 84, "block_q2_K must be 84 bytes");

// One thread per 256-value super-block. Each 16-value group is a half of one
// Q8_0 block and owns one `scales` byte; its codes live in `qs` at
// 32*(g/8) + 16*(g%2) + l, plane g/2 % 4.
//
// shortcut: the min/max fit per group is coarser than ggml's iterative
// make_qkx2_quants. The screen only ranks rows, and the picks are rescored
// exactly, so a slightly worse fit costs nothing here.
__global__ void QuantizeQ2KKernel(const Q8_0Block* __restrict__ src,
                                  Q2KBlock* __restrict__ dst,
                                  std::size_t blocks) {
  const std::size_t b = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (b >= blocks) {
    return;
  }
  const Q8_0Block* in = src + b * 8;
  Q2KBlock out{};
  float scale[16];
  float offset[16];
  float max_scale = 0.0F;
  float max_offset = 0.0F;
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const Q8_0Block& blk = in[g >> 1];
    const std::int8_t* qs = blk.qs + 16 * (g & 1);
    const float d = __half2float(blk.d);
    float lo = INFINITY;
    float hi = -INFINITY;
    for (int l = 0; l < 16; ++l) {
      const float v = d * static_cast<float>(qs[l]);
      lo = fminf(lo, v);
      hi = fmaxf(hi, v);
    }
    offset[g] = fmaxf(0.0F, -lo);
    scale[g] = (hi + offset[g]) / 3.0F;
    max_scale = fmaxf(max_scale, scale[g]);
    max_offset = fmaxf(max_offset, offset[g]);
  }
  out.d = __float2half(max_scale / 15.0F);
  out.dmin = __float2half(max_offset / 15.0F);
  const float d = __half2float(out.d);
  const float dmin = __half2float(out.dmin);
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    int sc = 0;
    int mn = 0;
    if (d > 0.0F) {
      sc = __float2int_rn(scale[g] / d);
      sc = sc < 0 ? 0 : (sc > 15 ? 15 : sc);
    }
    if (dmin > 0.0F) {
      mn = __float2int_rn(offset[g] / dmin);
      mn = mn < 0 ? 0 : (mn > 15 ? 15 : mn);
    }
    out.scales[g] = static_cast<std::uint8_t>((mn << 4) | sc);
  }
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    const float dg = d * (out.scales[g] & 0xF);
    const float mg = dmin * (out.scales[g] >> 4);
    const Q8_0Block& blk = in[g >> 1];
    const std::int8_t* qs = blk.qs + 16 * (g & 1);
    const float bd = __half2float(blk.d);
    std::uint8_t* dst_qs = out.qs + 32 * (g >> 3) + 16 * (g & 1);
    const int shift = 2 * ((g >> 1) & 3);
    for (int l = 0; l < 16; ++l) {
      int code = 0;
      if (dg > 0.0F) {
        const float v = bd * static_cast<float>(qs[l]);
        code = __float2int_rn((v + mg) / dg);
        code = code < 0 ? 0 : (code > 3 ? 3 : code);
      }
      dst_qs[l] |= static_cast<std::uint8_t>(code << shift);
    }
  }
  dst[b] = out;
}

}  // namespace

void QuantizeHeadQ2K(const void* q8, void* q2k, std::uint32_t rows,
                     std::uint32_t cols, hipStream_t stream) {
  const std::size_t blocks = static_cast<std::size_t>(rows) * (cols / 256);
  if (blocks == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const auto grid = static_cast<std::uint32_t>((blocks + kThreads - 1) /
                                              kThreads);
  QuantizeQ2KKernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const Q8_0Block*>(q8), static_cast<Q2KBlock*>(q2k), blocks);
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_
