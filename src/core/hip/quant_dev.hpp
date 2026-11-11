#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

#include "ggml_dequant.hpp"

namespace gufo::hip::quant {

using gufo::quant::kIq3sGrid;

// —————————————————————————————————————————————————————————————————————————
// Canonical block layouts (single source of truth; match ggml_dequant.hpp)
// —————————————————————————————————————————————————————————————————————————

struct block_iq4_nl {
  uint16_t d;
  uint8_t qs[16];
};

struct block_q4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};

struct block_q5_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};

struct block_q6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};

struct block_iq4_xs {
  uint16_t d;
  uint16_t scales_h;
  uint8_t scales_l[4];
  uint8_t qs[128];
};

struct block_iq3_s {
  uint16_t d;
  uint8_t qs[64];
  uint8_t qh[8];
  uint8_t signs[32];
  uint8_t scales[4];
};

// —————————————————————————————————————————————————————————————————————————
// Decode kernels (device-side, matching libosmos kernels/quant.h arithmetic)
// —————————————————————————————————————————————————————————————————————————

#if defined(__HIPCC__)
#define FN __device__ static inline
#else
#define FN static inline
#endif

FN float lane_exp(float x)
{
  if (x > 88.72283905f) return INFINITY;
  const float k = rintf(x * 1.44269504089f);
  float r = fmaf(-k, 0.693145751953125f, x);
  r = fmaf(-k, 1.428606765330187e-06f, r);
  float p = 1.0f / 5040.0f;
  p = fmaf(p, r, 1.0f / 720.0f);
  p = fmaf(p, r, 1.0f / 120.0f);
  p = fmaf(p, r, 1.0f / 24.0f);
  p = fmaf(p, r, 0.5f);
  p = fmaf(p, r, 1.0f);
  p = fmaf(p, r, 1.0f);
  return ldexpf(p, (int)k);
}

FN float lane_q4_scale_min(std::size_t idx, const uint8_t* packed,
                           uint8_t& scale, uint8_t& minimum)
{
  if (idx < 4) {
    scale = packed[idx] & 0x3FU;
    minimum = packed[idx + 4] & 0x3FU;
    return;
  }
  scale = static_cast<uint8_t>((packed[idx + 4] & 0x0FU) |
                               ((packed[idx - 4] >> 6U) << 4U));
  minimum = static_cast<uint8_t>((packed[idx + 4] >> 4U) |
                                 ((packed[idx] >> 6U) << 4U));
  return;
}

FN float q4_value(const block_q4_K& block, std::size_t index)
{
#pragma HIP UNROLL
  const std::size_t group = index / 32;
  const std::size_t pair = group / 2;
  const bool high_nibble = (group & 1U) != 0;
  const std::size_t lane = index % 32;
  const uint8_t packed = block.qs[(pair * 32) + lane];
  const uint8_t quant = high_nibble ? packed >> 4U : packed & 0x0FU;
  uint8_t scale = 0, minimum = 0;
  lane_q4_scale_min(group, block.scales, scale, minimum);
  return Fp16ToFloat(block.d) * static_cast<float>(scale) *
         static_cast<float>(quant) -
       Fp16ToFloat(block.dmin) * static_cast<float>(minimum);
}

FN float q5_value(const block_q5_K& block, std::size_t index)
{
  const std::size_t gg = index / 64;   // 0..3
  const std::size_t wv = index % 64;   // 0..63
  const std::size_t lane = wv % 32;    // 0..31
  const bool lohalf = (wv < 32);
  const uint8_t qb = block.qs[(gg * 32) + lane];
  const uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
  const uint8_t qhb = block.qh[lane];
  const int bit = static_cast<int>(2 * gg) + (lohalf ? 0 : 1);  // 0..7
  const uint8_t quant =
      static_cast<uint8_t>(quant4 + ((qhb >> bit) & 1U ? 16U : 0U));  // 0..31
  const std::size_t sis = (2 * gg) + (lohalf ? 0 : 1);             // 0..7
  uint8_t sc = 0, m = 0;
  lane_q4_scale_min(sis, block.scales, sc, m);
  return Fp16ToFloat(block.d) * static_cast<float>(sc) *
         static_cast<float>(quant) -
       Fp16ToFloat(block.dmin) * static_cast<float>(m);
}

FN float q6_value(const block_q6_K& block, std::size_t index)
{
  const std::size_t half = index / 128;
  const std::size_t within_half = index % 128;
  const std::size_t segment = within_half / 32;
  const std::size_t lane = within_half % 32;
  const uint8_t qh = block.qh[(half * 32) + lane];

  uint8_t low = 0, high = 0;
  switch (segment) {
    case 0:
      low = block.ql[lane] & 0x0FU;
      high = qh & 0x03U;
      break;
    case 1:
      low = block.ql[32 + lane] & 0x0FU;
      high = (qh >> 2U) & 0x03U;
      break;
    case 2:
      low = block.ql[lane] >> 4U;
      high = (qh >> 4U) & 0x03U;
      break;
    default:
      low = block.ql[32 + lane] >> 4U;
      high = (qh >> 6U) & 0x03U;
      break;
  }

  const std::size_t scale_index = (half * 8) + (lane / 16) + (segment * 2);
  const int quant = static_cast<int>((high << 4U) | low) - 32;
  return Fp16ToFloat(block.d) * static_cast<float>(block.scales[scale_index]) *
         static_cast<float>(quant);
}

FN float iq4_nl_value(const block_iq4_nl& block, std::size_t index)
{
#pragma HIP UNROLL
  const std::size_t lane = index % 16;
  const bool high_nibble = index >= 16;
  const uint8_t packed = block.qs[lane];
  const uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  return Fp16ToFloat(block.d) * static_cast<float>(kValuesIq4Nl[code]);
}

FN float iq4_xs_value(const block_iq4_xs& block, std::size_t index)
{
  const std::size_t ib = index / 32;   // 0..7 sub-block
  const std::size_t within = index % 32;
  const std::size_t lane = within % 16;
  const bool high_nibble = within >= 16;
  const uint8_t packed = block.qs[(ib * 16) + lane];
  const uint8_t code = high_nibble ? (packed >> 4U) : (packed & 0x0FU);
  const int ls =
      static_cast<int>((block.scales_l[ib / 2] >> (4U * (ib % 2))) & 0x0FU) |
      static_cast<int>(((block.scales_h >> (2U * ib)) & 0x03U) << 4U);
  return Fp16ToFloat(block.d) * static_cast<float>(ls - 32) *
         static_cast<float>(kValuesIq4Nl[code]);
}

FN float iq3_s_value(const block_iq3_s& block, std::size_t index)
{
  const std::size_t ib32 = index / 32;    // 0..7 sub-block
  const std::size_t within = index % 32;   // 0..31
  const std::size_t l = within / 8;        // 0..3 group of eight
  const std::size_t j = within % 8;        // 0..7 element in the group
  const std::size_t half = j / 4;          // 0 -> grid1, 1 -> grid2
  const std::size_t jj = j % 4;

  const uint8_t qh_byte = block.qh[ib32];
  const std::size_t qs_index = (ib32 * 8) + (2 * l) + half;
  const int shift = static_cast<int>(8 - (2 * l) - half);
  const uint32_t grid_index =
      static_cast<uint32_t>(block.qs[qs_index]) |
      ((static_cast<uint32_t>(qh_byte) << shift) & 256U);
  const auto* grid = reinterpret_cast<const uint32_t*>(kIq3sGrid);

  const uint8_t sign_byte = block.signs[(ib32 * 4) + l];
  const bool negate = (sign_byte & kSignMaskIq2xs[j]) != 0;

  const uint8_t scale_byte = block.scales[ib32 / 2];
  const int scale_nibble = ib32 % 2 == 0
      ? static_cast<int>(scale_byte & 0x0FU)
      : static_cast<unsigned>(scale_byte >> 4U);
  const float db = Fp16ToFloat(block.d) * static_cast<float>(1 + (2 * scale_nibble));

  const float magnitude = static_cast<float>(grid[jj]);
  return negate ? -db * magnitude : db * magnitude;
}

// —————————————————————————————————————————————————————————————————————————

// —————————————————————————————————————————————————————————————————————————

// —————————————————————————————————————————————————————————————————————————

constexpr uint8_t kSignMaskIq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// —————————————————————————————————————————————————————————————————————————

// —————————————————————————————————————————————————————————————————————————

FN float Fp16ToFloat(uint16_t h) noexcept
{
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1F;
  const uint32_t mant = h & 0x3FF;

  if (exp == 0) {
    if (mant == 0) {
      float zero = 0.0f;
      std::memcpy(&zero, &sign, sizeof(float));
      return zero;
    }
    const float f = (static_cast<float>(mant) / 1024.0f) *
                    std::ldexp(1.0f, -14);
    return (h & 0x8000) ? -f : f;
  }
  if (exp == 31) {
    const uint32_t val = sign | 0x7F800000U | (mant << 13);
    float f = 0.0f;
    std::memcpy(&f, &val, sizeof(float));
    return f;
  }
  const uint32_t val = sign | ((exp + 112) << 23) | (mant << 13);
  float f = 0.0f;
  std::memcpy(&f, &val, sizeof(float));
  return f;
}

// —————————————————————————————————————————————————————————————————————————

// —————————————————————————————————————————————————————————————————————————

constexpr int8_t kValuesIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22,
                                       -10, 1, 13, 25, 38, 53, 69, 89, 113};

// —————————————————————————————————————————————————————————————————————————

}  // namespace gufo::hip::quant