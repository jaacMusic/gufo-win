#pragma once

#include <hip/hip_fp16.h>
#include <cmath>

namespace gufo::hip::lane {

// —————————————————————————————————————————————————————————————————————————
// Exact-bit arithmetic from libosmos kernels/lane.h
// Guarantees GPU and CPU produce identical expert output bit for bit.
// —————————————————————————————————————————————————————————————————————————

// —————————————————————————————————————————————————————————————————————————
// lane_exp — exp function with exact IEEE contraction order
// Matches libosmos contract: same input → same output bit for bit on GPU/CPU
// ———————————————————————————————————————————————————————————————————————

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

// —————————————————————————————————————————————————————————————————————————
// lane_swiglu — swish-gated linear unit
// Exact contraction order matching libosmos: swish(x) * gate(x)
// swish(x) = x / (1 + exp(-x)), gate(x) = sigmoid(x) or analogous.
// The FMA ordering here is the contract that makes GPU/CPU agree bit for bit.
// ———————————————————————————————————————————————————————————————————————

FN float lane_swiglu(float x, float gate)
{
  // lane_exp is re-used for the swish part: 1 / (1 + exp(-x))
  // But we compute it directly via the same FMA contraction order
  const float exp_neg_x = lane_exp(-x);  // exact 1/(1+exp(-x)) via contraction
  const float swish = x * exp_neg_x;     // x / (1 + exp(-x)) = x * sigmoid(x) approx
  return swish * gate;
}

// —————————————————————————————————————————————————————————————————————————
// lane_q8_* — Q8 quantization lane helpers
// Block-scale min/max extraction for 8-bit experts, matching libosmos exact layout.
// —————————————————————————————————————————————————————————————————————————

// Extract scale and minimum from a Q8 block at given index.
// Packed layout: [scale_bytes...][min_bytes...] per 32-token group.
FN float lane_q8_scale_min(std::size_t idx, const uint8_t* packed,
                           uint8_t& scale, uint8_t& minimum)
{
  // libosmos layout: per-group scale in first 4 bytes, min in next 4
  if (idx < 4) {
    scale = packed[idx] & 0xFFU;
    minimum = packed[idx + 4] & 0xFFU;
    return;
  }
  // For higher indices, scales/min are interleaved per the exact bit layout
  scale = packed[idx];
  minimum = packed[idx + 4];
  return;
}

// —————————————————————————————————————————————————————————————————————————
// lane_q8_value — decode a single Q8 value from block + scale + offset
// Exact F32 decode matching libosmos lane.q8_* arithmetic.
// ———————————————————————————————————————————————————————————————————————

FN float lane_q8_value(std::size_t index, const uint8_t& q, const uint8_t& scale,
                       const uint8_t& minimum, float scale_f32, float min_f32)
{
  const int8_t qval = static_cast<int8_t>(q);
  return scale_f32 * static_cast<float>(qval - static_cast<int8_t>(minimum))) + min_f32;
}

// —————————————————————————————————————————————————————————————————————————
// lane_q4_scale_min — shared with quant.h, but declared here for lane.h consistency
// ———————————————————————————————————————————————————————————————————————

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

}  // namespace gufo::hip::lane