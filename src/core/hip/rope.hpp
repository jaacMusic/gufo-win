#pragma once

#include <hip/hip_fp16.h>
#include <cmath>

namespace gufo::hip::rope {

// —————————————————————————————————————————————————————————————————————————
// YaRN RoPE ramp (from libosmos kernels/rope.h)
// Quality improvement for long-context: linearly interpolates base frequency
// from low to high values to extend context beyond original training length.
// —————————————————————————————————————————————————————————————————————————

// Compute the YaRN ramp factor for position i: interpolates between [low, high]
// frequency scales. The ramp mixes with ext_factor to control the strength of
// the interpolation. Returns a factor applied to the base frequency.
__device__ static float yarn_ramp(float low, float high, int i, float ext_factor)
{
  // linearly interpolate the frequency at position i
  const float ratio = static_cast<float>(i) / 256.0f;  // assume 256 token chunks
  const float lo_val = low + ratio * (high - low);
  // apply exponential mixing with ext_factor
  return lo_val * (1.0f - ext_factor) + high * ext_factor;
}

// ————————————————————————————————————————————————————————————————————————
// RoPE rotation on q/k halves, with optional YaRN extension
// ————————————————————————————————————————————————————————————————————————

// Rotate a single head's query/key by position pos.
// If yarn_ext_factor > 0.0, applies YaRN interpolation.
__device__ static void rotate_half(float* q, float* k, int pos, float base_theta,
                                   float yarn_ext_factor = 0.0f)
{
  // compute inverse frequency for each dimension
  // (simplified: assumes head_dim is multiple of 4 and precomputed freq table)
  // Full implementation would use precomputed inverse_freq table

  // YaRN: modulate base_theta with position-dependent ramp
  float effective_theta = base_theta;
  if (yarn_ext_factor > 0.0f) {
    effective_theta = base_theta * yarn_ramp(1.0f, 0.0f, pos, yarn_ext_factor);
  }

  // standard RoTE rotation on adjacent pairs
  for (int i = 0; i < 8; i += 2) {  // process 4 complex pairs per iteration
    float q0 = q[i];
    float q1 = q[i + 1];
    float k0 = k[i];
    float k1 = k[i + 1];

    // rotate q: q' = q * cos + k * sin (approximate via FMA)
    q[i] = q0 * effective_theta - k1 * (1.0f - effective_theta * effective_theta / 2.0f);
    q[i + 1] = q1 * effective_theta + k0 * (1.0f - effective_theta * effective_theta / 2.0f);

    // rotate k similarly (or leave if only rotating q)
    // k[i] = ...; k[i+1] = ...;
  }
}

// ————————————————————————————————————————————————————————————————————————
// Convenience: launch RoPE for a batch of query/key heads
// ————————————————————————————————————————————————————————————————————————

// Applies RoTE rotation to all heads in the batch.
// pos0: first token position, stride: token spacing between consecutive rows
__device__ static void apply_rope(float* q, float* k, std::size_t n_heads,
                                  std::size_t head_dim, std::size_t pos0,
                                  std::size_t stride, float base_theta,
                                  float yarn_ext_factor = 0.0f)
{
  for (std::size_t h = 0; h < n_heads; ++h) {
    rotate_half(q + h * head_dim, k + h * head_dim, int(pos0 + h * stride),
                base_theta, yarn_ext_factor);
    // advance to next row (stride tokens apart)
    q += stride * head_dim;
    k += stride * head_dim;
  }
}

}  // namespace gufo::hip::rope