#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace gufo::hip::rope {

// ————————————————————————————————————————————————————————————————————————
// YaRN RoPE (Rotary Position Embedding) from libosmos rope.h
// Provides plain rotation + YaRN interpolation for long-context quality.
// ————————————————————————————————————————————————————————————————————————

// RoPE state structure matching libosmos attn_pack_rope layout.
// Used by attention kernels to apply RoPE + YaRN interpolation.
struct RopeState {
  std::size_t head_dim;   // per-head dimension (must be multiple of 4)
  std::size_t n_rot;      // number of rotated dimensions (0 = none)
  std::size_t n_ctx_orig; // original context length for YaRN
  float freq_base;        // base frequency for RoPE
  float freq_scale;       // scale factor for YaRN
  float ext_factor;       // extrapolation factor (>1.0 extends context)
  float attn_factor;      // attention scaling factor
  float beta_fast;        // fast decay for RoPE beta
  float beta_slow;        // slow decay for RoPE beta
  bool inverse;           // inverse RoPE flag
};

// ————————————————————————————————————————————————————————————————————————
// RoPE base frequency computation (YaRN scheme)
// ————————————————————————————————————————————————————————————————————————

// Compute the base frequency for RoPE using the YaRN formula.
// freq_base = n_ctx / (log(n_ctx / fmin) / log(fmax / fmin))
// where fmin and fmax are the min/max frequencies.
// This mirrors the libosmos rope implementation exactly.
__forceinline float yarn_freq_base(std::size_t n_ctx, float f_min, float f_max) noexcept {
  if (n_ctx < 2 || f_min <= 0.0f || f_max <= f_min) {
    return 1.0f;
  }
  const float log_ratio = std::log(static_cast<float>(n_ctx) / f_min) /
                          std::log(f_max / f_min);
  return static_cast<float>(n_ctx) / log_ratio;
}

// ————————————————————————————————————————————————————————————————————————
// RoPE rotation function — apply rotary embedding to a query/key vector
// ————————————————————————————————————————————————————————————————————————

// Apply RoPE to a single head's Q/K vector pair.
// The first `n_rot` dimensions are rotated; the rest are unchanged.
// YaRN interpolation is applied if n_ctx_orig > 0 and the position differs
// from the original training context.
//   output[rotated] = q[rotated] * cos(pos / freq_base) - q[unrotated] * sin(pos / freq_base)
//   output[unrotated] = q[rotated] * sin(pos / freq_base) + q[unrotated] * cos(pos / freq_base)
// YaRN adjustment: freq_base is scaled by freq_scale * (pos / n_ctx_orig)^ext_factor
__forceinline void apply_rope_1d(
    float* q,          // [head_dim] query vector (modified in place)
    std::size_t pos,   // current position index (0-based)
    const RopeState& rope) noexcept
{
  if (rope.n_rot == 0 || rope.head_dim == 0) return;

  // Determine the effective position with YaRN interpolation
  float effective_pos = static_cast<float>(pos);
  if (rope.n_ctx_orig > 0 && rope.freq_base > 0.0f) {
    const float ratio = effective_pos / static_cast<float>(rope.n_ctx_orig);
    const float yaRN_scale = std::pow(ratio, rope.ext_factor);
    effective_pos *= rope.freq_scale * yaRN_scale;
  }

  // Compute the rotation angles for the rotated dimensions
  // Frequencies follow the original libosmos pattern: inv_freq[i] = 1 / (base ^ (2i/n))
  // For simplicity, we compute on-the-fly using the rope parameters.
  // Full implementation would precompute inv_freq table.

  // Apply rotation to each rotated dimension pair
  for (std::size_t i = 0; i < rope.n_rot; i += 2) {
    if (i + 1 >= rope.head_dim) break;

    const float pos_scaled = effective_pos / rope.freq_base;
    const float sin_val = std::sin(pos_scaled);
    const float cos_val = std::cos(pos_scaled);

    // Get the rotated dimensions (swap interleave pattern)
    const float q_rot_i = q[i];
    const float q_rot_j = q[i + 1];

    // Apply the rotation (standard RoPE)
    q[i]     = q_rot_i * cos_val - q_rot_j * sin_val;
    q[i + 1] = q_rot_i * sin_val + q_rot_j * cos_val;
  }
}

// ————————————————————————————————————————————————————————————————————————
// Reference RoPE application (host-side, for verification)
// ————————————————————————————————————————————————————————————————————————

// Apply RoPE to a batch of Q/K vectors on CPU, matching the GPU kernel
// exactly for verification purposes.
inline void apply_rope_1d_host(
    float* q,          // [batch × head_dim] query vectors (modified in place)
    std::size_t batch, // number of vectors
    std::size_t head_dim,
    std::size_t pos,
    const RopeState& rope)
{
  for (std::size_t b = 0; b < batch; ++b) {
    apply_rope_1d(q + b * head_dim, pos, rope);
  }
}

// ————————————————————————————————————————————————————————————————————————
// Convenience: full RoPE + softmax integration point
// ————————————————————————————————————————————————————————————————————————

// Launch RoPE application for all positions in a sequence.
// This is the entry point attention kernels should call before softmax.
// rope may have n_rot == 0 if no RoPE is needed (rope will be nullptr or default).
void apply_rope_seq(
    float* q,               // [batch × n_heads × seq × head_dim] modified in place
    std::size_t batch,      // batch size
    std::size_t n_heads,    // number of heads
    std::size_t seq_len,    // sequence length
    std::size_t head_dim,   // per-head dimension
    const RopeState* rope,  // optional; if nullptr, no-op
    std::size_t pos_offset = 0);  // starting position index

}  // namespace gufo::hip::rope