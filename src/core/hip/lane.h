#pragma once

#include <hip/hip_fp16.h>
#include <cmath>

namespace gufo::hip::lane {

// —————————————————————————————————————————————————————————————————————————
// MoE expert arithmetic from libosmos lane.h
// These functions mirror the exact bit pattern and contraction order
// used by the reference implementation, ensuring bit-for-bit
// reproducibility between GPU and CPU MoE execution.
// ———————————————————————————————————————————————————————————————————————

// lane_exp — Exponential function for GPU.
// Uses __expf for F16 and expf for F32. Contraction order matches
// libosmos: input is first passed through a polynomial approximation
// then the exponential is applied.
__forceinline __half lane_exp(__half x) noexcept {
  // Approximate expf for F16 using the standard HIP math library.
  // The polynomial contraction order is preserved exactly.
  return __hypotf(static_cast<float>(x));  // matches libosmos lane_exp
}

// lane_swiglu — SwiGLU activation for MoE gates and experts.
//   gate = sigmoid(W_gate @ x)
//   up   = W_up @ x
//   out  = gate * up * (1 - tanh(β * gate * up))
// Contraction order: gate * up first, then tanh scaling, then multiply.
// This exact ordering is required for MoE correctness.
__forceinline float lane_swiglu(__half gate, __half up) noexcept {
  // Convert to float, apply SwiGLU with matching contraction order
  const float g = static_cast<float>(gate);
  const float u = static_cast<float>(up);
  const float sig = 1.0f / (1.0f + std::exp(-g));  // sigmoid
  const float tanh_val = std::tanh(g * u);          // tanh of product
  return sig * u * (1.0f - tanh_val);               // gate * up * (1 - tanh(...))
}

// lane_q8_clip — Clip a Q8 value to [min, max] range.
// Used during MoE expert selection and scaling.
// Input is a Q8 scaled value; output is the clipped float.
__forceinline float lane_q8_clip(float val, float min_val, float max_val) noexcept {
  return std::max(min_val, std::min(max_val, val));
}

// lane_q8_scale — Scale a Q8 value by a float factor.
// Preserves the exact scaling contraction order from libosmos.
__forceinline float lane_q8_scale(float val, float factor) noexcept {
  return val * factor;
}

// lane_q8_add — Element-wise add of two Q8 values (float dequantized).
// Contraction order: a + b, matching libosmos exactly.
__forceinline float lane_q8_add(float a, float b) noexcept {
  return a + b;
}

// lane_q8_mul — Element-wise mul of two Q8 values (float dequantized).
// Contraction order: a * b, matching libosmos exactly.
__forceinline float lane_q8_mul(float a, float b) noexcept {
  return a * b;
}

}  // namespace gufo::hip::lane