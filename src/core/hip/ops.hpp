#pragma once

#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace gufo::hip::ops {

// —————————————————————————————————————————————————————————————————————————
// Attention score kernel from libosmos kernels/ops.h
// Scaled dot-product: softmax(QK^T / sqrt(d_k)) @ V
// Reference implementation for gfx1151; production uses WMMA N64 group-4.
// ———————————————————————————————————————————————————————————————————————

// Compute scaled dot-product attention scores: scale = 1.0 / sqrt(dim)
// Q: [batch*n_heads, seq_len_q, dim], K: [batch*n_heads, seq_len_k, dim]
// out: [batch*n_heads, seq_len_q, seq_len_k] pre-softmax scores
// scale_f32 = 1.0f / sqrtf(dim)
__device__ float* attention_scores(
    float* out, const float* Q, const float* K, std::size_t batch,
    std::size_t n_heads, std::size_t dim, std::size_t seq_len_q,
    std::size_t seq_len_k, float scale_f32) {
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < n_heads; ++h) {
      const auto q_base = Q + (b * n_heads + h) * seq_len_q * dim;
      const auto k_base = K + (b * n_heads + h) * seq_len_k * dim;
      auto o_base = out + (b * n_heads + h) * seq_len_q * seq_len_k;
      for (std::size_t i = 0; i < seq_len_q; ++i) {
        for (std::size_t j = 0; j < seq_len_k; ++j) {
          float acc = 0.0f;
          for (std::size_t kk = 0; kk < dim; ++kk) {
            acc += q_base[i * dim + kk] * k_base[j * dim + kk];
          }
          o_base[i * seq_len_k + j] = acc * scale_f32;
        }
      }
    }
  }
  return out;
}

// ———————————————————————————————————————————————————————————————————————
// Softmax over the last dimension (the attention score dimension)
// Applied row-by-row; each row has "size" elements.
// Writes softmax output back to out[0..size-1] and returns the max
// per row in m[0..batch-1] (optional; may be nullptr).
// ———————————————————————————————————————————————————————————————————————

__device__ void softmax(float* out, const float* in, std::size_t size,
                        std::size_t batch, float* max_val) {
  for (std::size_t b = 0; b < batch; ++b) {
    // Find max for numerical stability
    float m = -INFINITY;
    for (std::size_t i = 0; i < size; ++i) {
      m = fmaxf(m, in[b * size + i]);
    }
    if (max_val) max_val[b] = m;

    // Exponentiate and normalize
    float sum = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
      out[b * size + i] = expf(in[b * size + i] - m);
      sum += out[b * size + i];
    }
    for (std::size_t i = 0; i < size; ++i) {
      out[b * size + i] /= sum;
    }
  }
}

// ———————————————————————————————————————————————————————————————————————
// Top-k selection on attention scores
// Keep only the top-k scores per row, zeroing the rest.
// out_scores: output scores (modified in-place)
// out_indices: optional output indices of top-k selections
// ———————————————————————————————————————————————————————————————————————

// Select top-k scores per row; k may exceed size (in which case all kept).
// Modifies scores in-place; indices output if idx != nullptr.
__device__ void topk_scores(float* scores, std::size_t k, std::size_t size,
                            int* indices) {
  if (k >= size) return;  // keep all entries

  // Partial selection: for each row, move top-k to the front
  for (std::size_t b = 0; b < (size / 32 + 1); ++b) {  // simplified: one row
    // Simple sort-based selection for the row
    for (std::size_t i = 0; i < size; ++i) {
      for (std::size_t j = i + 1; j < size; ++j) {
        std::size_t ii = b * size + i;
        std::size_t jj = b * size + j;
        if (scores[jj] > scores[ii]) {
          std::swap(scores[ii], scores[jj]);
          if (indices) std::swap(indices[ii], indices[jj]);
        }
      }
    }
  }
  // Zero out the bottom (size - k) entries per row
  for (std::size_t b = 0; b < 1; ++b) {  // one row for now
    for (std::size_t i = k; i < size; ++i) {
      scores[b * size + i] = 0.0f;
    }
  }
}

// ———————————————————————————————————————————————————————————————————————
// Simple attention output: softmax_scores @ V
// scores: [batch*n_heads, seq_len_q, seq_len_k] softmax probabilities
// V: [batch*n_heads, seq_len_k, head_dim]
// out: [batch*n_heads, seq_len_q, head_dim]
// ———————————————————————————————————————————————————————————————————————

__device__ float* attention_output(
    float* out, const float* scores, const float* V,
    std::size_t batch, std::size_t n_heads, std::size_t seq_len_q,
    std::size_t head_dim) {
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t h = 0; h < n_heads; ++h) {
      const auto s_base = scores + (b * n_heads + h) * seq_len_q * /*seq_len_k*/ 64; // simplified
      const auto v_base = V + (b * n_heads + h) * 64 * head_dim;  // simplified: seq_len_k=64
      auto o_base = out + (b * n_heads + h) * head_dim;
      for (std::size_t i = 0; i < head_dim; ++i) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < 64; ++j) {  // simplified: seq_len_k=64
          acc += s_base[i * 64 + j] * v_base[j * head_dim + i];
        }
        o_base[i] = acc;
      }
    }
  }
  return out;
}

}  // namespace gufo::hip::ops