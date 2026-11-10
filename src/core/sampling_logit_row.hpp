#ifndef GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_
#define GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_

// Internal to the sampler: the full-vocabulary block scan behind top-k.
// Include only from sampling.cpp and its harnesses.

#include <array>
#include <cstddef>
#include <limits>
#include <span>

#include "src/core/cpu_features.hpp"

#if (defined(__clang__) || defined(__GNUC__)) && defined(__x86_64__)
#include <immintrin.h>
#endif

namespace gufo::sampling {

constexpr std::size_t kSkipBlock = 64;

#if (defined(__clang__) || defined(__GNUC__)) && defined(__x86_64__)
// The scalar loop drops NaN values (`value > maximum ? value : maximum` keeps
// `maximum` when `value` is NaN), so the vector form keeps the value in the
// first operand and the accumulator second: _mm512_max_ps returns the
// accumulator whenever the value is NaN, and accumulators never hold NaN.
__attribute__((target("avx512f"))) inline float BlockMaximumAvx512(
    const float* p, std::size_t n) noexcept {
  const __m512 neg_inf =
      _mm512_set1_ps(-std::numeric_limits<float>::infinity());
  __m512 a0 = neg_inf;
  __m512 a1 = neg_inf;
  __m512 a2 = neg_inf;
  __m512 a3 = neg_inf;
  std::size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    a0 = _mm512_max_ps(_mm512_loadu_ps(p + i), a0);
    a1 = _mm512_max_ps(_mm512_loadu_ps(p + i + 16), a1);
    a2 = _mm512_max_ps(_mm512_loadu_ps(p + i + 32), a2);
    a3 = _mm512_max_ps(_mm512_loadu_ps(p + i + 48), a3);
  }
  a0 = _mm512_max_ps(a1, a0);
  a0 = _mm512_max_ps(a2, a0);
  a0 = _mm512_max_ps(a3, a0);
  for (; i + 16 <= n; i += 16)
    a0 = _mm512_max_ps(_mm512_loadu_ps(p + i), a0);
  if (i < n) {
    const __mmask16 tail = static_cast<__mmask16>((1U << (n - i)) - 1U);
    a0 = _mm512_mask_max_ps(a0, tail, _mm512_maskz_loadu_ps(tail, p + i), a0);
  }
  return _mm512_reduce_max_ps(a0);
}
#endif

/// Largest value in a block, ignoring NaN; +inf is returned as is.
[[nodiscard]] inline float BlockMaximum(std::span<const float> block) noexcept {
#if (defined(__clang__) || defined(__GNUC__)) && defined(__x86_64__)
  static const bool has_avx512f = gufo::HasAvx512f();
  if (has_avx512f)
    return BlockMaximumAvx512(block.data(), block.size());
#endif
  if (block.size() == kSkipBlock) {
    // A fixed trip count with independent lanes lets the compiler vectorize.
    std::array<float, 8> lanes;
    lanes.fill(-std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < kSkipBlock; i += lanes.size())
      for (std::size_t lane = 0; lane < lanes.size(); ++lane)
        lanes[lane] =
            block[i + lane] > lanes[lane] ? block[i + lane] : lanes[lane];
    float maximum = lanes[0];
    for (const float lane : lanes)
      maximum = lane > maximum ? lane : maximum;
    return maximum;
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (const float value : block)
    maximum = value > maximum ? value : maximum;
  return maximum;
}

}  // namespace gufo::sampling

#endif  // GUFO_CORE_SAMPLING_LOGIT_ROW_HPP_
