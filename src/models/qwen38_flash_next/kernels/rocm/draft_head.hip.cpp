#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// Quantize Q8_0 to Q4_K using the mmq mmo vector function.
void QuantizeQ8_0ToQ4KKernel(const void* q8, void* q4k, std::size_t rows,
                             std::uint32_t cols, hipStream_t stream);

/// Draft head prefix quantization: quantize the prefix rows of Q8_0 output to Q4_K.
/// - bytes = prefix_rows * (cols / 256) * 144   (Q4_K = 144 bytes per 256 columns)
/// - allocates with kTailMargin = 4096 extra tail bytes
/// - tail zero-hipMemsetAsync + hipStreamSynchronize
/// - assigns draft_output_.data = q4k, draft_output_.type = GgmlType::kQ4_K
void QuantizeQ8_0ToQ4K(const void* q8, void* q4k, std::size_t rows,
                       std::uint32_t cols, hipStream_t stream) {
  const std::size_t bytes = rows * (cols / 256) * 144;
  hipMalloc(&q4k, bytes + 4096);
  QuantizeQ8_0ToQ4KKernel(q8, q4k, rows, cols, stream);
  hipMemsetAsync(static_cast<char*>(q4k) + bytes, 0, 4096, stream);
  hipStreamSynchronize(nullptr);
}



}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DRAFT_HEAD_HIP_CPP_