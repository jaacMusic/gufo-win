#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Check(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}

struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

}  // namespace

int main() {
  try {
    using gufo::models::qwen38_flash_next::rocm::BlasLt;
    std::string error;
    auto warmed = BlasLt::Create(nullptr, &error);
    Require(warmed != nullptr, error);
    constexpr int max_n = 4096;
    for (const auto [type, m, k] :
         {std::tuple{HIP_R_16F, 96, 2560}, std::tuple{HIP_R_16F, 513, 2560},
          std::tuple{HIP_R_16F, 17, 2560}, std::tuple{HIP_R_16F, 33, 96},
          std::tuple{HIP_R_16BF, 64, 2560}, std::tuple{HIP_R_16BF, 128, 2560},
          std::tuple{HIP_R_16BF, 256, 2560},
          std::tuple{HIP_R_16BF, 512, 2560}}) {
      // Prepare before the inputs exist, then reuse the plan in Gemm.
      Require(warmed->Prepare(type, m, max_n, k, &error), error);
      std::vector<__half> weights(static_cast<std::size_t>(m) * k);
      std::vector<__half> input(static_cast<std::size_t>(max_n) * k);
      unsigned rng = 123;
      for (auto* values : {&weights, &input}) {
        for (auto& v : *values) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          v = __float2half_rn(static_cast<float>(int(rng % 8193) - 4096) /
                              4096.0F);
        }
      }
      Buffer a(weights.size() * sizeof(__half));
      Buffer b(input.size() * sizeof(__half));
      Buffer c(static_cast<std::size_t>(m) * max_n * sizeof(float));
      Check(hipMemcpy(a.data, weights.data(), weights.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(b.data, input.data(), input.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      if (k == 2560) {
        // Exercise an unaligned cache boundary, a wide prefill split and the
        // short tails a checkpoint restore finishes a prompt with. A row's
        // sums must not depend on its position in the batch: the indexer's
        // BF16 scores feed a top-k whose near-ties would otherwise flip with
        // the prefill chunking, and F16 router and recurrent-gate errors
        // amplify across layers.
        for (const auto [total, boundary] :
             {std::pair{136, 94}, std::pair{2048, 1025}, std::pair{3487, 777},
              std::pair{1619, 1612}, std::pair{1619, 1618},
              std::pair{1619, 1523}, std::pair{300, 293}}) {
          std::vector<float> bulk(std::size_t(m) * total), split(bulk.size());
          Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                               type, m, total, k, &error),
                  error);
          Check(hipMemcpy(bulk.data(), c.data, bulk.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          for (const auto [off, n] : {std::pair{0, boundary},
                                      std::pair{boundary, total - boundary}}) {
            Require(
                warmed->Gemm(
                    a.data, static_cast<__half*>(b.data) + std::size_t(off) * k,
                    static_cast<float*>(c.data) + std::size_t(off) * m, type, m,
                    n, k, &error),
                error);
          }
          Check(hipMemcpy(split.data(), c.data, split.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          Require(bulk == split,
                  std::string(type == HIP_R_16BF ? "BF16" : "F16") +
                      " m=" + std::to_string(m) +
                      " projection changed across "
                      "prefill chunks at " +
                      std::to_string(boundary) + " of " +
                      std::to_string(total));
        }
      }
      // Vary shape order and use ragged sizes between the calibrated widths.
      // F16 only: the CPU reference below reads the weights as F16 halves.
      if (type != HIP_R_16F)
        continue;
      for (int n : {4096, 2048, 9, 1024, 512, 129, 128, 513, 256}) {
        auto fresh = BlasLt::Create(nullptr, &error);
        Require(fresh != nullptr, error);
        std::vector<float> reference(static_cast<std::size_t>(m) * n);
        std::vector<float> actual(reference.size());
        for (int pass = 0; pass < 3; ++pass) {
          auto& blas = pass == 1 ? fresh : warmed;
          Require(blas->Gemm(a.data, b.data, static_cast<float*>(c.data),
                             HIP_R_16F, m, n, k, &error),
                  error);
          Check(hipDeviceSynchronize());
          Check(hipMemcpy(actual.data(), c.data, actual.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          if (pass == 0) {
            reference = actual;
          } else {
            Require(std::memcmp(reference.data(), actual.data(),
                                actual.size() * sizeof(float)) == 0,
                    "projection changed across instances or plan reuse");
          }
        }
        for (int sample = 0; sample < 32; ++sample) {
          const int row = (sample * 79) % n;
          const int col = (sample * 127) % m;
          double expected = 0.0;
          for (int j = 0; j < k; ++j) {
            expected += static_cast<double>(
                            __half2float(weights[std::size_t(col) * k + j])) *
                        __half2float(input[std::size_t(row) * k + j]);
          }
          const float result = actual[std::size_t(row) * m + col];
          Require(std::isfinite(result) && std::abs(result - expected) < 0.002,
                  "projection differs from the F64 reference");
        }
      }
    }
    std::cout << "PASS: F16 projection replay, chunk boundaries, ragged shapes "
                 "and F64\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
