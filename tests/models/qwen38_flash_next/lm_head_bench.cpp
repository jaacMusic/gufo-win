// Times the output-head projection (Q8_0, vocab x hidden) through the two
// production entry points: the Q8_0 GEMV that prefill logits take (one row),
// and DenseF16Gemm, which a wider logit batch would use. Prints milliseconds
// and the implied weight-stream rate, so the head's share of a pass can be
// priced without a model load.
//
// lm_head_bench [vocab] [hidden] [rows] [replays]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

void* Device(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return p;
}

void* Q8Weights(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> host(blocks * 34);
  for (std::size_t b = 0; b < blocks; ++b) {
    const __half d =
        __float2half_rn(static_cast<float>(1 + Next(&seed) % 13) / 2048.0F);
    std::memcpy(host.data() + b * 34, &d, sizeof(d));
    for (std::size_t j = 0; j < 32; ++j)
      host[b * 34 + 2 + j] =
          static_cast<std::uint8_t>(static_cast<int>(Next(&seed) % 255) - 127);
  }
  void* p = Device(host.size());
  CheckHip(hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice),
           "weights");
  return p;
}

__half* Halves(std::size_t count, std::uint32_t seed, float scale) {
  std::vector<__half> host(count);
  for (__half& v : host)
    v = __float2half_rn(
        scale *
        static_cast<float>(static_cast<int>(Next(&seed) & 0xFFFF) - 32768) /
        32768.0F);
  auto* p = static_cast<__half*>(Device(count * sizeof(__half)));
  CheckHip(
      hipMemcpy(p, host.data(), count * sizeof(__half), hipMemcpyHostToDevice),
      "halves");
  return p;
}

/// Median milliseconds of one launch of `body`, warmed first.
double MedianMs(hipStream_t stream, int replays,
                const std::function<void()>& body) {
  hipEvent_t start, stop;
  CheckHip(hipEventCreate(&start), "event");
  CheckHip(hipEventCreate(&stop), "event");
  for (int i = 0; i < 2; ++i)
    body();
  std::vector<double> times;
  for (int i = 0; i < replays; ++i) {
    CheckHip(hipEventRecord(start, stream), "record");
    body();
    CheckHip(hipEventRecord(stop, stream), "record");
    CheckHip(hipEventSynchronize(stop), "sync");
    float ms = 0.0F;
    CheckHip(hipEventElapsedTime(&ms, start, stop), "elapsed");
    times.push_back(ms);
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

void Row(const char* what, double ms, double gb) {
  std::printf("%-28s %8.2f ms  %7.1f GB/s\n", what, ms, gb * 1e3 / ms);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int vocab = argc > 1 ? std::atoi(argv[1]) : 151936;
    const int hidden = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int rows = argc > 3 ? std::atoi(argv[3]) : 255;
    const int replays = argc > 4 ? std::atoi(argv[4]) : 9;
    hipStream_t stream = nullptr;
    CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    const double gb = static_cast<double>(vocab) * hidden * 34.0 / 32.0 * 1e-9;

    void* weights =
        Q8Weights(static_cast<std::size_t>(vocab) * hidden / 32, 0x1EAD0001U);
    auto* x = static_cast<float*>(Device(sizeof(float) * hidden));
    void* x_q8 = Device(qfn_mmq_q8_1_bytes(1, hidden));
    auto* out = static_cast<float*>(Device(sizeof(float) * vocab));
    CheckHip(hipMemcpy(x, std::vector<float>(hidden, 0.25F).data(),
                       sizeof(float) * hidden, hipMemcpyHostToDevice),
             "input");
    if (qfn_mmq_quantize_q8_1(x, x_q8, 1, hidden, stream) != 0)
      throw std::runtime_error("quantize rejected");
    Row("q8_0 gemv 1 row",
        MedianMs(stream, replays,
                 [&] {
                   if (qfn_mmq_q8_0_dense_vec_preq(weights, nullptr, x_q8, out,
                                                   vocab, 1, hidden,
                                                   stream) != 0)
                     throw std::runtime_error("gemv rejected");
                 }),
        gb);
    // The same GEMV through the non-vector wide entry point, as a reference.
    Row("q8_0 dense f32 1 row",
        MedianMs(stream, replays,
                 [&] {
                   if (qfn_mmq_q8_0_dense(weights, x, out, vocab, 1, hidden,
                                          stream) != 0)
                     throw std::runtime_error("dense rejected");
                 }),
        gb);

    auto* x_half =
        Halves(static_cast<std::size_t>(rows) * hidden, 0x1EAD0002U, 1.0F);
    auto* wide = static_cast<float*>(
        Device(sizeof(float) * static_cast<std::size_t>(rows) * vocab));
    char label[48];
    std::snprintf(label, sizeof(label), "dense f16 gemm %d rows", rows);
    Row(label,
        MedianMs(stream, replays,
                 [&] {
                   if (!q::DenseF16Gemm(weights, x_half, wide, rows, vocab,
                                        hidden, stream))
                     throw std::runtime_error("f16 gemm rejected");
                 }),
        gb);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
