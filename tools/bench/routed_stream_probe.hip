// Can RoutedF16GEMM's weight stream saturate DRAM on gfx1151?
//
// The production kernel stages BK=2 k-blocks per iteration and one stage
// ahead, so each lane issues a single 16 B code load per stage per row tile
// (32 lanes = 16 rows x 2 lanes, 4 KB per CTA per stage).  This probe replays
// that exact Q4_K row layout and thread mapping (144 B super-blocks: 16 B
// header + 8 x 16 B code blocks) at three fetch widths, so the gap between
// them is memory-level parallelism and not compute.
//
//   clang++ -x hip --offload-arch=gfx1151 -O2 -I <rocm include> ^
//     -I <repo>/compat/win32/hip_include routed_stream_probe.hip ^
//     -o routed_stream_probe.exe -L<rocm lib> -lamdhip64

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GPU_CHECK(x)                                                    \
  do {                                                                  \
    auto err_ = (x);                                                    \
    if (err_ != hipSuccess) {                                           \
      std::printf("hip error %s at %s:%d\n", hipGetErrorString(err_),   \
                  __FILE__, __LINE__);                                  \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// kStride: k-blocks each lane reads per stage (1 = production BK=2).
template <int kStride>
__launch_bounds__(256) __global__ void RoutedWeights(
    const std::uint8_t* __restrict__ w, float* __restrict__ y, int m, int k) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_local = wave * 16 + (lane >> 1);
  const int row = blockIdx.x * 128 + row_local;  // m tile, 128 rows
  if (row >= m) return;
  const std::uint8_t* rp = w + (std::size_t) (blockIdx.y * m + row) * 1440;
  const int num_kb = k / 32;  // 80 k-blocks for k = 2560
  float acc = 0.0F;
  for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * kStride) {
    const int kb = kb0 + (lane & 1) * kStride;
    uint4 v[kStride];
#pragma unroll
    for (int s = 0; s < kStride; ++s) {
      const int b = kb + s;
      const int off = b < num_kb ? ((b >> 3) * 144) + 16 + ((b & 7) * 16) : 0;
      v[s] = *reinterpret_cast<const uint4*>(rp + off);
    }
#pragma unroll
    for (int s = 0; s < kStride; ++s) {
      const std::uint32_t* q = reinterpret_cast<const std::uint32_t*>(&v[s]);
#pragma unroll
      for (int i = 0; i < 4; ++i) acc += (float) q[i];
    }
  }
  if (lane == 0) y[blockIdx.x * gridDim.y + blockIdx.y] += acc;
}

// HC-combine stream: one CTA per token, a 4 x hidden F32 residual read and
// write-back, a broadcast block_out read and a narrowed F16 norm write.
// Replays HcCombineVec4Kernel's float4 pattern at the pp2048 shape (2048
// tokens, hidden 2560) to see whether the ~10% combine bucket streams at the
// routed ceiling (~198 GB/s above) or has headroom.
__launch_bounds__(256) __global__ void HcCombineStream(
    float* __restrict__ res, const float* __restrict__ block_out,
    __half* __restrict__ xn, int hidden) {
  const std::size_t hc = 4 * (std::size_t)hidden;
  float* dst = res + (std::size_t)blockIdx.x * hc;
  const float* src = block_out + (std::size_t)blockIdx.x * hidden;
  __half2* out = reinterpret_cast<__half2*>(xn + (std::size_t)blockIdx.x * hc);
  float acc = 0.0F;
  for (std::uint32_t c = threadIdx.x; (std::size_t)c * 4 + 3 < hc;
       c += blockDim.x) {
    const std::uint32_t e = c * 4;
    const std::uint32_t i = e - (e / (std::uint32_t)hidden) * hidden;
    const float4 r = *reinterpret_cast<const float4*>(dst + e);
    const float4 b = *reinterpret_cast<const float4*>(src + i);
    const float4 v = {r.x + b.x, r.y + b.y, r.z + b.z, r.w + b.w};
    *reinterpret_cast<float4*>(dst + e) = v;
    const __half2 h0 = __floats2half2_rn(v.x, v.y);
    const __half2 h1 = __floats2half2_rn(v.z, v.w);
    out[c * 2] = h0;
    out[c * 2 + 1] = h1;
    acc += v.x + __low2float(h0) + __high2float(h1);
  }
  if (acc == 12345.678F) res[threadIdx.x] = acc;  // keep the loads alive
}

int main() {
  GPU_CHECK(hipSetDevice(0));
  const int experts = 512;   // pp2048: one 64-row tile per expert
  const int m = 640;         // gate (and up) rows
  const int k = 2560;
  const std::size_t bytes = (std::size_t) experts * m * 1440;
  std::uint8_t* dw;
  float* dy;
  GPU_CHECK(hipMalloc(&dw, bytes));
  GPU_CHECK(hipMalloc(&dy, 1 << 22));
  std::vector<std::uint8_t> hw(bytes);
  for (std::size_t i = 0; i < bytes; ++i) hw[i] = (std::uint8_t) (i * 31 + 7);
  GPU_CHECK(hipMemcpy(dw, hw.data(), bytes, hipMemcpyHostToDevice));
  const double gb = (double) bytes / 1e9;
  std::printf("%d experts x %d rows x 1440 B = %.2f GB per matrix\n", experts, m,
              gb);

  const int tiles = (m + 127) / 128;  // 5 per expert
  const auto run = [&](auto kernel, int stride_blocks, const char* name) {
    const dim3 grid(tiles, experts);
    auto launch = [&] { kernel<<<grid, 256>>>(dw, dy, m, k); };
    launch();
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    float best = 1e30F;
    for (int rep = 0; rep < 9; ++rep) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      launch();
      GPU_CHECK(hipDeviceSynchronize());
      const auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(
          best, std::chrono::duration<float, std::milli>(t1 - t0).count());
    }
    // Rows past m are skipped, so count the traffic the kernel really reads.
    const double live = (double) experts * (tiles * 128 > m ? m : tiles * 128) *
                        1440 / 1e9;
    std::printf("  %-22s %2d B/lane/stage: %7.3f ms  %6.1f GB/s\n", name,
                stride_blocks * 16, best, live / best * 1e3);
  };
  run(RoutedWeights<1>, 1, "production BK=2");
  run(RoutedWeights<2>, 2, "BK=4");
  run(RoutedWeights<4>, 4, "BK=8");

  // HC combine at the pp2048 d0 shape: 2048 CTAs, one per token, best of 9.
  {
    const int tokens = 2048;
    const int hidden = 2560;
    const std::size_t hc = 4 * (std::size_t)hidden;
    float* cres;
    float* cblock;
    __half* cxn;
    GPU_CHECK(hipMalloc(&cres, tokens * hc * sizeof(float)));
    GPU_CHECK(hipMalloc(&cblock, tokens * hidden * sizeof(float)));
    GPU_CHECK(hipMalloc(&cxn, tokens * hc * sizeof(__half)));
    GPU_CHECK(hipMemset(cres, 0, tokens * hc * sizeof(float)));
    GPU_CHECK(hipMemset(cblock, 0, tokens * hidden * sizeof(float)));
    const double cgb =
        (double)tokens * (2.0 * hc * sizeof(float) +
                          hidden * sizeof(float) + hc * sizeof(__half)) /
        1e9;
    std::printf("HC combine: %d tokens x %d hidden = %.2f GB per call\n",
                tokens, hidden, cgb);
    HcCombineStream<<<tokens, 256>>>(cres, cblock, cxn, hidden);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    float best = 1e30F;
    for (int rep = 0; rep < 9; ++rep) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      HcCombineStream<<<tokens, 256>>>(cres, cblock, cxn, hidden);
      GPU_CHECK(hipDeviceSynchronize());
      const auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(
          best, std::chrono::duration<float, std::milli>(t1 - t0).count());
    }
    std::printf("  %-22s float4 RMW+F16 out: %7.3f ms  %6.1f GB/s"
                "  (47 calls = %.0f ms/pass)\n",
                "HC combine", best, cgb / best * 1e3, best * 47);
  }
  return 0;
}
