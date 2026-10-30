// Row-stride aliasing on gfx1151: what a 1 KB-multiple activation stride costs
// a prefill GEMM's X plane, and what padding it removes.
//
// Strata ships this as STRATA_PF_PAD (their gfx11 default, paired with
// STRATA_PF_GEMM: `ldx` = a padded activation row stride).  The measurement
// below is why our tree does not: our activation strides are 5120 B (hidden
// 2560 halves), 1280 B (expert_ff) and 640 B (hc_low_rank), and the X plane is
// ~3 GB of DRAM per pass against 111 GB of weights, so the ceiling is ~1%.
// See docs/STRATA-SPEC-REVIEW.md for the rejection.
//
// Two patterns, both from the production kernels:
//   GemmBFetch   - DenseF16GEMMKernel's B plane: 128 token rows per block, two
//                  threads per row, 4 x uint4 per thread per stage.
//   RoutedGather - RoutedF16GEMMKernel's activation gather: one thread per
//                  (row, k-block), 16 B each, rows through rows_in.
//
//   clang++ -x hip --offload-arch=gfx1151 -O2 -I <rocm include> ^
//     -I <repo>/compat/win32/hip_include stride_alias_bench.hip ^
//     -o stride_alias_bench.exe -L<rocm lib> -lamdhip64
//
// GB/s counts each row once; the repeats across row-blocks are L2 hits, so the
// absolute number is the floor of what the pattern sustains, not DRAM traffic.

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

__launch_bounds__(256) __global__ void GemmBFetch(const __half* __restrict__ x,
                                                  float* __restrict__ y, int k,
                                                  int ldx, int batch) {
  const int t_local = (threadIdx.x >> 1);  // 128 rows, 2 threads each
  const int kk = threadIdx.x & 1;
  const int tok = blockIdx.y * 128 + t_local;
  const int num_kb = k / 32;
  float acc = 0.0F;
  const __half* row = x + (std::size_t) (tok < batch ? tok : 0) * ldx;
  for (int kb0 = 0; kb0 < num_kb; kb0 += 2) {
    const int kb = kb0 + kk;
    if (kb >= num_kb) break;
    const auto* src = reinterpret_cast<const uint4*>(row + kb * 32);
    uint4 v[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) v[c] = src[c];
    const __half2* h = reinterpret_cast<const __half2*>(v);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += __low2float(h[i]) + __high2float(h[i]);
  }
  if (threadIdx.x == 0) y[blockIdx.x * gridDim.y + blockIdx.y] = acc;
}

__launch_bounds__(256) __global__ void RoutedGather(
    const __half* __restrict__ x, const int* __restrict__ rows_in,
    float* __restrict__ y, int k, int ldx) {
  const int row = blockIdx.x * 256 + threadIdx.x;
  const int num_kb = k / 32;
  float acc = 0.0F;
  const __half* src = x + (std::size_t) rows_in[row] * ldx;
  for (int kb = 0; kb < num_kb; ++kb) {
    const uint4 v = *reinterpret_cast<const uint4*>(src + kb * 32);
    const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += __low2float(h[i]) + __high2float(h[i]);
  }
  y[row] = acc;
}

int main() {
  GPU_CHECK(hipSetDevice(0));
  const int rows = 2048;  // pp2048 token rows
  const int pad_max = 512;
  const int cases[] = {2560, 2048, 1024, 10240};
  const int pads[] = {0, 8, 16, 32, 64, 128};
  __half* dx;
  float* dy;
  int* drows;
  GPU_CHECK(hipMalloc(&dy, 65536 * 4));
  GPU_CHECK(hipMalloc(&drows, 20480 * 4));
  std::vector<int> hrows(20480);
  for (int i = 0; i < 20480; ++i) hrows[i] = i % rows;
  GPU_CHECK(hipMemcpy(drows, hrows.data(), 20480 * 4, hipMemcpyHostToDevice));

  for (int k : cases) {
    GPU_CHECK(hipMalloc(&dx, (std::size_t) rows * (k + pad_max) * 2));
    std::vector<__half> hx((std::size_t) rows * (k + pad_max), __float2half(1.0F));
    GPU_CHECK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    std::printf("k %5d halves (%6d B stride, %%4K %4d)\n", k, k * 2,
                (k * 2) % 4096);
    for (int pad : pads) {
      const int ldx = k + pad;
      float best[2] = {1e30F, 1e30F};
      for (int rep = 0; rep < 7; ++rep) {
        for (int pattern = 0; pattern < 2; ++pattern) {
          const auto t0 = std::chrono::high_resolution_clock::now();
          if (pattern == 0) {
            GemmBFetch<<<dim3(4, rows / 128), 256>>>(dx, dy, k, ldx, rows);
          } else {
            RoutedGather<<<80, 256>>>(dx, drows, dy, k, ldx);
          }
          GPU_CHECK(hipGetLastError());
          GPU_CHECK(hipDeviceSynchronize());
          const auto t1 = std::chrono::high_resolution_clock::now();
          best[pattern] = std::min(
              best[pattern],
              std::chrono::duration<float, std::milli>(t1 - t0).count());
        }
      }
      const double gb_dense = (double) rows * k * 2 / 1e9;
      const double gb_routed = (double) 20480 * k * 2 / 1e9;
      std::printf(
          "  pad %3d (%6d B): gemm-b %7.1f GB/s | gather %7.1f GB/s\n", pad,
          ldx * 2, gb_dense / best[0] * 1e3, gb_routed / best[1] * 1e3);
    }
    GPU_CHECK(hipFree(dx));
  }
  GPU_CHECK(hipFree(drows));
  return 0;
}
