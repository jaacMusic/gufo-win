#ifndef GUFO_CORE_CPU_FEATURES_HPP_
#define GUFO_CORE_CPU_FEATURES_HPP_

#if (defined(__clang__) || defined(__GNUC__)) && defined(__x86_64__)
#if defined(_WIN32)
#include <intrin.h>
#endif

namespace gufo {

/// AVX-512F is usable: the CPU has it and the OS saves the ZMM/opmask state.
/// Windows clang links no __cpu_model for __builtin_cpu_supports, so the
/// Windows build reads the same CPUID/XCR0 leaves itself.
inline bool HasAvx512f() noexcept {
#if defined(_WIN32)
  int r[4];
  __cpuidex(r, 1, 0);
  constexpr int kOsxsave = 1 << 27;
  constexpr int kAvx = 1 << 28;
  if ((r[2] & (kOsxsave | kAvx)) != (kOsxsave | kAvx) ||
      (_xgetbv(0) & 0xE6) != 0xE6) {
    return false;
  }
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 16)) != 0;
#else
  return __builtin_cpu_supports("avx512f");
#endif
}

}  // namespace gufo

#endif

#endif  // GUFO_CORE_CPU_FEATURES_HPP_
