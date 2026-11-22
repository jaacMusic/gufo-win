#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, pinned Flash-Next Q4 target/Q8 MTP. 2026-09-20 measurements
// scaled by 0.80 (2026-11-22): kernels got faster but the raw re-measured
// table over-drafts and loses ~3% tok/s, while the unscaled stale table
// forfeits the kernel wins; the 0.80 scale lands the controller at 53.9
// vs 46.8 stale and 44.6 raw-fresh tok/s on the greedy chat probe.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary decode).
// Raw rows: qwen38_flash_next_gpu_probe --cost-audit 0. Allocation and C1
// graph capture are warmed outside the measurement; each row is the median
// of three complete cycles. Use --depth N to audit one depth. See benchmarks.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {30.7F, 37.8F, 45.4F, 53.8F, 62.7F, 68.6F, 75.6F, 82.1F},        // C1
        {36.9F, 47.7F, 59.2F, 72.4F, 87.2F, 97.0F, 108.7F, 116.6F},      // C2
        {46.1F, 64.5F, 86.9F, 105.1F, 129.5F, 146.8F, 170.8F, 187.4F},   // C4
        {54.7F, 83.8F, 112.2F, 140.4F, 169.3F, 203.4F, 235.0F, 260.6F},  // C6
        {63.0F, 97.5F, 133.9F, 172.6F, 211.8F, 249.0F, 291.3F, 328.9F},  // C8
    },
    {
        // Context 4096
        {31.8F, 39.4F, 47.8F, 55.8F, 63.3F, 71.3F, 78.9F, 87.0F},         // C1
        {39.3F, 51.4F, 64.4F, 77.7F, 93.3F, 104.7F, 116.2F, 125.8F},      // C2
        {49.8F, 70.2F, 94.9F, 113.9F, 138.4F, 158.3F, 182.7F, 204.3F},    // C4
        {60.1F, 92.3F, 125.2F, 154.5F, 186.8F, 223.1F, 256.1F, 286.5F},   // C6
        {70.3F, 107.4F, 151.8F, 190.9F, 236.8F, 274.2F, 321.7F, 364.8F},  // C8
    },
    {
        // Context 32768
        {34.0F, 41.5F, 49.2F, 56.9F, 66.0F, 74.0F, 82.9F, 90.1F},         // C1
        {43.9F, 55.2F, 66.6F, 79.7F, 96.4F, 108.2F, 121.1F, 131.0F},      // C2
        {58.9F, 77.5F, 100.7F, 120.1F, 148.2F, 169.8F, 195.1F, 213.2F},   // C4
        {73.6F, 102.9F, 133.6F, 165.0F, 199.6F, 238.6F, 271.3F, 296.6F},  // C6
        {88.1F, 122.0F, 161.4F, 204.8F, 257.5F, 300.9F, 345.6F, 392.7F},  // C8
    },
};

// C1 sampled-prefix head, same weights, 2026-10-23. Fixed token histories at
// depths 0/4096/32768, median of three warmed complete cycles on ROCm 7.2.3;
// matched ROCm 10.1 cycles agree within 2%. Full-head and concurrent costs
// above remain separate: pricing the smaller head as a full projection
// unnecessarily shortens profitable draft chains. See the model's ROCM.md.
// Full target-row transfers also retain the original curve in the engine.
inline constexpr float kMtpSampledSubsetMilliseconds[3][8] = {
    {34.4F, 42.3F, 47.1F, 53.1F, 59.3F, 64.5F, 72.0F, 78.9F},
    {35.1F, 42.9F, 48.7F, 54.8F, 60.8F, 66.5F, 74.3F, 81.5F},
    {35.2F, 42.9F, 48.9F, 55.1F, 60.9F, 66.5F, 74.4F, 81.5F},
};

// Seeded sampled policy uses fixed configured capacity for reproducibility;
// greedy batch policy uses these curves to bootstrap each physical occupancy,
// and a greedy or unseeded sampled request uses the row for the requests
// decoding with it (MtpPolicyConcurrency).
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency,
    bool sampled_subset = false) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto interval = context <= 4096 ? 0 : 1;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : static_cast<float>(std::min(context, 262144U) - 4096) / 28672.0F;
  const auto* low_row = sampled_subset && cohort == 0
                            ? kMtpSampledSubsetMilliseconds[interval]
                            : kMtpCycleMilliseconds[interval][cohort];
  const auto* high_row = sampled_subset && cohort == 0
                             ? kMtpSampledSubsetMilliseconds[interval + 1]
                             : kMtpCycleMilliseconds[interval + 1][cohort];
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = low_row[i];
    const auto high = high_row[i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal = high - high_row[i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  return costs;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
