#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

// gfx1151, pinned Flash-Next Q4 target/Q8 MTP. 2026-09-20 measurements
// scaled by 0.80 (2026-11-22) and again by 0.90 (2026-11-22, after the
// lm_head Q4_1 decode copy and the 16-way attention split made cycles
// cheaper; bench 2048/128 MTP d0 53.5 -> 54.5, @16k 44.5 tok/s): kernels
// got faster but the raw re-measured table over-drafts and loses ~3%
// tok/s, while the unscaled stale table forfeits the kernel wins; the
// scale lands the controller at its optimum on the greedy chat probe.
// Rows are widths 1..8 (width 1 is headless MTP catch-up plus ordinary decode).
// Raw rows: qwen38_flash_next_gpu_probe --cost-audit 0. Allocation and C1
// graph capture are warmed outside the measurement; each row is the median
// of three complete cycles. Use --depth N to audit one depth. See benchmarks.
inline constexpr float kMtpCycleMilliseconds[3][5][8] = {
    {
        // Context 0
        {27.6F, 34.0F, 40.9F, 48.4F, 56.4F, 61.7F, 68.0F, 73.9F},        // C1
        {33.2F, 42.9F, 53.3F, 65.2F, 78.5F, 87.3F, 97.8F, 104.9F},      // C2
        {41.5F, 58.1F, 78.2F, 94.6F, 116.5F, 132.1F, 153.7F, 168.7F},   // C4
        {49.2F, 75.4F, 101.0F, 126.4F, 152.4F, 183.1F, 211.5F, 234.5F},  // C6
        {56.7F, 87.8F, 120.5F, 155.3F, 190.6F, 224.1F, 262.2F, 296.0F},  // C8
    },
    {
        // Context 4096
        {28.6F, 35.5F, 43.0F, 50.2F, 57.0F, 64.2F, 71.0F, 78.3F},         // C1
        {35.4F, 46.3F, 58.0F, 69.9F, 84.0F, 94.2F, 104.6F, 113.2F},      // C2
        {44.8F, 63.2F, 85.4F, 102.5F, 124.6F, 142.5F, 164.4F, 183.9F},    // C4
        {54.1F, 83.1F, 112.7F, 139.1F, 168.1F, 200.8F, 230.5F, 257.9F},   // C6
        {63.3F, 96.7F, 136.6F, 171.8F, 213.1F, 246.8F, 289.5F, 328.3F},  // C8
    },
    {
        // Context 32768
        {30.6F, 37.4F, 44.3F, 51.2F, 59.4F, 66.6F, 74.6F, 81.1F},         // C1
        {39.5F, 49.7F, 59.9F, 71.7F, 86.8F, 97.4F, 109.0F, 117.9F},      // C2
        {53.0F, 69.8F, 90.6F, 108.1F, 133.4F, 152.8F, 175.6F, 191.9F},   // C4
        {66.2F, 92.6F, 120.2F, 148.5F, 179.6F, 214.7F, 244.2F, 266.9F},  // C6
        {79.3F, 109.8F, 145.3F, 184.3F, 231.8F, 270.8F, 311.0F, 353.4F},  // C8
    },
};

// C1 sampled-prefix head, same weights, 2026-10-23. Fixed token histories at
// depths 0/4096/32768, median of three warmed complete cycles on ROCm 7.2.3;
// matched ROCm 10.1 cycles agree within 2%. Full-head and concurrent costs
// above remain separate: pricing the smaller head as a full projection
// unnecessarily shortens profitable draft chains. See the model's ROCM.md.
// Full target-row transfers also retain the original curve in the engine.
inline constexpr float kMtpSampledSubsetMilliseconds[3][8] = {
    {31.0F, 38.1F, 42.4F, 47.8F, 53.4F, 58.1F, 64.8F, 71.0F},
    {31.6F, 38.6F, 43.8F, 49.3F, 54.7F, 59.9F, 66.9F, 73.4F},
    {31.7F, 38.6F, 44.0F, 49.6F, 54.8F, 59.9F, 67.0F, 73.4F},
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
