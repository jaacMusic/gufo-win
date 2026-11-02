#pragma once

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::serve {

// Memoizes the previous prompt tokenization. When the next prompt byte-extends
// it at a special-token boundary, only the suffix is encoded; the result is
// identical to a full re-encode because each special-delimited segment is
// tokenized independently. One instance per runner.
class PromptTokenCache {
public:
  [[nodiscard]] std::vector<tokenization::TokenId> Encode(
      std::string rendered, const tokenization::QwenTokenizer& tokenizer,
      const tokenization::TokenizerOptions& options) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!memo_.boundaries.empty()) {
      // Largest checkpoint whose bytes are a prefix of the new prompt.
      for (std::size_t i = memo_.boundaries.size(); i-- > 0;) {
        const auto& boundary = memo_.boundaries[i];
        if (boundary.byte_end > rendered.size()) {
          continue;
        }
        if (std::memcmp(rendered.data(), memo_.rendered.data(),
                        boundary.byte_end) != 0) {
          continue;
        }
        std::vector<tokenization::TokenBoundary> suffix_boundaries;
        auto suffix = tokenizer.Encode(rendered.substr(boundary.byte_end),
                                       options, &suffix_boundaries);
        std::vector<tokenization::TokenId> tokens(
            memo_.tokens.begin(),
            memo_.tokens.begin() +
                static_cast<std::ptrdiff_t>(boundary.token_count));
        tokens.insert(tokens.end(), suffix.begin(), suffix.end());
        for (auto& entry : suffix_boundaries) {
          entry.byte_end += boundary.byte_end;
          entry.token_count += boundary.token_count;
        }
        memo_ =
            Entry{std::move(rendered), tokens, std::move(suffix_boundaries)};
        return tokens;
      }
    }
    std::vector<tokenization::TokenBoundary> boundaries;
    auto tokens = tokenizer.Encode(rendered, options, &boundaries);
    memo_ = Entry{std::move(rendered), tokens, std::move(boundaries)};
    return tokens;
  }

private:
  struct Entry {
    std::string rendered;
    std::vector<tokenization::TokenId> tokens;
    std::vector<tokenization::TokenBoundary> boundaries;
  };

  std::mutex mutex_;
  Entry memo_;
};

}  // namespace gufo::serve
