#ifndef GUFO_CORE_EOG_BOUNDARY_HPP_
#define GUFO_CORE_EOG_BOUNDARY_HPP_

#include <string_view>

namespace gufo::sampling {

/// Tracks the Markdown and LaTeX spans that cannot legally end an answer.
///
/// Feed it the decoded piece of every accepted token. While SpanOpen() is
/// true the caller keeps the stop tokens out of the distribution: a model that
/// samples an end-of-generation token inside an unclosed ``` fence, `code`,
/// $math$, $$display$$, \(..\) or \[..\] would otherwise cut the answer in the
/// middle. Each call ends a delimiter run, so "``" starts inline code and "```"
/// toggles a fence.
class EogBoundary {
public:
  void Reset() noexcept { *this = EogBoundary{}; }

  void AcceptPiece(std::string_view piece) noexcept {
    for (const char c : piece) {
      if (c != '`')
        FlushBackticks();
      if (c != '$')
        FlushDollars();
      if (c == '`') {
        ++pending_backticks_;
        pending_backslash_ = false;
        continue;
      }
      if (c == '$' && !pending_backslash_ && !fence_open_ &&
          !inline_code_open_) {
        ++pending_dollars_;
        pending_backslash_ = false;
        continue;
      }
      if (pending_backslash_ && c == '\\') {
        // An escaped backslash is literal: \\[ does not open math.
        pending_backslash_ = false;
        continue;
      }
      if (pending_backslash_ && !fence_open_ && !inline_code_open_) {
        if (c == '(') {
          latex_math_ = 1;
        } else if (c == '[') {
          latex_math_ = 2;
        } else if ((c == ')' && latex_math_ == 1) ||
                   (c == ']' && latex_math_ == 2)) {
          latex_math_ = 0;
        }
      }
      pending_backslash_ = c == '\\';
    }
    // shortcut: a delimiter run split across pieces is read as inline code and
    // tokenizers emit fences whole, so a piece boundary ends the run.
    FlushBackticks();
    FlushDollars();
  }

  /// True when an unclosed span runs to the end of the emitted text.
  [[nodiscard]] bool SpanOpen() const noexcept {
    return fence_open_ || inline_code_open_ || dollar_math_open_ ||
           display_math_open_ || latex_math_ != 0;
  }

private:
  void FlushBackticks() noexcept {
    if (pending_backticks_ >= 3) {
      fence_open_ = !fence_open_;
    } else if (pending_backticks_ != 0 && !fence_open_) {
      inline_code_open_ = !inline_code_open_;
    }
    pending_backticks_ = 0;
  }

  void FlushDollars() noexcept {
    if (!fence_open_ && !inline_code_open_) {
      if (pending_dollars_ >= 2) {
        display_math_open_ = !display_math_open_;
      } else if (pending_dollars_ == 1) {
        dollar_math_open_ = !dollar_math_open_;
      }
    }
    pending_dollars_ = 0;
  }

  bool fence_open_ = false;
  bool inline_code_open_ = false;
  bool dollar_math_open_ = false;
  bool display_math_open_ = false;
  bool pending_backslash_ = false;
  int latex_math_ = 0;
  int pending_backticks_ = 0;
  int pending_dollars_ = 0;
};

}  // namespace gufo::sampling

#endif  // GUFO_CORE_EOG_BOUNDARY_HPP_
