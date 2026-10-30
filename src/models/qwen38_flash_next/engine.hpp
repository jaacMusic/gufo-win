#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/prompt.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/prompt_lookup.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen38_flash_next {

struct ModelWeights;
struct MtpWeights;
struct MtpCandidateLogits;
namespace rocm {
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

/// Tokens the MTP draft head may propose.
enum class DraftVocabulary {
  kFull,
  /// Special tokens plus every token whose text is ASCII (English, code,
  /// numbers, markup and the unaccented pieces of Latin-script languages)
  /// or ASCII with common typographic marks: about half of the vocabulary,
  /// so each draft step reads half of the output head.
  kLatinText,
};

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Prompt tokens per prefill pass, capped by max_context.
  /// gfx1151, served repository prompts: 4096-token chunks prefill 24K-102K
  /// tokens 6-11% faster than 2048 (pp4096 +5% at d0, +6% at d32K); 8192
  /// against 4096 at d0 is +2.2% at pp8192 and +2.6% at pp32768 for -1.6% at
  /// pp4096, decode unchanged. Each chunk's n-gram gather doubles; next-chunk
  /// prefetch overlaps that read. Prefill logits do not depend on the chunk
  /// size (pre-budget queries always take the dense attention tiles).
  static constexpr std::uint32_t kDefaultPrefillChunkTokens = 8192;
  std::uint32_t prefill_chunk_tokens = kDefaultPrefillChunkTokens;
  /// Optional MTP draft sidecar (`mtp-...-shared-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Maximum proposals per cycle; acceptance history selects the length.
  std::uint32_t max_draft_tokens = kMaxMtpDraftTokens;
  std::string vision_model_path;
  /// Fixed serving capacity used by the calibrated MTP cost model. Keeping
  /// it independent of scheduler timing preserves seeded request replay.
  std::uint32_t decode_concurrency = 1;
  /// MTP draft policies. Verification is unchanged, so outputs keep
  /// the target distribution (greedy text is byte-identical); they change
  /// how many tokens a cycle proposes and how cheaply.
  /// Sampled single-session drafting stops on a chain-survival estimate
  /// from the draft head's own confidence instead of the length controller.
  bool mtp_survival = false;
  /// Single-session prompt lookup: after an MTP draft, a match of 12+ tokens
  /// in the conversation proposes the tokens that followed it (PromptLookup).
  bool prompt_lookup = true;
  DraftVocabulary draft_vocabulary = DraftVocabulary::kFull;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions with shared projection batches and independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  /// Trunk tensors the official GGUF keeps in F32 but this artifact stores
  /// as Q8_0 (see `ExpandedQ8_0Tensors`). The official MTP sidecar ships its
  /// inject as Q8_0, so the draft block is not counted.
  [[nodiscard]] std::size_t ExpandedQ8_0Tensors() const noexcept;
  [[nodiscard]] std::uint32_t DecodeConcurrency() const noexcept {
    return options_.decode_concurrency;
  }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const tokenization::QwenTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state, including the configured rollback cap.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;
  [[nodiscard]] const std::shared_ptr<qwen::vision::Encoder>& VisionEncoder()
      const noexcept {
    return vision_;
  }

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::unique_ptr<NgramTable> ngram_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  std::shared_ptr<qwen::vision::Encoder> vision_;
  MtpBatchController batch_policy_;
  /// MTP drafts may use the small head (draft_vocab.hpp).
  bool draft_subset_{false};
  /// Special tokens at the start of every session's draft table.
  std::uint32_t draft_specials_{0};
  /// Prefix and special tokens: always in the small head.
  [[nodiscard]] bool InDraftSubset(std::int32_t token) const;

  friend class Session;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest. `next` may name
  /// the tokens the following Sync will append, so their n-gram rows can be
  /// read during this one; it never changes the result.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr,
                          std::span<const std::int32_t> next = {});
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  /// Appends `tokens` through the decode arithmetic: one token as a decode
  /// step, more as a fully accepted speculative verification (at most
  /// MaxVerifyWidth()). With `prefill`, through the prompt arithmetic
  /// instead (a session without MTP). `rows` receives every token's
  /// next-token logits (tokens.size() * vocab). For numerical A/B of the
  /// kernels.
  [[nodiscard]] bool TeacherForce(std::span<const std::int32_t> tokens,
                                  std::vector<float>* rows,
                                  std::string* error_msg = nullptr,
                                  bool prefill = false);
  [[nodiscard]] std::uint32_t MaxVerifyWidth() const noexcept;
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Greedy decoding verifies deterministic drafts. Sampled decoding draws
  /// compact MTP proposals and uses target/draft rejection with a residual
  /// correction. History contains only committed tokens; stochastic RNG
  /// consumption includes proposal and verification draws. A
  /// one-token budget or a model without MTP uses ordinary decoding.
  /// Benchmarks may continue past EOS by setting stop_at_eos to false.
  /// A greedy step prices its drafts as one of `occupancy` decoding requests,
  /// as DecodeBatch does for an unplanned cohort of that width.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true,
                                std::uint32_t occupancy = 1);
  struct BatchOutcome {
    bool completed{false};
    std::string error{};
  };
  struct DecodeRequest {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    BatchOutcome* outcome{nullptr};
  };
  /// A false return may include completed peers. Inspect each outcome; only
  /// sessions whose persistent state was partly mutated become invalid.
  [[nodiscard]] static bool DecodeBatch(std::span<const DecodeRequest> requests,
                                        std::string* error_msg = nullptr);
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ && EnsureFrontier() ? std::span<const float>(logits_)
                                      : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Cancellation is checked at model, layer and snapshot-copy boundaries.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  void ConfigureVision(std::shared_ptr<const qwen::vision::Prompt> prompt);
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy() noexcept {
    draft_length_.Reset();
    accept_table_ = {};
    // A fully reused prompt skips SyncImpl; it still starts a new request, so
    // do not inherit a copy cursor or widen its first step.
    lookup_.BeginRequest();
  }

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
    /// Of drafted/accepted: proposals copied from the context.
    std::uint64_t lookup{0};
    std::uint64_t lookup_accepted{0};
    /// Cycles whose draft scored the small head (draft_vocab.hpp).
    std::uint64_t small_head_cycles{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }
  /// PLE rows gathered by this execution slot since creation; zero without
  /// a PLE table.
  [[nodiscard]] NgramTotals NgramRowTotals() const noexcept;

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 17;
  /// Bytes a snapshot of the current context occupies.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  /// Captures the whole context (tokens, device caches and recurrent
  /// state, draft-block state, last logits) into host memory. The
  /// speculative length controller is preserved for stochastic replay.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Replaces this session's context with a snapshot of the same model.
  /// Image snapshots require matching consumed images via ConfigureVision;
  /// the attached prompt may append future images. Pixels are not serialized.
  /// Text snapshots clear images; reattach the request after restoring.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
            bool prefill = false, std::span<const std::int32_t> after = {});
  /// Trunk rows the draft block may still read: [hidden_base_, size).
  [[nodiscard]] std::uint32_t KeptHiddenRows() const noexcept;
  bool DraftCatchUp(std::int32_t next_token, bool propose,
                    std::string* error_msg,
                    MtpCandidateLogits* candidates = nullptr,
                    bool draft_subset = false);
  bool DraftReplay(std::int32_t next_token, std::vector<std::int32_t>* replay,
                   std::int32_t* hidden_row, std::string* error_msg) const;
  /// `propose` selects, per request, whether the catch-up also computes the
  /// next neural proposal. A false entry keeps only the predictor's KV.
  static bool DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const std::uint8_t> propose = {});
  template<class Request>
  static bool RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg);
  static bool DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg);
  static bool EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  struct PendingDecode;
  bool PrepareDecode(const DecodeRequest& request, PendingDecode* pending,
                     std::string* error_msg, bool defer_head = false,
                     std::optional<std::uint32_t> batch_drafts = {},
                     std::uint32_t occupancy = 1);
  static void AppendDraft(PendingDecode& pending);
  /// Appends copies within the prepared budget, stopping at EOG when enabled.
  /// `require_resume` copies only from an established source with a full
  /// match: the caller then skips the neural proposal entirely.
  bool AppendLookup(PendingDecode& pending, bool stop_at_eos,
                    std::size_t minimum = 1, bool require_resume = false);
  bool FinishDecode(const DecodeRequest& request, const PendingDecode& pending,
                    std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  mutable std::vector<float> logits_;
  /// A sampled cycle's rollback left the frontier row in flight (see
  /// FinishDecode): logits_ is stale until EnsureFrontier fetches it.
  mutable bool frontier_pending_{false};
  bool EnsureFrontier(std::string* error_msg = nullptr) const;
  /// GPU-selected top candidates of logits_, valid only from a sampled MTP
  /// cycle's FinishDecode until the next PrepareDecode consumes them.
  MtpCandidateLogits anchor_candidates_{};
  bool anchor_candidates_valid_{false};
  std::int32_t draft_token_{0};
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  MtpLengthController draft_length_;
  MtpAcceptTable accept_table_;
  /// Index of tokens_ for ModelOptions::prompt_lookup.
  PromptLookup lookup_;
  SpeculativeStats stats_;
  std::shared_ptr<const qwen::vision::Prompt> image_prompt_;
  [[nodiscard]] std::span<const std::uint8_t> ImageIdentity(
      std::size_t token_count) const;
  bool valid_{true};
  /// This request's draft vocabulary; the device table follows it.
  MtpDraftVocab draft_vocab_;
  bool draft_vocab_stale_{false};
  std::vector<std::int32_t> draft_added_;
  /// After the token history is replaced: the table restarts from it.
  void ForgetDraftVocab() noexcept { draft_vocab_stale_ = true; }
  /// Brings the draft table up to the committed tokens.
  bool SyncDraftVocab(std::string* error_msg);
  [[nodiscard]] bool MtpEnabled() const noexcept;
};

/// Immutable host copy of a session context. The same bytes restore in
/// memory and persist to disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot() = default;
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_.get(), size_};
  }
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::unique_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};

  friend class Session;
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ENGINE_HPP_
