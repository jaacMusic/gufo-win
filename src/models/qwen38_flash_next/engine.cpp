#include "src/models/qwen38_flash_next/engine.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/models/qwen38_flash_next/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/npu_retry.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#ifdef GUFO_NPU
#include <map>
#include <string_view>

#include "src/cli/serve/logging.hpp"
#include "src/core/experimental_authorization.hpp"
#include "src/models/qwen38_flash_next/npu/server_prefill.hpp"
#include "src/models/qwen38_flash_next/npu/staging.hpp"
#include "src/models/qwen38_flash_next/npu/chunk_retry.hpp"
#include "src/models/qwen38_flash_next/npu/request_stats.hpp"
#endif

namespace gufo::models::qwen38_flash_next {
namespace {

// Sampled drafting stops once the chain is unlikely to survive: each
// proposal multiplies an acceptance estimate looked up from the draft head's
// own top probability, and drafting ends when the product falls below
// kMtpSurvivalFloor (the dropped proposal is not verified). The decision
// uses only draft-side probabilities, so rejection-sampling verification
// keeps the target distribution exact. Prior table: P(accept | reached) by
// draft depth (first, later) and top-probability bin, measured over 2130
// full 7-draft cycles on Windows (2026-09-25); replayed offline it beat the
// per-cycle length controller by ~7.7% on held-out cycles. The runtime
// estimate comes from Session::accept_table_, seeded with this prior and
// refined online from every verified cycle.
inline constexpr std::array<float, 8> kMtpTopBins = {0.30F, 0.50F, 0.70F, 0.80F,
                                                     0.90F, 0.95F, 0.99F, 2.0F};
inline constexpr std::array<std::array<float, 8>, 2> kMtpAcceptByTop = {{
    {0.400F, 0.506F, 0.615F, 0.595F, 0.635F, 0.774F, 0.825F, 0.947F},
    {0.337F, 0.484F, 0.522F, 0.584F, 0.640F, 0.711F, 0.702F, 0.935F},
}};
constexpr float kMtpSurvivalFloor = 0.40F;

std::size_t MtpTopBin(const MtpProposal& proposal) {
  float top = 0;
  for (std::size_t i = 0; i < proposal.size; ++i) {
    top = std::max(top, proposal.probabilities[i]);
  }
  std::size_t bin = 0;
  while (top >= kMtpTopBins[bin]) {
    ++bin;
  }
  return bin;
}

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

/// DraftVocabulary::kLatinText: special tokens, tokens whose bytes are
/// printable ASCII (or tab / newline / carriage return), and tokens that are
/// complete UTF-8 made of ASCII plus common typographic marks. Pieces of
/// multi-byte characters (most CJK, Cyrillic, accented letters) are left out.
std::vector<std::int32_t> LatinTextVocabulary(
    const tokenization::QwenTokenizer& tokenizer) {
  constexpr char32_t kTypographic[] = {
      0x2014, 0x2013, 0x201C, 0x201D, 0x2018, 0x2019, 0x2026, 0x2022, 0x2192,
      0x2190, 0x2248, 0x00D7, 0x00F7, 0x00B1, 0x00B0, 0x00B7, 0x00E9};
  const auto ascii = [](unsigned char b) {
    return (b >= 0x20 && b < 0x7F) || b == '\t' || b == '\n' || b == '\r';
  };
  std::vector<std::int32_t> ids;
  for (std::size_t id = 0; id < tokenizer.GetVocabSize(); ++id) {
    const auto token = static_cast<tokenization::TokenId>(id);
    if (tokenizer.IsSpecialToken(token)) {
      ids.push_back(static_cast<std::int32_t>(id));
      continue;
    }
    const std::string text = tokenizer.DecodeTokenCopy(token);
    bool keep = !text.empty();
    for (std::size_t i = 0; keep && i < text.size();) {
      const auto b = static_cast<unsigned char>(text[i]);
      if (b < 0x80) {
        keep = ascii(b);
        ++i;
        continue;
      }
      // A complete 2- or 3-byte UTF-8 sequence for an allowed mark.
      const std::size_t length = (b & 0xE0) == 0xC0   ? 2
                                 : (b & 0xF0) == 0xE0 ? 3
                                                      : 0;
      if (length == 0 || i + length > text.size()) {
        keep = false;
        break;
      }
      char32_t cp = length == 2 ? (b & 0x1F) : (b & 0x0F);
      for (std::size_t j = 1; j < length; ++j) {
        const auto c = static_cast<unsigned char>(text[i + j]);
        if ((c & 0xC0) != 0x80) {
          keep = false;
          break;
        }
        cp = (cp << 6) | (c & 0x3F);
      }
      keep = keep && std::find(std::begin(kTypographic), std::end(kTypographic),
                               cp) != std::end(kTypographic);
      i += length;
    }
    if (keep) {
      ids.push_back(static_cast<std::int32_t>(id));
    }
  }
  return ids;
}

constexpr std::array<char, 8> kSessionSnapshotMagic{'Q', 'F', 'N', 'S',
                                                    'E', 'S', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t image_identity_bytes;
  std::uint32_t policy_concurrency;
  std::uint32_t reserved{0};
  PromptLookup::State lookup;
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size,
                                       std::size_t image_bytes) {
  return sizeof(SessionSnapshotHeader) + image_bytes +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (options.prefill_chunk_tokens == 0) {
    AssignError(error_msg, "prefill chunk must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg,
                "Flash-Next MTP supports at most seven draft tokens");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  try {
    m->vision_ = qwen::vision::Encoder::Open(
        model_path, options.vision_model_path, c.hidden_size);
  } catch (const std::exception& e) {
    AssignError(error_msg, e.what());
    return nullptr;
  }
  m->tokenizer_ =
      tokenization::QwenTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (c.ple_layer >= 0) {
    const auto& t = m->weights_->ple_table;
    m->ngram_ = NgramTable::Open(
        m->reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, error_msg);
    if (!m->ngram_) {
      return nullptr;
    }
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
#ifdef GUFO_NPU
  // Opt-in only: the NPU path computes in 8-bit block floating point and is
  // not byte-exact, so it starts only for an explicit authorizing switch.
  std::shared_ptr<npu::ServerPrefill> npu_prefill;
  if (const char* on = std::getenv("GUFO_NPU_PREFILL");
      on != nullptr && std::string_view(on) == "1") {
    try {
      const auto authorization = core::ExperimentalAuthorization::Granted(
          core::ExperimentalFeature::Prefill, "GUFO_NPU_PREFILL=1");
      std::map<std::string, std::string> values;
      for (const char* name : {"GUFO_NPU_DIR", "GUFO_NPU_SLICES",
                               "GUFO_NPU_GATE_SLICES", "GUFO_NPU_SHEXP",
                               "GUFO_NPU_TIMEOUT_MS", "GUFO_NPU_KEEPALIVE",
                               "GUFO_NPU_KEEPALIVE_MS", "GUFO_NPU_GUARD",
                               "GUFO_NPU_HAL_MONITOR", "GUFO_NPU_CHIP_POWER",
                               "GUFO_NPU_MEM_TRACE"}) {
        if (const char* value = std::getenv(name))
          values.emplace(name, value);
      }
      npu_prefill = std::make_shared<npu::ServerPrefill>(
          authorization, npu::ParseKnobs(authorization, values));
      if (!npu_prefill->Start())
        npu_prefill.reset();
    } catch (const std::exception& e) {
      server::Logger::Warn("npu", e.what());
      npu_prefill.reset();
    }
  }
#endif
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_,
                                         m->mtp_weights_.get(),
                                         m->mtp_reader_.get(), error_msg
#ifdef GUFO_NPU
                                         ,
                                         npu_prefill
#endif
  );
  if (!m->device_) {
    return nullptr;
  }
  if (m->mtp_weights_ && m->device_->output().rows > kMtpDraftVocab) {
    std::vector<std::int32_t> specials;
    for (std::uint32_t id = kMtpDraftVocab; id < m->device_->output().rows;
         ++id)
      if (m->tokenizer_->IsSpecialToken(id))
        specials.push_back(static_cast<std::int32_t>(id));
    // A vocabulary with more special tokens than table rows drafts over the
    // full head.
    if (specials.size() < kMtpDraftExtraRows) {
      m->draft_specials_ = static_cast<std::uint32_t>(specials.size());
      if (!m->device_->SetDraftVocab(kMtpDraftVocab, std::move(specials),
                                     error_msg))
        return nullptr;
      m->draft_subset_ = true;
    }
  }
  rocm::Executor::Options exec;
  exec.max_batch = m->PrefillCapacity();
  exec.max_logit_rows =
      m->mtp_weights_
          ? static_cast<std::uint32_t>(std::min<std::uint64_t>(
                exec.max_batch, std::uint64_t{options.max_draft_tokens} + 1))
          : 1;
  exec.max_speculative = exec.max_logit_rows;
  if (m->mtp_weights_ &&
      options.draft_vocabulary == DraftVocabulary::kLatinText) {
    exec.draft_vocab = LatinTextVocabulary(*m->tokenizer_);
  }
  m->executor_ =
      rocm::Executor::Create(*m->device_, m->ngram_.get(), exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  if (vision_) {
    // Image-capable slots can first serve text. Prepare the stable RoPE
    // descriptor before any graphs capture a null pointer: replacing it
    // later destroys those graphs during image admission.
    try {
      native->PrepareVision(executor_->stream());
    } catch (const std::exception& exception) {
      AssignError(error_msg, exception.what());
      return nullptr;
    }
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(options_.prefill_chunk_tokens, options_.max_context);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

std::size_t Model::ExpandedQ8_0Tensors() const noexcept {
  std::size_t count = 0;
  for (const LayerWeights& layer : weights_->layers) {
    count += qwen38_flash_next::ExpandedQ8_0Tensors(layer);
  }
  return count;
}

bool Model::InDraftSubset(std::int32_t token) const {
  return token >= 0 && (static_cast<std::uint32_t>(token) < kMtpDraftVocab ||
                        tokenizer_->IsSpecialToken(
                            static_cast<tokenization::TokenId>(token)));
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("Qwen3.8-Flash-Next"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes() + (vision_ ? vision_->ResidentBytes() : 0);
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  const std::size_t vision =
      vision_ ? std::size_t{context} * (config().hidden_size * sizeof(float) +
                                        3 * sizeof(std::int32_t)) +
                    64
              : 0;
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0) +
         vision;
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

NgramTotals Session::NgramRowTotals() const noexcept {
  return session_->NgramRowTotals();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      draft_length_(model_->options_.max_draft_tokens,
                    model_->DecodeConcurrency()),
      draft_vocab_(kMtpDraftExtraRows - model_->draft_specials_) {
  logits_.resize(model_->VocabSize());
  accept_table_.Seed(kMtpAcceptByTop, 32.0F);
}

bool Session::SyncDraftVocab(std::string* error_msg) {
  if (!model_->draft_subset_ || !MtpEnabled())
    return true;
  auto& exec = *model_->executor_;
  const auto first = model_->draft_specials_;
  if (draft_vocab_stale_ || draft_vocab_.scanned() > tokens_.size()) {
    if (!exec.ClearDraftRows(*session_, first, error_msg))
      return false;
    draft_vocab_.Reset();
    draft_vocab_stale_ = false;
  }
  const auto tabled = static_cast<std::uint32_t>(draft_vocab_.rare().size());
  draft_added_.clear();
  draft_vocab_.CatchUp(
      tokens_, [&](std::int32_t token) { return model_->InDraftSubset(token); },
      &draft_added_);
  return exec.AppendDraftRows(*session_, first + tabled, draft_added_,
                              error_msg);
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

bool Session::EnsureFrontier(std::string* error_msg) const {
  if (!frontier_pending_) {
    return true;
  }
  frontier_pending_ = false;
  return model_->executor_->FinishFrontier(logits_.data(), error_msg);
}

void Session::Reset() {
  (void)EnsureFrontier();
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  lookup_.Clear();
  ForgetDraftVocab();
  hidden_base_ = 0;
  draft_length_.Reset();
  model_->executor_->MtpRewind(*session_, 0);
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

void Session::ConfigureVision(
    std::shared_ptr<const qwen::vision::Prompt> prompt) {
  const auto identity = prompt ? prompt->IdentityForPrefix(tokens_.size())
                               : std::span<const std::uint8_t>{};
  if (!tokens_.empty() &&
      !std::ranges::equal(identity, ImageIdentity(tokens_.size())))
    Reset();
  const bool was_valid = valid_;
  valid_ = false;
  session_->ConfigureVision(prompt, model_->vision_,
                            model_->executor_->stream());
  image_prompt_ = std::move(prompt);
  valid_ = was_valid;
}

std::span<const std::uint8_t> Session::ImageIdentity(
    std::size_t token_count) const {
  return image_prompt_ ? image_prompt_->IdentityForPrefix(token_count)
                       : std::span<const std::uint8_t>{};
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize(),
                                  ImageIdentity(tokens_.size()).size()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  if (!EnsureFrontier(error_msg)) {
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const auto identity = ImageIdentity(token_count);
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes = SessionSnapshotHostBytes(
      token_count, model_->VocabSize(), identity.size());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSessionSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .image_identity_bytes = static_cast<std::uint32_t>(identity.size()),
      .policy_concurrency = model_->DecodeConcurrency(),
      .lookup = lookup_.Save(),
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  if (!identity.empty())
    std::memcpy(out, identity.data(), identity.size());
  out += identity.size();
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  if (!model_->executor_->SaveSnapshot(
          *session_, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshot(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  if (!EnsureFrontier(error_msg)) {
    return false;
  }
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSessionSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      (header.image_identity_bytes != 0 && header.image_identity_bytes != 32) ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size,
                                                 header.image_identity_bytes) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::uint8_t> image_identity(in,
                                           in + header.image_identity_bytes);
  if (!image_identity.empty() &&
      !std::ranges::equal(image_identity, ImageIdentity(header.token_count))) {
    AssignError(error_msg,
                "image snapshot requires its matching prompt attachment");
    return false;
  }
  in += header.image_identity_bytes;
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  if (image_identity.empty())
    ConfigureVision(nullptr);
  valid_ = false;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  auto next_drafts = MtpEnabled()
                         ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count)
                         : 0;
  if (MtpEnabled() && model_->draft_subset_) {
    // The next request's sampler/head is not known yet. Avoid discarding
    // existing scratch that either eligible policy can use after restore.
    next_drafts = std::max(next_drafts,
                           restored_policy.Choose(remaining ? remaining - 1 : 0,
                                                  header.token_count, 0, true));
  }
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg, next_drafts)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  // Reject malformed lookup state outright (it would misalign the cursor).
  if (!PromptLookup::Valid(header.lookup, tokens_.size())) {
    Reset();
    AssignError(error_msg, "session snapshot lookup state is inconsistent");
    return false;
  }
  lookup_.Restore(header.lookup);
  ForgetDraftVocab();
  logits_ = std::move(logits);
  anchor_candidates_valid_ = false;
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  stats_ = {};
  valid_ = true;
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {
  // Snapshot copies first-touch hundreds of MiB. Let Linux back the interior
  // with transparent huge pages instead of faulting one 4 KiB page at a time.
  // Advise only complete pages belonging to this allocation; this is optional
  // and does not pin memory or change the serialized payload.
  const long page = sysconf(_SC_PAGESIZE);
  if (page > 0) {
    const auto address = reinterpret_cast<std::uintptr_t>(data_.get());
    const auto skip = (page - address % page) % page;
    if (size > skip) {
      const auto length = (size - skip) / page * page;
      if (length != 0)
        (void)madvise(data_.get() + skip, length, MADV_HUGEPAGE);
    }
  }
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_) {
    return false;
  }
  std::memcpy(destination.data(), data_.get(), size_);
  return true;
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates, bool draft_subset) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && candidates == nullptr ? &draft_token_ : nullptr,
           .candidates = candidates,
           .kv_only = !propose,
           .draft_subset = draft_subset},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg,
                                std::span<const std::uint8_t> propose) {
  if (requests.empty())
    return true;
  if (!propose.empty() && propose.size() != requests.size()) {
    AssignError(error_msg, "draft catch-up head count differs from requests");
    return false;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    if (!session.MtpEnabled())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row,
                     propose.empty() || !propose[i]});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill, std::span<const std::int32_t> after) {
  if (!EnsureFrontier(error_msg)) {
    return false;
  }
  rocm::Executor& exec = *model_->executor_;
#ifdef GUFO_NPU
  if (npu::ProcessFailed()) {
    exec.ReleaseNpuRetryBackup();
  }
#endif
  for (std::size_t off = 0; off < tokens.size();) {
    const auto remaining = tokens.size() - off;
    // Fits-through capacity lets a final chunk absorb the 128-token tail.
    const auto capacity = remaining <= exec.max_batch()
                              ? exec.max_batch()
                              : model_->PrefillCapacity();
    bool npu_on = false;
#ifdef GUFO_NPU
    const auto npu_prefill = model_->device_->npu_prefill();
    npu_on = prefill && npu_prefill && npu_prefill->ready();
#endif
    // A lane takes whole 4096-token groups, so a chunk keeps the wider GPU
    // batching and the lane takes every group in it (up to four).
    const std::size_t room = std::min<std::size_t>(capacity, remaining);
    const std::size_t n = npu_on && room > 4096 ? room - room % 4096 : room;
    const auto chunk = tokens.subspan(off, n);
    if (MtpEnabled() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    // The chunk this loop, or the following Sync, takes next.
    const auto rest = off + n < tokens.size() ? tokens.subspan(off + n) : after;
    const auto next = rest.first(std::min<std::size_t>(
        rest.size() <= exec.max_batch() ? exec.max_batch()
                                        : model_->PrefillCapacity(),
        rest.size()));
    anchor_candidates_valid_ = false;
#ifdef GUFO_NPU
    // L-REL-2: a chunk the NPU may take is backed up so it can run again on the
    // GPU. Chunks that cannot use a lane (not whole 4096 groups, a hold that
    // stays on) are not.
    const auto* pref = npu_on ? npu_prefill.get() : nullptr;
    bool backed_up = false;
    if (pref &&
        npu::ShouldBackupChunk(
            true,
            npu::QualifiedChunk(static_cast<unsigned>(n), exec.max_batch()),
            pref->HoldWillStay())) {
      std::string backup_error;
      backed_up = exec.SaveNpuRetryBackup(*session_, &backup_error);
      static std::atomic<bool> unavailable_logged{false};
      if (!backed_up && !unavailable_logged.exchange(true)) {
        server::Logger::Warn("npu",
                             "GPU retry of a faulted NPU chunk is unavailable: " +
                                 backup_error);
      }
    }
    const bool failed_before = npu::ProcessFailed();
    const auto serial_before = pref ? pref->chunk_serial() : 0;
    const auto noted_before = npu::RequestChunksNoted();
#endif
    bool forwarded = exec.Forward(*session_, chunk, 1, logits_.data(), mode,
                                  error_msg, /*verify_candidates=*/false, next);
#ifdef GUFO_NPU
    if (backed_up) {
      double copy_ms = -1;
      if (!exec.NpuRetryBackupMs(&copy_ms))
        copy_ms = -1;
      const bool used = npu::ChunkUsedThisAttempt(serial_before,
                                                  pref->chunk_serial(),
                                                  pref->ChunkUsed());
      if (used &&
          npu::BackupBelongsToChunk(noted_before, npu::RequestChunksNoted()))
        npu::NoteChunkBackup(copy_ms);
      if (!forwarded &&
          npu::ShouldRetryChunk(failed_before, npu::ProcessFailed(), used,
                                true)) {
        const auto chunk_number =
            npu::Stats().chunks.load(std::memory_order_relaxed);
        const auto started = std::chrono::steady_clock::now();
        std::string retry_error;
        forwarded = exec.RestoreNpuRetryBackup(*session_, &retry_error);
        forwarded =
            forwarded && exec.Forward(*session_, chunk, 1, logits_.data(),
                                      mode, &retry_error,
                                      /*verify_candidates=*/false, next);
        if (forwarded) {
          npu::Faults().gpu_retries.fetch_add(1, std::memory_order_relaxed);
          server::Logger::Warn(
              "npu", npu::FormatChunkRetryWarning(
                         chunk_number, copy_ms,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count()));
        } else if (error_msg) {
          *error_msg += "; GPU retry failed: " + retry_error;
        }
      }
    }
    // The NPU is off for the process once it failed: the buffer is dead weight.
    if (npu::ProcessFailed())
      exec.ReleaseNpuRetryBackup();
#endif
    if (!forwarded) {
      return false;
    }
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n - std::min<std::size_t>(n, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
    off += n;
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt, std::string* error_msg,
                   std::span<const std::int32_t> next) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  // Every Sync starts a new request: reset the copy cursor so the first
  // speculative step does not copy across the request boundary.
  lookup_.BeginRequest();
  if (!valid_)
    Reset();
  // Recurrent state cannot be rewound, so any divergence restarts the
  // session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true, next);
  valid_ = ok;
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

std::uint32_t Session::MaxVerifyWidth() const noexcept {
  return MtpEnabled() ? model_->executor_->max_speculative() : 1;
}

bool Session::TeacherForce(std::span<const std::int32_t> tokens,
                           std::vector<float>* rows, std::string* error_msg,
                           bool prefill) {
  if (!valid_ || tokens_.empty()) {
    AssignError(error_msg, "teacher forcing needs a synced session");
    return false;
  }
  if (!EnsureFrontier(error_msg)) {
    return false;
  }
  const std::size_t n = tokens.size();
  if (n == 0 || (prefill ? MtpEnabled() : n > MaxVerifyWidth()) ||
      tokens_.size() + n > ContextSize()) {
    AssignError(error_msg, "teacher forcing width or context out of range");
    return false;
  }
  rocm::Executor& exec = *model_->executor_;
  const std::size_t vocab = model_->VocabSize();
  rows->resize(n * vocab);
  valid_ = false;
  anchor_candidates_valid_ = false;
  // The same Forward calls DecodeStep makes: a one-token decode, or a verify
  // pass whose every row is kept (the draft head is never consulted).
  // Prefill arithmetic is the same at every chunk width.
  const auto mode = prefill  ? rocm::Executor::ForwardMode::kPrefill
                    : n == 1 ? rocm::Executor::ForwardMode::kDecode
                             : rocm::Executor::ForwardMode::kVerify;
  if (!exec.Forward(*session_, tokens, static_cast<std::uint32_t>(n),
                    rows->data(), mode, error_msg) ||
      (mode == rocm::Executor::ForwardMode::kVerify &&
       !exec.Rollback(*session_, static_cast<std::uint32_t>(n), error_msg))) {
    return false;
  }
  std::copy_n(rows->data() + (n - 1) * vocab, vocab, logits_.begin());
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
  valid_ = true;
  return true;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  // Sampled verification reads candidate lists, so the verified rows stay on
  // the GPU; FinishDecode fetches them only for a fallback.
  bool rows_on_device{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
  /// Trailing chain tokens copied by prompt lookup (after the MTP drafts).
  std::uint32_t lookup{0};
  /// Copy budget for this cycle (0 disables copying for the step).
  std::size_t lookup_cap{0};
  /// Committed index the copied tokens continue.
  std::size_t lookup_start{0};
  bool draft_subset{false};  ///< this cycle drafts through the subset head
  bool sampled_subset_cost{false};
  std::uint32_t policy_concurrency{0};
};

namespace {

/// The exact target distribution of `row` from its candidate list, or
/// nothing when the list cannot prove it (then the caller uses the row).
std::optional<sampling::SamplingDistribution> FastDistribution(
    const sampling::SamplerState& sampler, const MtpCandidateLogits& list,
    std::span<const float> row) {
  return sampler.DistributionFromTop(std::span(list.logits).first(list.size),
                                     std::span(list.ids).first(list.size),
                                     row.size());
}

/// True when every finite candidate still matches `row` bit for bit: the
/// list was selected from this very row.
bool CandidatesMatch(const MtpCandidateLogits& list,
                     std::span<const float> row) {
  for (std::size_t i = 0; i < list.size; ++i) {
    if (!std::isfinite(list.logits[i])) {
      continue;
    }
    if (list.ids[i] >= row.size() ||
        std::memcmp(&list.logits[i], &row[list.ids[i]], sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::AppendLookup(PendingDecode& pending, bool stop_at_eos,
                           std::size_t minimum, bool require_resume) {
  const auto cap = pending.lookup_cap;
  if (pending.chain.size() >= cap) {
    return false;
  }
  if (stop_at_eos && model_->IsStopToken(pending.chain.back())) {
    return false;
  }
  const auto match =
      lookup_.Find(tokens_, pending.chain,
                   pending.sampled ? PromptLookup::kSampledMinMatch
                                   : PromptLookup::kMinMatch,
                   require_resume);
  if (match.length == 0) {
    return false;
  }
  const std::size_t available = std::min(cap - pending.chain.size(),
                                         lookup_.IndexedTokens() - match.start);
  if (available < minimum)
    return false;
  std::size_t count = 0;
  for (; count < available; ++count) {
    const std::int32_t token = tokens_[match.start + count];
    pending.chain.push_back(token);
    if (pending.sampled) {
      // A point-mass proposal: accepted with the target's p(token), and a
      // rejection resamples from the target with the token removed.
      MtpProposal proposal;
      proposal.ids[0] = static_cast<sampling::TokenId>(token);
      proposal.probabilities[0] = 1.0F;
      proposal.size = 1;
      proposal.token = proposal.ids[0];
      proposal.probability = 1.0F;
      pending.proposals.push_back(proposal);
    }
    if (stop_at_eos && model_->IsStopToken(token)) {
      ++count;
      break;
    }
  }
  pending.lookup = static_cast<std::uint32_t>(count);
  pending.lookup_start = match.start;
  pending.width = pending.chain.size();
  return count != 0;
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts,
                            std::uint32_t occupancy) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  // ModelOptions::mtp_survival: sampled single-session decoding drafts up to
  // the limit and stops on the survival estimate (see kMtpSurvivalFloor);
  // the width is an upper bound.
  const bool survival = model_->options_.mtp_survival && !batch_drafts &&
                        !defer_head && sampler.config().uses_random_sampling();
  const auto concurrency = model_->DecodeConcurrency();
  // Head eligibility is MtpDraftSubsetAllowed. Sampled cohorts are never
  // jointly planned (DecodeBatchImpl), so a sampled request decides the same
  // way alone and in any batch. Greedy requests draft small whenever they
  // cannot collide with another request's table. Every request keeps the
  // established length controller and AR backoff.
  const bool allow_subset =
      MtpDraftSubsetAllowed(sampler.config().uses_random_sampling(),
                            concurrency, occupancy, batch_drafts.has_value());
  // Unplanned greedy and unseeded sampled cycles are priced at the requests
  // decoding now; seeded sampled ones keep the configured concurrency so they
  // replay the same widths, and a planned batch keeps the row its plan used.
  pending->policy_concurrency = MtpPolicyConcurrency(
      sampler.config().uses_random_sampling(), sampler.config().seed >= 0,
      batch_drafts.has_value(), occupancy);
  std::size_t width =
      MtpEnabled() && cap > 1
          ? 1 + (batch_drafts ? std::min<std::uint32_t>(*batch_drafts, cap - 1)
                 : survival   ? std::min<std::uint32_t>(
                                  kMaxMtpDraftTokens,
                                  static_cast<std::uint32_t>(cap - 1))
                            : draft_length_.Choose(
                                  static_cast<std::uint32_t>(cap - 1),
                                  static_cast<std::uint32_t>(tokens_.size()),
                                  pending->policy_concurrency))
          : cap;
  if (width == 0) {
    result->stop = true;
    return true;
  }
  // After a sampled MTP cycle logits_ is the kept verification row, whose
  // candidate list FinishDecode left behind: sample it without a full scan.
  std::optional<sampling::TokenId> fast_anchor;
  if (anchor_candidates_valid_) {
    anchor_candidates_valid_ = false;
    // A frontier row still in flight is the very row the verify graph
    // selected these candidates from (FinishDecode).
    if (frontier_pending_ || CandidatesMatch(anchor_candidates_, logits_)) {
      fast_anchor = sampler.SampleFromTop(
          std::span(anchor_candidates_.logits).first(anchor_candidates_.size),
          std::span(anchor_candidates_.ids).first(anchor_candidates_.size),
          logits_.size());
    }
  }
  if (!fast_anchor && !EnsureFrontier(error_msg)) {
    return false;
  }
  const auto anchor = static_cast<std::int32_t>(
      fast_anchor ? *fast_anchor : sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  if (!MtpEnabled() || width < 2) {
    if (MtpEnabled() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  // #332: GPU greedy now applies penalties, so temperature alone gates it.
  const bool gpu_greedy = sampler.config().temperature == 0.0F;
  const bool gpu_verification = gpu_greedy;
  if (allow_subset && !SyncDraftVocab(error_msg))
    return false;
  pending->draft_subset = allow_subset && model_->draft_subset_ &&
                          draft_vocab_.Fits(anchor, [&](std::int32_t token) {
                            return model_->InDraftSubset(token);
                          });
  // Wider verification slowed short resumed replies in matched controls.
  // Keep the established costs within two blocks of the remaining budget.
  // Full target rows (for example, active tool constraints) also keep their
  // original costs; the smaller-head curve does not price their transfers.
  // The fork prices sampled cycles whose verification reads GPU candidate
  // lists (the fast_sampling gate in FinishDecode); upstream's
  // CanCompactMtpVerification is not ported.
  const bool compact_sampled =
      sampled && platform::PlatformTuning().fast_sampling &&
      sampler.config().top_k > 0 && !sampler.config().penalties_enabled();
  pending->sampled_subset_cost =
      compact_sampled && pending->draft_subset &&
      std::min(max_tokens, room) > 2 * exec.max_speculative();
  if (pending->sampled_subset_cost && !batch_drafts) {
    // Price the head actually used, after vocabulary fallback and the copy
    // bypass. The policy depends only on committed history, not timings.
    width = 1 + draft_length_.Choose(static_cast<std::uint32_t>(cap - 1), base,
                                     pending->policy_concurrency, true);
  }
  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   sampled ? &pending->candidates : nullptr,
                                   pending->draft_subset)) {
    return false;
  }
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler;
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  // ModelOptions::prompt_lookup (single session): after each kept MTP
  // draft, a long enough match in the committed tokens fills the rest of
  // the chain up to the verify width and ends drafting. A request's first
  // speculative step copies nothing (see PromptLookup), so scheduling a
  // peer cannot change a sampled request's proposals or RNG sequence.
  const bool lookup =
      model_->options_.prompt_lookup && !batch_drafts && !defer_head;
  if (lookup) {
    pending->lookup_cap = lookup_.Step(base) ? cap : 0;
    if (pending->lookup_cap != 0) {
      lookup_.Extend(tokens_, sampled);
    }
  }
  if (!defer_head) {
    // Copies can supply the whole chain without the neural head. A sampled
    // copy needs an established source and a full match: replacing the
    // proposal on shared syntax alone regressed copying at temperature.
    if (lookup && AppendLookup(*pending, request.stop_at_eos, 3, sampled)) {
      return true;
    }
    float alive = 1.0F;
    while (pending->chain.size() < width) {
      AppendDraft(*pending);
      if (request.stop_at_eos && model_->IsStopToken(pending->chain.back())) {
        pending->width = pending->chain.size();
        break;
      }
      if (survival && !pending->proposals.empty()) {
        const std::size_t depth = pending->proposals.size() - 1;
        alive *= accept_table_.Estimate(depth == 0 ? 0 : 1,
                                        MtpTopBin(pending->proposals.back()));
        // The first proposal is always kept (a speculative cycle).
        if (depth > 0 && alive < kMtpSurvivalFloor) {
          pending->chain.pop_back();
          pending->proposals.pop_back();
          pending->width = pending->chain.size();
          break;
        }
      }
      if (lookup && AppendLookup(*pending, request.stop_at_eos, 1)) {
        break;
      }
      if (pending->chain.size() < width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = sampled ? &pending->candidates : nullptr,
               .draft_subset = pending->draft_subset},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), sampler,
                                 std::span(chain).subspan(1), error_msg)) {
    return false;
  }
  // Tuning::fast_sampling: sampled verification reads GPU-selected
  // candidate lists instead of scanning full host rows (exact, see
  // FastDistribution); without them it falls back to the rows.
  std::array<MtpCandidateLogits, kMaxMtpDraftTokens + 1> candidates{};
  std::string candidate_error;
  const bool have_candidates =
      sampled && platform::PlatformTuning().fast_sampling &&
      k <= candidates.size() && sampler.config().top_k > 0 &&
      !sampler.config().penalties_enabled() &&
      exec.VerifyCandidates(std::span(candidates).first(k), &candidate_error);
  // Full rows kept on the GPU are fetched once, only when a row is needed.
  bool rows_ready = !pending.rows_on_device;
  const auto ensure_rows = [&] {
    if (!rows_ready) {
      rows_ready =
          exec.DownloadVerification(k, verify_logits_.data(), error_msg);
    }
    return rows_ready;
  };
  if (sampled && !have_candidates && !ensure_rows()) {
    return false;
  }
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      const auto row = std::span<const float>(verify_logits_)
                           .subspan((keep - 1) * vocab, vocab);
      std::optional<sampling::SamplingDistribution> target;
      if (have_candidates) {
        target = FastDistribution(sampler, candidates[keep - 1], row);
      }
      if (!target && !ensure_rows()) {
        return false;
      }
      const auto verified =
          target
              ? VerifyMtpProposal(*target, vocab, proposals[keep - 1], sampler)
              : VerifyMtpProposal(row, proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  // Sampled cycles with candidate lists sample the next anchor from those
  // lists, so the frontier row stays in flight (EnsureFrontier fetches it
  // when anything else needs it) and the rollback is not waited for: the
  // next draft queues behind it.
  const bool defer_frontier =
      !gpu_verification && pending.rows_on_device && have_candidates;
  frontier_pending_ = false;
  if (!exec.Rollback(
          *session_, keep, error_msg,
          gpu_verification || pending.rows_on_device ? logits_.data() : nullptr,
          defer_frontier)) {
    return false;
  }
  frontier_pending_ = defer_frontier;
  if (!gpu_verification) {
    // Rows kept on the GPU: Rollback already wrote the frontier to logits_.
    if (!pending.rows_on_device) {
      std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                  logits_.begin());
    }
    // The next PrepareDecode samples its anchor from this row's list.
    anchor_candidates_valid_ = have_candidates;
    if (have_candidates) {
      anchor_candidates_ = candidates[keep - 1];
    }
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // Copied proposals trail the MTP drafts; the length controller sees only
  // the MTP part.
  const std::uint32_t mtp_drafts = k - 1 - pending.lookup;
  const std::uint32_t mtp_accepted = std::min(keep - 1, mtp_drafts);
  stats_.lookup += pending.lookup;
  stats_.lookup_accepted += keep - 1 - mtp_accepted;
  if (pending.lookup != 0) {
    // The copied text continues at lookup_start + accepted; a mostly
    // rejected copy keeps the next match long.
    lookup_.Observe(pending.lookup_start, pending.lookup,
                    keep - 1 - mtp_accepted);
  }
  // A fully accepted neural tail is hidden behind the copies: observe the
  // full kept length so the controller does not over-draft after copies.
  const bool mtp_all = mtp_accepted == mtp_drafts;
  const std::uint32_t observed = mtp_all ? keep - 1 : mtp_accepted;
  stats_.small_head_cycles += pending.draft_subset;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(
      observed, result->stop || mtp_all ? observed : mtp_drafts, base,
      pending.policy_concurrency, pending.sampled_subset_cost);
  // Same per-proposal evidence refines the online accept table. Bin each
  // proposal by its draft top-probability; every drafted MTP proposal up to
  // the kept prefix was accepted, the next one (if drafted) was rejected.
  if (!pending.proposals.empty()) {
    const std::size_t mtp_proposals = std::min(
        pending.proposals.size(), static_cast<std::size_t>(mtp_drafts));
    for (std::size_t i = 0; i < mtp_proposals; ++i) {
      accept_table_.Observe(i == 0 ? 0 : 1, MtpTopBin(pending.proposals[i]),
                            i < mtp_accepted);
    }
  }

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos,
                         std::uint32_t occupancy) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  if (!PrepareDecode(request, &pending, error_msg, false, std::nullopt,
                     occupancy)) {
    return false;
  }
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  // Tuning::fast_sampling: sampled verification rows stay on the GPU; the
  // candidate lists replace them unless a row is needed (DownloadVerification).
  pending.rows_on_device =
      pending.speculative && !pending.gpu_verification && pending.sampled &&
      platform::PlatformTuning().fast_sampling && sampler.config().top_k > 0 &&
      !sampler.config().penalties_enabled();
  float* logits = !pending.speculative ? logits_.data()
                  : pending.gpu_verification || pending.rows_on_device
                      ? nullptr
                      : verify_logits_.data();
  if (!pending.speculative) {
    frontier_pending_ = false;  // the forward rewrites the whole row
  }
  if (!model_->executor_->Forward(
          *session_, pending.chain, pending.chain.size(), logits,
          pending.speculative ? rocm::Executor::ForwardMode::kVerify
                              : rocm::Executor::ForwardMode::kDecode,
          error_msg, pending.rows_on_device)) {
    return false;
  }
  const bool ok = FinishDecode(request, pending, error_msg);
  valid_ = ok;
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::vector<Request> active;
  active.reserve(requests.size());
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, &shared_error);
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos);
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->DecodeStep(
        r.max_tokens, *r.sampler, r.result, &r.outcome->error, r.stop_at_eos);
    return r.outcome->completed;
  }
  for (const auto& r : requests) {
    if (!r.session->EnsureFrontier(error_msg)) {
      return false;
    }
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  if (std::ranges::all_of(
          requests, [](const auto& r) { return r.session->MtpEnabled(); }) &&
      std::ranges::none_of(requests, [](const auto& request) {
        return request.sampler->config().uses_random_sampling();
      })) {
    std::array<MtpBatchController::Row, 8> rows{};
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& r = requests[i];
      const auto cap = std::min<std::size_t>(
          r.max_tokens, r.session->ContextSize() - r.session->Position());
      rows[i] = {&r.session->draft_length_,
                 static_cast<std::uint32_t>(
                     std::min<std::size_t>(cap, exec.max_speculative())) -
                     (cap != 0)};
      batch_context = std::max(batch_context, r.session->Position());
    }
    batch_drafts =
        policy.Choose(std::span(rows).first(requests.size()), batch_context);
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(
              r, &pending[i], &r.outcome->error, true, batch_drafts,
              static_cast<std::uint32_t>(requests.size())))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  std::vector<std::uint8_t> propose;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty()) {
      catchup.push_back({requests[i].session, pending[i].chain.front()});
      propose.push_back(pending[i].speculative &&
                        pending[i].chain.size() < pending[i].width);
    }
  }
  if (!DraftCatchUpBatch(catchup, error_msg, propose))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  std::vector<rocm::Executor::MtpHeadItem> heads;
  std::vector<rocm::Executor::MtpBatchItem> bodies;
  for (;;) {
    heads.clear();
    bodies.clear();
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back({requests[i].session->session_.get(),
                         {.token = p.sampled ? nullptr : &p.draft,
                          .candidates = p.sampled ? &p.candidates : nullptr,
                          .draft_subset = p.draft_subset}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  // Lookup-only chains (no neural drafts) are longer than the shared width;
  // trim them so one batch verify stays shape-compatible with the rest.
  if (batch_drafts) {
    const bool mixed_lookup = std::ranges::any_of(pending, [](const auto& p) {
      return !p.chain.empty() && p.lookup == 0;
    });
    if (mixed_lookup) {
      for (auto& p : pending) {
        if (p.lookup == 0)
          continue;
        const auto cap =
            std::max<std::size_t>(1 + *batch_drafts, (p.lookup_cap + 1) / 2);
        if (p.chain.size() > cap) {
          p.lookup -= static_cast<std::uint32_t>(p.chain.size() - cap);
          p.chain.resize(cap);
          p.width = cap;
        }
      }
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    const auto row_offset = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    session.anchor_candidates_valid_ = false;
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                 &r.outcome->error) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  const bool used_lookup =
      std::ranges::any_of(pending, [](const auto& p) { return p.lookup != 0; });
  if (used_lookup)
    policy.SkipCycle();
  if (!used_lookup && batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  for (const auto& r : requests) {
    if (!r.session->EnsureFrontier(error_msg)) {
      return false;
    }
  }
  auto& exec = *requests.front().session->model_->executor_;
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      session.anchor_candidates_valid_ = false;
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::qwen38_flash_next
