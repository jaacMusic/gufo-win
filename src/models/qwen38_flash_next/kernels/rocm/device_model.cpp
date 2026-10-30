#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <thread>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#ifdef GUFO_NPU
#include "src/cli/serve/logging.hpp"
#include "src/models/qwen38_flash_next/npu/linear_z.hpp"
#include "src/models/qwen38_flash_next/npu/projection.hpp"
#include "src/models/qwen38_flash_next/npu/server_prefill.hpp"
#endif

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
  /// Expands Q8_0 weights to F32 instead of narrowing F32 to F16.
  bool expand_q8{false};
  std::size_t source_bytes{0};
};

/// Re-encodes Q6_K rows as Q8_0 (fp16 scale + 32 int8 per block).
/// Q8_0's per-32 step is about a quarter of Q6_K's per-16 step, so the added
/// rounding is small next to the Q6_K quantization it sits on.
void Q6KRowsToQ8_0(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                   std::size_t cols) {
  const std::size_t src_row = cols / 256 * 210;
  const std::size_t dst_row = cols / 32 * 34;
  std::vector<float> values(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    gufo::quant::DequantizeQ6_K(src + r * src_row, values.data(), cols);
    std::uint8_t* out = dst + r * dst_row;
    for (std::size_t b = 0; b < cols / 32; ++b) {
      const float* x = values.data() + b * 32;
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
      const auto scale = static_cast<_Float16>(amax / 127.0F);
      const float d = static_cast<float>(scale);
      const float inverse = d != 0.0F ? 1.0F / d : 0.0F;
      std::memcpy(out + b * 34, &scale, sizeof(scale));
      auto* q = reinterpret_cast<std::int8_t*>(out + b * 34 + 2);
      for (int i = 0; i < 32; ++i) {
        const float v = std::nearbyint(x[i] * inverse);
        q[i] = static_cast<std::int8_t>(std::clamp(v, -127.0F, 127.0F));
      }
    }
  }
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  /// Layer() leaves the routed experts for a later Experts() pass.
  bool defer_experts{false};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Uploads a narrow projection the quantized tier does not cover (the
  /// hyper-connection inject) as F32. A Q8_0 tensor is uploaded as is and
  /// expanded once the disk pipeline drains, so it runs the F32 kernels with
  /// its dequantized values instead of quantizing their activations.
  DeviceTensor CopyF32(const TensorRef& t) {
    if (t.empty() || !ok || t.type != core::GgmlType::kQ8_0) {
      return Copy(t);
    }
    DeviceTensor d = Copy(t);
    const std::size_t count = t.cols * t.rows * t.experts;
    void* f32 = nullptr;
    if (!ok ||
        hipMalloc(&f32, count * sizeof(float) + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name));
      return {};
    }
    allocations.push_back(f32);
    bytes += count * sizeof(float) + kTailMargin;
    conversions.push_back({d.data, f32, count, true, t.SizeBytes()});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(f32) + count * 4, 0,
                         kTailMargin, nullptr);
    d.data = f32;
    d.type = core::GgmlType::kF32;
    return d;
  }

  /// A Q6_K matrix (the output head of unsloth UD-IQ4_XS) re-encoded as
  /// Q8_0 on the host, so it runs on the tuned Q8_0 dense tier instead of a
  /// path the HIP kernels do not have.
  DeviceTensor CopyQ6KAsQ8_0(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.cols % 256 != 0 || t.experts != 1) {
      Fail("Q6_K tensor " + std::string(t.name) + " has an unsupported shape");
      return d;
    }
    const std::size_t src_row = t.cols / 256 * 210;
    const std::size_t dst_row = t.cols / 32 * 34;
    const std::size_t size = dst_row * t.rows;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name));
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    constexpr std::size_t kChunkRows = 16384;
    std::vector<std::uint8_t> host(kChunkRows * dst_row);
    const std::size_t workers =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    for (std::size_t r0 = 0; r0 < t.rows; r0 += kChunkRows) {
      const std::size_t rows = std::min<std::size_t>(kChunkRows, t.rows - r0);
      const std::size_t per = (rows + workers - 1) / workers;
      std::vector<std::jthread> pool;
      for (std::size_t w = 0; w < workers && w * per < rows; ++w) {
        const std::size_t begin = w * per;
        const std::size_t count = std::min(per, rows - begin);
        pool.emplace_back([&, begin, count] {
          Q6KRowsToQ8_0(src + (r0 + begin) * src_row,
                        host.data() + begin * dst_row, count, t.cols);
        });
      }
      pool.clear();
      if (hipMemcpy(static_cast<std::uint8_t*>(ptr) + r0 * dst_row, host.data(),
                    rows * dst_row, hipMemcpyHostToDevice) != hipSuccess) {
        Fail("upload failed for " + std::string(t.name));
        return d;
      }
    }
    (void)hipMemset(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV, unless `expand_q8` asks for the F32 stack's F16
  /// route over its dequantized values.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts,
                     bool expand_q8 = false) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0 && !expand_q8) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    void* f32 = ptr;
    std::size_t f32_bytes = size;
    if (type == core::GgmlType::kQ8_0) {
      const std::size_t elements = rows * (*parts.begin())->cols;
      f32_bytes = elements * sizeof(float);
      if (hipMalloc(&f32, f32_bytes + kTailMargin) != hipSuccess) {
        Fail("stacked tensor allocation failed");
        return d;
      }
      allocations.push_back(f32);
      bytes += f32_bytes + kTailMargin;
      conversions.push_back({ptr, f32, elements, true, size});
    }
    const std::size_t count = f32_bytes / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({f32, half, count, false, f32_bytes});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), CopyF32(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta}, true);
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    if (!defer_experts) {
      Experts(l, d);
    }
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  void Experts(const LayerWeights& l, DeviceLayer& d) {
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 || output_bytes_ % output_.rows != 0)
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg
#ifdef GUFO_NPU
    ,
    std::shared_ptr<npu::ServerPrefill> prefill
#endif
) {
  // The CPU reference also reads Q6_K, but the production embedding, dense
  // and routed kernels do not. Reject it before allocating device weights;
  // the embedding and output head are re-encoded as Q8_0 instead.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K || &t == &w.token_embd ||
        &t == &w.output)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
#ifdef GUFO_NPU
  // Stage before GPU uploads. Only a successfully started authorized service
  // can be injected; direct model callers default to no capability. A staging
  // failure disables the NPU lane only: the GPU path still loads the model.
  // Only the linear-z and shared-expert lanes have call sites; the query/gate
  // plan would be staged and made resident without ever being dispatched.
  const bool npu_shapes =
      w.config.num_layers == 48 && w.config.full_attention_interval == 4 &&
      w.config.hidden_size == 2560 && w.config.SsmConvChannels() == 10240 &&
      w.config.SsmValueDim() == 6144;
  std::vector<std::shared_ptr<const npu::LinearZWeights>> z;
  std::vector<std::shared_ptr<const npu::ProjectionWeights>> shared;
  if (prefill && prefill->ready() && npu_shapes) {
    try {
      z.resize(w.layers.size());
      shared.resize(w.layers.size());
      const auto matrix = [](const TensorRef& t) {
        return npu::Q8Matrix{
            t.type,
            t.rows,
            t.cols,
            {static_cast<const std::byte*>(t.data), t.SizeBytes()}};
      };
      for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        // Validate metadata before constructing any byte span/SizeBytes.
        if (!l.linear || l.ssm_qkv.experts != 1 || l.ssm_gate.experts != 1 ||
            l.ssm_qkv.empty() || l.ssm_gate.empty() ||
            l.ssm_qkv.type != core::GgmlType::kQ8_0 ||
            l.ssm_gate.type != core::GgmlType::kQ8_0 ||
            l.ssm_qkv.cols != npu::kK || l.ssm_gate.cols != npu::kK ||
            l.ssm_qkv.rows != 10240 || l.ssm_gate.rows != 6144)
          continue;
        z[i] = npu::PrepareLinearZ(matrix(l.ssm_qkv), matrix(l.ssm_gate),
                                   prefill->knobs().slices);
      }
      const auto q8 = [](const TensorRef& t, unsigned rows) {
        return !t.empty() && t.experts == 1 &&
               t.type == core::GgmlType::kQ8_0 && t.cols == npu::kK &&
               t.rows == rows;
      };
      for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        if (w.config.shared_expert_ff > 0 &&
            w.config.shared_expert_ff <= 1024 &&
            w.config.shared_expert_ff % 32 == 0 &&
            q8(l.shexp_gate, w.config.shared_expert_ff) &&
            q8(l.shexp_up, w.config.shared_expert_ff) &&
            !l.shexp_down.empty() && l.shexp_down.experts == 1 &&
            l.shexp_down.type == core::GgmlType::kQ8_0 &&
            l.shexp_down.rows == 2560 &&
            l.shexp_down.cols == w.config.shared_expert_ff)
          shared[i] =
              npu::PrepareSharedExpert(matrix(l.shexp_gate), matrix(l.shexp_up),
                                       prefill->knobs().shared_expert);
      }
      for (const auto& weights : z)
        if (weights)
          for (const auto& b : weights->b)
            prefill->session().PrepareWeight(std::as_bytes(std::span(b)));
      for (const auto& weights : shared)
        if (weights)
          for (const auto& b : weights->b)
            prefill->session().PrepareWeight(std::as_bytes(std::span(b)));
      server::Logger::Info(
          "npu",
          "linear-z qualified layers=" +
              std::to_string(std::count_if(
                  z.begin(), z.end(), [](const auto& p) { return bool(p); })) +
              "; slices=" + std::to_string(prefill->knobs().slices) +
              "; shared gate/up qualified layers=" +
              std::to_string(
                  std::count_if(shared.begin(), shared.end(),
                                [](const auto& p) { return bool(p); })) +
              "; other weight shapes and non-full prefill use GPU "
              "fallback");
      m->npu_prefill_ = std::move(prefill);
    } catch (const std::exception& e) {
      z.clear();
      shared.clear();
      prefill->Reject(e.what());
      server::Logger::Warn("npu", std::string("linear-z staging failed (") +
                                      e.what() +
                                      "); falling back to the GPU "
                                      "path");
    }
  } else if (prefill && prefill->ready()) {
    server::Logger::Warn("npu",
                         "model shape not qualified for linear-z; "
                         "falling back to the GPU path");
  }
#endif
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg};
  const auto head = [&](const TensorRef& t) {
    return t.type == core::GgmlType::kQ6_K ? up.CopyQ6KAsQ8_0(t) : up.Copy(t);
  };
  // Tuning::hot_first_upload: allocation order decides placement. Once most
  // of the GPU memory is taken, later allocations get slower memory (up to
  // ~20% lower GEMV bandwidth at the tail on gfx1151 under Windows).
  // Everything read in full on every token goes first: the output head,
  // then every dense projection (target and MTP). The routed experts follow
  // (each byte is read by a few percent of tokens), and the token embedding,
  // a row gather, comes last.
  const bool hot_first = platform::PlatformTuning().hot_first_upload;
  const bool tied = w.output.data == w.token_embd.data;
  if (tied || !hot_first) {
    m->token_embd_ = head(w.token_embd);
  }
  m->output_ = tied ? m->token_embd_ : head(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  up.defer_experts = hot_first;
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
#ifdef GUFO_NPU
    if (m->layers_.size() <= z.size()) {
      m->layers_.back().npu_z = std::move(z[m->layers_.size() - 1]);
      m->layers_.back().npu_shared = std::move(shared[m->layers_.size() - 1]);
    }
#endif
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
    up.shard_base = 0;
  }
  if (hot_first) {
    for (std::size_t i = 0; i < w.layers.size() && up.ok; ++i) {
      up.Experts(w.layers[i], m->layers_[i]);
    }
    if (mtp != nullptr && up.ok) {
      up.shard_base = shard_count;
      up.Experts(mtp->block, m->mtp_);
      up.shard_base = 0;
    }
    if (!tied) {
      m->token_embd_ = head(w.token_embd);
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  m->output_bytes_ = w.output.SizeBytes();
  for (const auto& c : conversions) {
    if (c.expand_q8) {
      ExpandQ8_0(c.source, static_cast<float*>(c.destination), c.count,
                 nullptr);
    } else {
      NarrowActivations(static_cast<const float*>(c.source), c.destination,
                        false, c.count, nullptr);
    }
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.source_bytes + kTailMargin;
  }

  // Q4_1 decode copies: requantize hot decode matrices so the decode GEMV
  // reads 20 instead of 34 bytes per 32 elements. Opt-in "off"; default
  // "balanced" copies the small dense stack + shared expert;
  // GUFO_DECODE_COPY=q4 also covers hyper-connections and attention output. The
  // Q8_0 weights stay resident and serve prefill and wide batches, so copies
  // only add memory.
  {
    const std::string mode = getenv("GUFO_DECODE_COPY") != nullptr
                                 ? getenv("GUFO_DECODE_COPY")
                                 : "balanced";
    if (mode == "off") {
      fprintf(stderr, "decode Q4_1 copies: off\n");
    } else {
      std::string classes = getenv("GUFO_DECODE_COPY_CLASSES") != nullptr
                                ? getenv("GUFO_DECODE_COPY_CLASSES")
                                : "ssm_in,ssm_out,attn_qkv,shexp";
      const bool q4_on = mode.empty() || mode == "balanced" || mode == "q4";
      const auto wants = [&classes](const char* cls) {
        return classes.find(cls) != std::string::npos;
      };
      constexpr std::size_t kMinElements = 2560 * 6144;
      constexpr std::size_t kHcElements = 320 * 10240;
      constexpr std::size_t kBlkBytes = 20;  // Q4_1
      std::size_t copied = 0;
      const auto add = [&](DeviceTensor& t, std::size_t min_elements) {
        if (t.type != core::GgmlType::kQ8_0 || t.data == nullptr ||
            t.cols % 32 != 0) {
          return;
        }
        const std::size_t rows = std::size_t{t.rows} * t.experts;
        const std::size_t blocks = rows * t.cols / 32;
        if (rows * t.cols < min_elements || blocks == 0 ||
            rows * t.cols % 32 != 0) {
          return;
        }
        void* copy = nullptr;
        if (hipMalloc(&copy, blocks * kBlkBytes + kTailMargin) != hipSuccess) {
          return;
        }
        m->allocations_.push_back(copy);
        m->bytes_ += blocks * kBlkBytes + kTailMargin;
        rocm::RequantQ8_0ToQ4_1(t.data, copy, blocks, nullptr);
        if (hipGetLastError() != hipSuccess) {
          (void)hipFree(copy);
          return;
        }
        t.decode_data = copy;
        t.decode_type = core::GgmlType::kQ4_1;
        copied += blocks * kBlkBytes;
      };
      const auto wants_layer = [&](DeviceLayer& l) {
        if (wants("ssm_in")) {
          add(l.ssm_in, kMinElements);
        }
        if (wants("ssm_out")) {
          add(l.ssm_out, kMinElements);
        }
        if (wants("attn_qkv")) {
          add(l.attn_qkv, kMinElements);
        }
        if (wants("attn_out")) {
          add(l.attn_out, kMinElements);
        }
        if (wants("hc")) {
          for (DeviceMixer* mx : {&l.hc_attn, &l.hc_ffn}) {
            add(mx->down, kHcElements);
            add(mx->up, kHcElements);
          }
        }
        if (wants("shexp")) {
          add(l.shexp_up, 1);
          add(l.shexp_gate, 1);
          add(l.shexp_down, 1);
        }
      };
      for (auto& l : m->layers_) {
        wants_layer(l);
      }
      const auto status = hipDeviceSynchronize();
      if (status != hipSuccess) {
        if (error_msg != nullptr) {
          *error_msg = "decode weight conversion failed: " +
                       std::string(hipGetErrorString(status));
        }
        return nullptr;
      }
      fprintf(stderr, "decode Q4_1 copies: %.2f GiB\n",
              double(copied) / (1024.0 * 1024.0 * 1024.0));
    }
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
