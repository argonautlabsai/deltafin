#include "provider_pilot.h"

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/div.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/round.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/where.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr float kPilotRmsEpsilon = 1.0e-5F;
constexpr std::uint32_t kK3Hidden = 7168;
constexpr std::uint32_t kK3Experts = 896;
constexpr std::uint32_t kFirstMoeLayer = 1;
constexpr std::uint32_t kLastMoeLayer = 92;
constexpr std::size_t kPilotMaxPositions = 64;
constexpr std::array<std::int64_t, 1> kLastDimension = {-1};
constexpr std::array<std::int64_t, 1> kRouterRowDimension = {1};

void require_f32(const at::Tensor& tensor, const at::IntArrayRef shape,
                 const at::Device& device, const char* name) {
  if (!tensor.defined() || tensor.scalar_type() != at::kFloat ||
      tensor.device() != device || !tensor.is_contiguous() ||
      tensor.sizes() != shape) {
    throw std::invalid_argument(std::string(name) +
                                " violates its contiguous fp32 shape/device contract");
  }
}

void validate_router(const PilotRouterT1& router, const at::Device& device,
                     const bool exact_k3) {
  if (router.layer_index < kFirstMoeLayer ||
      router.layer_index > kLastMoeLayer || router.generation == 0) {
    throw std::invalid_argument(
        "pilot router layer/generation is outside K3's loaded-layer contract");
  }
  if (router.hidden_size == 0 || router.expert_count < kPilotTopK ||
      router.expert_count >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::uint16_t>::max()) +
              1U) {
    throw std::invalid_argument(
        "pilot router dimensions are empty, undersized, or exceed uint16 IDs");
  }
  if (exact_k3 &&
      (router.hidden_size != kK3Hidden || router.expert_count != kK3Experts)) {
    throw std::invalid_argument(
        "production pilot router accepts only exact K3 geometry");
  }

  const std::int64_t hidden = static_cast<std::int64_t>(router.hidden_size);
  const std::int64_t experts = static_cast<std::int64_t>(router.expert_count);
  require_f32(router.post_attention_norm, {hidden}, device,
              "pilot post-attention norm");
  require_f32(router.correction_bias, {experts}, device,
              "pilot correction bias");

  if (router.packed_int8_qualified) {
    if (!router.router.quantized.defined() ||
        router.router.quantized.scalar_type() != at::kChar ||
        router.router.quantized.device() != device ||
        !router.router.quantized.is_contiguous() ||
        router.router.quantized.sizes() !=
            at::IntArrayRef({experts, hidden}) ||
        !router.router.row_scales.defined() ||
        router.router.row_scales.scalar_type() != at::kFloat ||
        router.router.row_scales.device() != device ||
        !router.router.row_scales.is_contiguous() ||
        router.router.row_scales.sizes() != at::IntArrayRef({experts})) {
      throw std::invalid_argument(
          "pilot router violates its qualified row-int8 shape/device contract");
    }
  } else {
    require_f32(router.router.dense_f32, {experts, hidden}, device,
                "pilot dense router");
  }
}

at::Tensor pilot_rms_norm(const at::Tensor& input,
                          const at::Tensor& weight) {
  // Keep the operator ordering bit-for-bit aligned with KimiRMSNorm.forward
  // and the authoritative provider MoE implementation. Input is already fp32.
  const at::Tensor variance =
      at::mean(at::pow(input, 2), kLastDimension, true);
  const at::Tensor normalized =
      at::mul(input, at::rsqrt(at::add(variance, kPilotRmsEpsilon)));
  return at::mul(weight, normalized);
}

at::Tensor pilot_router_linear(const at::Tensor& input,
                               const PilotRouterT1& router) {
  if (router.packed_int8_qualified) {
    // This is the same private ATen operator admitted for the authoritative
    // resident spine. There is no silent arithmetic fallback after admission:
    // the optional prediction fails and demand I/O remains authoritative.
    return at::_weight_int8pack_mm(input, router.router.quantized,
                                   router.router.row_scales);
  }
  return at::linear(input, router.router.dense_f32, std::nullopt);
}

struct Candidate {
  std::uint16_t expert = 0;
  float score = 0.0F;
};

}  // namespace

std::size_t pilot_topk_width() noexcept {
  static const std::size_t width = [] {
    const char* value = std::getenv("K3_PILOT_TOPK");
    if (value == nullptr || *value == '\0') {
      return kPilotTopK;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == nullptr || *end != '\0' ||
        parsed < static_cast<long>(kPilotTopK) ||
        parsed > static_cast<long>(kPilotMaxPrefetch)) {
      std::fprintf(stderr,
                   "deltafin: K3_PILOT_TOPK must be an integer in [%zu, %zu]; "
                   "using %zu\n",
                   kPilotTopK, kPilotMaxPrefetch, kPilotTopK);
      return kPilotTopK;
    }
    const auto chosen = static_cast<std::size_t>(parsed);
    if (chosen != kPilotTopK) {
      std::fprintf(stderr, "deltafin: pilot prediction width %zu (default %zu)\n",
                   chosen, kPilotTopK);
    }
    return chosen;
  }();
  return width;
}

PilotRouterT1 clone_compact_pilot_router_t1(
    const MoeSpineT1& authoritative,
    const at::Tensor& post_attention_norm, const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  if (authoritative.layer_index < kFirstMoeLayer ||
      authoritative.layer_index > kLastMoeLayer ||
      authoritative.generation == 0 || authoritative.geometry.hidden == 0 ||
      authoritative.geometry.experts < kPilotTopK ||
      authoritative.geometry.experts >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::uint16_t>::max()) +
              1U ||
      (exact_k3 &&
       (authoritative.geometry.hidden != kK3Hidden ||
        authoritative.geometry.experts != kK3Experts))) {
    throw std::invalid_argument(
        "compact pilot clone received an invalid authoritative geometry");
  }
  const auto hidden =
      static_cast<std::int64_t>(authoritative.geometry.hidden);
  const auto experts =
      static_cast<std::int64_t>(authoritative.geometry.experts);
  const at::Device device = post_attention_norm.device();
  require_f32(post_attention_norm, {hidden}, device,
              "compact pilot post-attention norm");
  require_f32(authoritative.router_correction_bias, {experts}, device,
              "compact pilot correction bias");

  MoeRowInt8Matrix compact;
  if (authoritative.packed_int8_qualified) {
    if (!authoritative.router.quantized.defined() ||
        authoritative.router.quantized.scalar_type() != at::kChar ||
        authoritative.router.quantized.device() != device ||
        !authoritative.router.quantized.is_contiguous() ||
        authoritative.router.quantized.sizes() !=
            at::IntArrayRef({experts, hidden}) ||
        !authoritative.router.row_scales.defined() ||
        authoritative.router.row_scales.scalar_type() != at::kFloat ||
        authoritative.router.row_scales.device() != device ||
        !authoritative.router.row_scales.is_contiguous() ||
        authoritative.router.row_scales.sizes() !=
            at::IntArrayRef({experts})) {
      throw std::invalid_argument(
          "compact pilot clone received an invalid row-int8 router");
    }
    compact.quantized = authoritative.router.quantized.clone().contiguous();
    compact.row_scales =
        authoritative.router.row_scales.clone().contiguous();
  } else {
    at::Tensor dense;
    if (authoritative.router.original_bf16.defined()) {
      if (authoritative.router.dense_f32.defined() ||
          !original_bf16_matrix_matches(
              authoritative.router.original_bf16, device,
              static_cast<std::size_t>(experts),
              static_cast<std::size_t>(hidden))) {
        throw std::invalid_argument(
            "compact pilot clone received an invalid original-BF16 router");
      }
      // This temporary expansion exists only while creating the detached q8
      // scheduling clone. The authoritative router remains original BF16 and
      // every model-output route still uses the direct exact-weight carrier.
      dense = materialize_original_bf16_f32(
          authoritative.router.original_bf16);
    } else {
      require_f32(authoritative.router.dense_f32, {experts, hidden}, device,
                  "compact pilot dense router");
      dense = authoritative.router.dense_f32;
    }
    const at::Tensor peaks = at::amax(
        at::abs(dense), kRouterRowDimension, false);
    const at::Tensor scales = at::where(
        at::gt(peaks, 0.0F), at::div(peaks, 127.0F), at::ones_like(peaks));
    compact.quantized = at::clamp(
                            at::round(at::div(
                                dense, scales.unsqueeze(1))),
                            -127.0F, 127.0F)
                            .to(at::kChar)
                            .contiguous();
    compact.row_scales = scales.contiguous();
  }

  PilotRouterT1 result{
      .layer_index = authoritative.layer_index,
      .generation = authoritative.generation,
      .hidden_size = authoritative.geometry.hidden,
      .expert_count = authoritative.geometry.experts,
      .packed_int8_qualified = true,
      .post_attention_norm = post_attention_norm.clone().contiguous(),
      .router = std::move(compact),
      .correction_bias =
          authoritative.router_correction_bias.clone().contiguous(),
  };
  // Reuse the prediction validator before publication. This checks all four
  // detached tensors without dispatching any router arithmetic.
  validate_router(result, device, exact_k3);
  return result;
}

namespace {

// K3_PILOT_DUMP=<path>: append one JSONL line per SINGLE-ROW pilot
// prediction carrying the pilot's RANKED top-64 candidates —
// {"n":<per-layer ordinal>,"layer":<target>,"ids":[64],"w":[64]}.
// This must live HERE, before the width-capped topk: every downstream
// surface (hint ABI, ExpertPrefetchPlan, the 32-slot arena) is capped at
// kPilotMaxPrefetch and sorts ids ascending, so rank and breadth beyond 32
// exist only in this function. Instrumentation-only: the .to(kCPU) is a
// per-layer MPS drain (the exact sync the mailbox path removed), so a run
// with the dump enabled is never a benchmark. Offline join contract: for a
// given target layer, the n-th dump line pairs with the n-th SINGLE-ROW
// router-trace record of that layer in step order; an aborted pass writes
// a dump line but no trace record and silently shifts the join — dump runs
// must complete cleanly (k3-pilot-containment.py refuses mismatched
// counts). Use an absolute path: this fopen resolves against cwd, not the
// model root.
constexpr std::int64_t kPilotDumpWidth = 64;

std::FILE* pilot_dump_file() noexcept {
  static std::FILE* const file = []() -> std::FILE* {
    const char* path = std::getenv("K3_PILOT_DUMP");
    if (path == nullptr || *path == '\0') {
      return nullptr;
    }
    std::FILE* handle = std::fopen(path, "a");
    if (handle == nullptr) {
      std::fprintf(stderr,
                   "deltafin: K3_PILOT_DUMP cannot open %s; dump disabled\n",
                   path);
    }
    return handle;
  }();
  return file;
}

// noexcept + internal catch-all: the sole production caller is the
// noexcept try_predict_pilot_router_rows, which swallows every exception
// into nullopt — a throwing dump would silently disable pilot prefetch.
void dump_pilot_ranked(const at::Tensor& choice,
                       const std::uint32_t layer_index) noexcept {
  std::FILE* const file = pilot_dump_file();
  if (file == nullptr || choice.size(0) != 1 || layer_index > kLastMoeLayer) {
    return;
  }
  try {
    const std::int64_t width =
        std::min<std::int64_t>(kPilotDumpWidth, choice.size(1));
    const auto ranked =
        at::topk(choice, width, -1, /*largest=*/true, /*sorted=*/true);
    const at::Tensor values = std::get<0>(ranked).to(at::kCPU).contiguous();
    const at::Tensor ids = std::get<1>(ranked).to(at::kCPU).contiguous();
    const float* value_ptr = values.const_data_ptr<float>();
    const std::int64_t* id_ptr = ids.const_data_ptr<std::int64_t>();

    static std::mutex mutex;
    static std::array<std::uint64_t, kLastMoeLayer + 1> ordinals{};
    const std::lock_guard<std::mutex> guard(mutex);
    std::string line;
    line.reserve(static_cast<std::size_t>(width) * 18 + 48);
    char scratch[64];
    std::snprintf(scratch, sizeof scratch, "{\"n\":%llu,\"layer\":%u,\"ids\":[",
                  static_cast<unsigned long long>(ordinals[layer_index]++),
                  layer_index);
    line += scratch;
    for (std::int64_t index = 0; index < width; ++index) {
      std::snprintf(scratch, sizeof scratch, index == 0 ? "%lld" : ",%lld",
                    static_cast<long long>(id_ptr[index]));
      line += scratch;
    }
    line += "],\"w\":[";
    for (std::int64_t index = 0; index < width; ++index) {
      std::snprintf(scratch, sizeof scratch, index == 0 ? "%.5f" : ",%.5f",
                    static_cast<double>(value_ptr[index]));
      line += scratch;
    }
    line += "]}\n";
    if (std::fwrite(line.data(), 1, line.size(), file) == line.size()) {
      std::fflush(file);
    }
  } catch (...) {
  }
}

}  // namespace

PilotPredictionRows predict_pilot_router_rows(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  if (!lookahead_source.defined() ||
      lookahead_source.scalar_type() != at::kFloat ||
      !lookahead_source.is_contiguous() || lookahead_source.dim() != 2 ||
      lookahead_source.size(0) < 1 ||
      lookahead_source.size(0) >
          static_cast<std::int64_t>(kPilotMaxPositions) ||
      lookahead_source.size(1) !=
          static_cast<std::int64_t>(router.hidden_size)) {
    throw std::invalid_argument(
        "pilot prediction requires contiguous fp32 lookahead source [1..64,H]");
  }
  validate_router(router, lookahead_source.device(), exact_k3);

  const auto width = static_cast<std::int64_t>(pilot_topk_width());
  const at::Tensor normalized =
      pilot_rms_norm(lookahead_source, router.post_attention_norm);
  const at::Tensor logits = pilot_router_linear(normalized, router);
  const at::Tensor scores = at::sigmoid(logits);
  const at::Tensor choice = at::add(scores, router.correction_bias);
  dump_pilot_ranked(choice, router.layer_index);
  auto [choice_scores, expert_ids] =
      at::topk(choice, width, -1, true, false);

  if (!expert_ids.defined() || expert_ids.scalar_type() != at::kLong ||
      expert_ids.device() != lookahead_source.device() ||
      !expert_ids.is_contiguous() ||
      expert_ids.sizes() !=
          at::IntArrayRef({lookahead_source.size(0), width}) ||
      !choice_scores.defined() || choice_scores.scalar_type() != at::kFloat ||
      choice_scores.device() != lookahead_source.device() ||
      !choice_scores.is_contiguous() ||
      choice_scores.sizes() !=
          at::IntArrayRef({lookahead_source.size(0), width})) {
    throw std::runtime_error(
        "ATen pilot topk violated its device-resident output contract");
  }

  return PilotPredictionRows{
      router.layer_index,
      router.generation,
      router.expert_count,
      static_cast<std::uint16_t>(lookahead_source.size(0)),
      std::move(expert_ids),
      std::move(choice_scores)};
}

std::optional<PilotPredictionRows> try_predict_pilot_router_rows(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    const bool exact_k3) noexcept {
  try {
    return predict_pilot_router_rows(lookahead_source, router, exact_k3);
  } catch (...) {
    return std::nullopt;
  }
}

PilotPredictionT1 predict_pilot_router_t1(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    const bool exact_k3) {
  if (!lookahead_source.defined() || lookahead_source.dim() != 2 ||
      lookahead_source.size(0) != 1) {
    throw std::invalid_argument(
        "decode pilot prediction requires exactly one source row");
  }
  PilotPredictionRows rows =
      predict_pilot_router_rows(lookahead_source, router, exact_k3);
  return PilotPredictionT1{rows.layer_index, rows.generation,
                           rows.expert_count, std::move(rows.expert_ids),
                           std::move(rows.choice_scores)};
}

std::optional<PilotPredictionT1> try_predict_pilot_router_t1(
    const at::Tensor& lookahead_source, const PilotRouterT1& router,
    const bool exact_k3) noexcept {
  try {
    return predict_pilot_router_t1(lookahead_source, router, exact_k3);
  } catch (...) {
    return std::nullopt;
  }
}

float pilot_prior_bonus() noexcept {
  static const float bonus = [] {
    const char* value = std::getenv("K3_PILOT_PRIOR");
    if (value == nullptr || *value == '\0') {
      return 0.0F;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    // Fail closed to disabled: a malformed or out-of-range bonus must never
    // silently reshape the read schedule.
    if (end == value || *end != '\0' || !std::isfinite(parsed) ||
        parsed < 0.0 || parsed > 10.0) {
      std::fprintf(stderr,
                   "deltafin: K3_PILOT_PRIOR must be a finite bonus in "
                   "[0,10]; prior disabled\n");
      return 0.0F;
    }
    if (parsed > 0.0) {
      std::fprintf(stderr, "deltafin: pilot previous-token prior bonus %.3f\n",
                   parsed);
    }
    return static_cast<float>(parsed);
  }();
  return bonus;
}

namespace {

struct PilotPriorSlot {
  std::array<std::uint16_t, kPilotTopK> experts{};
  std::uint8_t count = 0;
};

std::mutex& pilot_prior_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::array<PilotPriorSlot, kPilotPriorMaxLayers>& pilot_prior_slots() {
  static std::array<PilotPriorSlot, kPilotPriorMaxLayers> slots{};
  return slots;
}

}  // namespace

void pilot_prior_record_route(const std::uint32_t layer,
                              const std::uint16_t* const experts,
                              const std::size_t count) noexcept {
  // Decode only: a prefill pass routes many positions at once and its edge
  // list is not "the previous token's routing" for any single position.
  if (pilot_prior_bonus() <= 0.0F || experts == nullptr ||
      count != kPilotTopK || layer >= kPilotPriorMaxLayers) {
    return;
  }
  const std::lock_guard<std::mutex> guard(pilot_prior_mutex());
  PilotPriorSlot& slot = pilot_prior_slots()[layer];
  std::copy_n(experts, kPilotTopK, slot.experts.begin());
  slot.count = static_cast<std::uint8_t>(kPilotTopK);
}

namespace {

// Copy out under the lock so scoring never holds it.
PilotPriorSlot pilot_prior_snapshot(const std::uint32_t layer) noexcept {
  if (layer >= kPilotPriorMaxLayers) {
    return PilotPriorSlot{};
  }
  const std::lock_guard<std::mutex> guard(pilot_prior_mutex());
  return pilot_prior_slots()[layer];
}

}  // namespace

CanonicalPilotPrefetchT1 canonicalize_pilot_prefetch_t1(
    const std::span<const std::int64_t> expert_ids,
    const std::span<const float> choice_scores,
    const std::uint32_t expert_count, const std::size_t cap) {
  return canonicalize_pilot_prefetch_rows(
      expert_ids, choice_scores, 1, expert_count,
      std::min(cap, pilot_topk_width()));
}

CanonicalPilotPrefetchT1 canonicalize_pilot_prefetch_rows(
    const std::span<const std::int64_t> expert_ids,
    const std::span<const float> choice_scores,
    const std::size_t position_count, const std::uint32_t expert_count,
    const std::size_t cap, const std::uint32_t target_layer) {
  // Per-row candidate width is carried by the span shape: exactly the
  // pilot's topk width, which K3_PILOT_TOPK may widen up to the prefetch
  // bound. Rows stay rectangular; the union cap below is unchanged.
  const std::size_t row_width =
      position_count == 0 ? 0 : expert_ids.size() / position_count;
  if (position_count == 0 || position_count > kPilotMaxPositions ||
      expert_count < kPilotTopK ||
      expert_count >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::uint16_t>::max()) +
              1U ||
      row_width < kPilotTopK || row_width > kPilotMaxPrefetch ||
      expert_ids.size() != position_count * row_width ||
      (!choice_scores.empty() &&
       choice_scores.size() != expert_ids.size())) {
    throw std::invalid_argument(
        "pilot materialization violates its rows/top-k/expert-count contract");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(std::min<std::size_t>(
      expert_count, position_count * row_width));
  for (std::size_t index = 0; index < expert_ids.size(); ++index) {
    const std::int64_t expert = expert_ids[index];
    const float score = choice_scores.empty() ? 0.0F : choice_scores[index];
    if (expert < 0 ||
        expert >= static_cast<std::int64_t>(expert_count) ||
        !std::isfinite(score)) {
      throw std::invalid_argument(
          "pilot candidate contains an invalid expert ID or choice score");
    }
    const std::size_t row_start = (index / row_width) * row_width;
    for (std::size_t row_index = row_start; row_index < index; ++row_index) {
      if (expert_ids[row_index] == expert) {
        throw std::invalid_argument(
            "one pilot candidate row repeats an expert");
      }
    }
    const auto existing = std::find_if(
        candidates.begin(), candidates.end(), [expert](const Candidate& item) {
          return item.expert == static_cast<std::uint16_t>(expert);
        });
    if (existing == candidates.end()) {
      candidates.push_back(
          Candidate{static_cast<std::uint16_t>(expert), score});
    } else if (!choice_scores.empty() && score > existing->score) {
      existing->score = score;
    }
  }

  // K3_PILOT_PRIOR: bias the ranking toward experts this layer routed to on
  // the PREVIOUS token before the cap is applied. This changes WHICH
  // candidates survive the cap, never HOW MANY — the read count is `bounded`
  // either way, which is what separates this from the K3_PILOT_TOPK breadth
  // experiment that measured -8.9%/-16.5% by issuing extra reads.
  const float prior_bonus = pilot_prior_bonus();
  if (prior_bonus > 0.0F && target_layer != kPilotPriorNoLayer &&
      !choice_scores.empty()) {
    const PilotPriorSlot prior = pilot_prior_snapshot(target_layer);
    if (prior.count == kPilotTopK) {
      for (Candidate& candidate : candidates) {
        if (std::find(prior.experts.begin(), prior.experts.end(),
                      candidate.expert) != prior.experts.end()) {
          candidate.score += prior_bonus;
        }
      }
    }
  }

  const std::size_t bounded =
      std::min({cap, kPilotMaxPrefetch, candidates.size()});
  if (choice_scores.empty() && candidates.size() > bounded) {
    throw std::invalid_argument(
        "capped pilot union requires materialized choice scores");
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& left, const Candidate& right) {
              if (left.score != right.score) {
                return left.score > right.score;
              }
              return left.expert < right.expert;
            });
  std::sort(candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(bounded),
            [](const Candidate& left, const Candidate& right) {
              return left.expert < right.expert;
            });

  CanonicalPilotPrefetchT1 result;
  result.count = bounded;
  result.candidate_count = candidates.size();
  for (std::size_t index = 0; index < bounded; ++index) {
    result.expert_ids[index] = candidates[index].expert;
  }
  return result;
}

}  // namespace deltafin::provider_internal
