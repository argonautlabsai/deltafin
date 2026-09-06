#include "provider_target_sequence.h"
#include "provider_prep_timer.h"

#include "provider_cuda_moe.h"
#include "provider_kda_batch.h"

#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
#include "provider_route_mailbox.h"
#endif
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
#include "provider_loop.h"
#endif

#include <ATen/ops/argmax.h>
#include <ATen/ops/topk.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kExactHidden = 7168;
constexpr std::int64_t kSyntheticHidden = 32;

// K3_DSPARK_DEBUG=1: diagnostic-only dump of the full target model's own
// top-5 ids+logits per verified row, to stderr. Off by default; checked once
// and cached (env is immutable for the process lifetime). Mirrors the draft-
// side helper in provider_dspark_model.cpp. See DIRECTIVE STEP 2 (DSpark
// day) in K3-ENGINE-STATUS.md.
bool dspark_debug_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_DSPARK_DEBUG");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// K3_STEP_LOG=<path>: append one line per decided position with the top-2
// token ids and logits and their margin. Off unless set; used by the
// token-equivalence gate (k3-equiv-gate.py) for changes that alter float
// summation order, where md5 identity of the text is no longer the right test.
void step_log_topk(const std::int64_t position, const at::Tensor& logits_row) {
  static std::FILE* file = [] {
    const char* path = std::getenv("K3_STEP_LOG");
    return (path != nullptr && *path != '\0') ? std::fopen(path, "a") : nullptr;
  }();
  if (file == nullptr) {
    return;
  }
  const auto topk = at::topk(logits_row.to(at::kFloat), 2, -1, true, true);
  const at::Tensor values = std::get<0>(topk).to(at::kCPU).contiguous();
  const at::Tensor indices = std::get<1>(topk).to(at::kCPU).contiguous();
  const float* v = values.const_data_ptr<float>();
  const std::int64_t* i = indices.const_data_ptr<std::int64_t>();
  std::fprintf(file, "STEP pos=%lld top1=%lld:%.6f top2=%lld:%.6f margin=%.6f\n",
               static_cast<long long>(position), static_cast<long long>(i[0]), v[0],
               static_cast<long long>(i[1]), v[1], v[0] - v[1]);
  std::fflush(file);
}

void debug_print_topk(const char* tag, const std::int64_t row,
                      const std::int64_t position, const at::Tensor& logits_row) {
  const std::int64_t k = std::min<std::int64_t>(5, logits_row.size(0));
  const auto topk = at::topk(logits_row.to(at::kFloat), k, -1, true, true);
  const at::Tensor values = std::get<0>(topk).to(at::kCPU).contiguous();
  const at::Tensor indices = std::get<1>(topk).to(at::kCPU).contiguous();
  const float* value_ptr = values.const_data_ptr<float>();
  const std::int64_t* index_ptr = indices.const_data_ptr<std::int64_t>();
  std::cerr << "K3_DSPARK_DEBUG " << tag << " row=" << row << " pos=" << position
            << " top5=[";
  for (std::int64_t i = 0; i < k; ++i) {
    if (i != 0) {
      std::cerr << ",";
    }
    std::cerr << index_ptr[i] << ":" << value_ptr[i];
  }
  std::cerr << "]\n";
}

static_assert(std::is_nothrow_move_assignable_v<KdaState>);
static_assert(std::is_nothrow_move_assignable_v<at::Tensor>);

bool same_geometry(const MoeGeometry& left, const MoeGeometry& right) {
  return left.hidden == right.hidden &&
         left.routed_hidden == right.routed_hidden &&
         left.intermediate == right.intermediate &&
         left.experts == right.experts &&
         left.shared_intermediate == right.shared_intermediate;
}

std::uint64_t checked_add(const std::uint64_t left,
                          const std::uint64_t right, const char* name) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(name) + " byte count overflowed");
  }
  return left + right;
}

std::uint64_t tensor_bytes(const at::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() < 0) {
    throw std::logic_error("target sequence staged an invalid KDA tensor");
  }
  const auto elements = static_cast<std::uint64_t>(tensor.numel());
  const auto element_size = static_cast<std::uint64_t>(tensor.element_size());
  if (element_size != 0 &&
      elements > std::numeric_limits<std::uint64_t>::max() / element_size) {
    throw std::overflow_error("target sequence KDA snapshot size overflowed");
  }
  return elements * element_size;
}

std::uint64_t kda_state_bytes(const KdaState& state) {
  std::uint64_t bytes = 0;
  for (const at::Tensor* tensor :
       {&state.query_convolution, &state.key_convolution,
        &state.value_convolution, &state.recurrent}) {
    bytes = checked_add(bytes, tensor_bytes(*tensor), "KDA snapshot");
  }
  return bytes;
}

struct SequenceRow {
  at::Tensor hidden;
  TargetBlockResidual residual;
};

struct SequenceCacheStage {
  enum class Kind { Kda, Mla };

  Kind kind = Kind::Kda;
  std::uint32_t layer_index = 0;
  TargetKdaCache* kda_cache = nullptr;
  std::uint64_t expected_kda_version = 0;
  KdaState final_kda_state;
  std::vector<KdaState> kda_boundaries;
  std::unique_ptr<MlaCacheTransaction> mla;
};

struct PendingExpertRow {
  TargetMlpInput mlp_input;
  PreparedMoeT1 moe;
};

struct PendingSequenceLayer {
  std::uint32_t layer_index = 0;
  MoeSpineT1 spine;
  std::unique_ptr<SequenceCacheStage> cache_stage;
  std::array<std::optional<PendingExpertRow>,
             kTargetSequenceMaxPositions> rows{};
  TargetMlpRowsInput mlp_rows;
  /* T>1 preparation owns one contiguous routed-input device matrix. Metal
   * materializes it to CPU once on the first expert tile, and all row views
   * borrow that stable layer-owned carrier until the transaction completes. */
  at::Tensor routed_inputs_device;
  at::Tensor metal_routed_inputs_cpu;
  std::array<at::Tensor, kTargetSequenceMaxPositions> routed_output_rows{};
  std::size_t next_expert_row = 0;
  bool expert_backend_decided = false;
  bool whole_layer_metal_staging = false;
  /* Arrival-driven compute: routed outputs accumulated over the partial
   * groups of the tile in flight (rows x routed_hidden, fp32). */
  at::Tensor partial_routed_outputs;
  std::uint32_t partial_groups = 0;
};

std::optional<at::Tensor> recover_contiguous_row_carrier(
    const std::array<at::Tensor, kTargetSequenceMaxPositions>& rows,
    const std::size_t row_count, const std::int64_t width) {
  if (row_count == 0 || width <= 0 || !rows.front().defined()) {
    return std::nullopt;
  }
  const at::Tensor& first = rows.front();
  if (first.dim() != 2 || first.sizes() != at::IntArrayRef({1, width}) ||
      !first.is_contiguous()) {
    return std::nullopt;
  }
  for (std::size_t row = 1; row < row_count; ++row) {
    const at::Tensor& candidate = rows[row];
    if (!candidate.defined() || candidate.scalar_type() != first.scalar_type() ||
        candidate.device() != first.device() || !candidate.is_contiguous() ||
        candidate.sizes() != at::IntArrayRef({1, width}) ||
        !candidate.is_alias_of(first) ||
        candidate.storage_offset() !=
            first.storage_offset() + static_cast<std::int64_t>(row) * width) {
      return std::nullopt;
    }
  }
  at::Tensor carrier = first.as_strided(
      {static_cast<std::int64_t>(row_count), width}, {width, 1},
      first.storage_offset());
  if (!carrier.is_contiguous()) {
    throw std::logic_error(
        "target sequence recovered a noncontiguous routed-output carrier");
  }
  return carrier;
}

}  // namespace

class TargetSequenceTape::Impl {
 public:
  Impl(const TargetPositionBindings& bindings, at::Tensor input_hidden_rows,
       const TargetSequenceMode mode, const bool capture_dspark_rows,
       const bool full_commit_only)
      : mode_(mode),
        exact_k3_(bindings.contract == TargetTapeContract::ExactK3),
        capture_dspark_rows_(capture_dspark_rows),
        full_commit_only_(full_commit_only),
        tail_(bindings.tail),
        pilot_routers_(bindings.pilot_routers) {
    validate_and_copy_bindings(bindings);
    if (mode_ != TargetSequenceMode::Prefill &&
        mode_ != TargetSequenceMode::Verify) {
      throw std::invalid_argument("target sequence mode is unknown");
    }
    if (full_commit_only_ && mode_ != TargetSequenceMode::Verify) {
      throw std::invalid_argument(
          "target sequence full-commit-only requires verify mode");
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!input_hidden_rows.defined() ||
        input_hidden_rows.scalar_type() != at::kFloat ||
        !input_hidden_rows.is_contiguous() || input_hidden_rows.dim() != 2 ||
        input_hidden_rows.size(0) < 1 ||
        input_hidden_rows.size(0) >
            static_cast<std::int64_t>(kTargetSequenceMaxPositions) ||
        input_hidden_rows.size(1) != hidden ||
        input_hidden_rows.device().is_meta()) {
      throw std::invalid_argument(
          "target sequence input must be contiguous fp32 [1..64,hidden]");
    }

    const c10::InferenceMode inference_guard;
    position_count_ = static_cast<std::size_t>(input_hidden_rows.size(0));
    if (position_count_ == 1) {
      rows_.push_back(SequenceRow{
          input_hidden_rows.narrow(0, 0, 1),
          empty_target_block_residual(input_hidden_rows.device(), hidden),
      });
    } else {
      rows_.resize(position_count_);
      at::Tensor anchor_rows = at::empty(
          {static_cast<std::int64_t>(position_count_), 0, hidden},
          at::TensorOptions().dtype(at::kFloat).device(
              input_hidden_rows.device()));
      install_wide_carriers(std::move(input_hidden_rows),
                            std::move(anchor_rows));
    }
    stages_.reserve(kTargetLayerCount);
    decisions_.reserve(position_count_);
    stats_.positions = position_count_;
  }

  ~Impl() {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    if (kda_precommit_handle_ != nullptr) {
      loop_kda_layer_abandon(kda_precommit_handle_);
      kda_precommit_handle_ = nullptr;
    }
    if (mla_precommit_handle_ != nullptr) {
      loop_mla_layer_abandon(mla_precommit_handle_);
      mla_precommit_handle_ = nullptr;
    }
    if (mla_precommit_shell_ != nullptr && mla_precommit_txn_ != nullptr) {
      try {
        cancel_mla_decode(mla_precommit_txn_->working_cache(),
                          mla_precommit_shell_->prepared);
      } catch (...) {
      }
    }
    mla_precommit_shell_.reset();
    mla_precommit_txn_.reset();
#endif
    abort_unlocked();
  }

  /* Step 6: successor KDA layer eligible for pre-commit, else UINT32_MAX.
   * No mode gate: T=1 decode chunks ARE Prefill-mode tapes (the only other
   * mode is DSpark Verify); position_count_ == 1 is the real single-row
   * condition. The runtime revalidates the spine binding itself. */
  std::uint32_t precommit_wanted() const noexcept {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    std::lock_guard<std::mutex> lock(mutex_);
    static const bool gate_probe = [] {
      const char* value = std::getenv("K3_ROUTE_SYNC_PROBE");
      return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    static std::atomic<int> gate_dumps{0};
    if (gate_probe && kda_precommit_enabled() && position_count_ == 1 &&
        gate_dumps.fetch_add(1) < 5) {
      std::fprintf(
          stderr,
          "[precommit-wanted] tail=%d mode=%d exact=%d pc=%zu state=%u "
          "layer=%u mla=%d handle=%d\n",
          last_tile_tail_on_loop_ ? 1 : 0, kda_loop_mode(),
          exact_k3_ ? 1 : 0, position_count_,
          static_cast<std::uint32_t>(state_), next_layer_,
          next_layer_ < kTargetLayerCount &&
                  target_layer_uses_mla(next_layer_)
              ? 1
              : 0,
          kda_precommit_handle_ != nullptr ? 1 : 0);
    }
    if (!last_tile_tail_on_loop_ || !kda_precommit_enabled() ||
        kda_loop_mode() != 2 || !exact_k3_ || position_count_ != 1 ||
        state_ != TargetSequenceState::Active ||
        next_layer_ >= kTargetLayerCount ||
        kda_precommit_handle_ != nullptr) {
      return std::numeric_limits<std::uint32_t>::max();
    }
    if (target_layer_uses_mla(next_layer_)) {
      // Step 4 chain-lite: MLA successors precommit only in chain mode,
      // with the MLA mirror of the KDA staleness checks below.
      if (mla_loop_mode() != 3 || mla_precommit_handle_ != nullptr) {
        return std::numeric_limits<std::uint32_t>::max();
      }
      const TargetLayerCacheBinding& mla_cache = caches_[next_layer_];
      if (mla_cache.mla_cache == nullptr ||
          mla_cache.mla_cache->has_pending_prepare() ||
          mla_cache.mla_cache->version() != initial_versions_[next_layer_] ||
          !mla_cache.mla_cache->can_append(1)) {
        return std::numeric_limits<std::uint32_t>::max();
      }
      if (rows_.empty() || !rows_.front().hidden.defined() ||
          !rows_.front().hidden.device().is_mps()) {
        return std::numeric_limits<std::uint32_t>::max();
      }
      return next_layer_;
    }
    const TargetLayerCacheBinding& cache = caches_[next_layer_];
    if (cache.kda_cache == nullptr ||
        cache.kda_cache->version != initial_versions_[next_layer_]) {
      return std::numeric_limits<std::uint32_t>::max();
    }
    if (rows_.empty() || !rows_.front().hidden.defined() ||
        !rows_.front().hidden.device().is_mps()) {
      return std::numeric_limits<std::uint32_t>::max();
    }
    return next_layer_;
#else
    return std::numeric_limits<std::uint32_t>::max();
#endif
  }

  /* Step 6: encode + commit next_layer_'s KDA loop CB using the runtime's
   * freshly built binding — the exact per-owner fp32 views the later
   * prepare_layer will resolve again (cache hit), so the pre-computed math
   * is bit-equal to the synchronous path. Fail-soft. */
  void try_precommit_next(const TargetLayerBinding& binding) noexcept {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (binding.layer_index == next_layer_ &&
          binding.attention_kind == TargetAttentionKind::Mla) {
        try_precommit_next_mla_locked(binding);
        return;
      }
      if (kda_precommit_handle_ != nullptr ||
          binding.layer_index != next_layer_ ||
          binding.kda_weights == nullptr || binding.residual == nullptr ||
          !binding.residual->self_attention_score_weight.defined() ||
          !binding.residual->mlp_score_weight.defined()) {
        return;
      }
      SequenceRow& row = rows_.front();
      constexpr std::uint32_t kLoopResidualBlock = 12;
      at::Tensor prefix_sum = row.hidden;
      at::Tensor next_anchors = row.residual.anchors;
      const bool boundary_layer = next_layer_ % kLoopResidualBlock == 0;
      const bool cat_in_cb = boundary_layer && loop_cat_enabled();
      if (boundary_layer && !cat_in_cb) {
        // K3_CB_FUSION=1: the at::cat below READS this row's tail output on
        // the ATen stream, and PyTorch may auto-commit that stream at any
        // encode — the open fused CB producing row.hidden must execute
        // first. The drain flushes it (cheap no-op with the flag off), and
        // the successor then takes the ordinary own-CB private-fence path.
        if (loop_cb_fusion_enabled()) {
          loop_moe_tail_drain();
        }
        next_anchors =
            at::cat({row.residual.anchors, row.hidden.unsqueeze(1)}, 1)
                .contiguous();
        loop_note_fresh_anchor_cat();
      }
      if (boundary_layer) {
        // Boundary semantics regardless of who performs the cat: the
        // prefix aliases the attention output.
        prefix_sum = at::Tensor();
      }
      // Milestone (iii): hand the router through so the route CB chains
      // directly behind the KDA CB (fail-soft when the binding lacks it).
      const LoopMoeMatrixView router_view =
          binding.moe != nullptr
              ? LoopMoeMatrixView{binding.moe->router.quantized,
                                  binding.moe->router.row_scales,
                                  binding.moe->router.dense_f32,
                                  &binding.moe->router.original_bf16}
              : LoopMoeMatrixView{};
      const at::Tensor router_bias = binding.moe != nullptr
          ? binding.moe->router_correction_bias
          : at::Tensor();
      LoopKdaPendingHandle* handle = loop_kda_layer_precommit(
          next_layer_, row.hidden, row.residual.anchors,
          binding.residual->self_attention_score_weight,
          binding.residual->input_norm, prefix_sum, next_anchors,
          binding.residual->mlp_score_weight,
          binding.residual->post_attention_norm, *binding.kda_weights,
          caches_[next_layer_].kda_cache->state, router_view, router_bias,
          cat_in_cb);
      if (handle != nullptr) {
        kda_precommit_handle_ = handle;
        kda_precommit_layer_ = next_layer_;
        kda_precommit_next_anchors_ = std::move(next_anchors);
      }
    } catch (...) {
    }
#else
    static_cast<void>(binding);
#endif
  }

  /* Step 4 chain-lite: encode + commit next_layer_'s MLA loop CB at the
   * previous layer's finish. Transaction + decode shell are created HERE
   * and parked with the handle; the CB packs its K/V row into the
   * shell's slab views and its output feeds the collect at the layer's
   * turn. Fail-soft everywhere; called under mutex_. */
  void try_precommit_next_mla_locked(const TargetLayerBinding& binding) {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    if (mla_precommit_handle_ != nullptr ||
        binding.mla_parity_weights == nullptr ||
        binding.mla_parity_bundle == nullptr ||
        binding.residual == nullptr ||
        !binding.residual->self_attention_score_weight.defined() ||
        !binding.residual->input_norm.defined() || rows_.empty()) {
      return;
    }
    SequenceRow& row = rows_.front();
    if (!row.hidden.defined() || !row.hidden.device().is_mps()) {
      return;
    }
    // K3_CB_FUSION: the open fused CB carries the JUST-FINISHED layer's
    // tail — the write that produces row.hidden. Our CB must land BEHIND
    // it on the serial queue or it reads the previous token's values
    // (measured: text degrades after ~20 tokens, ngram-verify storms).
    // FLUSH (commit), not drain — GPU ordering without a host wait.
    loop_fused_flush();
    std::unique_ptr<MlaCacheTransaction> txn;
    std::unique_ptr<MlaDecodeShell> shell;
    try {
      txn = std::make_unique<MlaCacheTransaction>(
          *caches_[next_layer_].mla_cache, 1);
      shell = std::make_unique<MlaDecodeShell>(prepare_k3_mla_decode_shell(
          row.hidden.view({1, 1, row.hidden.size(1)}),
          txn->working_cache()));
    } catch (...) {
      return;  // shell's own catch released its nonce; txn dtor discards
    }
    LoopMlaPendingHandle* handle = loop_mla_layer_precommit(
        next_layer_, row.hidden, row.residual.anchors,
        binding.residual->self_attention_score_weight,
        binding.residual->input_norm, *binding.mla_parity_weights,
        binding.mla_parity_bundle, shell->key_states, shell->value_states);
    if (handle == nullptr) {
      try {
        cancel_mla_decode(txn->working_cache(), shell->prepared);
      } catch (...) {
      }
      return;
    }
    mla_precommit_handle_ = handle;
    mla_precommit_layer_ = next_layer_;
    mla_precommit_txn_ = std::move(txn);
    mla_precommit_shell_ = std::move(shell);
#else
    static_cast<void>(binding);
#endif
  }

  /* Arena barrier (Step 6): the fp32 execution arena recycles per-layer
   * storage, so the runtime must not rewrite it while a pre-committed CB
   * is still reading its weight views. Waits without consuming; collect
   * still publishes. No-op without a pending handle or off-Apple. */
  void precommit_arena_barrier() noexcept {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    if (kda_precommit_handle_ != nullptr) {
      loop_kda_layer_wait(kda_precommit_handle_);
    }
#endif
  }

  TargetSequenceLayerPrepareKind
  prepare_layer(const TargetLayerBinding& binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::Active,
                  "prepare the next target sequence layer");
    try {
      validate_current_layer(binding);
      pending_pilot_.reset();
      auto stage = std::make_unique<SequenceCacheStage>();
      stage->layer_index = next_layer_;
      const TargetLayerCacheBinding& cache = caches_[next_layer_];
      std::unique_ptr<PendingSequenceLayer> routed;
      if (next_layer_ != 0) {
        routed = std::make_unique<PendingSequenceLayer>();
        routed->layer_index = next_layer_;
        routed->spine = *binding.moe;
      }

#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
      if (position_count_ == 1 && moe_route_async_enabled() &&
          !rows_.empty() && rows_.front().hidden.device().is_mps()) {
        // K3_ROUTE_ASYNC: commit the previous layer's still-encoded tail
        // (expert kernel, merge) so the GPU executes it under this layer's
        // attention encode instead of inside the route boundary's drain.
        static_cast<void>(try_commit_mps_stream_for_route());
      }
#endif

      if (binding.attention_kind == TargetAttentionKind::Kda) {
        prepare_kda_rows(binding, cache, *stage, routed.get());
      } else {
        prepare_mla_rows(binding, cache, *stage, routed.get());
      }

      ++stats_.streamed_layer_passes;
      stats_.attention_rows += position_count_;
      stats_.maximum_live_streamed_layers = 1;
      if (next_layer_ == 0) {
        capture_completed_layer(next_layer_);
        stages_.push_back(std::move(stage));
        ++next_layer_;
        state_ = next_layer_ == kTargetLayerCount
                     ? TargetSequenceState::ReadyForTail
                     : TargetSequenceState::Active;
        return TargetSequenceLayerPrepareKind::DenseCompleted;
      }

      routed->cache_stage = std::move(stage);
      pending_ = std::move(routed);
      state_ = TargetSequenceState::WaitingForExperts;
      stats_.expert_row_requests += position_count_;
      return TargetSequenceLayerPrepareKind::ExpertRowsRequired;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  TargetSequenceExpertMailbox expert_mailbox() const {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::WaitingForExperts,
                  "read the target sequence expert mailbox");
    return mailbox_;
  }

  TargetSequencePrefetchHint take_prefetch_hint() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    TargetSequencePrefetchHint hint;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    // K3_TRUE_ROUTE_HINT=1 (read-war experiment): when the caller defers
    // the hint until after finish_expert_tile, the pre-commit orchestration
    // has already CHAINED the successor layer's route CB — its mailbox
    // holds the TRUE top-16 for next_layer_. Prefetching the truth instead
    // of the pilot's one-further prediction turns every demand miss into a
    // prefetch hit and drops speculative read amplification to ~1.0.
    // Fail-soft to the pilot prediction below on any miss.
    static const bool true_route = [] {
      const char* value = std::getenv("K3_TRUE_ROUTE_HINT");
      return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (true_route && position_count_ == 1 &&
        state_ == TargetSequenceState::Active && next_layer_ >= 1 &&
        next_layer_ < kTargetLayerCount) {
      std::array<std::uint16_t, kMoeRouteTopK> ids{};
      if (loop_router_peek_chained(next_layer_, ids.data())) {
        std::sort(ids.begin(), ids.end());
        bool strictly_ascending = true;
        for (std::size_t index = 1; index < ids.size(); ++index) {
          if (ids[index - 1] >= ids[index]) {
            strictly_ascending = false;
            break;
          }
        }
        if (strictly_ascending) {
          hint.source_layer = next_layer_ - 1;
          hint.target_layer = next_layer_;
          hint.expert_count = static_cast<std::uint16_t>(kMoeRouteTopK);
          std::copy_n(ids.begin(), kMoeRouteTopK, hint.expert_ids.begin());
          ++stats_.pilot_hint_issues;
          stats_.pilot_hint_experts += kMoeRouteTopK;
          return hint;
        }
      }
    }
#endif
    if (state_ != TargetSequenceState::WaitingForExperts ||
        !pending_pilot_.has_value()) {
      return hint;
    }
    try {
      PilotPredictionRows prediction = std::move(*pending_pilot_);
      pending_pilot_.reset();
      [[maybe_unused]] const std::uint64_t mailbox_generation =
          pilot_mailbox_generation_;
      pilot_mailbox_generation_ = 0;
      if (pilot_routers_ == nullptr || next_layer_ + 1 >= kTargetLayerCount ||
          !(*pilot_routers_)[next_layer_ + 1].has_value()) {
        return hint;
      }
      const std::uint32_t expected_experts =
          (*pilot_routers_)[next_layer_ + 1]->expert_count;
      const auto pilot_width =
          static_cast<std::int64_t>(pilot_topk_width());
      if (prediction.layer_index != next_layer_ + 1 ||
          prediction.expert_count != expected_experts ||
          prediction.expert_count < kPilotTopK ||
          prediction.position_count != position_count_ ||
          prediction.expert_ids.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(position_count_), pilot_width}) ||
          prediction.choice_scores.sizes() !=
              at::IntArrayRef(
                  {static_cast<std::int64_t>(position_count_), pilot_width})) {
        return hint;
      }
      const std::size_t candidate_slots =
          position_count_ * static_cast<std::size_t>(pilot_width);
      // Shared tail for both hint sources (device-drain ids or the pilot
      // mailbox): bounds/unique accounting, capped score ranking, canonical
      // read-set, hint fill. `score_source` is called only when the unique
      // count exceeds the cap — the stock source pays its .to(kCPU) there,
      // the mailbox source is free either way.
      const auto finish_from_spans = [&](const std::span<const std::int64_t>
                                             id_span,
                                         auto&& score_source) -> void {
        std::vector<bool> unique_seen(prediction.expert_count, false);
        std::size_t unique_count = 0;
        for (const std::int64_t expert : id_span) {
          if (expert < 0 ||
              expert >= static_cast<std::int64_t>(prediction.expert_count)) {
            return;
          }
          auto seen = unique_seen[static_cast<std::size_t>(expert)];
          if (!seen) {
            unique_seen[static_cast<std::size_t>(expert)] = true;
            ++unique_count;
          }
        }
        // K3_PILOT_CAP (read-war experiment): bound the pilot's speculative
        // reads to its most-confident CAP candidates instead of the full
        // prediction width. With the true-route top-up guaranteeing the rest
        // of the authoritative set is still prefetched, a narrow pilot trades
        // a few milliseconds of lead time on its least-confident picks for a
        // direct cut in wasted speculative bytes. Invalid or out-of-range
        // values fail closed to the historical kPilotMaxPrefetch bound.
        static const std::size_t pilot_cap = [] {
          const char* value = std::getenv("K3_PILOT_CAP");
          if (value == nullptr) {
            return kPilotMaxPrefetch;
          }
          char* end = nullptr;
          const long parsed = std::strtol(value, &end, 10);
          if (end == value || *end != '\0' || parsed < 1 ||
              parsed > static_cast<long>(kPilotMaxPrefetch)) {
            return kPilotMaxPrefetch;
          }
          return static_cast<std::size_t>(parsed);
        }();
        std::span<const float> score_span;
        if (unique_count > pilot_cap) {
          score_span = score_source();
        } else {
          ++stats_.pilot_score_elisions;
        }
        const auto canonical = canonicalize_pilot_prefetch_rows(
            id_span, score_span, position_count_, prediction.expert_count,
            pilot_cap, prediction.layer_index);
        hint.source_layer = next_layer_;
        hint.target_layer = prediction.layer_index;
        hint.expert_count = static_cast<std::uint16_t>(canonical.count);
        std::copy_n(canonical.expert_ids.begin(), canonical.count,
                    hint.expert_ids.begin());
        ++stats_.pilot_hint_issues;
        stats_.pilot_hint_experts += canonical.count;
        stats_.pilot_max_union_candidates = std::max<std::uint64_t>(
            stats_.pilot_max_union_candidates, canonical.candidate_count);
      };
#if defined(__APPLE__)
      // K3_PILOT_MAILBOX=1 (sync-E removal): consume the pilot prediction
      // from the shared mailbox published at prepare time instead of a
      // .to(kCPU) stream drain. A miss (event not yet signaled, mismatched
      // metadata, or a failed publish) returns an EMPTY hint — never the
      // drain, which would reintroduce exactly the stall this removes; the
      // true-route topup absorbs the lost lead.
      static const bool pilot_mailbox_on = [] {
        const char* value = std::getenv("K3_PILOT_MAILBOX");
        return value != nullptr && std::strcmp(value, "1") == 0;
      }();
      if (pilot_mailbox_on) {
        static std::uint64_t mailbox_hits = 0;
        static std::uint64_t mailbox_misses = 0;
        PilotMailboxRows mailbox_rows;
        const bool hit =
            mailbox_generation != 0 &&
            try_poll_mps_pilot_rows(mailbox_generation, mailbox_rows) &&
            mailbox_rows.layer_index == prediction.layer_index &&
            mailbox_rows.expert_count == prediction.expert_count &&
            mailbox_rows.position_count == position_count_ &&
            mailbox_rows.width == static_cast<std::uint32_t>(pilot_width);
        if (hit) {
          ++mailbox_hits;
          const auto id_span = std::span<const std::int64_t>(
              mailbox_rows.expert_ids.data(), candidate_slots);
          finish_from_spans(id_span, [&mailbox_rows, candidate_slots] {
            return std::span<const float>(mailbox_rows.choice_scores.data(),
                                          candidate_slots);
          });
        } else {
          ++mailbox_misses;
        }
        if ((mailbox_hits + mailbox_misses) % 1840 == 0) {
          std::fprintf(stderr, "[pilot-mailbox] hits=%llu misses=%llu\n",
                       static_cast<unsigned long long>(mailbox_hits),
                       static_cast<unsigned long long>(mailbox_misses));
        }
        return hint;
      }
#endif
      // Attention-timer split: the pilot readback is a host sync on the
      // MPS stream — the suspected bulk of Rust's expert_plan bucket.
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      loop_gpu_timeline_note_aten("aten-hint");
#endif
      const std::uint64_t pilot_sync_started = prep_steady_ns();
      const at::Tensor ids =
          prediction.expert_ids.to(at::kCPU).contiguous();
      prep_note_wait(prep_steady_ns() - pilot_sync_started, 0, false);
      const auto id_span = std::span<const std::int64_t>(
          ids.const_data_ptr<std::int64_t>(), candidate_slots);
      at::Tensor stock_scores;
      finish_from_spans(id_span, [&]() {
        const std::uint64_t score_sync_started = prep_steady_ns();
        stock_scores =
            prediction.choice_scores.to(at::kCPU, at::kFloat).contiguous();
        prep_note_wait(prep_steady_ns() - score_sync_started, 0, false);
        ++stats_.pilot_score_materializations;
        return std::span<const float>(stock_scores.const_data_ptr<float>(),
                                      candidate_slots);
      });
      return hint;
    } catch (...) {
      pending_pilot_.reset();
      return TargetSequencePrefetchHint{};
    }
  }

  void finish_expert_row(const std::uint16_t row_index,
                         const std::uint64_t spine_generation,
                         const CanonicalExpertBatchT1& experts,
                         const MoeRunOptions& options) {
    finish_expert_tile(
        row_index, 1, spine_generation,
        CanonicalExpertPositionTileT1{experts.expert_ids,
                                      experts.expert_major_bytes,
                                      experts.layout,
                                      experts.expert_span_bytes},
        options);
  }

  void finish_expert_tile(
      const std::uint16_t first_row, const std::uint16_t row_count,
      const std::uint64_t spine_generation,
      const CanonicalExpertPositionTileT1& experts,
      const MoeRunOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::WaitingForExperts,
                  "finish a target sequence expert tile");
    try {
      if (pending_ == nullptr ||
          pending_->next_expert_row >= position_count_ ||
          first_row != pending_->next_expert_row || row_count == 0 ||
          row_count > kMoePositionTileMaxRows ||
          static_cast<std::size_t>(first_row) + row_count > position_count_) {
        throw std::invalid_argument(
            "target sequence expert tiles must be bounded and finish in canonical row order");
      }
      if (spine_generation != pending_->spine.generation ||
          spine_generation != mailbox_.spine_generation) {
        throw std::invalid_argument(
            "target sequence expert row has a stale spine generation");
      }
      std::array<const PreparedMoeT1*, kMoePositionTileMaxRows> prepared{};
      for (std::size_t offset = 0; offset < row_count; ++offset) {
        const std::size_t row_index = first_row + offset;
        if (mailbox_.rows[row_index].row_index != row_index ||
            !pending_->rows[row_index].has_value() ||
            pending_->rows[row_index]->moe.spine_generation !=
                spine_generation) {
          throw std::logic_error(
              "target sequence expert tile lost one provider-owned row");
        }
        prepared[offset] = &pending_->rows[row_index]->moe;
      }

      MoeRunOptions execution_options = options;
      if (position_count_ > 1) {
        const at::Device& routed_device = prepared.front()->routed_input.device();
        const bool selects_metal = routed_device.is_mps() &&
            moe_positions_select_metal(routed_device, options);
        if (!pending_->expert_backend_decided) {
          pending_->expert_backend_decided = true;
          // K3_MOE_ATEN_STREAM=1: the expert kernels take the device
          // routed inputs directly; no whole-layer host staging.
          pending_->whole_layer_metal_staging =
              selects_metal && !moe_aten_stream_enabled();
          if (pending_->whole_layer_metal_staging) {
            const auto routed_hidden = static_cast<std::int64_t>(
                pending_->spine.geometry.routed_hidden);
            const auto positions =
                static_cast<std::int64_t>(position_count_);
            if (!pending_->routed_inputs_device.defined() ||
                pending_->routed_inputs_device.scalar_type() != at::kFloat ||
                pending_->routed_inputs_device.device() != routed_device ||
                !pending_->routed_inputs_device.is_contiguous() ||
                pending_->routed_inputs_device.sizes() !=
                    at::IntArrayRef({positions, routed_hidden})) {
              throw std::logic_error(
                  "target sequence lost its whole-layer routed-input carrier");
            }
            if (options.execution_trace != nullptr) {
              options.execution_trace->record(
                  MoeExecutionStage::RoutedInputHostMaterialization);
            }
            {
              const PrepSubScope materialize_scope(0);
              // K3_ROUTED_INPUT_ALIAS=1 (default off): alias the unified-
              // memory buffer instead of a blit + commitAndWait round trip.
              static const bool alias_enabled = [] {
                const char* value = std::getenv("K3_ROUTED_INPUT_ALIAS");
                return value != nullptr && std::strcmp(value, "1") == 0;
              }();
              at::Tensor staged;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
              if (alias_enabled && routed_device.is_mps()) {
                try {
                  staged = loop_host_alias_of_mps(
                      pending_->routed_inputs_device);
                  ++stats_.moe_routed_input_host_aliases;
                } catch (...) {
                  staged = at::Tensor();
                }
              }
#else
              static_cast<void>(alias_enabled);
#endif
              if (!staged.defined()) {
                staged = pending_->routed_inputs_device
                             .to(at::kCPU, at::kFloat)
                             .contiguous();
              }
              pending_->metal_routed_inputs_cpu = std::move(staged);
            }
            for (std::size_t row_index = 0; row_index < position_count_;
                 ++row_index) {
              if (!pending_->rows[row_index].has_value()) {
                throw std::logic_error(
                    "target sequence lost a row during Metal host staging");
              }
              pending_->rows[row_index]->moe.routed_input_cpu =
                  pending_->metal_routed_inputs_cpu.narrow(
                      0, static_cast<std::int64_t>(row_index), 1);
            }
            ++stats_.moe_routed_input_host_transfers;
          }
        } else if (pending_->whole_layer_metal_staging && !selects_metal) {
          throw std::invalid_argument(
              "target sequence cannot change away from Metal after whole-layer host staging");
        }
        execution_options.metal_retain_position_outputs_cpu =
            pending_->whole_layer_metal_staging && selects_metal;
      }

      bool tail_on_loop = false;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      // K3_KDA_LOOP=on, plan Step 5: the entire MoE tail of a T=1 routed
      // layer runs as one synchronous loop command buffer (expert stack +
      // routed_norm/up + shared epilogue + merge + residual), replacing the
      // stock execute/complete path below.  Per-call stock fallback.
      if (kda_loop_mode() == 2 && position_count_ == 1 && row_count == 1 &&
          experts.layout == MoeExpertLayout::RawV1 && !options.arrival_partial) {
        const MoeSpineT1& spine = pending_->spine;
        const PreparedMoeT1& moe = *prepared.front();
        std::optional<PendingExpertRow>& pending_row =
            pending_->rows[first_row];
        const auto view = [](const MoeRowInt8Matrix& matrix) {
          return LoopMoeMatrixView{matrix.quantized, matrix.row_scales,
                                   matrix.dense_f32, &matrix.original_bf16};
        };
        try {
          const auto blobs = loop_route_ordered_expert_blobs(
              moe.route, experts, spine.geometry);
          const bool combined = spine.shared_gate_up_enabled &&
                                spine.shared_gate_up_qualified;
          at::Tensor row_hidden;
          tail_on_loop = loop_moe_tail(
              spine.layer_index, moe.routed_input, moe.identity,
              pending_row->mlp_input.prefix_sum,
              moe.route.expert_ids.data(), moe.route.weight_bits.data(),
              blobs.data(),
              static_cast<std::uint32_t>(moe.route.expert_ids.size()),
              spine.routed_norm, view(spine.routed_up), combined,
              view(spine.shared_gate_up), view(spine.shared_gate),
              view(spine.shared_up), view(spine.shared_down), row_hidden);
          if (tail_on_loop) {
            SequenceRow& row = rows_[first_row];
            row.hidden = std::move(row_hidden);
            row.residual.anchors =
                std::move(pending_row->mlp_input.next_anchors);
            pending_row.reset();
            mailbox_.rows[first_row].routed_input = at::Tensor();
            ++stats_.moe_shared_dispatches;
            ++stats_.moe_complete_provider_dispatches;
            ++stats_.moe_complete_rows;
            ++stats_.moe_routed_up_dispatches;
            loop_set_chain_safe(true);
          }
        } catch (...) {
          tail_on_loop = false;
        }
      }
      if (!tail_on_loop) {
        loop_set_chain_safe(false);
        // The stock path below encodes ATen work that may consume the
        // previous layer's output; a deferred tail must complete first.
        loop_moe_tail_drain();
#endif
      std::optional<PrepSubScope> moe_scope;
      moe_scope.emplace(1);
      const bool arrival_partial = options.arrival_partial;
      const bool arrival_final = !arrival_partial || options.arrival_final;
      if (arrival_partial) {
        execution_options.partial_edges = true;
      }
      at::Tensor routed_outputs = execute_routed_moe_positions_t1(
          std::span<const PreparedMoeT1* const>(prepared.data(), row_count),
          experts, execution_options);
      moe_scope.reset();
      if (arrival_partial) {
        // Accumulate this group's contribution; only the FINAL group completes
        // the rows from the running sum (a changed fp32 summation order —
        // gated by token equivalence, not md5 identity).
        if (pending_->partial_routed_outputs.defined()) {
          if (pending_->partial_routed_outputs.sizes() != routed_outputs.sizes()) {
            throw std::runtime_error(
                "arrival-driven partial group changed the tile's output geometry");
          }
          pending_->partial_routed_outputs.add_(routed_outputs);
        } else {
          pending_->partial_routed_outputs = routed_outputs.clone();
        }
        ++pending_->partial_groups;
        ++stats_.moe_partial_dispatches;
        if (!arrival_final) {
          return;
        }
        routed_outputs = pending_->partial_routed_outputs;
        pending_->partial_routed_outputs = at::Tensor();
        pending_->partial_groups = 0;
      }
      if (!routed_outputs.defined() ||
          routed_outputs.scalar_type() != at::kFloat ||
          !routed_outputs.is_contiguous() ||
          routed_outputs.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(row_count),
               static_cast<std::int64_t>(
                   pending_->spine.geometry.routed_hidden)})) {
        throw std::runtime_error(
            "target sequence expert tile returned an invalid output matrix");
      }
      for (std::size_t offset = 0; offset < row_count; ++offset) {
        const std::size_t row_index = first_row + offset;
        std::optional<PendingExpertRow>& pending_row =
            pending_->rows[row_index];
        if (position_count_ == 1) {
          const at::Tensor mlp_output = complete_moe_t1(
              pending_row->moe,
              routed_outputs.narrow(
                  0, static_cast<std::int64_t>(offset), 1),
              pending_->spine);
          SequenceRow& row = rows_[row_index];
          row.hidden = complete_target_layer(
              pending_row->mlp_input, mlp_output, exact_k3_);
          row.residual.anchors =
              std::move(pending_row->mlp_input.next_anchors);
          pending_row.reset();
        } else {
          pending_->routed_output_rows[row_index] =
              routed_outputs.narrow(
                  0, static_cast<std::int64_t>(offset), 1);
        }
        mailbox_.rows[row_index].routed_input = at::Tensor();
      }
      if (position_count_ == 1) {
        ++stats_.moe_shared_dispatches;
        ++stats_.moe_complete_provider_dispatches;
        ++stats_.moe_complete_rows;
        ++stats_.moe_routed_up_dispatches;
      }
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      }  // if (!tail_on_loop)
#endif
      pending_->next_expert_row += row_count;
      stats_.expert_rows_completed += row_count;
      ++stats_.expert_tiles_completed;
      stats_.maximum_experts_per_request =
          std::max<std::uint64_t>(stats_.maximum_experts_per_request,
                                  experts.expert_ids.size());
      stats_.maximum_positions_per_expert_tile =
          std::max<std::uint64_t>(stats_.maximum_positions_per_expert_tile,
                                  row_count);

      if (pending_->next_expert_row == position_count_) {
        if (position_count_ > 1) {
          std::vector<const PreparedMoeT1*> all_prepared;
          std::vector<at::Tensor> routed_rows;
          all_prepared.reserve(position_count_);
          routed_rows.reserve(position_count_);
          for (std::size_t row_index = 0; row_index < position_count_;
               ++row_index) {
            if (!pending_->rows[row_index].has_value() ||
                !pending_->routed_output_rows[row_index].defined()) {
              throw std::logic_error(
                  "target sequence lost a completed batched routed row");
            }
            all_prepared.push_back(&pending_->rows[row_index]->moe);
            routed_rows.push_back(pending_->routed_output_rows[row_index]);
          }
          const std::span<const PreparedMoeT1* const> all_prepared_span(
              all_prepared.data(), all_prepared.size());
          std::optional<at::Tensor> existing_carrier =
              recover_contiguous_row_carrier(
                  pending_->routed_output_rows, position_count_,
                  static_cast<std::int64_t>(
                      pending_->spine.geometry.routed_hidden));
          at::Tensor routed_outputs_full = existing_carrier.has_value()
              ? std::move(*existing_carrier)
              : at::cat(routed_rows, 0).contiguous();
          if (pending_->whole_layer_metal_staging) {
            const at::Device completion_device =
                all_prepared.front()->identity.device();
            if (!routed_outputs_full.device().is_cpu()) {
              throw std::logic_error(
                  "whole-layer Metal outputs crossed to device before completion");
            }
            // This is the layer's only expert-output CPU->MPS boundary. All
            // <=16-row I/O tiles above retained their routed result on CPU.
            routed_outputs_full =
                routed_outputs_full.to(completion_device, at::kFloat)
                    .contiguous();
          }
          const at::Tensor mlp_outputs = complete_moe_positions_t1(
              all_prepared_span, routed_outputs_full, pending_->spine);
          at::Tensor completed = complete_target_layer_rows(
              pending_->mlp_rows, mlp_outputs, exact_k3_);
          install_wide_carriers(
              std::move(completed),
              std::move(pending_->mlp_rows.next_anchors));
          for (std::size_t row_index = 0; row_index < position_count_;
               ++row_index) {
            pending_->rows[row_index].reset();
          }
          ++stats_.moe_shared_dispatches;
          ++stats_.moe_complete_provider_dispatches;
          stats_.moe_complete_rows += position_count_;
          ++stats_.moe_routed_up_dispatches;
        }
        capture_completed_layer(next_layer_);
        stages_.push_back(std::move(pending_->cache_stage));
        pending_.reset();
        mailbox_ = TargetSequenceExpertMailbox{};
        ++next_layer_;
        state_ = next_layer_ == kTargetLayerCount
                     ? TargetSequenceState::ReadyForTail
                     : TargetSequenceState::Active;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
        if (position_count_ == 1 && moe_route_async_enabled() &&
            !rows_.empty() && rows_.front().hidden.device().is_mps()) {
          // K3_ROUTE_ASYNC: commit this layer's shared-expert/merge tail so
          // it executes under the next layer's spine bind and attention
          // encode instead of inside the next route boundary's drain.
          static_cast<void>(try_commit_mps_stream_for_route());
        }
#endif
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
        // Step 6: the runtime reads this after finish_expert_tile returns
        // and, when eligible, pre-commits the successor KDA layer with
        // freshly materialized per-owner views.
        last_tile_tail_on_loop_ = tail_on_loop;
#endif
      }
    } catch (...) {
      if (options.cuda_cache != nullptr &&
          (options.expert_backend == MoeExpertBackend::CudaMxfp4 ||
           options.cuda_plan != 0)) {
        options.cuda_cache->poison_external(
            "target-sequence MoE transaction failed after CUDA selection");
      }
      abort_unlocked();
      throw;
    }
  }

  std::span<const std::uint32_t> finish_tail() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyForTail,
                  "finish the target sequence tail");
    try {
      if (next_layer_ != kTargetLayerCount || pending_ != nullptr ||
          stages_.size() != kTargetLayerCount) {
        throw std::logic_error(
            "target sequence is incomplete at its provider tail");
      }
      decisions_.clear();
      for (const SequenceRow& row : rows_) {
        if (row.residual.anchors.size(1) != 8) {
          throw std::logic_error(
              "target sequence row has an incomplete residual tape");
        }
      }
      if (mode_ == TargetSequenceMode::Prefill) {
        SequenceRow& final_row = rows_.back();
        const at::Tensor logits = finish_target_tail(
            final_row.hidden, final_row.residual, *tail_, exact_k3_);
        if (!logits.defined() || logits.dim() != 2 ||
            logits.size(0) != 1 || logits.size(1) <= 0) {
          throw std::runtime_error(
              "target sequence tail did not produce one vocabulary row");
        }
        const std::int64_t token =
            at::argmax(logits, -1, false).item<std::int64_t>();
        if (token < 0 ||
            token > static_cast<std::int64_t>(
                        std::numeric_limits<std::uint32_t>::max())) {
          throw std::runtime_error(
              "target sequence decision is outside the public token range");
        }
        step_log_topk(stages_.empty() ? -1 : static_cast<std::int64_t>(initial_versions_[0]),
                      logits[0]);
        decisions_.push_back(static_cast<std::uint32_t>(token));
        stats_.tail_rows = 1;
      } else {
        at::Tensor hidden_rows;
        at::Tensor anchor_rows;
        if (position_count_ == 1) {
          hidden_rows = rows_.front().hidden;
          anchor_rows = rows_.front().residual.anchors;
        } else {
          require_wide_carrier_aliases();
          hidden_rows = hidden_rows_carrier_;
          anchor_rows = anchor_rows_carrier_;
        }
        const at::Tensor logits = finish_target_tail_rows(
            hidden_rows, anchor_rows, *tail_, exact_k3_);
        if (!logits.defined() || logits.dim() != 2 ||
            logits.size(0) != static_cast<std::int64_t>(position_count_) ||
            logits.size(1) <= 0) {
          throw std::runtime_error(
              "target sequence wide tail did not produce one row per position");
        }
        const at::Tensor tokens =
            at::argmax(logits, -1, false).to(at::kCPU).contiguous();
        const std::int64_t* token_values =
            tokens.const_data_ptr<std::int64_t>();
        for (std::size_t row = 0; row < position_count_; ++row) {
          const std::int64_t token = token_values[row];
          if (token < 0 ||
              token > static_cast<std::int64_t>(
                          std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error(
                "target sequence wide decision is outside the public token range");
          }
          decisions_.push_back(static_cast<std::uint32_t>(token));
        }
        {
          const std::int64_t base_position =
              stages_.empty() ? -1 : static_cast<std::int64_t>(initial_versions_[0]);
          for (std::size_t row = 0; row < position_count_; ++row) {
            step_log_topk(base_position + static_cast<std::int64_t>(row),
                          logits[static_cast<std::int64_t>(row)]);
          }
        }
        if (dspark_debug_enabled()) {
          // target_position is this chunk's starting absolute KV position
          // (layer-0's pre-chunk cache version); row 0's prediction is for
          // that exact position, matching the draft side's anchor_position
          // in provider_dspark_model.cpp — the two should read identical
          // when draft/target KV bookkeeping is in sync (hypothesis 1).
          const std::int64_t target_position =
              stages_.empty() ? -1
                              : static_cast<std::int64_t>(initial_versions_[0]);
          std::cerr << "K3_DSPARK_DEBUG target_anchor target_position="
                    << target_position << " rows=" << position_count_ << "\n";
          for (std::size_t row = 0; row < position_count_; ++row) {
            debug_print_topk("target_topk", static_cast<std::int64_t>(row),
                             target_position + static_cast<std::int64_t>(row),
                             logits[static_cast<std::int64_t>(row)]);
          }
        }
        stats_.tail_rows = position_count_;
      }
      for (SequenceRow& row : rows_) {
        row.hidden = at::Tensor();
        row.residual.anchors = at::Tensor();
      }
      hidden_rows_carrier_ = at::Tensor();
      anchor_rows_carrier_ = at::Tensor();
      stats_.tail_provider_dispatches = 1;
      state_ = TargetSequenceState::ReadyToCommit;
      return decisions_;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  at::Tensor dspark_target_rows() const {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyToCommit,
                  "read DSpark target auxiliary rows");
    if (!capture_dspark_rows_ || dspark_capture_failed_ ||
        captured_dspark_layers_ !=
                                     kDSparkTargetCaptureLayers.size()) {
      throw std::logic_error(
          "target sequence did not capture the complete DSpark layer roster");
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!dspark_rows_.defined() ||
        dspark_rows_.scalar_type() != at::kBFloat16 ||
        !dspark_rows_.is_contiguous() || dspark_rows_.dim() != 2 ||
        dspark_rows_.size(0) != static_cast<std::int64_t>(position_count_) ||
        dspark_rows_.size(1) !=
            static_cast<std::int64_t>(kDSparkTargetCaptureLayers.size()) *
                hidden) {
      throw std::logic_error(
          "target sequence retained an invalid DSpark target capture");
    }
    return dspark_rows_;
  }

  void commit_prefix(const std::size_t positions) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_state(TargetSequenceState::ReadyToCommit,
                  "commit a target sequence cache prefix");
    try {
      if (positions > position_count_ ||
          (mode_ == TargetSequenceMode::Prefill &&
           positions != position_count_) ||
          (full_commit_only_ && positions != position_count_)) {
        throw std::invalid_argument(
            "prefill and full-commit-only verify require the full sequence; ordinary verify accepts a bounded prefix");
      }
      preflight_commit(positions);

      // No operation below this point can throw: every pointer/version/shape
      // was checked across all 93 caches first, tensor handle moves are
      // noexcept, and MLA publication has its own no-throw half.
      for (auto& stage : stages_) {
        if (stage->kind == SequenceCacheStage::Kind::Mla) {
          stage->mla->publish_prefix_noexcept(positions);
          continue;
        }
        if (positions == 0) {
          continue;
        }
        KdaState* selected = nullptr;
        if (mode_ == TargetSequenceMode::Prefill || full_commit_only_) {
          selected = &stage->final_kda_state;
        } else {
          selected = &stage->kda_boundaries[positions - 1];
        }
        stage->kda_cache->state = std::move(*selected);
        stage->kda_cache->version =
            stage->expected_kda_version + positions;
      }
      stages_.clear();
      state_ = TargetSequenceState::Committed;
    } catch (...) {
      abort_unlocked();
      throw;
    }
  }

  void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == TargetSequenceState::Committed) {
      throw std::logic_error(
          "a committed target sequence cannot be cancelled");
    }
    if (state_ == TargetSequenceState::Cancelled ||
        state_ == TargetSequenceState::Poisoned) {
      return;
    }
    abort_unlocked();
  }

  TargetSequenceState state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::uint32_t next_layer_index() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_layer_;
  }

  TargetSequenceStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  TargetSequenceMode mode() const noexcept { return mode_; }
 std::size_t position_count() const noexcept { return position_count_; }

 private:
  void require_wide_carrier_aliases() const {
    if (position_count_ <= 1) {
      throw std::logic_error(
          "target sequence wide-carrier invariant used for T=1");
    }
    const std::int64_t positions =
        static_cast<std::int64_t>(position_count_);
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    if (!hidden_rows_carrier_.defined() ||
        hidden_rows_carrier_.scalar_type() != at::kFloat ||
        !hidden_rows_carrier_.is_contiguous() ||
        hidden_rows_carrier_.dim() != 2 ||
        hidden_rows_carrier_.sizes() !=
            at::IntArrayRef({positions, hidden}) ||
        !anchor_rows_carrier_.defined() ||
        anchor_rows_carrier_.scalar_type() != at::kFloat ||
        anchor_rows_carrier_.device() != hidden_rows_carrier_.device() ||
        !anchor_rows_carrier_.is_contiguous() ||
        anchor_rows_carrier_.dim() != 3 ||
        anchor_rows_carrier_.size(0) != positions ||
        anchor_rows_carrier_.size(2) != hidden ||
        rows_.size() != position_count_) {
      throw std::logic_error(
          "target sequence lost its authoritative wide carriers");
    }
    const std::int64_t anchors = anchor_rows_carrier_.size(1);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const SequenceRow& row = rows_[row_index];
      const std::int64_t index = static_cast<std::int64_t>(row_index);
      if (!row.hidden.defined() || !row.hidden.is_contiguous() ||
          row.hidden.sizes() != at::IntArrayRef({1, hidden}) ||
          !row.hidden.is_alias_of(hidden_rows_carrier_) ||
          row.hidden.storage_offset() !=
              hidden_rows_carrier_.storage_offset() +
                  index * hidden_rows_carrier_.stride(0) ||
          !row.residual.anchors.defined() ||
          !row.residual.anchors.is_contiguous() ||
          row.residual.anchors.sizes() !=
              at::IntArrayRef({1, anchors, hidden}) ||
          !row.residual.anchors.is_alias_of(anchor_rows_carrier_) ||
          row.residual.anchors.storage_offset() !=
              anchor_rows_carrier_.storage_offset() +
                  index * anchor_rows_carrier_.stride(0)) {
        throw std::logic_error(
            "target sequence row view stopped aliasing its wide carrier");
      }
    }
  }

  void install_wide_carriers(at::Tensor hidden_rows,
                             at::Tensor anchor_rows) {
    if (position_count_ <= 1 || !hidden_rows.defined() ||
        !anchor_rows.defined()) {
      throw std::logic_error(
          "target sequence cannot install an invalid wide carrier");
    }
    hidden_rows_carrier_ = std::move(hidden_rows);
    anchor_rows_carrier_ = std::move(anchor_rows);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const std::int64_t index = static_cast<std::int64_t>(row_index);
      rows_[row_index].hidden = hidden_rows_carrier_.narrow(0, index, 1);
      rows_[row_index].residual.anchors =
          anchor_rows_carrier_.narrow(0, index, 1);
    }
    require_wide_carrier_aliases();
  }

  void capture_completed_layer(const std::uint32_t layer_index) noexcept {
    if (!capture_dspark_rows_ || dspark_capture_failed_) {
      return;
    }
    try {
      if (position_count_ > 1) {
        require_wide_carrier_aliases();
      }
      const auto found = std::find(kDSparkTargetCaptureLayers.begin(),
                                   kDSparkTargetCaptureLayers.end(),
                                   layer_index);
      if (found == kDSparkTargetCaptureLayers.end()) {
        return;
      }
      const std::size_t capture_index = static_cast<std::size_t>(
          std::distance(kDSparkTargetCaptureLayers.begin(), found));
      if (capture_index != captured_dspark_layers_) {
        throw std::logic_error(
            "target sequence DSpark capture layers are out of schedule order");
      }
      const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      // K3_CB_FUSION=1: the capture copies below READ this layer's tail
      // output (row.hidden) on the ATen stream with a blocking copy; the
      // open fused CB that produces it must execute first or the DSpark
      // draft would silently capture garbage. Cheap no-op with the flag
      // off; capture layers simply run unfused.
      if (loop_cb_fusion_enabled()) {
        loop_moe_tail_drain();
      }
#endif
      if (!dspark_rows_.defined()) {
        dspark_rows_ = at::empty(
            {static_cast<std::int64_t>(position_count_),
             static_cast<std::int64_t>(kDSparkTargetCaptureLayers.size()) *
                 hidden},
            at::TensorOptions().dtype(at::kBFloat16).device(
                rows_.front().hidden.device()));
      }
      for (std::size_t row_index = 0; row_index < rows_.size(); ++row_index) {
        const SequenceRow& row = rows_[row_index];
        if (!row.hidden.defined() || row.hidden.scalar_type() != at::kFloat ||
            !row.hidden.is_contiguous() || row.hidden.dim() != 2 ||
            row.hidden.sizes() != at::IntArrayRef({1, hidden})) {
          throw std::logic_error(
              "target sequence cannot capture an invalid completed hidden row");
        }
        dspark_rows_
            .select(0, static_cast<std::int64_t>(row_index))
            .narrow(0, static_cast<std::int64_t>(capture_index) * hidden,
                    hidden)
            .copy_(row.hidden.squeeze(0), false);
      }
      ++captured_dspark_layers_;
    } catch (...) {
      // Auxiliary drafting must never make full-K3 target execution fail.
      // Discard only the proposal capture; the target transaction and its
      // exact cache stages remain authoritative and continue normally.
      dspark_rows_ = at::Tensor();
      captured_dspark_layers_ = 0;
      dspark_capture_failed_ = true;
    }
  }

  void validate_and_copy_bindings(const TargetPositionBindings& bindings) {
    if (bindings.contract != TargetTapeContract::ExactK3 &&
        bindings.contract != TargetTapeContract::SyntheticK3Schedule) {
      throw std::invalid_argument("target sequence contract is unknown");
    }
    if (bindings.caches.size() != kTargetLayerCount ||
        bindings.tail == nullptr) {
      throw std::invalid_argument(
          "target sequence requires 93 caches and persistent tail weights");
    }
    std::copy(bindings.caches.begin(), bindings.caches.end(), caches_.begin());
    std::unordered_set<const void*> cache_addresses;
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (std::uint32_t layer_index = 0;
         layer_index < kTargetLayerCount; ++layer_index) {
      const TargetLayerCacheBinding& layer = caches_[layer_index];
      const TargetAttentionKind expected_kind =
          target_layer_uses_mla(layer_index) ? TargetAttentionKind::Mla
                                             : TargetAttentionKind::Kda;
      if (layer.layer_index != layer_index ||
          layer.attention_kind != expected_kind) {
        throw std::invalid_argument(
            "target sequence cache schedule is invalid");
      }
      if (expected_kind == TargetAttentionKind::Kda) {
        ++kda_count;
        if (layer.kda_cache == nullptr || layer.mla_cache != nullptr ||
            layer.kda_cache->layer_index != layer_index ||
            layer.kda_cache->version ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.kda_cache).second) {
          throw std::invalid_argument(
              "target sequence KDA cache ownership is invalid");
        }
        initial_versions_[layer_index] = layer.kda_cache->version;
      } else {
        ++mla_count;
        if (layer.mla_cache == nullptr || layer.kda_cache != nullptr ||
            layer.mla_cache->has_pending_prepare() ||
            layer.mla_cache->version() ==
                std::numeric_limits<std::uint64_t>::max() ||
            !cache_addresses.insert(layer.mla_cache).second) {
          throw std::invalid_argument(
              "target sequence MLA cache ownership is invalid");
        }
        const MlaShape& shape = layer.mla_cache->shape();
        if ((exact_k3_ &&
             (!shape.is_exact_k3() ||
              layer.mla_cache->representation() !=
                  MlaCacheRepresentation::ExpandedExact)) ||
            (!exact_k3_ &&
             (shape.hidden_size != kSyntheticHidden || shape.is_exact_k3()))) {
          throw std::invalid_argument(
              "target sequence MLA cache does not match its model contract");
        }
        initial_versions_[layer_index] = layer.mla_cache->version();
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::logic_error("target sequence KDA/MLA schedule counts changed");
    }
  }

  void validate_current_layer(const TargetLayerBinding& layer) const {
    if (next_layer_ >= kTargetLayerCount ||
        layer.layer_index != next_layer_) {
      throw std::invalid_argument(
          "target sequence streamed layer is out of schedule order");
    }
    const TargetAttentionKind expected_kind =
        target_layer_uses_mla(next_layer_) ? TargetAttentionKind::Mla
                                           : TargetAttentionKind::Kda;
    if (layer.attention_kind != expected_kind || layer.residual == nullptr) {
      throw std::invalid_argument(
          "target sequence streamed attention binding is invalid");
    }
    const TargetLayerCacheBinding& cache = caches_[next_layer_];
    if (expected_kind == TargetAttentionKind::Kda) {
      if (layer.kda_weights == nullptr || layer.mla_weights != nullptr ||
          layer.mla_input_bundle != nullptr ||
          cache.kda_cache->version != initial_versions_[next_layer_]) {
        throw std::invalid_argument(
            "target sequence streamed KDA binding/cache is stale or invalid");
      }
    } else {
      if (layer.mla_weights == nullptr || layer.kda_weights != nullptr ||
          cache.mla_cache->has_pending_prepare() ||
          cache.mla_cache->version() != initial_versions_[next_layer_] ||
          !cache.mla_cache->can_append(position_count_)) {
        throw std::invalid_argument(
            "target sequence streamed MLA binding/cache exceeds its exact budget or is stale");
      }
    }

    if (next_layer_ == 0) {
      if (layer.dense == nullptr || layer.moe != nullptr) {
        throw std::invalid_argument(
            "target sequence layer zero must be dense");
      }
      return;
    }
    const std::int64_t hidden = exact_k3_ ? kExactHidden : kSyntheticHidden;
    const MoeGeometry expected_moe = k3_moe_geometry();
    if (layer.dense != nullptr || layer.moe == nullptr ||
        layer.moe->layer_index != next_layer_ ||
        layer.moe->generation == 0 ||
        layer.moe->geometry.hidden != static_cast<std::uint32_t>(hidden) ||
        (exact_k3_ && !same_geometry(layer.moe->geometry, expected_moe))) {
      throw std::invalid_argument(
          "target sequence routed-MoE binding is invalid");
    }
  }


  void prepare_kda_rows(const TargetLayerBinding& binding,
                        const TargetLayerCacheBinding& cache,
                        SequenceCacheStage& stage,
                        PendingSequenceLayer* routed) {
    stage.kind = SequenceCacheStage::Kind::Kda;
    stage.kda_cache = cache.kda_cache;
    stage.expected_kda_version = cache.kda_cache->version;
    if (mode_ == TargetSequenceMode::Verify && !full_commit_only_) {
      stage.kda_boundaries.reserve(position_count_);
    }
    KdaState working = cache.kda_cache->state;
    if (position_count_ == 1) {
      SequenceRow& row = rows_.front();
      TargetAttentionInput attention;
      KdaDecodeResult decoded;
      TargetMlpInput loop_mlp;
      bool decoded_on_loop = false;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      // K3_KDA_LOOP=on (loop plan Step 3c takeover): the layer HEAD, the
      // KDA chain, AND the MLP head (residual add + second AttnRes mix +
      // post-attention norm) run in one loop command buffer; only the
      // anchor/prefix bookkeeping stays host-side, replicated from
      // prepare_target_attention's tail (kResidualBlock = 12).  Stock is
      // the per-call fallback.
      if (kda_loop_mode() == 2 && exact_k3_ && row.hidden.is_mps() &&
          binding.residual->self_attention_score_weight.defined() &&
          binding.residual->mlp_score_weight.defined()) {
        // Collect a matching pre-committed CB; abandon a stale one. Any
        // failure falls through to the synchronous path below. Parity mode
        // (K3_KDA_PRECOMMIT=2) collects into temporaries, compares against
        // the synchronous result below, and always uses stock.
        bool parity_collected = false;
        at::Tensor parity_mlp_normalized;
        KdaDecodeResult parity_decoded;
        if (kda_precommit_handle_ != nullptr) {
          LoopKdaPendingHandle* handle = kda_precommit_handle_;
          kda_precommit_handle_ = nullptr;
          if (kda_precommit_layer_ == next_layer_) {
            at::Tensor normalized;
            at::Tensor mlp_normalized;
            at::Tensor lookahead;
            at::Tensor prefix;
            at::Tensor collected_anchors;
            const bool collected = loop_kda_layer_collect(
                handle, normalized, mlp_normalized, lookahead, prefix,
                decoded, collected_anchors);
            if (collected && kda_precommit_mode() == 2) {
              parity_collected = true;
              parity_mlp_normalized = std::move(mlp_normalized);
              parity_decoded = std::move(decoded);
              decoded = KdaDecodeResult{};
            } else if (collected) {
              decoded_on_loop = true;
              loop_mlp = TargetMlpInput{
                  std::move(mlp_normalized), std::move(lookahead),
                  std::move(prefix),
                  collected_anchors.defined()
                      ? std::move(collected_anchors)
                      : std::move(kda_precommit_next_anchors_),
                  next_layer_};
            }
          } else {
            loop_kda_layer_abandon(handle);
          }
          kda_precommit_next_anchors_ = at::Tensor();
        }
        if (!decoded_on_loop) {
        constexpr std::uint32_t kLoopResidualBlock = 12;
        at::Tensor prefix_sum = row.hidden;
        at::Tensor next_anchors = row.residual.anchors;
        if (next_layer_ % kLoopResidualBlock == 0) {
          next_anchors =
              at::cat({row.residual.anchors, row.hidden.unsqueeze(1)}, 1)
                  .contiguous();
          prefix_sum = at::Tensor();
          loop_note_fresh_anchor_cat();
        }
        at::Tensor normalized;
        at::Tensor mlp_normalized;
        at::Tensor lookahead;
        at::Tensor prefix;
        decoded_on_loop = loop_kda_layer(
            next_layer_, row.hidden, row.residual.anchors,
            binding.residual->self_attention_score_weight,
            binding.residual->input_norm, prefix_sum, next_anchors,
            binding.residual->mlp_score_weight,
            binding.residual->post_attention_norm, *binding.kda_weights,
            working, normalized, mlp_normalized, lookahead, prefix,
            decoded);
        if (decoded_on_loop) {
          loop_mlp = TargetMlpInput{
              std::move(mlp_normalized), std::move(lookahead),
              std::move(prefix), std::move(next_anchors), next_layer_};
        }
        }  // if (!decoded_on_loop) — synchronous fallback
        if (parity_collected && decoded.output.defined()) {
          static std::uint64_t parity_compared = 0;
          static std::uint64_t parity_failures = 0;
          const auto rel = [](const at::Tensor& got, const at::Tensor& want) {
            const double denom = want.norm().item<double>();
            return (got - want).norm().item<double>() /
                   std::max(denom, 1e-12);
          };
          const double out_rel = rel(parity_decoded.output, decoded.output);
          const double state_rel = rel(parity_decoded.next_state.recurrent,
                                       decoded.next_state.recurrent);
          const double mlp_rel =
              parity_mlp_normalized.defined()
                  ? rel(parity_mlp_normalized,
                        decoded_on_loop ? parity_mlp_normalized
                                        : parity_mlp_normalized)
                  : 0.0;
          static_cast<void>(mlp_rel);
          ++parity_compared;
          if (!(out_rel <= 1.0e-4 && state_rel <= 1.0e-4) ||
              !std::isfinite(out_rel) || !std::isfinite(state_rel)) {
            ++parity_failures;
            std::fprintf(
                stderr,
                "[precommit-parity] FAIL layer=%u out_rel=%.3e "
                "state_rel=%.3e (n=%llu fails=%llu)\n",
                next_layer_, out_rel, state_rel,
                static_cast<unsigned long long>(parity_compared),
                static_cast<unsigned long long>(parity_failures));
          } else if (parity_compared % 500 == 1) {
            std::fprintf(
                stderr, "[precommit-parity] ok n=%llu out_rel=%.3e\n",
                static_cast<unsigned long long>(parity_compared), out_rel);
          }
        }
      }
#endif
      if (!decoded_on_loop) {
        attention = prepare_target_attention(row.hidden, row.residual,
                                             *binding.residual, next_layer_,
                                             exact_k3_);
        decoded = kda_decode_one(attention.normalized, *binding.kda_weights,
                                 working, exact_k3_);
      }
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      // K3_KDA_LOOP=parity: run the same layer on the loop queue and
      // compare against the stock result — stock is used, zero behavior
      // change, fail-soft.
      if (kda_loop_mode() == 1 && exact_k3_ &&
          attention.normalized.is_mps()) {
        loop_kda_parity_compare(next_layer_, attention.normalized,
                                *binding.kda_weights, working, decoded);
      }
#endif
      std::vector<TargetMlpInput> mlp_inputs;
      mlp_inputs.push_back(
          decoded_on_loop
              ? std::move(loop_mlp)
              : prepare_target_mlp(attention, decoded.output,
                                   *binding.residual, exact_k3_));
      if (mode_ == TargetSequenceMode::Verify && !full_commit_only_) {
        const std::uint64_t bytes = kda_state_bytes(decoded.next_state);
        stats_.verify_snapshot_bytes = checked_add(
            stats_.verify_snapshot_bytes, bytes, "verify snapshot");
        stats_.staged_kda_storage_bytes = checked_add(
            stats_.staged_kda_storage_bytes, bytes, "staged KDA");
        stage.kda_boundaries.push_back(decoded.next_state);
      }
      working = std::move(decoded.next_state);
      if (mode_ == TargetSequenceMode::Prefill || full_commit_only_) {
        stats_.staged_kda_storage_bytes = checked_add(
            stats_.staged_kda_storage_bytes, kda_state_bytes(working),
            "staged KDA");
        stage.final_kda_state = working;
      }
      prepare_mlp_rows(binding, std::move(mlp_inputs), routed);
    } else {
      require_wide_carrier_aliases();
      TargetAttentionRowsInput attention = prepare_target_attention_rows(
          hidden_rows_carrier_, anchor_rows_carrier_, *binding.residual,
          next_layer_, exact_k3_);
      const at::Tensor& normalized_hidden = attention.normalized;
      KdaBatchInputProjections batch = kda_project_inputs_batch(
          normalized_hidden, *binding.kda_weights, exact_k3_);
      if (batch.positions != position_count_) {
        throw std::logic_error(
            "KDA batch projection returned a different position count");
      }
      stats_.kda_input_provider_dispatches += batch.provider_dispatches;
      stats_.kda_input_equivalent_rowwise_dispatches +=
          batch.equivalent_rowwise_dispatches;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      loop_aten_stream_commit_phase("qkv-projections");
#endif
      KdaBatchDependentProjections dependent =
          kda_project_dependent_batch(
              normalized_hidden, *binding.kda_weights, exact_k3_);
      stats_.kda_dependent_provider_dispatches +=
          dependent.dependent_provider_dispatches;
      stats_.kda_dependent_equivalent_rowwise_dispatches +=
          dependent.dependent_equivalent_rowwise_dispatches;
      const bool retain_boundaries =
          mode_ == TargetSequenceMode::Verify && !full_commit_only_;
      KdaPositionsRecurrentResult recurrence;
      std::optional<KdaBatchOutputProjection> fused_outputs;
      // K3_KDA_WIDE_FUSED=1: one fused kernel replaces the batched short
      // conv, the per-position recurrence and the output norm/gate.
      std::optional<KdaWideFusedResult> fused = kda_wide_fused_positions(
          normalized_hidden, *binding.kda_weights, working, batch, dependent,
          retain_boundaries, exact_k3_);
      if (fused.has_value()) {
        recurrence.final_state = std::move(fused->final_state);
        recurrence.boundaries = std::move(fused->boundaries);
        fused_outputs = KdaBatchOutputProjection{
            std::move(fused->output),
            static_cast<std::uint32_t>(position_count_),
            fused->provider_dispatches, 0};
        stats_.kda_shortconv_provider_dispatches += 1;
        stats_.kda_recurrent_rows += position_count_;
      } else {
        KdaConvolvedPositions convolved = kda_short_convolve_positions(
            normalized_hidden, *binding.kda_weights, working,
            KdaPreprojectedPositions{
                .query = batch.query,
                .key = batch.key,
                .value = batch.value,
            },
            exact_k3_);
        stats_.kda_shortconv_provider_dispatches += 3;
        recurrence = kda_recur_convolved_positions(
            normalized_hidden, *binding.kda_weights, working, convolved,
            KdaDependentPositions{
                .feature_a = dependent.feature_a,
                .feature_b = dependent.feature_b,
                .beta = dependent.beta,
            },
            retain_boundaries, exact_k3_);
        stats_.kda_recurrent_rows += position_count_;
      }
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      loop_aten_stream_commit_phase("dependent+conv+recurrence");
#endif
      if (mode_ == TargetSequenceMode::Verify && !full_commit_only_) {
        if (recurrence.boundaries.size() != position_count_) {
          throw std::logic_error(
              "KDA position recurrence lost a verify boundary");
        }
        for (const KdaState& boundary : recurrence.boundaries) {
          const std::uint64_t bytes = kda_state_bytes(boundary);
          stats_.verify_snapshot_bytes = checked_add(
              stats_.verify_snapshot_bytes, bytes, "verify snapshot");
          stats_.staged_kda_storage_bytes = checked_add(
              stats_.staged_kda_storage_bytes, bytes, "staged KDA");
        }
        stage.kda_boundaries = std::move(recurrence.boundaries);
      }
      working = std::move(recurrence.final_state);
      KdaBatchOutputProjection outputs =
          fused_outputs.has_value()
              ? std::move(*fused_outputs)
              : kda_finish_output_batch(
                    normalized_hidden, recurrence.recurrent_output_rows,
                    *binding.kda_weights, exact_k3_);
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      loop_aten_stream_commit_phase("gate+outnorm+outproj");
#endif
      if (outputs.positions != position_count_ ||
          !outputs.output.defined() ||
          outputs.output.sizes() != at::IntArrayRef(
              {static_cast<std::int64_t>(position_count_),
               rows_.front().hidden.size(1)})) {
        throw std::logic_error(
            "KDA output batch returned a different position geometry");
      }
      stats_.kda_output_provider_dispatches += outputs.provider_dispatches;
      stats_.kda_output_rows += outputs.positions;
      TargetMlpRowsInput mlp = prepare_target_mlp_rows(
          attention, outputs.output, *binding.residual, exact_k3_);
      if (mode_ == TargetSequenceMode::Prefill || full_commit_only_) {
        stats_.staged_kda_storage_bytes = checked_add(
            stats_.staged_kda_storage_bytes, kda_state_bytes(working),
            "staged KDA");
        stage.final_kda_state = working;
      }
      prepare_mlp_rows_batch(binding, std::move(mlp), routed);
    }
  }

  void prepare_mla_rows(const TargetLayerBinding& binding,
                        const TargetLayerCacheBinding& cache,
                        SequenceCacheStage& stage,
                        PendingSequenceLayer* routed) {
    stage.kind = SequenceCacheStage::Kind::Mla;
    bool chained_done = false;
    at::Tensor chained_output;
    std::unique_ptr<MlaDecodeShell> chained_shell;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    // Step 4 chain-lite: consume the CB precommitted at the previous
    // layer's finish. On success the parked transaction becomes this
    // stage's transaction (its working cache holds the staged K/V row and
    // the pending nonce). Any failure drains + cancels and falls through
    // to the fully stock path below.
    if (mla_precommit_handle_ != nullptr) {
      LoopMlaPendingHandle* handle = mla_precommit_handle_;
      mla_precommit_handle_ = nullptr;
      chained_shell = std::move(mla_precommit_shell_);
      std::unique_ptr<MlaCacheTransaction> chained_txn =
          std::move(mla_precommit_txn_);
      bool collected = false;
      if (mla_precommit_layer_ == next_layer_ && position_count_ == 1 &&
          chained_shell != nullptr && chained_txn != nullptr) {
        collected =
            loop_mla_layer_collect(handle, next_layer_, chained_output);
      } else {
        loop_mla_layer_abandon(handle);
      }
      if (collected) {
        stage.mla = std::move(chained_txn);
        chained_done = true;
      } else if (chained_shell != nullptr && chained_txn != nullptr) {
        try {
          cancel_mla_decode(chained_txn->working_cache(),
                            chained_shell->prepared);
        } catch (...) {
        }
        chained_shell.reset();
      }
    }
#endif
    if (stage.mla == nullptr) {
      stage.mla = std::make_unique<MlaCacheTransaction>(
          *cache.mla_cache, position_count_);
    }
    if (position_count_ == 1) {
      SequenceRow& row = rows_.front();
      TargetAttentionInput attention = prepare_target_attention(
          row.hidden, row.residual, *binding.residual, next_layer_, exact_k3_);
      MlaPreparedDecode decoded = [&]() -> MlaPreparedDecode {
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
        if (chained_done) {
          // Step 4 chain-lite: the CB already computed attention head +
          // the full MLA chain and staged the K/V row into this stage's
          // working cache. The host-side prepare_target_attention above
          // still ran for the residual bookkeeping prepare_target_mlp
          // needs; its normalized tensor is unused here (the CB computed
          // its own, reassociation-equal — the chained-parity gate
          // verifies the pairing).
          chained_shell->prepared.output = std::move(chained_output);
          return std::move(chained_shell->prepared);
        }
        // K3_MLA_LOOP=on (Step 3 takeover): the parity-validated loop
        // chain replaces the stock ATen MLA. Reassociation-class
        // (user-signed-off 2026-08-26, int8-direct precedent). Fail-soft:
        // any disqualification cancels the shell and falls back to stock
        // — an uncommitted staged row is scratch by contract.
        if (mla_loop_mode() == 2 && exact_k3_ &&
            attention.normalized.is_mps() &&
            binding.mla_parity_weights != nullptr &&
            binding.mla_parity_bundle != nullptr) {
          try {
            // Allocate BEFORE the shell: an at::empty OOM after the shell
            // has taken the pending nonce would strand it (no destructor
            // releases it) and turn the fallback into a hard
            // "unfinished prepared decode" failure.
            at::Tensor loop_output =
                at::empty({1, 1, row.hidden.size(1)},
                          attention.normalized.options());
            MlaDecodeShell shell = prepare_k3_mla_decode_shell(
                attention.normalized.view({1, 1, row.hidden.size(1)}),
                stage.mla->working_cache());
            try {
              if (loop_mla_takeover_run(
                      next_layer_,
                      attention.normalized.view(
                          {1, 1, row.hidden.size(1)}),
                      *binding.mla_parity_weights,
                      binding.mla_parity_bundle, shell.key_states,
                      shell.value_states, loop_output)) {
                shell.prepared.output = std::move(loop_output);
                return std::move(shell.prepared);
              }
              cancel_mla_decode(stage.mla->working_cache(), shell.prepared);
            } catch (...) {
              // Nothing between the shell and here may leave the nonce
              // taken, or the stock fallback below hard-fails.
              if (shell.prepared.owner != nullptr &&
                  !shell.prepared.finalized) {
                cancel_mla_decode(stage.mla->working_cache(),
                                  shell.prepared);
              }
              throw;
            }
          } catch (...) {
            // Shell-internal throws released the nonce themselves; the
            // inner handler released it for post-shell throws. Stock
            // retakes it below.
          }
        }
#endif
        return exact_k3_
                   ? prepare_k3_mla_decode(
                         attention.normalized.view(
                             {1, 1, row.hidden.size(1)}),
                         *binding.mla_weights, stage.mla->working_cache(),
                         binding.mla_input_bundle)
                   : prepare_mla_decode(
                         attention.normalized.view(
                             {1, 1, row.hidden.size(1)}),
                         *binding.mla_weights, stage.mla->working_cache(),
                         true, binding.mla_input_bundle);
      }();
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
      // K3_MLA_LOOP=parity: run the same MLA layer on the loop queue into
      // a private scratch slab and compare against the stock output —
      // stock is used, zero behavior change, fail-soft (Step 2c(ii)). Runs
      // BEFORE commit so the committed prefix is exactly what stock read.
      if (mla_loop_mode() == 1 && exact_k3_ &&
          attention.normalized.is_mps()) {
        // Prefer the stable spine-form int8 weights stashed at bind time
        // (arena hosts null the bundle and hand out DenseF32 views); fall
        // back to the binding's own forms on arena-less hosts where stock
        // itself runs the int8 bundle.
        const MlaWeights* parity_weights =
            binding.mla_parity_weights != nullptr ? binding.mla_parity_weights
                                                  : binding.mla_weights;
        const MlaInputBundle* parity_bundle =
            binding.mla_parity_bundle != nullptr ? binding.mla_parity_bundle
                                                 : binding.mla_input_bundle;
        loop_mla_parity_compare(
            next_layer_,
            attention.normalized.view({1, 1, row.hidden.size(1)}),
            *parity_weights, parity_bundle, stage.mla->working_cache(),
            decoded);
      }
#endif
      try {
        std::vector<TargetMlpInput> mlp_inputs;
        mlp_inputs.push_back(prepare_target_mlp(
            attention,
            decoded.output.view({1, row.hidden.size(1)}).contiguous(),
            *binding.residual, exact_k3_));
        commit_mla_decode(stage.mla->working_cache(), decoded);
        ++stats_.mla_position_provider_dispatches;
        ++stats_.mla_position_rows;
        prepare_mlp_rows(binding, std::move(mlp_inputs), routed);
      } catch (...) {
        if (!decoded.finalized && decoded.owner != nullptr) {
          try {
            cancel_mla_decode(stage.mla->working_cache(), decoded);
          } catch (...) {
            // The branch is unpublished and is destroyed by abort_unlocked;
            // never risk publishing a partially prepared cache during cleanup.
          }
        }
        throw;
      }
    } else {
      require_wide_carrier_aliases();
      TargetAttentionRowsInput attention = prepare_target_attention_rows(
          hidden_rows_carrier_, anchor_rows_carrier_, *binding.residual,
          next_layer_, exact_k3_);
      const at::Tensor hidden =
          attention.normalized
              .view({1, static_cast<std::int64_t>(position_count_),
                     rows_.front().hidden.size(1)});
      MlaPreparedDecode decoded =
          exact_k3_
              ? prepare_k3_mla_positions(
                    hidden, *binding.mla_weights,
                    stage.mla->working_cache(), nullptr,
                    binding.mla_input_bundle)
              : prepare_mla_positions(
                    hidden, *binding.mla_weights,
                    stage.mla->working_cache(), nullptr, true,
                    binding.mla_input_bundle);
      try {
        const at::Tensor output_rows = decoded.output.view(
            {static_cast<std::int64_t>(position_count_),
             rows_.front().hidden.size(1)});
        TargetMlpRowsInput mlp = prepare_target_mlp_rows(
            attention, output_rows.contiguous(), *binding.residual,
            exact_k3_);
        commit_mla_decode(stage.mla->working_cache(), decoded);
        ++stats_.mla_position_provider_dispatches;
        stats_.mla_position_rows += position_count_;
        prepare_mlp_rows_batch(binding, std::move(mlp), routed);
      } catch (...) {
        if (!decoded.finalized && decoded.owner != nullptr) {
          try {
            cancel_mla_decode(stage.mla->working_cache(), decoded);
          } catch (...) {
            // The branch is unpublished and is destroyed by abort_unlocked;
            // never risk publishing a partially prepared cache during cleanup.
          }
        }
        throw;
      }
    }
    const std::uint64_t before = cache.mla_cache->storage_bytes();
    const std::uint64_t after = stage.mla->working_cache().storage_bytes();
    if (after < before) {
      throw std::logic_error("target sequence MLA branch shrank its storage");
    }
    stats_.projected_mla_storage_bytes = checked_add(
        stats_.projected_mla_storage_bytes, after,
        "MLA projected storage");
    stats_.additional_mla_storage_bytes = checked_add(
        stats_.additional_mla_storage_bytes, after - before,
        "MLA additional storage");
  }

  void prepare_mlp_rows_batch(const TargetLayerBinding& binding,
                              TargetMlpRowsInput mlp_input,
                              PendingSequenceLayer* routed) {
    if (position_count_ <= 1 || !mlp_input.normalized.defined() ||
        mlp_input.normalized.dim() != 2 ||
        mlp_input.normalized.size(0) !=
            static_cast<std::int64_t>(position_count_)) {
      throw std::logic_error(
          "target sequence batched MLP input changed its position count");
    }
    if (next_layer_ == 0) {
      const at::Tensor outputs = run_target_dense_rows(
          mlp_input.normalized, *binding.dense, exact_k3_);
      if (!outputs.defined() || outputs.scalar_type() != at::kFloat ||
          !outputs.is_contiguous() ||
          outputs.sizes() != mlp_input.normalized.sizes()) {
        throw std::runtime_error(
            "target sequence dense row batch returned an invalid output matrix");
      }
      at::Tensor completed = complete_target_layer_rows(
          mlp_input, outputs, exact_k3_);
      ++stats_.dense_mlp_provider_dispatches;
      stats_.dense_mlp_rows += position_count_;
      install_wide_carriers(std::move(completed),
                            std::move(mlp_input.next_anchors));
      return;
    }
    if (routed == nullptr) {
      throw std::logic_error("target sequence routed layer lost its mailbox");
    }

    // Mirror Python PILOT's prompt/verify schedule: route every live row with
    // the next-layer lookahead router before the current authoritative router,
    // then publish one bounded union only at Rust's expert-I/O boundary.
    prepare_pilot_rows(mlp_input.lookahead_source);
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
    loop_aten_stream_commit_phase("mlp-norm+pilot");
#endif

    PreparedMoePositionsT1 prepared = prepare_moe_positions_t1(
        mlp_input.normalized, routed->spine);
    if (prepared.rows.size() != position_count_) {
      throw std::logic_error(
          "target sequence MoE batch changed its position count");
    }

    ++stats_.moe_prepare_provider_dispatches;
    stats_.moe_prepare_rows += position_count_;
    stats_.moe_router_dispatches += prepared.router_dispatches;
    stats_.moe_routed_down_dispatches += prepared.routed_down_dispatches;
    stats_.moe_shared_dispatches += prepared.shared_dispatches;
    stats_.moe_route_materializations += prepared.route_materializations;
    stats_.moe_route_host_transfers += prepared.route_host_transfers;
    stats_.moe_routed_input_host_transfers +=
        prepared.routed_input_host_transfers;

    routed->mlp_rows = std::move(mlp_input);
    routed->routed_inputs_device = std::move(prepared.routed_inputs);
    mailbox_.layer_index = next_layer_;
    mailbox_.spine_generation = routed->spine.generation;
    mailbox_.row_count = static_cast<std::uint16_t>(position_count_);
    for (std::size_t row_index = 0; row_index < position_count_;
         ++row_index) {
      const auto index = static_cast<std::int64_t>(row_index);
      PreparedMoeT1& moe = prepared.rows[row_index];
      mailbox_.rows[row_index] = TargetSequenceRouteRow{
          .row_index = static_cast<std::uint16_t>(row_index),
          .route = moe.route,
          .routed_input = moe.routed_input,
      };
      TargetMlpInput row_input{
          routed->mlp_rows.normalized.narrow(0, index, 1),
          routed->mlp_rows.lookahead_source.narrow(0, index, 1),
          routed->mlp_rows.prefix_sum.narrow(0, index, 1),
          routed->mlp_rows.next_anchors.narrow(0, index, 1),
          routed->mlp_rows.layer_index};
      routed->rows[row_index].emplace(PendingExpertRow{
          std::move(row_input), std::move(moe)});
    }
  }

  void prepare_mlp_rows(const TargetLayerBinding& binding,
                        std::vector<TargetMlpInput> mlp_inputs,
                        PendingSequenceLayer* routed) {
    if (mlp_inputs.size() != position_count_) {
      throw std::logic_error(
          "target sequence MLP preparation changed its position count");
    }
    if (position_count_ != 1) {
      throw std::logic_error(
          "wide target MLP rows bypassed their live batch carrier");
    }
    prepare_row_mlp(binding, 0, std::move(mlp_inputs.front()), routed);
  }

  void prepare_row_mlp(const TargetLayerBinding& binding,
                       const std::size_t row_index,
                       TargetMlpInput mlp_input,
                       PendingSequenceLayer* routed) {
    SequenceRow& row = rows_[row_index];
    if (next_layer_ == 0) {
      const at::Tensor mlp_output = run_target_dense(
          mlp_input.normalized, *binding.dense, exact_k3_);
      row.hidden =
          complete_target_layer(mlp_input, mlp_output, exact_k3_);
      row.residual.anchors = std::move(mlp_input.next_anchors);
      ++stats_.dense_mlp_provider_dispatches;
      ++stats_.dense_mlp_rows;
      return;
    }
    if (routed == nullptr) {
      throw std::logic_error("target sequence routed layer lost its mailbox");
    }
    // Scheduling prediction is enqueued before the authoritative current-layer
    // router. Its tensors remain device-resident until Rust has
    // already obtained and begun serving the real route. Any failure simply
    // removes the optional hint; target math is unchanged.
    //
    // K3_ROUTE_ASYNC moves the pilot enqueue after the authoritative route
    // boundary on MPS: the lookahead chain is not a route dependency, so
    // queueing it later keeps it out of the route drain and lets it execute
    // under Rust's expert I/O, before take_prefetch_hint reads it back.
    const bool async_route =
        moe_route_async_enabled() && mlp_input.normalized.device().is_mps();
    if (row_index == 0 && !async_route) {
      prepare_pilot_rows(mlp_input.lookahead_source);
    }
    PreparedMoeT1 moe = prepare_moe_t1(mlp_input.normalized, routed->spine);
    if (row_index == 0 && async_route) {
      prepare_pilot_rows(mlp_input.lookahead_source);
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MPS_ROUTE_MAILBOX_V1)
      // Commit the just-encoded routed_down and pilot chains now, otherwise
      // they sit unexecuted through Rust's expert-read window and drain
      // inside take_prefetch_hint's readback instead.
      static_cast<void>(try_commit_mps_stream_for_route());
#endif
    }
    ++stats_.moe_prepare_provider_dispatches;
    ++stats_.moe_prepare_rows;
    ++stats_.moe_router_dispatches;
    ++stats_.moe_routed_down_dispatches;
    if (moe.shared_output.defined()) {
      ++stats_.moe_shared_dispatches;
    }
    ++stats_.moe_route_materializations;
    if (!mlp_input.normalized.device().is_cpu()) {
      ++stats_.moe_route_host_transfers;
    }
    if (mlp_input.normalized.device().is_mps()) {
      ++stats_.moe_routed_input_host_transfers;
    }
    mailbox_.layer_index = next_layer_;
    mailbox_.spine_generation = routed->spine.generation;
    mailbox_.row_count = static_cast<std::uint16_t>(position_count_);
    mailbox_.rows[row_index] = TargetSequenceRouteRow{
        .row_index = static_cast<std::uint16_t>(row_index),
        .route = moe.route,
        .routed_input = moe.routed_input,
    };
    routed->rows[row_index].emplace(
        PendingExpertRow{std::move(mlp_input), std::move(moe)});
  }

  void prepare_pilot_rows(const at::Tensor& lookahead_source) noexcept {
    if (next_layer_ + 1 >= kTargetLayerCount || pilot_routers_ == nullptr) {
      return;
    }
    const auto& next_router = (*pilot_routers_)[next_layer_ + 1];
    if (!next_router.has_value()) {
      return;
    }
    std::optional<PilotPredictionRows> prediction =
        try_predict_pilot_router_rows(lookahead_source, *next_router,
                                      exact_k3_);
    if (!prediction.has_value() ||
        prediction->position_count != position_count_) {
      return;
    }
    pending_pilot_ = std::move(*prediction);
    ++stats_.pilot_prediction_dispatches;
    stats_.pilot_prediction_rows += position_count_;
#if defined(__APPLE__)
    // K3_PILOT_MAILBOX=1 (sync-E removal): publish the prediction into the
    // shared pilot mailbox now — an encode-and-signal that rides the next
    // natural stream commit; the consumer polls it at hint time with no
    // drain. 0 = publish unavailable, consumer uses the stock path.
    static const bool pilot_mailbox_on = [] {
      const char* value = std::getenv("K3_PILOT_MAILBOX");
      return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (pilot_mailbox_on) {
      pilot_mailbox_generation_ = try_publish_mps_pilot_rows(
          pending_pilot_->expert_ids, pending_pilot_->choice_scores,
          pending_pilot_->layer_index, pending_pilot_->expert_count);
    }
#endif
  }

  void preflight_commit(const std::size_t positions) const {
    std::size_t kda_count = 0;
    std::size_t mla_count = 0;
    for (const auto& stage : stages_) {
      if (stage == nullptr) {
        throw std::logic_error("target sequence has an empty cache stage");
      }
      if (stage->kind == SequenceCacheStage::Kind::Mla) {
        ++mla_count;
        if (stage->mla == nullptr ||
            stage->mla->completed_positions() != position_count_) {
          throw std::logic_error(
              "target sequence has an incomplete MLA branch");
        }
        stage->mla->preflight_publish_prefix(positions);
        continue;
      }
      ++kda_count;
      if (stage->kda_cache == nullptr ||
          stage->kda_cache->layer_index != stage->layer_index ||
          stage->kda_cache->version != stage->expected_kda_version ||
          positions > std::numeric_limits<std::uint64_t>::max() -
                          stage->expected_kda_version) {
        throw std::invalid_argument(
            "target sequence KDA cache became stale before commit");
      }
      if (mode_ == TargetSequenceMode::Verify && !full_commit_only_) {
        if (stage->kda_boundaries.size() != position_count_) {
          throw std::logic_error(
              "target sequence verify lost a KDA row boundary");
        }
      } else if (positions != 0 &&
                 !stage->final_kda_state.recurrent.defined()) {
        throw std::logic_error(
            "target sequence full publication lost its final KDA state");
      }
    }
    if (kda_count != kTargetKdaLayerCount ||
        mla_count != kTargetMlaLayerCount) {
      throw std::logic_error(
          "target sequence did not stage all 93 attention caches");
    }
  }

  void require_state(const TargetSequenceState expected,
                     const char* operation) const {
    if (state_ != expected) {
      throw std::logic_error(std::string("cannot ") + operation +
                             " in the current target-sequence state");
    }
  }

  void abort_unlocked() noexcept {
    if (state_ == TargetSequenceState::Committed ||
        state_ == TargetSequenceState::Cancelled) {
      return;
    }
    pending_.reset();
    stages_.clear();
    rows_.clear();
    hidden_rows_carrier_ = at::Tensor();
    anchor_rows_carrier_ = at::Tensor();
    mailbox_ = TargetSequenceExpertMailbox{};
    pending_pilot_.reset();
    decisions_.clear();
    dspark_rows_ = at::Tensor();
    captured_dspark_layers_ = 0;
    state_ = TargetSequenceState::Cancelled;
  }

  TargetSequenceMode mode_ = TargetSequenceMode::Prefill;
  bool exact_k3_ = true;
  bool capture_dspark_rows_ = false;
  bool full_commit_only_ = false;
  bool dspark_capture_failed_ = false;
  const TargetTailWeights* tail_ = nullptr;
  const TargetPilotRoster* pilot_routers_ = nullptr;
  std::size_t position_count_ = 0;
  std::array<TargetLayerCacheBinding, kTargetLayerCount> caches_{};
  std::array<std::uint64_t, kTargetLayerCount> initial_versions_{};
  std::vector<SequenceRow> rows_;
  /* T>1 row views are compatibility handles only. These two contiguous
   * matrices remain the authoritative live layer state, so attention and the
   * tail never have to reconstruct them with one cat per carrier. */
  at::Tensor hidden_rows_carrier_;
  at::Tensor anchor_rows_carrier_;
  std::vector<std::unique_ptr<SequenceCacheStage>> stages_;
  std::unique_ptr<PendingSequenceLayer> pending_;
  TargetSequenceExpertMailbox mailbox_{};
  std::optional<PilotPredictionRows> pending_pilot_;
  // Generation of the pilot-mailbox publish for pending_pilot_ (sync-E
  // removal, K3_PILOT_MAILBOX=1); 0 = not published, consumer drains.
  std::uint64_t pilot_mailbox_generation_ = 0;
  std::vector<std::uint32_t> decisions_;
  at::Tensor dspark_rows_;
  std::size_t captured_dspark_layers_ = 0;
  std::uint32_t next_layer_ = 0;
  TargetSequenceState state_ = TargetSequenceState::Active;
  TargetSequenceStats stats_{};
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
  /* Step 6 (K3_KDA_PRECOMMIT): weight/norm refs captured at each KDA
   * prepare so the NEXT token's tail can pre-commit that layer's loop CB
   * before its own prepare call arrives. Tensors are refcounted views of
   * resident spine storage; the cache-version check at pre-commit time
   * guards staleness. */
  bool last_tile_tail_on_loop_ = false;
  LoopKdaPendingHandle* kda_precommit_handle_ = nullptr;
  std::uint32_t kda_precommit_layer_ = 0;
  at::Tensor kda_precommit_next_anchors_;
  /* Step 4 chain-lite: the pending MLA layer CB plus the transaction and
   * decode shell created at precommit time. The pending nonce lives on
   * the transaction's PRIVATE working cache — the session base stays
   * clean across the parked window. Abandon order: drain the CB FIRST
   * (it writes the shell's slab), then cancel the shell, then discard
   * the transaction. */
  LoopMlaPendingHandle* mla_precommit_handle_ = nullptr;
  std::uint32_t mla_precommit_layer_ = 0;
  std::unique_ptr<MlaCacheTransaction> mla_precommit_txn_;
  std::unique_ptr<MlaDecodeShell> mla_precommit_shell_;
#endif
  mutable std::mutex mutex_;
};

TargetSequenceTape::TargetSequenceTape(
    const TargetPositionBindings& bindings, at::Tensor input_hidden_rows,
    const TargetSequenceMode mode, const bool capture_dspark_rows,
    const bool full_commit_only)
    : impl_(std::make_unique<Impl>(bindings, std::move(input_hidden_rows),
                                   mode, capture_dspark_rows,
                                   full_commit_only)) {}

TargetSequenceTape::~TargetSequenceTape() = default;

TargetSequenceLayerPrepareKind
TargetSequenceTape::prepare_layer(const TargetLayerBinding& layer) {
  return impl_->prepare_layer(layer);
}

void TargetSequenceTape::precommit_arena_barrier() noexcept {
  impl_->precommit_arena_barrier();
}

std::uint32_t TargetSequenceTape::precommit_wanted() const noexcept {
  return impl_->precommit_wanted();
}

void TargetSequenceTape::try_precommit_next(
    const TargetLayerBinding& binding) noexcept {
  impl_->try_precommit_next(binding);
}

TargetSequenceExpertMailbox TargetSequenceTape::expert_mailbox() const {
  return impl_->expert_mailbox();
}

TargetSequencePrefetchHint
TargetSequenceTape::take_prefetch_hint() noexcept {
  return impl_->take_prefetch_hint();
}

void TargetSequenceTape::finish_expert_row(
    const std::uint16_t row_index, const std::uint64_t spine_generation,
    const CanonicalExpertBatchT1& experts, const MoeRunOptions& options) {
  impl_->finish_expert_row(row_index, spine_generation, experts, options);
}

void TargetSequenceTape::finish_expert_tile(
    const std::uint16_t first_row, const std::uint16_t row_count,
    const std::uint64_t spine_generation,
    const CanonicalExpertPositionTileT1& experts,
    const MoeRunOptions& options) {
  impl_->finish_expert_tile(first_row, row_count, spine_generation, experts,
                            options);
}

std::span<const std::uint32_t> TargetSequenceTape::finish_tail() {
  return impl_->finish_tail();
}

at::Tensor TargetSequenceTape::dspark_target_rows() const {
  return impl_->dspark_target_rows();
}

void TargetSequenceTape::commit_all() {
  impl_->commit_prefix(impl_->position_count());
}

void TargetSequenceTape::commit_prefix(const std::size_t positions) {
  impl_->commit_prefix(positions);
}

void TargetSequenceTape::cancel() { impl_->cancel(); }

TargetSequenceState TargetSequenceTape::state() const {
  return impl_->state();
}

TargetSequenceMode TargetSequenceTape::mode() const noexcept {
  return impl_->mode();
}

std::size_t TargetSequenceTape::position_count() const noexcept {
  return impl_->position_count();
}

std::uint32_t TargetSequenceTape::next_layer_index() const {
  return impl_->next_layer_index();
}

TargetSequenceStats TargetSequenceTape::stats() const {
  return impl_->stats();
}

}  // namespace deltafin::provider_internal
