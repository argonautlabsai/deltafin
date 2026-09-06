#ifndef DELTAFIN_PROVIDER_TARGET_SEQUENCE_H
#define DELTAFIN_PROVIDER_TARGET_SEQUENCE_H

#include "provider_target_tape.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace deltafin::provider_internal {

constexpr std::size_t kTargetSequenceMaxPositions = 64;
constexpr std::array<std::uint32_t, 5> kDSparkTargetCaptureLayers{
    2, 23, 47, 71, 89};

enum class TargetSequenceMode : std::uint32_t {
  /* Ordinary prompt ingestion: only the complete chunk may be committed. */
  Prefill = 1,
  /* Wide target verification: any completed row prefix may be committed. */
  Verify = 2,
};

enum class TargetSequenceState : std::uint32_t {
  Active = 1,
  WaitingForExperts = 2,
  ReadyForTail = 3,
  ReadyToCommit = 4,
  Committed = 5,
  Cancelled = 6,
  Poisoned = 7,
};

/*
 * Fixed provider-owned route mailbox.  Rust eventually needs only the IDs and
 * raw fp32 weight bits; the routed activation remains an ATen tensor owned by
 * the linked provider.  A row is intentionally one bounded expert tile: only
 * its canonical 16-expert set has to be resident while it executes, never the
 * potentially hundreds-of-experts union across a 64-position prompt chunk.
 */
struct TargetSequenceRouteRow {
  std::uint16_t row_index = 0;
  MoeRouteT1 route;
  at::Tensor routed_input;
};

struct TargetSequenceExpertMailbox {
  std::uint32_t layer_index = 0;
  std::uint64_t spine_generation = 0;
  std::uint16_t row_count = 0;
  std::array<TargetSequenceRouteRow, kTargetSequenceMaxPositions> rows{};
};

/* Scheduling-only disk-read hint. IDs are canonical ascending and never
 * replace, reorder, or weight the authoritative mailbox route. */
struct TargetSequencePrefetchHint {
  std::uint32_t source_layer = 0;
  std::uint32_t target_layer = 0;
  std::uint16_t expert_count = 0;
  std::array<std::uint16_t, kPilotMaxPrefetch> expert_ids{};
};

enum class TargetSequenceLayerPrepareKind : std::uint32_t {
  DenseCompleted = 1,
  ExpertRowsRequired = 2,
};

struct TargetSequenceStats {
  std::uint64_t positions = 0;
  std::uint64_t streamed_layer_passes = 0;
  std::uint64_t attention_rows = 0;
  std::uint64_t expert_row_requests = 0;
  std::uint64_t expert_rows_completed = 0;
  std::uint64_t expert_tiles_completed = 0;
  std::uint64_t moe_partial_dispatches = 0;
  std::uint64_t tail_rows = 0;
  std::uint64_t tail_provider_dispatches = 0;
  std::uint64_t maximum_live_streamed_layers = 0;
  std::uint64_t maximum_experts_per_request = 0;
  std::uint64_t maximum_positions_per_expert_tile = 0;
  std::uint64_t staged_kda_storage_bytes = 0;
  std::uint64_t verify_snapshot_bytes = 0;
  std::uint64_t projected_mla_storage_bytes = 0;
  std::uint64_t additional_mla_storage_bytes = 0;
  /* Internal qualification counters; the stable public ABI ignores these. */
  std::uint64_t dense_mlp_provider_dispatches = 0;
  std::uint64_t dense_mlp_rows = 0;
  std::uint64_t kda_input_provider_dispatches = 0;
  std::uint64_t kda_input_equivalent_rowwise_dispatches = 0;
  std::uint64_t kda_dependent_provider_dispatches = 0;
  std::uint64_t kda_dependent_equivalent_rowwise_dispatches = 0;
  std::uint64_t kda_shortconv_provider_dispatches = 0;
  std::uint64_t kda_recurrent_rows = 0;
  std::uint64_t kda_output_provider_dispatches = 0;
  std::uint64_t kda_output_rows = 0;
  std::uint64_t mla_position_provider_dispatches = 0;
  std::uint64_t mla_position_rows = 0;
  std::uint64_t moe_prepare_provider_dispatches = 0;
  std::uint64_t moe_prepare_rows = 0;
  std::uint64_t moe_router_dispatches = 0;
  std::uint64_t moe_routed_down_dispatches = 0;
  std::uint64_t moe_shared_dispatches = 0;
  std::uint64_t moe_route_materializations = 0;
  std::uint64_t moe_route_host_transfers = 0;
  std::uint64_t moe_routed_input_host_transfers = 0;
  /// K3_ROUTED_INPUT_ALIAS=1: transfers served by a unified-memory alias.
  std::uint64_t moe_routed_input_host_aliases = 0;
  std::uint64_t moe_complete_provider_dispatches = 0;
  std::uint64_t moe_complete_rows = 0;
  std::uint64_t moe_routed_up_dispatches = 0;
  std::uint64_t pilot_prediction_dispatches = 0;
  std::uint64_t pilot_prediction_rows = 0;
  std::uint64_t pilot_hint_issues = 0;
  std::uint64_t pilot_hint_experts = 0;
  std::uint64_t pilot_max_union_candidates = 0;
  std::uint64_t pilot_score_materializations = 0;
  std::uint64_t pilot_score_elisions = 0;
};

/*
 * A compiled, layer-major transaction over 1..64 target positions.  The
 * caller supplies one resident/transient layer at a time; prepare_layer uses
 * it for every row before the caller advances the spine stream.  No pointer to
 * a KDA/MLA/residual layer binding survives that call.  Routed layers retain a
 * tensor-handle copy of only their one MoeSpineT1 until all row expert tiles
 * complete.
 *
 * Persistent caches are never mutated during compute.  Prefill publishes all
 * rows only after every layer and tail succeeds.  Verify retains KDA row
 * boundaries and MLA's contiguous candidate rows, allowing an accepted prefix
 * to publish without replaying target math.  The memory cost of those KDA
 * boundaries is reported exactly in stats().verify_snapshot_bytes.  The
 * optional full-commit-only Verify contract instead retains exactly one final
 * KDA state per layer and rejects every partial publication.
 */
class TargetSequenceTape {
 public:
  TargetSequenceTape(const TargetPositionBindings& bindings,
                     at::Tensor input_hidden_rows, TargetSequenceMode mode,
                     bool capture_dspark_rows = false,
                     bool full_commit_only = false);
  ~TargetSequenceTape();

  TargetSequenceTape(const TargetSequenceTape&) = delete;
  TargetSequenceTape& operator=(const TargetSequenceTape&) = delete;
  TargetSequenceTape(TargetSequenceTape&&) = delete;
  TargetSequenceTape& operator=(TargetSequenceTape&&) = delete;

  [[nodiscard]] TargetSequenceLayerPrepareKind
  prepare_layer(const TargetLayerBinding& layer);

  [[nodiscard]] TargetSequenceExpertMailbox expert_mailbox() const;
  [[nodiscard]] TargetSequencePrefetchHint take_prefetch_hint() noexcept;
  /* Step 6 arena barrier: waits out any pre-committed KDA loop CB before
   * the caller rewrites fp32 execution-arena storage. No-op without a
   * pending pre-commit. Never throws. */
  void precommit_arena_barrier() noexcept;
  /* Step 6 pre-commit orchestration (runtime-driven so the weights are
   * fresh per-owner views, never cached across tapes): after a loop tail
   * completes a layer, precommit_wanted() names the successor KDA layer
   * eligible for early encoding (UINT32_MAX = none), and the runtime calls
   * try_precommit_next() with a freshly built binding for that layer —
   * the same views prepare_layer will later resolve (cache hit), so the
   * pre-computed math is bit-equal to the synchronous path. */
  [[nodiscard]] std::uint32_t precommit_wanted() const noexcept;
  void try_precommit_next(const TargetLayerBinding& binding) noexcept;
  void finish_expert_row(std::uint16_t row_index,
                         std::uint64_t spine_generation,
                         const CanonicalExpertBatchT1& experts,
                         const MoeRunOptions& options);
  void finish_expert_tile(
      std::uint16_t first_row, std::uint16_t row_count,
      std::uint64_t spine_generation,
      const CanonicalExpertPositionTileT1& experts,
      const MoeRunOptions& options);

  [[nodiscard]] std::span<const std::uint32_t> finish_tail();
  [[nodiscard]] at::Tensor dspark_target_rows() const;
  void commit_all();
  void commit_prefix(std::size_t positions);
  void cancel();

  [[nodiscard]] TargetSequenceState state() const;
  [[nodiscard]] TargetSequenceMode mode() const noexcept;
  [[nodiscard]] std::size_t position_count() const noexcept;
  [[nodiscard]] std::uint32_t next_layer_index() const;
  [[nodiscard]] TargetSequenceStats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deltafin::provider_internal

#endif
