#ifndef DELTAFIN_PROVIDER_TARGET_TAPE_H
#define DELTAFIN_PROVIDER_TARGET_TAPE_H

#include "provider_kda.h"
#include "provider_mla.h"
#include "provider_moe.h"
#include "provider_pilot.h"
#include "provider_target.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>

namespace deltafin::provider_internal {

constexpr std::size_t kTargetLayerCount = 93;
constexpr std::size_t kTargetKdaLayerCount = 69;
constexpr std::size_t kTargetMlaLayerCount = 24;
using TargetPilotRoster =
    std::array<std::optional<PilotRouterT1>, kTargetLayerCount>;

enum class TargetTapeContract : std::uint32_t {
  ExactK3 = 1,
  SyntheticK3Schedule = 2,
};

enum class TargetAttentionKind : std::uint32_t {
  Kda = 1,
  Mla = 2,
};

enum class TargetPositionState : std::uint32_t {
  Active = 1,
  WaitingForExperts = 2,
  ReadyForTail = 3,
  Committed = 4,
  Cancelled = 5,
  Poisoned = 6,
};

/* Persistent KDA state owned by the future provider session. */
struct TargetKdaCache {
  std::uint32_t layer_index = 0;
  std::uint64_t version = 0;
  KdaState state;
};

/* Cache objects persist for the session; streamed layer weights do not. */
struct TargetLayerCacheBinding {
  std::uint32_t layer_index = 0;
  TargetAttentionKind attention_kind = TargetAttentionKind::Kda;
  TargetKdaCache *kda_cache = nullptr;
  MlaCache *mla_cache = nullptr;
};

/*
 * One currently bound resident/transient layer. None of these pointers is
 * retained after prepare_layer returns. For a routed layer, the tape takes a
 * tensor-handle copy of its one MoeSpineT1 so that only that layer stays pinned
 * across synchronous expert I/O.
 */
struct TargetLayerBinding {
  std::uint32_t layer_index = 0;
  TargetAttentionKind attention_kind = TargetAttentionKind::Kda;
  const TargetResidualWeights *residual = nullptr;
  const KdaWeights *kda_weights = nullptr;
  const MlaWeights *mla_weights = nullptr;
  const MlaInputBundle *mla_input_bundle = nullptr;
  /* Stable spine-form MLA weights for K3_MLA_LOOP=parity only: when the
   * fp32 execution arena is active, mla_weights are per-layer arena VIEWS
   * (DenseF32, recycled every layer) and mla_input_bundle is nulled — the
   * loop parity must bind the resident row-int8 forms instead (the KDA
   * takeover's "bind resident int8, not the arena" lesson). Null unless
   * parity mode is on and the spine carries the qualified int8 bundle. */
  const MlaWeights *mla_parity_weights = nullptr;
  const MlaInputBundle *mla_parity_bundle = nullptr;
  const TargetDenseWeights *dense = nullptr;
  const MoeSpineT1 *moe = nullptr;
};

struct TargetPositionBindings {
  TargetTapeContract contract = TargetTapeContract::ExactK3;
  std::span<const TargetLayerCacheBinding> caches;
  const TargetTailWeights *tail = nullptr;
  /* Optional scheduling-only roster. The pointed-to array is session-owned,
   * address-stable, and may fill immutable layer slots after tape creation. */
  const TargetPilotRoster *pilot_routers = nullptr;
};

/* The only value that crosses the expert-I/O split. */
struct TargetRouteRequest {
  std::uint32_t layer_index = 0;
  std::uint64_t spine_generation = 0;
  MoeRouteT1 route;
};

enum class TargetLayerPrepareKind : std::uint32_t {
  DenseCompleted = 1,
  ExpertsRequired = 2,
};

struct TargetLayerPrepareResult {
  TargetLayerPrepareKind kind = TargetLayerPrepareKind::DenseCompleted;
  TargetRouteRequest route;
};

/*
 * One coarse, transactional decode position.  No cache update is published
 * while routes and expert bytes are being processed.  finish_greedy computes
 * the target-model logits internally, extracts one greedy token ID, preflights
 * every staged cache, and only then publishes all 93 attention updates.
 */
class TargetPositionTape {
public:
  /*
   * input_hidden is the exact fp32 row produced by the provider-owned BF16
   * embedding reader. The tape deliberately does not retain a full embedding
   * table merely to begin one position.
   */
  TargetPositionTape(const TargetPositionBindings &bindings,
                     at::Tensor input_hidden);
  ~TargetPositionTape();

  TargetPositionTape(const TargetPositionTape &) = delete;
  TargetPositionTape &operator=(const TargetPositionTape &) = delete;
  TargetPositionTape(TargetPositionTape &&) = delete;
  TargetPositionTape &operator=(TargetPositionTape &&) = delete;

  /*
   * Consumes exactly the next streamed layer. Layer zero completes inline;
   * layers 1..92 return the sole route value needed by the Rust expert reader.
   */
  [[nodiscard]] TargetLayerPrepareResult
  prepare_layer(const TargetLayerBinding &layer);

  void finish_moe_layer(std::uint32_t layer_index,
                        std::uint64_t spine_generation,
                        const CanonicalExpertBatchT1 &experts,
                        const MoeRunOptions &options);

  [[nodiscard]] std::uint32_t finish_greedy();
  void cancel();

  [[nodiscard]] TargetPositionState state() const;
  [[nodiscard]] std::uint32_t next_layer_index() const;
  [[nodiscard]] std::size_t staged_kda_count() const;
  [[nodiscard]] std::size_t staged_mla_count() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] constexpr bool
target_layer_uses_mla(const std::uint32_t layer_index) {
  const std::uint32_t ordinal = layer_index + 1;
  return ordinal == kTargetLayerCount || ordinal % 4 == 0;
}

} // namespace deltafin::provider_internal

#endif
