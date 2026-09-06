#ifndef DELTAFIN_PROVIDER_KDA_BATCH_H
#define DELTAFIN_PROVIDER_KDA_BATCH_H

#include "provider_kda.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace deltafin::provider_internal {

constexpr std::uint32_t kKdaBatchMaximumPositions = 64;

enum class KdaBatchProjectionPath : std::uint32_t {
  ThreeWayAdjacent = 1,
  Separate = 4,
};

/*
 * Row-independent KDA input projections for a bounded sequence tile.
 *
 * The KDA convolution windows and recurrent state remain causally ordered and
 * are deliberately outside this primitive. The public prompt schedule issues
 * separate T-wide query, key, and value projections, followed by T-wide
 * feature-A, feature-B, and beta projections. Physical weight adjacency is
 * not sufficient to fuse those calls: doing so can select a different backend
 * reduction schedule. Output-gate projection deliberately remains in the
 * post-recurrence stage below.
 */
struct KdaBatchInputProjections {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  KdaBatchProjectionPath path = KdaBatchProjectionPath::Separate;
  std::uint32_t positions = 0;
  std::uint32_t provider_dispatches = 0;
  std::uint32_t equivalent_rowwise_dispatches = 0;
  std::uint32_t established_separate_rowwise_dispatches = 0;
};

struct KdaBatchDependentProjections {
  at::Tensor feature_a;
  at::Tensor feature_b;
  at::Tensor beta;
  std::uint32_t positions = 0;
  std::uint32_t dependent_provider_dispatches = 0;
  std::uint32_t dependent_equivalent_rowwise_dispatches = 0;
};

KdaBatchInputProjections kda_project_inputs_batch(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    bool exact_k3);

KdaBatchDependentProjections kda_project_dependent_batch(
    const at::Tensor& hidden_rows, const KdaWeights& weights,
    bool exact_k3);

struct KdaBatchOutputProjection {
  at::Tensor output;
  std::uint32_t positions = 0;
  std::uint32_t provider_dispatches = 0;
  std::uint32_t equivalent_rowwise_dispatches = 0;
};

/* K3_KDA_WIDE_FUSED=1: the batched short conv + per-position recurrence +
 * output norm/gate/projection as one fused Metal kernel on the ATen stream
 * (Apple, exact K3, 2..16 positions). Returns nullopt when ineligible so the
 * caller runs the stock batched path. Boundary states mirror the stock path:
 * per-position conv-window views of the source plus the recurrent state. */
struct KdaWideFusedResult {
  at::Tensor output;
  KdaState final_state;
  std::vector<KdaState> boundaries;
  std::uint32_t provider_dispatches = 0;
};
std::optional<KdaWideFusedResult> kda_wide_fused_positions(
    const at::Tensor& normalized_hidden, const KdaWeights& weights,
    const KdaState& state, const KdaBatchInputProjections& batch,
    const KdaBatchDependentProjections& dependent, bool retain_boundaries,
    bool exact_k3);

/*
 * Live-order post-recurrence stage: one T-wide full-rank output-gate
 * projection, T-wide output norm/gating, then one T-wide final O projection.
 */
KdaBatchOutputProjection kda_finish_output_batch(
    const at::Tensor& hidden_rows,
    const at::Tensor& recurrent_output_rows, const KdaWeights& weights,
    bool exact_k3);

}  // namespace deltafin::provider_internal

#endif
