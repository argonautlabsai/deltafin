#ifndef DELTAFIN_PROVIDER_LOOP_H
#define DELTAFIN_PROVIDER_LOOP_H

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "provider_kda.h"
#include "provider_mla.h"
#include "provider_route_mailbox.h"

namespace deltafin::provider_internal {

/*
 * Bespoke decode-loop scaffold (K3_BESPOKE_LOOP=1, default off).
 *
 * This is worklist step 2 of the bespoke decode-loop design: a LoopEncoder
 * that owns a serial MTLCommandQueue, an MTLSharedEvent pool, two 192-byte
 * shared-storage route mailboxes (RouteMailboxT1 layout reused verbatim), a
 * device weights-table cache for spine tensor lookups, and an embedded
 * precompiled metallib.  No decode-path hook exists yet: the only kernels in
 * provider_loop_kernels.metal are deterministic canary copies that prove the
 * queue/event/mailbox/weights-table plumbing end-to-end.  The validated
 * k3-proto kernel suite replaces them in worklist steps 3-5.
 */

constexpr std::uint32_t kLoopMetalAbiV1 = 1;
constexpr std::uint32_t kLoopMetalEmbeddedLibraryV1 = 1U << 0;
constexpr std::uint32_t kLoopMetalOwnedSerialQueueV1 = 1U << 1;
constexpr std::uint32_t kLoopMetalSharedEventPoolV1 = 1U << 2;
constexpr std::uint32_t kLoopMetalSharedMailboxV1 = 1U << 3;
constexpr std::uint32_t kLoopMetalWeightsTableV1 = 1U << 4;
constexpr std::uint32_t kLoopMetalStorageOffsetV1 = 1U << 5;
constexpr std::uint32_t kLoopMetalRequiredCapabilitiesV1 =
    kLoopMetalEmbeddedLibraryV1 | kLoopMetalOwnedSerialQueueV1 |
    kLoopMetalSharedEventPoolV1 | kLoopMetalSharedMailboxV1 |
    kLoopMetalWeightsTableV1 | kLoopMetalStorageOffsetV1;

/* Loop-owned shared events: route readback, ATen-boundary fencing, DSpark
 * capture, and one spare, per the blueprint's command-buffer plan. */
constexpr std::uint32_t kLoopMetalSharedEventPoolSizeV1 = 4;

struct LoopMetalCapabilities {
  std::uint32_t abi_version = 0;
  std::uint32_t flags = 0;
  std::uint32_t event_pool_size = 0;
  std::uint32_t reserved = 0;
};

struct LoopMetalCanaryReport {
  std::uint32_t top_k = 0;
  std::uint32_t route_ids_matched = 0;
  std::uint32_t route_weights_matched = 0;
  std::uint32_t pilot_ids_matched = 0;
  std::uint32_t pilot_weights_matched = 0;
  std::uint32_t command_buffers = 0;
  std::uint32_t events_signaled = 0;
  std::uint32_t weights_table_hits = 0;
  std::uint32_t weights_table_invalidations = 0;
  std::uint32_t id_offset_elements = 0;
  std::uint32_t weight_offset_elements = 0;
  std::uint32_t reserved = 0;
  double max_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  /* MLA takeover Step 1: the ported fused attention core executed through
   * the LOOP's own encoder at production head geometry (96 heads x 192 key /
   * 128 value dims), compared against a double-precision reference computed
   * on the same inputs. Step 0 proved the kernels are present in the loop
   * metallib; this proves the loop's binding, threadgroup and dispatch path
   * drives them correctly, which is the part that is genuinely new — the
   * standalone library has its own encoder that this does not share. */
  std::uint32_t mla_heads_checked = 0;
  double mla_attention_relative_l2 = 0.0;
};

/* The scaffold canary copies bits through the loop queue, so it must be
 * byte-exact; the relL2 bound is the loop parity class the real kernels are
 * held to, kept here so the pass predicate does not change when they land. */
constexpr double kLoopMetalCanaryMaxRelL2 = 1.0e-6;

/* The fused MLA core accumulates in fp32 against an fp64 reference, so it is
 * held to the kernel's own validated bound rather than the bit-copy bound
 * above — the standalone library qualifies at ~1.6e-7. */
constexpr double kLoopMlaAttentionMaxRelL2 = 5.0e-6;

[[nodiscard]] bool loop_metal_canary_passes(const LoopMetalCanaryReport& report);

/* Aggregate counters of the device weights-table cache (spine tensor
 * lookups keyed by layer index, spine generation, and tensor slot). */
struct LoopWeightsTableStats {
  std::uint64_t entries = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t generation_invalidations = 0;
};

#if defined(__APPLE__)

/* Load and validate the loop-owned queue, event pool, mailboxes, and the
 * embedded provider_loop_kernels metallib on the current MPS device. */
[[nodiscard]] LoopMetalCapabilities loop_metal_capabilities_v1();

/*
 * Deterministic end-to-end scaffold qualification: two command buffers on the
 * loop-owned serial queue copy 16 expert ids and 16 fp32 weight bit patterns
 * from offset MPS tensor storages into the route and pilot mailboxes, each
 * command buffer signalling a distinct pool event awaited through the
 * listener+semaphore pattern shared with the route mailbox; then a
 * store/hit/generation-invalidation round trip of the weights table.  All
 * comparisons are byte-exact against a host reference; relL2 is additionally
 * reported against an fp64 widening of the same reference.
 */
[[nodiscard]] LoopMetalCanaryReport loop_metal_canary_v1();

/* True only when K3_BESPOKE_LOOP=1 AND the session-wide one-shot canary
 * qualification passed on this process's MPS device.  Any qualification
 * failure sticks the session to the stock ATen path. */
[[nodiscard]] bool bespoke_loop_decode_ready();

/*
 * Router side-queue canary (K3-SIDEQUEUE-PLAN.md Step 1).  Runs the ported
 * deltafin_loop_gemv_i8_bundle_v1 + deltafin_loop_router_top16_v1 chain on
 * the loop-owned queue and validates it against host references:
 *  - controlled-logit cases (random-with-margin, engineered 20-way tie
 *    block, full 896-way tie, very-negative logits, bias-affects-selection-
 *    only) must be PAIR-EXACT: ids identical to a host reimplementation of
 *    the documented tie rule over the same fp32 values, weights bitwise
 *    identical to the host replaying the kernel's own fp32 gather+renorm;
 *  - the full-chain case (int8 GEMV at the production 896x7168 shape into
 *    the router) holds GEMV logits to the validated <=1e-6 relL2 bound
 *    against an fp64 reference and re-checks pair-exactness against the
 *    GPU's own logits.
 * No decode-path involvement; qualification-only, like loop_metal_canary_v1.
 */
struct LoopRouterCanaryReport {
  std::uint32_t cases_run = 0;
  std::uint32_t cases_pair_exact = 0;
  std::uint32_t gemv_rows_compared = 0;
  std::uint32_t command_buffers = 0;
  std::uint32_t reserved = 0;
  double gemv_relative_l2_error = 0.0;
  double max_weight_absolute_error = 0.0;
};

constexpr double kLoopRouterCanaryMaxGemvRelL2 = 1.0e-6;

[[nodiscard]] bool loop_router_canary_passes(const LoopRouterCanaryReport& report);
[[nodiscard]] LoopRouterCanaryReport loop_router_canary_v1();

/*
 * Router side-queue mode (K3_ROUTE_SIDEQUEUE, K3-SIDEQUEUE-PLAN.md Step 2):
 * 0 = off (default); 1 = "shadow" — the side-queue route is computed and
 * compared against the stock route every layer, stock result used, zero
 * behavior change; 2 = "on" — reserved for the Step 3 takeover and treated
 * as off until it lands.
 */
[[nodiscard]] int route_sidequeue_mode();

/* Aggregate shadow-comparison counters.  An id-set mismatch is the alarm
 * condition (the side-queue route selected a different EXPERT SET than
 * stock); weight deltas up to ~1 ulp are expected (stock's ATen renorm and
 * the kernel's sequential fp32 renorm associate differently). */
struct LoopRouterShadowStats {
  std::uint64_t compared = 0;
  std::uint64_t id_set_mismatches = 0;
  std::uint64_t skipped = 0;
  double max_weight_abs_error = 0.0;
};

/*
 * Begins one shadow route: signals an ATen-side boundary event on the
 * current root command buffer (committed via the same non-blocking
 * COMMIT_AND_CONTINUE the route path already uses), then encodes
 * GEMV -> router_top16 on the loop queue behind encodeWaitForEvent.
 * Returns false (and counts a skip) when inputs do not qualify or a prior
 * shadow is still unpaired.  `compare` waits for the side-queue result,
 * order-blind-compares it against the stock route, and accumulates stats
 * (immediate stderr alarm on any id-set mismatch; periodic summary).
 */
[[nodiscard]] bool loop_router_shadow_begin(std::uint32_t layer_index,
                                            const at::Tensor& hidden,
                                            const at::Tensor& quantized,
                                            const at::Tensor& row_scales,
                                            const at::Tensor& bias);

/* BF16-router variant: `weight_bits` is the ExactBf16Storage slab tensor
 * (MPS bf16/int16 bits, contiguous), `element_offset` the router matrix's
 * origin within it.  The side queue runs the verbatim stock GEMV math, so
 * logits are bit-identical to the stock path and the id-set comparison is
 * exact by construction. */
[[nodiscard]] bool loop_router_shadow_begin_bf16(std::uint32_t layer_index,
                                                 const at::Tensor& hidden,
                                                 const at::Tensor& weight_bits,
                                                 std::size_t element_offset,
                                                 const at::Tensor& bias);

/* Dense-fp32 router variant (MoeRowInt8Matrix::dense_f32 materialization —
 * the live form on hosts where the provider has not passed the packed-int8
 * gate).  The stock path runs at::linear here, so logits differ from the
 * side queue at fp32 rounding level; the shadow soak measures the resulting
 * near-tie id-set divergence rate — exactly the takeover-gate data. */
[[nodiscard]] bool loop_router_shadow_begin_f32(std::uint32_t layer_index,
                                                const at::Tensor& hidden,
                                                const at::Tensor& dense_f32,
                                                const at::Tensor& bias);
void loop_router_shadow_compare(std::uint32_t layer_index,
                                const std::uint16_t* expert_ids,
                                const std::uint32_t* weight_bits);
[[nodiscard]] LoopRouterShadowStats loop_router_shadow_stats();

/*
 * Takeover consume half (K3_ROUTE_SIDEQUEUE=on, plan Step 3): waits the
 * pending side-queue completion event — a wait that covers only the
 * committed prefix plus the router chain, NOT the ATen queue's projection
 * backlog — and copies the route out of the loop mailbox.  Returns false
 * (fail-soft) on timeout, command error, or an out-of-range id; the caller
 * must then run the stock route chain instead.  Under K3_ROUTE_SYNC_PROBE=1
 * it aggregates the wait and prints a per-token mean beside the stock
 * probe's numbers.
 */
[[nodiscard]] bool loop_router_collect(std::uint32_t layer_index,
                                       std::uint16_t* expert_ids_out,
                                       std::uint32_t* weight_bits_out);

/*
 * Fused-KDA-core canary (loop plan Step 3a, first gate): runs the ported
 * deltafin_loop_kda_core_v1 on the loop queue against an fp64 host
 * reference of the full fused chain (conv4+SiLU, per-head L2, decay, delta
 * rule, RMS x o_norm x gate) over three cases — random, zero-state, and
 * saturated decay — comparing out, S_out, and convout at the kernel's
 * validated 5e-6 relL2 bound.  Qualification-only; no decode-path use yet.
 */
struct LoopKdaCanaryReport {
  std::uint32_t cases_run = 0;
  std::uint32_t cases_passed = 0;
  std::uint32_t command_buffers = 0;
  std::uint32_t reserved = 0;
  double max_out_relative_l2 = 0.0;
  double max_state_relative_l2 = 0.0;
  double max_conv_relative_l2 = 0.0;
};

constexpr double kLoopKdaCanaryMaxRelL2 = 5.0e-6;

[[nodiscard]] bool loop_kda_canary_passes(const LoopKdaCanaryReport& report);
[[nodiscard]] LoopKdaCanaryReport loop_kda_canary_v1();

/*
 * KDA loop mode (K3_KDA_LOOP, plan Step 3a): 0 = off (default); 1 =
 * "parity" — the full KDA decode chain (per-projection GEMVs + fused core
 * + o_proj) runs on the loop queue behind the proven ATen->loop fence for
 * every T=1 KDA layer and is compared against the stock kda_decode_one
 * result (output + all four next-state tensors), stock results used, zero
 * behavior change; 2 = "on" reserved for the takeover.
 */
[[nodiscard]] int kda_loop_mode();

/*
 * Chain-safety marker for the shared-boundary experiment
 * (K3_KDA_SHARED_BOUNDARY=1). True while every input a KDA loop CB reads is
 * the product of a host-waited loop command buffer or resident storage —
 * i.e. the previous MoE layer completed via loop_moe_tail. Any layer that
 * completes through the stock ATen path must clear it so the next KDA fence
 * is taken privately again. Off by default; scheduling-only, exactness
 * unaffected (the private fence remains on anchor layers and after any
 * stock completion).
 */
void loop_set_chain_safe(bool safe) noexcept;

/*
 * One-shot marker set by the sequence immediately after it encodes a fresh
 * ATen anchor concatenation (the every-kLoopResidualBlock at::cat). The next
 * KDA loop CB consumes it and takes the private ATen fence for that layer;
 * all other layers read only anchor tensors that were completed many fences
 * ago. Pairs with K3_KDA_SHARED_BOUNDARY.
 */
void loop_note_fresh_anchor_cat() noexcept;
/* Marks a fresh fp32-arena dequant encode on the ATen stream. Consumed by
 * the next fp32-form loop CB's fence decision (int8-form CBs neither fence
 * for it nor consume it) or by the pre-commit drain. */
void loop_note_fresh_dequant() noexcept;
/* Host-wait the current ATen MPS stream (COMMIT_AND_WAIT). Used by the
 * pre-commit orchestration to complete a freshly encoded dequant inside the
 * inter-layer host gap, so the chained loop CBs can skip the ATen fence and
 * start on the still-warm queue. */
void loop_drain_aten_stream() noexcept;

/* K3_ROUTED_INPUT_ALIAS=1: returns a CPU fp32 tensor that ALIASES the
 * unified-memory MTLBuffer behind a contiguous fp32 MPS tensor, after a
 * timed COMMIT_AND_WAIT drain of the ATen stream. Replaces the per-layer
 * `.to(kCPU)` blit round trip (measured 1.25 ms/layer, [kernel-sub]
 * materialize). The alias keeps the device tensor alive; the caller must
 * not enqueue device writes to it while the alias is read. Throws when the
 * storage is not shared-mode or the tensor is not a plain contiguous fp32
 * MPS tensor — callers fall back to the copy. */
at::Tensor loop_host_alias_of_mps(const at::Tensor& tensor);
/* Attention-timer split diagnostics (K3_GPU_TIMELINE=1): the stock ATen
 * prepare path never touches the loop queue, so its command buffers were
 * invisible to the GPU timeline. Note the ATen stream's current root CB
 * under `klass` (once per CB) and flush the timeline report from the
 * prepare export's own cadence. No-ops unless the timeline is on. */
void loop_gpu_timeline_note_aten(const char* klass) noexcept;
void loop_gpu_timeline_flush() noexcept;
/* K3_GPU_TIMELINE: decompose one ATen host sync into queue latency / GPU
 * execution / completion latency ([aten-sync] every 930 syncs). begin()
 * returns a retained token (or nullptr) that end() consumes; stamps are
 * loop_aten_sync_now_ns(). All no-ops unless the timeline is on. */
std::uint64_t loop_aten_sync_now_ns() noexcept;
void* loop_aten_sync_begin() noexcept;
void loop_aten_sync_end(void* token, std::uint64_t began_ns,
                        std::uint64_t ended_ns) noexcept;
void loop_aten_sync_note_prep(std::uint64_t ns) noexcept;
void loop_aten_sync_note_second(std::uint64_t ns) noexcept;
/* K3_KDA_WIDE_FUSED=1: fused wide-tile KDA (short conv + recurrence + output
 * norm/gate for 2..16 positions) encoded on the current ATen MPS stream.
 * All tensors contiguous fp32 MPS; s_bound may be undefined when !retain. */
bool loop_kda_wide_fused_enabled() noexcept;
/* K3_PREP_SPLIT_COMMITS=1 diagnostic: commit the ATen stream at a phase
 * boundary (no wait). */
void loop_aten_stream_commit_phase(const char* phase) noexcept;
/* K3_MOE_ATEN_STREAM=1: expert flat position batch encoded on the current
 * MPS stream with device x/out; returns the bridge status (0 = ok). */
int loop_moe_positions_flat_on_aten_stream(
    const at::Tensor& x, const at::Tensor& out,
    const std::uint8_t* const* expert_blobs, int n_edges,
    const int* position_offsets, int n_positions, const float* weights,
    bool synchronize);
/* K3_RESIDENCY_SET=1: pin spine allocations (heap or buffer) in one
 * MTLResidencySet attached to the ATen, loop and expert queues. */
bool loop_residency_set_enabled() noexcept;
int loop_residency_set_level() noexcept;
bool loop_residency_expert_buffers_enabled() noexcept;
void loop_residency_attach_queue(void* queue) noexcept;
void loop_residency_register_tensor(const at::Tensor& tensor) noexcept;
/* Level >= 2 only: activation heaps seen by the Metal bridges. */
void loop_residency_register_activation(const at::Tensor& tensor) noexcept;
void loop_residency_register_allocation(void* mtl_buffer) noexcept;
bool loop_residency_activations_enabled() noexcept;
void loop_kda_wide_fused_on_aten_stream(
    const at::Tensor& src_q, const at::Tensor& src_k, const at::Tensor& src_v,
    const at::Tensor& convw_q, const at::Tensor& convw_k,
    const at::Tensor& convw_v, const at::Tensor& a_log,
    const at::Tensor& dt_bias, const at::Tensor& fb, const at::Tensor& beta,
    const at::Tensor& gate, const at::Tensor& o_norm, const at::Tensor& s_in,
    const at::Tensor& s_out, const at::Tensor& s_bound, const at::Tensor& out,
    std::uint32_t positions, bool retain);
/* Consume the fresh-ATen one-shot (returns its prior value). The pre-commit
 * orchestration calls this after draining: the ordering the marker demands
 * has been satisfied synchronously. */
bool loop_consume_fresh_marker() noexcept;

/*
 * Step 6 (K3_KDA_PRECOMMIT=1): encode + commit one KDA layer's loop CB the
 * moment its inputs are final (the previous MoE tail's host wait), so the
 * GPU executes it under the inter-layer host gap instead of paying the
 * ~10ms queue re-arm at prepare time. The handle owns the committed CB and
 * its output tensors; exactly one may be in flight. collect() host-waits
 * and publishes outputs (consumes). abandon() waits and discards
 * (consumes) — unconditionally safe: the CB writes only fresh tensors,
 * never persistent state. nullptr from precommit = nothing committed
 * (fail-soft; caller proceeds synchronously).
 */
struct LoopKdaPendingHandle;
/* Weight-form view shared by the loop MoE tail and the chained route:
 * dense-f32 / row-int8 / original-BF16 dispatch per matrix. Declared here
 * (ahead of the pre-commit API) and used again by loop_moe_tail below. */
struct LoopMoeMatrixView {
  at::Tensor int8_weight;
  at::Tensor row_scales;
  at::Tensor dense_f32;
  const OriginalBf16Matrix* original_bf16 = nullptr;
};
/* 0 = off, 1 = on (collected results are used), 2 = parity (collected
 * results are compared against the synchronous path and discarded). */
[[nodiscard]] int kda_precommit_mode();
/* K3_KDA_INT8_DIRECT=1: pre-committed loop CBs consume the resident int8
 * spine weights directly (no fp32 arena dependence, no dequant ordering).
 * Reassociation-level numeric change vs the fp32-view path — gated on the
 * regenerated identity reference. */
[[nodiscard]] bool kda_int8_direct_enabled();
[[nodiscard]] bool kda_precommit_enabled();
/* cat_next_anchors_in_cb (K3_LOOP_CAT=1 boundary layers): next_anchors
 * carries the PREVIOUS anchor slab and the CB itself blit-concatenates
 * [next_anchors | hidden] into a fresh slab before any consumer — the
 * every-12-layers at::cat leaves the ATen stream entirely, so those
 * layers keep the shared boundary (and full CB fusion) instead of the
 * private fence the fresh-cat marker used to force. */
[[nodiscard]] LoopKdaPendingHandle* loop_kda_layer_precommit(
    std::uint32_t layer_index, const at::Tensor& hidden,
    const at::Tensor& anchors, const at::Tensor& score_weight,
    const at::Tensor& input_norm, const at::Tensor& prefix_sum,
    const at::Tensor& next_anchors, const at::Tensor& mlp_score_weight,
    const at::Tensor& post_attention_norm, const KdaWeights& weights,
    const KdaState& state_in, const LoopMoeMatrixView& router,
    const at::Tensor& router_bias, bool cat_next_anchors_in_cb);

/* K3_LOOP_CAT=1 read once per process. */
[[nodiscard]] bool loop_cat_enabled() noexcept;
/* Layer whose route CB is in flight on the side queue (chained or begun),
 * 0xFFFFFFFF when none. prepare_moe_t1 skips its own begin on a match. */
[[nodiscard]] std::uint32_t loop_router_pending_layer();
/* True-route hint peek: waits the chained route CB for `layer_index` (the
 * one-shot semaphore is coordinated with collect) and copies the mailbox's
 * 16 expert IDs WITHOUT consuming the route — collect still publishes it
 * for execution. Fail-soft false on mismatch/timeout/CB error. */
[[nodiscard]] bool loop_router_peek_chained(
    std::uint32_t layer_index, std::uint16_t* expert_ids_out) noexcept;
[[nodiscard]] bool loop_kda_layer_collect(LoopKdaPendingHandle* handle,
                                          at::Tensor& normalized_out,
                                          at::Tensor& mlp_normalized_out,
                                          at::Tensor& lookahead_out,
                                          at::Tensor& prefix_out,
                                          KdaDecodeResult& result,
                                          at::Tensor& next_anchors_out);
void loop_kda_layer_abandon(LoopKdaPendingHandle* handle) noexcept;

/*
 * Arena barrier: block until the pre-committed CB has finished executing,
 * WITHOUT consuming the handle (collect still publishes the outputs).
 * Required before any fp32 execution-arena materialization — the arena
 * recycles storage per layer, and rewriting it while the CB reads its
 * weight views is the measured corruption mode. Idempotent.
 */
void loop_kda_layer_wait(LoopKdaPendingHandle* handle) noexcept;

struct LoopKdaParityStats {
  std::uint64_t compared = 0;
  std::uint64_t failures = 0;
  std::uint64_t skipped = 0;
  double max_output_relative_l2 = 0.0;
  double max_state_relative_l2 = 0.0;
};

void loop_kda_parity_compare(std::uint32_t layer_index,
                             const at::Tensor& normalized,
                             const KdaWeights& weights,
                             const KdaState& state_in,
                             const KdaDecodeResult& stock);
[[nodiscard]] LoopKdaParityStats loop_kda_parity_stats();

/*
 * K3_MLA_LOOP mode (read once per process): 0 = off (default), 1 = "parity"
 * — the full ten-op MLA decode chain (Step 2c(ii): bundle GEMV, two
 * rmsnorms, q_b/kv_b GEMVs, kv pack, private-slab prefix copy + new-row
 * placement, flash attention part/combine, sigmoid gate, o_proj) runs on
 * the loop queue behind the proven ATen->loop fence for every T=1 MLA
 * layer and its [7168] output is compared against the stock
 * MlaPreparedDecode.output; stock results used, zero behavior change,
 * fail-soft. 2 = "1"/"on" — reserved for the Step 3 takeover (not built;
 * currently inert). The parity NEVER writes the live KV slab: both paths
 * run during parity, so it copies the committed prefix + its own packed
 * row into a private scratch slab (K3-SIDEQUEUE-PLAN.md Step 2a).
 */
[[nodiscard]] int mla_loop_mode();

struct LoopMlaParityStats {
  std::uint64_t compared = 0;
  std::uint64_t failures = 0;
  std::uint64_t skipped = 0;
  double max_output_relative_l2 = 0.0;
};

void loop_mla_parity_compare(std::uint32_t layer_index,
                             const at::Tensor& hidden,
                             const MlaWeights& weights,
                             const MlaInputBundle* input_bundle,
                             const MlaCache& cache,
                             const MlaPreparedDecode& stock);
[[nodiscard]] LoopMlaParityStats loop_mla_parity_stats();

/*
 * K3_MLA_LOOP=on (Step 3 takeover): the parity-validated ten-op chain runs
 * INSTEAD of the stock ATen MLA and its results are used. key_states /
 * value_states are the decode shell's views over the CHOSEN storage
 * ([1,96,S,width], S = committed length + 1, position pitch = the slab
 * capacity); the chain packs its new K/V row and blits it into row S-1 —
 * writing the live staging slot is the stock contract (uncommitted rows
 * are scratch until commit_mla_decode). The [7168] result lands in
 * `output` ([1,1,7168] fp32 MPS, caller-allocated). Synchronous: host-
 * waits its CB, so on `false` (any disqualification or GPU error) the
 * caller cancels the shell and falls back to stock — a partially written
 * staging row is harmless, the stock prepare rewrites it.
 */
struct LoopMlaTakeoverStats {
  std::uint64_t taken = 0;
  std::uint64_t fallbacks = 0;
};
[[nodiscard]] bool loop_mla_takeover_run(std::uint32_t layer_index,
                                         const at::Tensor& hidden,
                                         const MlaWeights& weights,
                                         const MlaInputBundle* input_bundle,
                                         const at::Tensor& key_states,
                                         const at::Tensor& value_states,
                                         const at::Tensor& output);
[[nodiscard]] LoopMlaTakeoverStats loop_mla_takeover_stats();

/*
 * K3_MLA_LOOP=chain (Step 4 "chain-lite"): the same validated chain, but
 * PRE-COMMITTED at the previous layer's finish so the GPU works through
 * the host gap — the per-wall fence + host-wait leave the critical path
 * (tax migration #4's fix). The CB additionally encodes the attention
 * head (residual mix + input rmsnorm) GPU-side from the L-1 tail's
 * output tensors, so no host round-trip feeds it. prepare_target_mlp
 * stays host-side (stock) in this rung; the full MLP-head fold is the
 * next rung if this one converts.
 *
 * The handle owns the committed CB and STRONG REFS to everything the
 * parked CB reads or writes: the weights views (MlaWeights by value),
 * the residual carrier tensors, the slab views (incl. the shell's grown
 * storages via key/value_states), and the fresh normalized/output
 * tensors. Exactly one may be in flight. collect() host-waits (usually
 * already signalled) and publishes output; abandon() DRAINS the CB
 * (waitUntilCompleted on timeout — it writes the live staging slot and
 * pooled buffers, so it must be DEAD before its tensors are released;
 * the bd597d8 review lesson). The caller owns the MlaCacheTransaction +
 * MlaDecodeShell created at precommit time and must cancel them after
 * abandon or a failed collect.
 */
struct LoopMlaPendingHandle;
[[nodiscard]] LoopMlaPendingHandle* loop_mla_layer_precommit(
    std::uint32_t layer_index, const at::Tensor& hidden,
    const at::Tensor& anchors, const at::Tensor& score_weight,
    const at::Tensor& input_norm, MlaWeights weights,
    const MlaInputBundle* input_bundle, const at::Tensor& key_states,
    const at::Tensor& value_states);
[[nodiscard]] bool loop_mla_layer_collect(LoopMlaPendingHandle* handle,
                                          std::uint32_t layer_index,
                                          at::Tensor& output_out);
void loop_mla_layer_abandon(LoopMlaPendingHandle* handle) noexcept;

struct LoopMlaChainStats {
  std::uint64_t chained = 0;
  std::uint64_t collected = 0;
  std::uint64_t abandoned = 0;
  std::uint64_t begin_fallbacks = 0;
};
[[nodiscard]] LoopMlaChainStats loop_mla_chain_stats();

/*
 * Takeover half (K3_KDA_LOOP=on, milestone iii): runs the same validated
 * chain but fills `result` directly — output and all four next-state
 * tensors are freshly allocated at::empty MPS storages written by the loop
 * queue (the staging/commit contract downstream is untouched).  Handles the
 * post-verify non-contiguous conv-state case with an on-the-fly
 * .contiguous() copy.  Returns false fail-soft (caller must run stock
 * kda_decode_one).  Under K3_ROUTE_SYNC_PROBE=1 aggregates the per-layer
 * host wait as [kda-loop] wait means.
 */
/*
 * 3b extension: the WHOLE layer head runs on the loop queue too — the
 * AttnRes anchor mix (when anchors are present) and the input RMSNorm are
 * encoded ahead of the KDA chain in the same command buffer, so the host
 * wait no longer covers any ATen-encoded prefix math.  `normalized_out`
 * receives the loop-computed post-norm hidden (the same tensor the MLP head
 * consumes downstream).  Anchor bookkeeping (boundary append / prefix_sum
 * semantics) stays with the caller.  `score_weight` must be the bind-time
 * cached self-attention product; callers fall back to stock when it is
 * undefined.
 */
/* 3c extension: the MLP head too — the residual add (aliased at boundary
 * layers exactly like stock), the second AttnRes mix over the POST-append
 * anchors with the cached mlp score weight, and the post-attention RMSNorm
 * all join the same command buffer.  Outputs mirror TargetMlpInput:
 * `mlp_normalized_out` (the router input), `lookahead_out` (the mixed
 * pre-norm residual the pilot consumes), and `prefix_out` (the residual
 * carrier; aliases the attention output at boundary layers). */
/*
 * CB_tail (plan Step 5, synchronous): the whole MoE tail of a routed T=1
 * layer in one loop command buffer — the MXFP4 expert stack (encoded via
 * metal_moe's own pipelines and wrap cache onto this CB), routed_norm RMS,
 * routed_up, the shared-expert epilogue (gate_up | gate+up -> SiTU ->
 * shared_down), the routed+shared merge, and the final residual add that
 * replaces complete_target_layer.  Committed with a HOST WAIT before
 * returning, so the expert-byte lease and every downstream ATen consumer
 * keep today's synchronous contract.  K3_TAIL_ASYNC=1 defers that wait:
 * the CB is stashed as the pending tail and drained at the next tail (or
 * eagerly via loop_moe_tail_drain wherever synchronous semantics are
 * required again); the Rust side parks the expert-byte lease one layer to
 * keep the staged host memory alive until the drain proves the GPU is
 * done with it.  Weight forms dispatch per matrix (dense-f32 / row-int8 /
 * original-BF16, via LoopMoeMatrixView declared above).  Fail-soft: false
 * means the caller must run the stock execute/complete path.
 */
[[nodiscard]] bool loop_moe_tail(
    std::uint32_t layer_index, const at::Tensor& routed_input,
    const at::Tensor& identity, const at::Tensor& prefix_sum,
    const std::uint16_t* expert_ids, const std::uint32_t* weight_bits,
    const std::uint8_t* const* expert_blobs, std::uint32_t expert_count,
    const at::Tensor& routed_norm, const LoopMoeMatrixView& routed_up,
    bool shared_combined, const LoopMoeMatrixView& shared_gate_up,
    const LoopMoeMatrixView& shared_gate,
    const LoopMoeMatrixView& shared_up,
    const LoopMoeMatrixView& shared_down, at::Tensor& row_hidden_out);

/* K3_TAIL_ASYNC=1 read once per process; false otherwise. */
[[nodiscard]] bool loop_tail_async_enabled() noexcept;

/* Sticky: a deferred tail failed after its wait was swallowed by a
 * fail-soft caller. The finish ABIs check this and raise the hard error
 * the synchronous contract would have produced. */
[[nodiscard]] bool loop_tail_poisoned() noexcept;

/* Wait for the pending deferred tail CB, if any, and surface its error.
 * Idempotent and cheap when nothing is pending. Every path that needs the
 * synchronous tail contract back (stock fallback, an ATen consumer of the
 * layer output, sequence finalization) must call this first. */
void loop_moe_tail_drain();

/* K3_CB_FUSION=1 read once per process; false otherwise. Forced off (with
 * a one-line notice) while K3_TAIL_ASYNC=1 — the two flags reshape the
 * same tail commit sites and compose unsafely. */
[[nodiscard]] bool loop_cb_fusion_enabled() noexcept;

/* Commit the open fused CB (the layer tail plus any successor KDA/route
 * encoders that rode into it), then perform the synchronous tail wait and
 * surface its error. Idempotent and cheap when no fused CB is open.
 * loop_moe_tail_drain() calls this first, so every existing
 * restore-sync-semantics site covers fusion unchanged; the precommit
 * orchestration hook calls it on every exit. */
void loop_fused_flush();

[[nodiscard]] bool loop_kda_layer(std::uint32_t layer_index,
                                  const at::Tensor& hidden,
                                  const at::Tensor& anchors,
                                  const at::Tensor& score_weight,
                                  const at::Tensor& input_norm,
                                  const at::Tensor& prefix_sum,
                                  const at::Tensor& next_anchors,
                                  const at::Tensor& mlp_score_weight,
                                  const at::Tensor& post_attention_norm,
                                  const KdaWeights& weights,
                                  const KdaState& state_in,
                                  at::Tensor& normalized_out,
                                  at::Tensor& mlp_normalized_out,
                                  at::Tensor& lookahead_out,
                                  at::Tensor& prefix_out,
                                  KdaDecodeResult& result);

/*
 * Device weights-table cache.  `store` validates and caches the device view
 * of one spine tensor (contiguous MPS fp32 or int8 with a live MTLBuffer)
 * under (layer_index, slot), stamped with the spine generation; it returns
 * false without caching when the tensor does not qualify.  `hit` is true only
 * for a live entry whose generation matches; a generation mismatch evicts the
 * stale entry (a spine rebind produced fresh storages) and reports a miss.
 */
[[nodiscard]] bool loop_weights_table_store(std::uint32_t layer_index,
                                            std::uint64_t generation,
                                            std::uint32_t slot,
                                            const at::Tensor& tensor);
[[nodiscard]] bool loop_weights_table_hit(std::uint32_t layer_index,
                                          std::uint64_t generation,
                                          std::uint32_t slot);
[[nodiscard]] LoopWeightsTableStats loop_weights_table_stats();
void loop_weights_table_reset();

#endif  // defined(__APPLE__)

}  // namespace deltafin::provider_internal

#endif
