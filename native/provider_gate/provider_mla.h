#ifndef DELTAFIN_PROVIDER_MLA_H
#define DELTAFIN_PROVIDER_MLA_H

#include "provider_bf16_cpu.h"

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace deltafin::provider_internal {

// The MLA tape accepts canonical original-BF16 storage, a dense-fp32 synthetic
// oracle, or the explicit non-weight-exact row-int8 spine.
enum class MlaLinearEncoding : std::uint32_t {
  DenseF32 = 1,
  RowI8F32Scale = 2,
  OriginalBf16 = 3,
};

struct MlaLinearWeight {
  MlaLinearEncoding encoding = MlaLinearEncoding::DenseF32;
  at::Tensor data;
  at::Tensor row_scale;
  OriginalBf16Matrix original_bf16;
};

struct MlaShape {
  std::int64_t hidden_size = 0;
  std::int64_t num_heads = 0;
  std::int64_t q_lora_rank = 0;
  std::int64_t kv_lora_rank = 0;
  std::int64_t qk_nope_head_dim = 0;
  std::int64_t qk_rope_head_dim = 0;
  std::int64_t value_head_dim = 0;
  std::int64_t max_context = 0;
  double rms_epsilon = 0.0;

  [[nodiscard]] static MlaShape k3();
  // Fixed compact contract used only by the explicitly synthetic doctor
  // session.  It keeps every packed projection dimension valid for the same
  // ATen provider calls as production without allocating K3-sized weights.
  [[nodiscard]] static MlaShape small_canary();
  void validate() const;
  [[nodiscard]] bool is_exact_k3() const;
  [[nodiscard]] std::int64_t query_head_dim() const;
};

struct MlaWeights {
  MlaLinearWeight query_a;
  at::Tensor query_a_norm;
  MlaLinearWeight query_b;
  MlaLinearWeight key_value_a;
  at::Tensor key_value_a_norm;
  MlaLinearWeight key_value_b;
  MlaLinearWeight output_gate;
  MlaLinearWeight output;
};

// One row-concatenated packed storage descriptor for the three MLA branches
// that read the unchanged layer input. The reviewed T=1 path may execute it as
// one projection. T>1 uses the individual zero-copy views in downloaded-model
// order (q_a/q_b, kv_a/kv_b, attention, gate); it never executes this bundle.
struct MlaInputBundle {
  MlaLinearWeight projection;
  std::int64_t query_a_rows = 0;
  std::int64_t key_value_a_rows = 0;
  std::int64_t output_gate_rows = 0;
};

// Optional structural probe for the focused provider tests. Production leaves
// this null, so tracing cannot add allocation or synchronization to inference.
// It records provider submission order, not merely arithmetic dependencies.
enum class MlaExecutionStage : std::uint8_t {
  InputBundle,
  QueryA,
  QueryB,
  KeyValueA,
  KeyValueB,
  Attention,
  OutputGate,
  Output,
};

struct MlaExecutionTrace {
  std::array<MlaExecutionStage, 8> stages{};
  std::size_t count = 0;

  void record(const MlaExecutionStage stage) {
    if (count >= stages.size()) {
      throw std::logic_error("MLA execution trace overflow");
    }
    stages[count++] = stage;
  }
};

struct MlaPreparedDecode;
struct MlaDecodeShell;
class MlaCacheTransaction;

enum class MlaCacheRepresentation : std::uint32_t {
  /*
   * The downloaded K3 forward expands each latent row before caching it.
   * This preserves that exact operation order, but costs 120 KiB per token
   * per production MLA layer.  The explicit storage budget below prevents
   * this temporary representation from pretending to support a 1M context.
   */
  ExpandedExact = 1,
  /*
   * The absorbed MLA form retains the normalized fp32 KV latent and the
   * unchanged shared positional key.  It never quantizes either cache arm.
   * The representation is fixed when the cache is constructed; callers may
   * not reinterpret or silently fall back to ExpandedExact after mutation.
   */
  CompactLatentF32 = 2,
};

// Provider-owned, bind-time view/materialization of kv_b.  Dense fp32 rows
// remain zero-copy views; row-int8 rows are dequantized exactly once at bind
// time instead of reconstructing roughly 50 MiB of absorbed weights per token.
struct MlaAbsorbedKeyValue {
  at::Tensor key_up;    // [heads, qk_nope_head_dim, kv_lora_rank], fp32
  at::Tensor value_up;  // [heads, value_head_dim, kv_lora_rank], fp32
  MlaLinearEncoding source_encoding = MlaLinearEncoding::DenseF32;
  std::int64_t num_heads = 0;
  std::int64_t qk_nope_head_dim = 0;
  std::int64_t value_head_dim = 0;
  std::int64_t kv_lora_rank = 0;
};

// One provider-owned cache per MLA layer/request.  Only the prefix [0,length)
// is committed.  A prepare may write a candidate token beyond that prefix,
// but no public view or later prepare can observe it until commit succeeds.
// Capacity grows geometrically, avoiding torch.cat's full persistent cache
// allocation on every decoded token.
class MlaCache {
 public:
  static constexpr std::uint64_t kDefaultExpandedStorageBudgetBytes =
      512ULL * 1024ULL * 1024ULL;
  // Kept for source compatibility with the reviewed expanded provider.
  static constexpr std::uint64_t kDefaultStorageBudgetBytes =
      kDefaultExpandedStorageBudgetBytes;
  // On the representation-aware constructor, zero selects the reviewed
  // representation default: 512 MiB for ExpandedExact and the exact bytes
  // required by shape.max_context for CompactLatentF32.
  static constexpr std::uint64_t kAutomaticStorageBudgetBytes = 0;

  explicit MlaCache(
      MlaShape shape, double growth_factor = 1.5,
      std::uint64_t storage_budget_bytes = kDefaultStorageBudgetBytes);
  MlaCache(
      MlaShape shape, MlaCacheRepresentation representation,
      double growth_factor = 1.5,
      std::uint64_t storage_budget_bytes = kAutomaticStorageBudgetBytes);

  MlaCache(const MlaCache&) = delete;
  MlaCache& operator=(const MlaCache&) = delete;
  MlaCache(MlaCache&&) = delete;
  MlaCache& operator=(MlaCache&&) = delete;

  [[nodiscard]] const MlaShape& shape() const noexcept;
  [[nodiscard]] std::int64_t length() const noexcept;
  [[nodiscard]] std::int64_t capacity() const noexcept;
  [[nodiscard]] std::uint64_t version() const noexcept;
  [[nodiscard]] double growth_factor() const noexcept;
  [[nodiscard]] MlaCacheRepresentation representation() const noexcept;
  [[nodiscard]] std::uint64_t bytes_per_position() const noexcept;
  [[nodiscard]] std::uint64_t storage_budget_bytes() const noexcept;
  [[nodiscard]] std::uint64_t storage_bytes() const noexcept;
  [[nodiscard]] std::uint64_t full_context_storage_bytes() const noexcept;
  [[nodiscard]] std::int64_t admitted_max_context() const noexcept;
  [[nodiscard]] bool can_append(std::size_t positions) const noexcept;
  [[nodiscard]] bool has_pending_prepare() const noexcept;
  // ExpandedExact exposes expanded key/value prefixes. CompactLatentF32
  // exposes normalized-latent/positional prefixes through these legacy names;
  // callers must branch on the immutable representation tag before use.
  [[nodiscard]] at::Tensor committed_keys() const;
  [[nodiscard]] at::Tensor committed_values() const;

  /*
   * Fork one complete committed boundary. The child shares immutable prefix
   * storage, but owns independent length/version/nonce metadata. Subsequent
   * sequence transactions either write beyond the shared logical prefix or
   * publish newly grown storage, so discarding the child restores the parent
   * without copying multi-gigabyte MLA slabs. The immutable representation
   * tag is inherited and cannot change across the fork.
   */
  [[nodiscard]] std::unique_ptr<MlaCache> fork_committed() const;

 private:
  friend struct MlaPreparedDecode;
  friend MlaPreparedDecode prepare_mla_decode(
      const at::Tensor&, const MlaWeights&, MlaCache&, bool,
      const MlaInputBundle*);
  friend MlaPreparedDecode prepare_mla_positions(
      const at::Tensor&, const MlaWeights&, MlaCache&,
      const MlaAbsorbedKeyValue*, bool, const MlaInputBundle*,
      MlaExecutionTrace*);
  friend void commit_mla_decode(MlaCache&, MlaPreparedDecode&);
  friend void cancel_mla_decode(MlaCache&, MlaPreparedDecode&);
  friend struct MlaDecodeShell;
  friend MlaDecodeShell prepare_k3_mla_decode_shell(const at::Tensor&,
                                                    MlaCache&);
  friend class MlaCacheTransaction;

  MlaShape shape_;
  const MlaCacheRepresentation representation_;
  double growth_factor_ = 1.5;
  std::uint64_t bytes_per_position_ = 0;
  std::uint64_t storage_budget_bytes_ = 0;
  std::uint64_t full_context_storage_bytes_ = 0;
  std::int64_t admitted_max_context_ = 0;
  at::Tensor key_storage_;
  at::Tensor value_storage_;
  std::int64_t length_ = 0;
  std::int64_t capacity_ = 0;
  std::uint64_t version_ = 0;
  std::uint64_t next_nonce_ = 1;
  std::uint64_t pending_nonce_ = 0;
};

/*
 * An unpublished cache branch used by a wider provider transaction.  It
 * shares the committed prefix storage until growth is necessary, advances
 * only its private length/version metadata, and can publish any completed
 * prefix after the caller has preflighted every layer. No representation-
 * specific tensor shape escapes this abstraction, so the same transaction
 * protocol covers both expanded and compact caches.
 */
class MlaCacheTransaction {
 public:
  MlaCacheTransaction(MlaCache& base, std::size_t maximum_positions);

  MlaCacheTransaction(const MlaCacheTransaction&) = delete;
  MlaCacheTransaction& operator=(const MlaCacheTransaction&) = delete;
  MlaCacheTransaction(MlaCacheTransaction&&) = delete;
  MlaCacheTransaction& operator=(MlaCacheTransaction&&) = delete;

  [[nodiscard]] MlaCache& working_cache() noexcept;
  [[nodiscard]] const MlaCache& working_cache() const noexcept;
  [[nodiscard]] std::size_t completed_positions() const noexcept;
  [[nodiscard]] std::int64_t base_length() const noexcept;
  [[nodiscard]] std::uint64_t expected_version() const noexcept;
  [[nodiscard]] bool active() const noexcept;
  void preflight_publish_prefix(std::size_t positions) const;
  // Requires a successful preflight for the same prefix. This second phase is
  // intentionally noexcept so a multi-layer cache publication cannot stop
  // after committing only an early subset of layers.
  void publish_prefix_noexcept(std::size_t positions) noexcept;
  void cancel_noexcept() noexcept;

 private:
  MlaCache* base_ = nullptr;
  MlaCache working_;
  std::int64_t base_length_ = 0;
  std::uint64_t expected_version_ = 0;
  std::size_t maximum_positions_ = 0;
  bool published_ = false;
};

// The result of the attention half of a split layer.  The KV mutation remains
// unpublished until finish_layer calls commit_mla_decode.  Destroying a ticket
// in the provider must call cancel_mla_decode first.
struct MlaPreparedDecode {
  MlaPreparedDecode() = default;
  MlaPreparedDecode(const MlaPreparedDecode&) = delete;
  MlaPreparedDecode& operator=(const MlaPreparedDecode&) = delete;
  MlaPreparedDecode(MlaPreparedDecode&& other) noexcept;
  MlaPreparedDecode& operator=(MlaPreparedDecode&&) noexcept = delete;

  at::Tensor output;
  at::Tensor grown_key_storage;
  at::Tensor grown_value_storage;
  std::uint64_t nonce = 0;
  std::uint64_t expected_version = 0;
  std::int64_t expected_length = 0;
  std::int64_t next_length = 0;
  std::int64_t next_capacity = 0;
  std::size_t position_count = 0;
  const MlaCache* owner = nullptr;
  bool uses_grown_storage = false;
  bool finalized = false;
};

[[nodiscard]] MlaPreparedDecode prepare_mla_decode(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    bool allow_exact_query_alias = true,
    const MlaInputBundle* input_bundle = nullptr);

/*
 * K3_MLA_LOOP=on (Step 3 takeover): cache bookkeeping WITHOUT compute.
 * Takes the pending nonce, decides growth (allocating + prefix-copying a
 * grown slab exactly like the full prepare), and assembles the ticket —
 * but stages no row and leaves prepared.output undefined. The loop chain
 * then writes the new K/V row into row `expected_length` of key_states/
 * value_states (views over the CHOSEN storage including the staging slot,
 * [1,96,next_length,width]) and the caller sets prepared.output before
 * commit_mla_decode. On failure the caller must cancel_mla_decode so the
 * stock prepare can retake the nonce; an uncommitted staged row is
 * scratch by contract, so a fallback after a partial GPU write is safe.
 * ExpandedExact + exact-K3 + T=1 only.
 */
struct MlaDecodeShell {
  MlaPreparedDecode prepared;
  at::Tensor key_states;
  at::Tensor value_states;
};
[[nodiscard]] MlaDecodeShell prepare_k3_mla_decode_shell(
    const at::Tensor& hidden, MlaCache& cache);

// Multi-position causal entry used by prefill and speculative verification.
// Hidden is fp32 [1,T,hidden], 1 <= T <= remaining context.  ExpandedExact
// ignores absorbed_key_value; CompactLatentF32 requires a validated bind-time
// descriptor and stores normalized latent/positional rows without publishing
// them until commit_mla_decode succeeds.
[[nodiscard]] MlaPreparedDecode prepare_mla_positions(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    const MlaAbsorbedKeyValue* absorbed_key_value,
    bool allow_exact_query_alias = true,
    const MlaInputBundle* input_bundle = nullptr,
    MlaExecutionTrace* execution_trace = nullptr);

// Bind-time construction of the absorbed fp32 key/value projection.  It is
// representation metadata, not a cache conversion, and never changes the
// downloaded projection values.
[[nodiscard]] MlaAbsorbedKeyValue absorb_mla_key_value(
    const MlaShape& shape, const MlaLinearWeight& key_value_b);

// Bind-time operation. It concatenates packed rows once, then rewrites the
// three component projections as views of the combined allocation so the
// qualified fast path and the fallback have equal payload residency.
[[nodiscard]] MlaInputBundle bundle_mla_input_weights(
    const MlaShape& shape, MlaWeights& weights);

// Production entry: fail closed unless both the immutable model dimensions and
// every projection representation match Deltafin's reviewed K3 spine.
[[nodiscard]] MlaPreparedDecode prepare_k3_mla_decode(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    const MlaInputBundle* input_bundle = nullptr);

// The multi-position production entry deliberately admits ExpandedExact only.
// CompactLatentF32 is an algebraic research implementation: moving kv_b
// across the score/value contractions changes fp32 reduction order, so merely
// retaining fp32 cache rows does not make its output bit-exact to K3.
//
// One reviewed exception to LibTorch-tape bit-identity exists: with the
// explicit env opt-in K3_MLA_METAL_ATTENTION=1 (default off) the exact-K3
// expanded T=1 MPS attention core runs the fused Metal kernel
// (provider_mla_attn_metal.h), which computes the same expanded math on the
// same fp32 slabs but reassociates fp32 reductions.  The layout, the cache
// transaction contract, and every other stage remain unchanged.
[[nodiscard]] MlaPreparedDecode prepare_k3_mla_positions(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    const MlaAbsorbedKeyValue* absorbed_key_value,
    const MlaInputBundle* input_bundle = nullptr);

void commit_mla_decode(MlaCache& cache, MlaPreparedDecode& prepared);
void cancel_mla_decode(MlaCache& cache, MlaPreparedDecode& prepared);

}  // namespace deltafin::provider_internal

#endif
