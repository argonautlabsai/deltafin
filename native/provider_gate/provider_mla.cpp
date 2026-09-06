#include "provider_mla.h"

#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MLA_ATTN_METAL_V1)
#include "provider_mla_attn_metal.h"
#endif

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/einsum.h>
#include <ATen/ops/full.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/zeros.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kK3HiddenSize = 7168;
constexpr std::int64_t kK3Heads = 96;
constexpr std::int64_t kK3QueryLoraRank = 1536;
constexpr std::int64_t kK3KeyValueLoraRank = 512;
constexpr std::int64_t kK3QueryNopeHeadDim = 128;
constexpr std::int64_t kK3QueryRopeHeadDim = 64;
constexpr std::int64_t kK3ValueHeadDim = 128;
constexpr std::int64_t kK3MaximumContext = 1048576;
constexpr double kK3RmsEpsilon = 1.0e-5;

void require_positive(const std::int64_t value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("MLA ") + name +
                                " must be positive");
  }
}

void require_tensor(const at::Tensor& tensor, const at::Device& device,
                    const at::ScalarType dtype,
                    const at::IntArrayRef expected_shape, const char* name) {
  if (!tensor.defined() || tensor.device() != device ||
      tensor.scalar_type() != dtype || tensor.sizes() != expected_shape) {
    throw std::invalid_argument(std::string("MLA ") + name +
                                " has an invalid shape, dtype, or device");
  }
}

void validate_linear(const MlaLinearWeight& weight, const at::Device& device,
                     const std::int64_t rows, const std::int64_t columns,
                     const char* name) {
  switch (weight.encoding) {
    case MlaLinearEncoding::DenseF32:
      require_tensor(weight.data, device, at::kFloat, {rows, columns}, name);
      if (weight.row_scale.defined()) {
        throw std::invalid_argument(std::string("dense MLA ") + name +
                                    " unexpectedly has row scales");
      }
      return;
    case MlaLinearEncoding::RowI8F32Scale:
      require_tensor(weight.data, device, at::kChar, {rows, columns}, name);
      require_tensor(weight.row_scale, device, at::kFloat, {rows},
                     "row scales");
      return;
    case MlaLinearEncoding::OriginalBf16:
      if (weight.data.defined() || weight.row_scale.defined() ||
          !original_bf16_matrix_matches(
              weight.original_bf16, device,
              static_cast<std::size_t>(rows),
              static_cast<std::size_t>(columns))) {
        throw std::invalid_argument(std::string("MLA ") + name +
                                    " has an invalid original-BF16 carrier");
      }
      return;
    default:
      throw std::invalid_argument(std::string("MLA ") + name +
                                  " has an unknown encoding");
  }
}

at::Tensor linear(const at::Tensor& input, const MlaLinearWeight& weight) {
  if (input.dim() != 2 && input.dim() != 3) {
    throw std::invalid_argument(
        "MLA linear input must have rank two or three");
  }
  const std::int64_t columns = input.size(-1);
  const std::int64_t rows = weight.original_bf16.defined()
      ? static_cast<std::int64_t>(weight.original_bf16.rows)
      : (weight.data.dim() == 2 ? weight.data.size(0) : 0);
  const std::int64_t weight_columns = weight.original_bf16.defined()
      ? static_cast<std::int64_t>(weight.original_bf16.columns)
      : (weight.data.dim() == 2 ? weight.data.size(1) : 0);
  if (rows <= 0 || weight_columns != columns) {
    throw std::invalid_argument("MLA linear input/weight dimensions disagree");
  }
  const at::Tensor flat = input.reshape({-1, columns});
  at::Tensor output;
  switch (weight.encoding) {
    case MlaLinearEncoding::DenseF32:
      output = at::linear(flat, weight.data, std::nullopt);
      break;
    case MlaLinearEncoding::RowI8F32Scale:
      output = at::_weight_int8pack_mm(flat, weight.data, weight.row_scale);
      break;
    case MlaLinearEncoding::OriginalBf16:
      output = original_bf16_linear(flat, weight.original_bf16);
      break;
    default:
      throw std::invalid_argument("MLA linear weight encoding is unknown");
  }
  // The native MLA tape has only [1,C] and [1,1,C] inputs.  Spell those
  // fixed shapes directly instead of allocating a shape vector for every
  // packed projection (roughly eight allocations per MLA layer/token).
  if (input.dim() == 2) {
    return output.view({input.size(0), rows});
  }
  return output.view({input.size(0), input.size(1), rows});
}

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight,
                    const double epsilon) {
  // Mirrors KimiRMSNorm exactly for Deltafin's fp32 activation contract:
  // x * rsqrt(mean(x.pow(2), -1, keepdim=True) + eps), then weight * x.
  const at::Tensor variance = at::mean(at::pow(input, 2), {-1}, true);
  const at::Tensor normalized = input * at::rsqrt(variance + epsilon);
  return weight * normalized;
}

void validate_weights(const MlaShape& shape, const MlaWeights& weights,
                      const at::Device& device) {
  const std::int64_t query_width =
      shape.num_heads * shape.query_head_dim();
  const std::int64_t value_width =
      shape.num_heads * shape.value_head_dim;
  const std::int64_t key_value_width = shape.kv_lora_rank +
                                       shape.qk_rope_head_dim;
  const std::int64_t expanded_key_value_width =
      shape.num_heads *
      (shape.qk_nope_head_dim + shape.value_head_dim);
  validate_linear(weights.query_a, device, shape.q_lora_rank,
                  shape.hidden_size, "query-a projection");
  require_tensor(weights.query_a_norm, device, at::kFloat,
                 {shape.q_lora_rank}, "query-a norm");
  validate_linear(weights.query_b, device, query_width, shape.q_lora_rank,
                  "query-b projection");
  validate_linear(weights.key_value_a, device, key_value_width,
                  shape.hidden_size, "key/value-a projection");
  require_tensor(weights.key_value_a_norm, device, at::kFloat,
                 {shape.kv_lora_rank}, "key/value-a norm");
  validate_linear(weights.key_value_b, device, expanded_key_value_width,
                  shape.kv_lora_rank, "key/value-b projection");
  validate_linear(weights.output_gate, device, value_width,
                  shape.hidden_size, "output-gate projection");
  validate_linear(weights.output, device, shape.hidden_size, value_width,
                  "output projection");
}

void validate_input_bundle(const MlaShape& shape,
                           const MlaInputBundle& bundle,
                           const at::Device& device) {
  const std::int64_t query_rows = shape.q_lora_rank;
  const std::int64_t key_value_rows =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  const std::int64_t gate_rows = shape.num_heads * shape.value_head_dim;
  if (bundle.query_a_rows != query_rows ||
      bundle.key_value_a_rows != key_value_rows ||
      bundle.output_gate_rows != gate_rows ||
      (bundle.projection.encoding != MlaLinearEncoding::RowI8F32Scale &&
       bundle.projection.encoding != MlaLinearEncoding::OriginalBf16)) {
    throw std::invalid_argument(
        "MLA same-input bundle does not match the layer contract");
  }
  validate_linear(bundle.projection, device,
                  query_rows + key_value_rows + gate_rows,
                  shape.hidden_size, "same-input projection bundle");
}

std::int64_t grown_capacity(const MlaCache& cache,
                            const std::int64_t needed) {
  if (needed <= 0 || needed > cache.shape().max_context ||
      needed > cache.admitted_max_context()) {
    throw std::invalid_argument(
        "exact expanded MLA cache would exceed its explicit storage budget; "
        "a reviewed latent-cache provider is required for this context");
  }
  if (cache.capacity() == 0) {
    const auto initial = static_cast<std::int64_t>(
        std::ceil(static_cast<double>(needed) * 1.25));
    return std::min(cache.admitted_max_context(),
                    std::max<std::int64_t>({16, needed, initial}));
  }
  const double grown =
      std::ceil(static_cast<double>(cache.capacity()) * cache.growth_factor());
  if (!std::isfinite(grown) ||
      grown > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("MLA cache capacity growth overflowed");
  }
  return std::min(cache.admitted_max_context(),
                  std::max(needed, static_cast<std::int64_t>(grown)));
}

std::uint64_t checked_cache_bytes_per_position(
    const MlaShape& shape, const MlaCacheRepresentation representation) {
  std::uint64_t elements = 0;
  switch (representation) {
    case MlaCacheRepresentation::ExpandedExact: {
      const auto heads = static_cast<std::uint64_t>(shape.num_heads);
      const auto key = static_cast<std::uint64_t>(shape.query_head_dim());
      const auto value = static_cast<std::uint64_t>(shape.value_head_dim);
      if (key > std::numeric_limits<std::uint64_t>::max() - value ||
          heads > std::numeric_limits<std::uint64_t>::max() / (key + value)) {
        throw std::invalid_argument(
            "expanded MLA cache byte geometry overflows");
      }
      elements = heads * (key + value);
      break;
    }
    case MlaCacheRepresentation::CompactLatentF32: {
      const auto latent = static_cast<std::uint64_t>(shape.kv_lora_rank);
      const auto positional =
          static_cast<std::uint64_t>(shape.qk_rope_head_dim);
      if (latent > std::numeric_limits<std::uint64_t>::max() - positional) {
        throw std::invalid_argument(
            "compact MLA cache byte geometry overflows");
      }
      elements = latent + positional;
      break;
    }
    default:
      throw std::invalid_argument("MLA cache representation is unknown");
  }
  if (elements >
      std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
    throw std::invalid_argument("MLA cache byte geometry overflows");
  }
  return elements * sizeof(float);
}

std::uint64_t checked_full_context_storage_bytes(
    const MlaShape& shape, const std::uint64_t bytes_per_position) {
  const auto context = static_cast<std::uint64_t>(shape.max_context);
  if (context >
      std::numeric_limits<std::uint64_t>::max() / bytes_per_position) {
    throw std::invalid_argument("MLA full-context byte geometry overflows");
  }
  return context * bytes_per_position;
}

std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
cache_storage_shapes(const MlaShape& shape,
                     const MlaCacheRepresentation representation,
                     const std::int64_t capacity) {
  switch (representation) {
    case MlaCacheRepresentation::ExpandedExact:
      return {
          {1, shape.num_heads, capacity, shape.query_head_dim()},
          {1, shape.num_heads, capacity, shape.value_head_dim},
      };
    case MlaCacheRepresentation::CompactLatentF32:
      return {
          {1, 1, capacity, shape.kv_lora_rank},
          {1, 1, capacity, shape.qk_rope_head_dim},
      };
    default:
      throw std::invalid_argument("MLA cache representation is unknown");
  }
}

void validate_absorbed_key_value(
    const MlaShape& shape, const MlaAbsorbedKeyValue& absorbed,
    const at::Device& device) {
  if (absorbed.num_heads != shape.num_heads ||
      absorbed.qk_nope_head_dim != shape.qk_nope_head_dim ||
      absorbed.value_head_dim != shape.value_head_dim ||
      absorbed.kv_lora_rank != shape.kv_lora_rank ||
      (absorbed.source_encoding != MlaLinearEncoding::DenseF32 &&
       absorbed.source_encoding != MlaLinearEncoding::RowI8F32Scale &&
       absorbed.source_encoding != MlaLinearEncoding::OriginalBf16)) {
    throw std::invalid_argument(
        "absorbed MLA key/value descriptor has an invalid immutable geometry");
  }
  require_tensor(absorbed.key_up, device, at::kFloat,
                 {shape.num_heads, shape.qk_nope_head_dim,
                  shape.kv_lora_rank},
                 "absorbed key projection");
  require_tensor(absorbed.value_up, device, at::kFloat,
                 {shape.num_heads, shape.value_head_dim,
                  shape.kv_lora_rank},
                 "absorbed value projection");
}

at::Tensor causal_block_mask(const at::Tensor& scores,
                             const std::int64_t past,
                             const std::int64_t positions) {
  const std::int64_t total = past + positions;
  at::Tensor mask = at::zeros({positions, total}, scores.options());
  if (positions > 1) {
    const at::Tensor future = at::triu(
        at::full({positions, positions},
                 -std::numeric_limits<float>::infinity(), scores.options()),
        1);
    mask.narrow(1, past, positions).copy_(future);
  }
  return mask.view({1, 1, positions, total});
}

}  // namespace

MlaShape MlaShape::k3() {
  return MlaShape{
      .hidden_size = kK3HiddenSize,
      .num_heads = kK3Heads,
      .q_lora_rank = kK3QueryLoraRank,
      .kv_lora_rank = kK3KeyValueLoraRank,
      .qk_nope_head_dim = kK3QueryNopeHeadDim,
      .qk_rope_head_dim = kK3QueryRopeHeadDim,
      .value_head_dim = kK3ValueHeadDim,
      .max_context = kK3MaximumContext,
      .rms_epsilon = kK3RmsEpsilon,
  };
}

MlaShape MlaShape::small_canary() {
  return MlaShape{
      .hidden_size = 32,
      .num_heads = 2,
      .q_lora_rank = 32,
      .kv_lora_rank = 32,
      .qk_nope_head_dim = 16,
      .qk_rope_head_dim = 32,
      .value_head_dim = 16,
      .max_context = 32,
      .rms_epsilon = kK3RmsEpsilon,
  };
}

void MlaShape::validate() const {
  require_positive(hidden_size, "hidden size");
  require_positive(num_heads, "head count");
  require_positive(q_lora_rank, "query LoRA rank");
  require_positive(kv_lora_rank, "key/value LoRA rank");
  require_positive(qk_nope_head_dim, "non-positional query/key head width");
  require_positive(qk_rope_head_dim, "positional query/key head width");
  require_positive(value_head_dim, "value head width");
  require_positive(max_context, "maximum context");
  if (!std::isfinite(rms_epsilon) || rms_epsilon <= 0.0) {
    throw std::invalid_argument("MLA RMS epsilon must be positive and finite");
  }
  const auto safe_multiply = [](const std::int64_t left,
                                const std::int64_t right,
                                const char* name) {
    if (left > std::numeric_limits<std::int64_t>::max() / right) {
      throw std::invalid_argument(std::string("MLA ") + name +
                                  " overflows int64");
    }
  };
  safe_multiply(num_heads, query_head_dim(), "query width");
  safe_multiply(num_heads, value_head_dim, "value width");
  if (qk_nope_head_dim >
      std::numeric_limits<std::int64_t>::max() - value_head_dim) {
    throw std::invalid_argument(
        "MLA expanded key/value head width overflows int64");
  }
  if (kv_lora_rank >
      std::numeric_limits<std::int64_t>::max() - qk_rope_head_dim) {
    throw std::invalid_argument(
        "MLA compressed key/value width overflows int64");
  }
  safe_multiply(num_heads, qk_nope_head_dim + value_head_dim,
                "expanded key/value width");
}

bool MlaShape::is_exact_k3() const {
  return hidden_size == kK3HiddenSize && num_heads == kK3Heads &&
         q_lora_rank == kK3QueryLoraRank &&
         kv_lora_rank == kK3KeyValueLoraRank &&
         qk_nope_head_dim == kK3QueryNopeHeadDim &&
         qk_rope_head_dim == kK3QueryRopeHeadDim &&
         value_head_dim == kK3ValueHeadDim &&
         max_context == kK3MaximumContext && rms_epsilon == kK3RmsEpsilon;
}

std::int64_t MlaShape::query_head_dim() const {
  if (qk_nope_head_dim >
      std::numeric_limits<std::int64_t>::max() - qk_rope_head_dim) {
    throw std::invalid_argument("MLA query head width overflows int64");
  }
  return qk_nope_head_dim + qk_rope_head_dim;
}

MlaCache::MlaCache(MlaShape shape, const double growth_factor,
                   const std::uint64_t storage_budget_bytes)
    : MlaCache(std::move(shape), MlaCacheRepresentation::ExpandedExact,
               growth_factor, storage_budget_bytes) {}

MlaCache::MlaCache(MlaShape shape,
                   const MlaCacheRepresentation representation,
                   const double growth_factor,
                   const std::uint64_t storage_budget_bytes)
    : shape_(std::move(shape)), representation_(representation),
      growth_factor_(growth_factor) {
  shape_.validate();
  if (!std::isfinite(growth_factor_) || growth_factor_ < 1.1) {
    throw std::invalid_argument(
        "MLA cache growth factor must be finite and at least 1.1");
  }
  bytes_per_position_ =
      checked_cache_bytes_per_position(shape_, representation_);
  full_context_storage_bytes_ =
      checked_full_context_storage_bytes(shape_, bytes_per_position_);
  if (storage_budget_bytes == kAutomaticStorageBudgetBytes) {
    storage_budget_bytes_ =
        representation_ == MlaCacheRepresentation::CompactLatentF32
            ? full_context_storage_bytes_
            : kDefaultExpandedStorageBudgetBytes;
  } else {
    storage_budget_bytes_ = storage_budget_bytes;
  }
  if (storage_budget_bytes_ < bytes_per_position_) {
    throw std::invalid_argument(
        "MLA cache budget cannot hold even one position");
  }
  const std::uint64_t admitted = storage_budget_bytes_ / bytes_per_position_;
  admitted_max_context_ = static_cast<std::int64_t>(std::min<std::uint64_t>(
      admitted, static_cast<std::uint64_t>(shape_.max_context)));
}

const MlaShape& MlaCache::shape() const noexcept { return shape_; }

std::int64_t MlaCache::length() const noexcept { return length_; }

std::int64_t MlaCache::capacity() const noexcept { return capacity_; }

std::uint64_t MlaCache::version() const noexcept { return version_; }

double MlaCache::growth_factor() const noexcept { return growth_factor_; }

MlaCacheRepresentation MlaCache::representation() const noexcept {
  return representation_;
}

std::uint64_t MlaCache::bytes_per_position() const noexcept {
  return bytes_per_position_;
}

std::uint64_t MlaCache::storage_budget_bytes() const noexcept {
  return storage_budget_bytes_;
}

std::uint64_t MlaCache::storage_bytes() const noexcept {
  return static_cast<std::uint64_t>(capacity_) * bytes_per_position_;
}

std::uint64_t MlaCache::full_context_storage_bytes() const noexcept {
  return full_context_storage_bytes_;
}

std::int64_t MlaCache::admitted_max_context() const noexcept {
  return admitted_max_context_;
}

bool MlaCache::can_append(const std::size_t positions) const noexcept {
  if (length_ < 0 || length_ > admitted_max_context_ ||
      positions > static_cast<std::size_t>(
                      std::numeric_limits<std::int64_t>::max())) {
    return false;
  }
  const auto count = static_cast<std::int64_t>(positions);
  return count <= admitted_max_context_ - length_;
}

bool MlaCache::has_pending_prepare() const noexcept {
  return pending_nonce_ != 0;
}

at::Tensor MlaCache::committed_keys() const {
  if (length_ == 0) {
    return {};
  }
  return key_storage_.narrow(2, 0, length_);
}

at::Tensor MlaCache::committed_values() const {
  if (length_ == 0) {
    return {};
  }
  return value_storage_.narrow(2, 0, length_);
}

std::unique_ptr<MlaCache> MlaCache::fork_committed() const {
  if (pending_nonce_ != 0 || length_ < 0 || capacity_ < length_ ||
      capacity_ > admitted_max_context_) {
    throw std::logic_error("cannot fork an incomplete MLA cache boundary");
  }
  auto child = std::make_unique<MlaCache>(
      shape_, representation_, growth_factor_, storage_budget_bytes_);
  child->key_storage_ = key_storage_;
  child->value_storage_ = value_storage_;
  child->length_ = length_;
  child->capacity_ = capacity_;
  child->version_ = version_;
  child->next_nonce_ = next_nonce_;
  return child;
}

MlaCacheTransaction::MlaCacheTransaction(MlaCache& base,
                                         const std::size_t maximum_positions)
    : base_(&base),
      working_(base.shape_, base.representation_, base.growth_factor_,
               base.storage_budget_bytes_),
      base_length_(base.length_), expected_version_(base.version_),
      maximum_positions_(maximum_positions) {
  if (maximum_positions_ == 0 || !base.can_append(maximum_positions_)) {
    throw std::invalid_argument(
        "MLA cache transaction exceeds its admitted position/storage bound");
  }
  if (base.pending_nonce_ != 0) {
    throw std::invalid_argument(
        "MLA cache transaction cannot fork a pending cache");
  }
  if (maximum_positions_ >
      std::numeric_limits<std::uint64_t>::max() - expected_version_) {
    throw std::invalid_argument("MLA cache transaction version would overflow");
  }
  working_.key_storage_ = base.key_storage_;
  working_.value_storage_ = base.value_storage_;
  working_.length_ = base.length_;
  working_.capacity_ = base.capacity_;
  working_.version_ = base.version_;
  working_.next_nonce_ = base.next_nonce_;
}

MlaCache& MlaCacheTransaction::working_cache() noexcept { return working_; }

const MlaCache& MlaCacheTransaction::working_cache() const noexcept {
  return working_;
}

std::size_t MlaCacheTransaction::completed_positions() const noexcept {
  if (working_.length_ < base_length_) {
    return 0;
  }
  return static_cast<std::size_t>(working_.length_ - base_length_);
}

std::int64_t MlaCacheTransaction::base_length() const noexcept {
  return base_length_;
}

std::uint64_t MlaCacheTransaction::expected_version() const noexcept {
  return expected_version_;
}

bool MlaCacheTransaction::active() const noexcept {
  return !published_ && base_ != nullptr;
}

void MlaCacheTransaction::preflight_publish_prefix(
    const std::size_t positions) const {
  if (published_ || base_ == nullptr || positions > maximum_positions_ ||
      positions > completed_positions()) {
    throw std::invalid_argument(
        "MLA cache transaction publish prefix is outside its completed branch");
  }
  if (base_->pending_nonce_ != 0 || base_->version_ != expected_version_ ||
      base_->length_ != base_length_) {
    throw std::invalid_argument("MLA cache transaction base became stale");
  }
  if (working_.pending_nonce_ != 0 ||
      working_.version_ != expected_version_ + completed_positions() ||
      working_.length_ !=
          base_length_ + static_cast<std::int64_t>(completed_positions()) ||
      working_.capacity_ < working_.length_ ||
      working_.capacity_ > working_.admitted_max_context_) {
    throw std::logic_error("MLA cache transaction branch is inconsistent");
  }
  if (positions != 0 &&
      (!working_.key_storage_.defined() ||
       !working_.value_storage_.defined())) {
    throw std::logic_error("MLA cache transaction has no publishable storage");
  }
}

void MlaCacheTransaction::publish_prefix_noexcept(
    const std::size_t positions) noexcept {
  static_assert(std::is_nothrow_move_assignable_v<at::Tensor>);
  if (positions != 0) {
    base_->key_storage_ = std::move(working_.key_storage_);
    base_->value_storage_ = std::move(working_.value_storage_);
    base_->capacity_ = working_.capacity_;
    base_->length_ =
        base_length_ + static_cast<std::int64_t>(positions);
    base_->version_ = expected_version_ + positions;
    base_->next_nonce_ = working_.next_nonce_;
  }
  published_ = true;
  base_ = nullptr;
}

void MlaCacheTransaction::cancel_noexcept() noexcept {
  published_ = true;
  base_ = nullptr;
}

MlaPreparedDecode::MlaPreparedDecode(MlaPreparedDecode&& other) noexcept
    : output(std::move(other.output)),
      grown_key_storage(std::move(other.grown_key_storage)),
      grown_value_storage(std::move(other.grown_value_storage)),
      nonce(other.nonce),
      expected_version(other.expected_version),
      expected_length(other.expected_length),
      next_length(other.next_length),
      next_capacity(other.next_capacity),
      position_count(other.position_count),
      owner(other.owner),
      uses_grown_storage(other.uses_grown_storage),
      finalized(other.finalized) {
  other.nonce = 0;
  other.position_count = 0;
  other.owner = nullptr;
  other.finalized = true;
}

MlaInputBundle bundle_mla_input_weights(const MlaShape& shape,
                                        MlaWeights& weights) {
  shape.validate();
  const at::Device device = weights.query_a.data.device();
  validate_linear(weights.query_a, device, shape.q_lora_rank,
                  shape.hidden_size, "query-a projection");
  validate_linear(weights.key_value_a, device,
                  shape.kv_lora_rank + shape.qk_rope_head_dim,
                  shape.hidden_size, "key/value-a projection");
  validate_linear(weights.output_gate, device,
                  shape.num_heads * shape.value_head_dim,
                  shape.hidden_size, "output-gate projection");
  for (const MlaLinearWeight* weight :
       {&weights.query_a, &weights.key_value_a, &weights.output_gate}) {
    if (weight->encoding != MlaLinearEncoding::RowI8F32Scale) {
      throw std::invalid_argument(
          "MLA same-input bundling requires row-int8 projections");
    }
  }

  MlaInputBundle bundle;
  bundle.query_a_rows = shape.q_lora_rank;
  bundle.key_value_a_rows =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  bundle.output_gate_rows = shape.num_heads * shape.value_head_dim;
  bundle.projection = MlaLinearWeight{
      .encoding = MlaLinearEncoding::RowI8F32Scale,
      .data = at::cat({weights.query_a.data, weights.key_value_a.data,
                       weights.output_gate.data},
                      0)
                  .contiguous(),
      .row_scale = at::cat({weights.query_a.row_scale,
                            weights.key_value_a.row_scale,
                            weights.output_gate.row_scale},
                           0)
                       .contiguous(),
      // Stated rather than left implicit: the fused projection carries no original bf16 matrix.
      // GCC's -Wmissing-field-initializers rejects the omission under -Werror, while Clang does
      // not, so leaving it out builds on macOS and fails on Linux.
      .original_bf16 = {},
  };

  std::int64_t row = 0;
  const auto view = [&](const std::int64_t rows) {
    MlaLinearWeight result{
        .encoding = MlaLinearEncoding::RowI8F32Scale,
        .data = bundle.projection.data.narrow(0, row, rows),
        .row_scale = bundle.projection.row_scale.narrow(0, row, rows),
        .original_bf16 = {},
    };
    row += rows;
    return result;
  };
  weights.query_a = view(bundle.query_a_rows);
  weights.key_value_a = view(bundle.key_value_a_rows);
  weights.output_gate = view(bundle.output_gate_rows);
  validate_input_bundle(shape, bundle, device);
  return bundle;
}

MlaAbsorbedKeyValue absorb_mla_key_value(
    const MlaShape& shape, const MlaLinearWeight& key_value_b) {
  shape.validate();
  const at::Device device = key_value_b.original_bf16.defined()
      ? original_bf16_device(key_value_b.original_bf16)
      : key_value_b.data.device();
  const std::int64_t row_width =
      shape.qk_nope_head_dim + shape.value_head_dim;
  validate_linear(key_value_b, device, shape.num_heads * row_width,
                  shape.kv_lora_rank, "key/value-b projection");

  at::Tensor effective;
  switch (key_value_b.encoding) {
    case MlaLinearEncoding::DenseF32:
      effective = key_value_b.data;
      break;
    case MlaLinearEncoding::RowI8F32Scale:
      effective = key_value_b.data.to(at::kFloat) *
                  key_value_b.row_scale.unsqueeze(1);
      break;
    case MlaLinearEncoding::OriginalBf16:
      effective = materialize_original_bf16_f32(key_value_b.original_bf16);
      break;
    default:
      throw std::invalid_argument(
          "cannot absorb an unknown MLA key/value encoding");
  }
  const at::Tensor packed = effective.view(
      {shape.num_heads, row_width, shape.kv_lora_rank});
  MlaAbsorbedKeyValue absorbed{
      .key_up = packed.narrow(1, 0, shape.qk_nope_head_dim),
      .value_up = packed.narrow(1, shape.qk_nope_head_dim,
                                shape.value_head_dim),
      .source_encoding = key_value_b.encoding,
      .num_heads = shape.num_heads,
      .qk_nope_head_dim = shape.qk_nope_head_dim,
      .value_head_dim = shape.value_head_dim,
      .kv_lora_rank = shape.kv_lora_rank,
  };
  validate_absorbed_key_value(shape, absorbed, device);
  return absorbed;
}

MlaPreparedDecode prepare_mla_positions(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    const MlaAbsorbedKeyValue* absorbed_key_value,
    const bool allow_exact_query_alias,
    const MlaInputBundle* input_bundle,
    MlaExecutionTrace* execution_trace) {
  const MlaShape& shape = cache.shape_;
  shape.validate();
  if (shape.is_exact_k3() &&
      cache.representation_ == MlaCacheRepresentation::CompactLatentF32) {
    throw std::invalid_argument(
        "exact K3 compact MLA execution is disabled: its absorbed fp32 "
        "contractions are not bit-exact");
  }
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) < 1 ||
      hidden.size(2) != shape.hidden_size || hidden.device().is_meta()) {
    throw std::invalid_argument(
        "MLA native tape requires fp32 [1,T,hidden] input");
  }
  const std::int64_t positions = hidden.size(1);
  if (!cache.can_append(static_cast<std::size_t>(positions))) {
    throw std::invalid_argument(
        "MLA positions exceed the cache context/storage admission bound");
  }
  validate_weights(shape, weights, hidden.device());
  if (input_bundle != nullptr) {
    validate_input_bundle(shape, *input_bundle, hidden.device());
  }
  if (cache.representation_ == MlaCacheRepresentation::CompactLatentF32) {
    if (absorbed_key_value == nullptr) {
      throw std::invalid_argument(
          "compact MLA cache requires a bind-time absorbed key/value descriptor");
    }
    validate_absorbed_key_value(shape, *absorbed_key_value, hidden.device());
    if (absorbed_key_value->source_encoding !=
        weights.key_value_b.encoding) {
      throw std::invalid_argument(
          "compact MLA descriptor/source projection encoding disagrees");
    }
  } else if (absorbed_key_value != nullptr) {
    throw std::invalid_argument(
        "expanded MLA cache must not receive compact projection metadata");
  }
  if (cache.pending_nonce_ != 0) {
    throw std::runtime_error(
        "MLA cache already has an unfinished prepared decode");
  }
  if (cache.next_nonce_ == 0 ||
      cache.next_nonce_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("MLA cache transaction nonce is exhausted");
  }
  const std::uint64_t nonce = cache.next_nonce_++;
  cache.pending_nonce_ = nonce;

  try {
    const auto trace = [&](const MlaExecutionStage stage) {
      if (execution_trace != nullptr) {
        execution_trace->record(stage);
      }
    };
    const std::int64_t query_head_dim = shape.query_head_dim();
    const std::int64_t query_width = shape.num_heads * query_head_dim;
    const std::int64_t value_width =
        shape.num_heads * shape.value_head_dim;

    at::Tensor query_a;
    at::Tensor compressed;
    at::Tensor raw_output_gate;
    at::Tensor raw_query;
    if (input_bundle != nullptr && positions == 1) {
      trace(MlaExecutionStage::InputBundle);
      const at::Tensor combined = linear(hidden, input_bundle->projection);
      std::int64_t row = 0;
      query_a = combined.narrow(-1, row, input_bundle->query_a_rows);
      row += input_bundle->query_a_rows;
      compressed =
          combined.narrow(-1, row, input_bundle->key_value_a_rows);
      row += input_bundle->key_value_a_rows;
      raw_output_gate =
          combined.narrow(-1, row, input_bundle->output_gate_rows);
      trace(MlaExecutionStage::QueryB);
      raw_query = linear(
          rms_norm(query_a, weights.query_a_norm, shape.rms_epsilon),
          weights.query_b);
    } else if (positions == 1) {
      // Preserve the reviewed one-position submission order exactly.
      trace(MlaExecutionStage::QueryA);
      query_a = linear(hidden, weights.query_a);
      trace(MlaExecutionStage::KeyValueA);
      compressed = linear(hidden, weights.key_value_a);
      trace(MlaExecutionStage::OutputGate);
      raw_output_gate = linear(hidden, weights.output_gate);
      trace(MlaExecutionStage::QueryB);
      raw_query = linear(
          rms_norm(query_a, weights.query_a_norm, shape.rms_epsilon),
          weights.query_b);
    } else {
      // The downloaded prompt/verification forward completes each dependent
      // projection chain before beginning the next one. In particular, the
      // output gate is not submitted until attention itself is complete.
      trace(MlaExecutionStage::QueryA);
      query_a = linear(hidden, weights.query_a);
      trace(MlaExecutionStage::QueryB);
      raw_query = linear(
          rms_norm(query_a, weights.query_a_norm, shape.rms_epsilon),
          weights.query_b);
      trace(MlaExecutionStage::KeyValueA);
      compressed = linear(hidden, weights.key_value_a);
    }
    at::Tensor query_states;
    if (positions == 1 && allow_exact_query_alias &&
        raw_query.is_contiguous()) {
      // At T=1 this is the canonical [B,H,1,D] view of the unchanged adjacent
      // q_nope/q_rope columns; no values are reordered or recomputed.
      query_states =
          raw_query.view({1, shape.num_heads, 1, query_head_dim});
      if (query_states.stride(0) != query_width ||
          query_states.stride(1) != query_head_dim ||
          query_states.stride(2) != query_head_dim ||
          query_states.stride(3) != 1) {
        query_states = at::Tensor();
      }
    }
    if (!query_states.defined()) {
      const at::Tensor original =
          raw_query.view({1, positions, shape.num_heads, query_head_dim})
              .transpose(1, 2);
      const at::Tensor query_nope =
          original.narrow(-1, 0, shape.qk_nope_head_dim);
      const at::Tensor query_rope = original.narrow(
          -1, shape.qk_nope_head_dim, shape.qk_rope_head_dim);
      query_states = at::cat({query_nope, query_rope}, -1);
    }

    const at::Tensor latent =
        compressed.narrow(-1, 0, shape.kv_lora_rank);
    const at::Tensor normalized_latent =
        rms_norm(latent, weights.key_value_a_norm, shape.rms_epsilon);
    const at::Tensor key_rope = compressed.narrow(
        -1, shape.kv_lora_rank, shape.qk_rope_head_dim);

    at::Tensor new_key;
    at::Tensor new_value;
    if (cache.representation_ == MlaCacheRepresentation::ExpandedExact) {
      trace(MlaExecutionStage::KeyValueB);
      const at::Tensor expanded =
          linear(normalized_latent, weights.key_value_b);
      const at::Tensor expanded_heads =
          expanded
              .view({1, positions, shape.num_heads,
                     shape.qk_nope_head_dim + shape.value_head_dim})
              .transpose(1, 2);
      const at::Tensor key_nope =
          expanded_heads.narrow(-1, 0, shape.qk_nope_head_dim);
      new_value = expanded_heads.narrow(
          -1, shape.qk_nope_head_dim, shape.value_head_dim);
      const at::Tensor expanded_rope =
          key_rope.view({1, 1, positions, shape.qk_rope_head_dim})
              .expand({1, shape.num_heads, positions,
                       shape.qk_rope_head_dim});
      // Despite the historical qk_rope names, the reviewed K3 forward does
      // not apply rotary embedding here: these adjacent features are retained
      // unchanged. Adding RoPE would silently change the model.
      new_key = at::cat({key_nope, expanded_rope}, -1);
    } else {
      // CompactLatentF32 never changes precision: both persistent arms are
      // provider-owned fp32 views copied from the exact normalized latent and
      // unchanged positional projection.
      new_key = normalized_latent.unsqueeze(1);
      new_value = key_rope.unsqueeze(1);
    }

    const std::int64_t needed = cache.length_ + positions;
    const bool grow = cache.capacity_ < needed;
    const std::int64_t capacity =
        grow ? grown_capacity(cache, needed) : cache.capacity_;
    at::Tensor key_storage = cache.key_storage_;
    at::Tensor value_storage = cache.value_storage_;
    if (grow) {
      const auto options = hidden.options().dtype(at::kFloat);
      const auto [key_shape, value_shape] =
          cache_storage_shapes(shape, cache.representation_, capacity);
      key_storage = at::empty(key_shape, options);
      value_storage = at::empty(value_shape, options);
      if (cache.length_ != 0) {
        key_storage.narrow(2, 0, cache.length_)
            .copy_(cache.committed_keys());
        value_storage.narrow(2, 0, cache.length_)
            .copy_(cache.committed_values());
      }
    } else if (!key_storage.defined() || !value_storage.defined()) {
      throw std::logic_error("MLA cache capacity exists without storage");
    }
    key_storage.narrow(2, cache.length_, positions).copy_(new_key);
    value_storage.narrow(2, cache.length_, positions).copy_(new_value);
    const at::Tensor key_states = key_storage.narrow(2, 0, needed);
    const at::Tensor value_states = value_storage.narrow(2, 0, needed);

    const double scaling = std::pow(
        static_cast<double>(query_head_dim), -0.5);
    trace(MlaExecutionStage::Attention);
    at::Tensor attention;
#if defined(__APPLE__) && defined(DELTAFIN_HAVE_MLA_ATTN_METAL_V1)
    // Explicit env opt-in (K3_MLA_METAL_ATTENTION=1): the fused Metal
    // attention core reads the same fp32 expanded slabs and computes the same
    // scores/softmax/value math, but reassociates fp32 reductions (simd_sum
    // trees, online softmax), so it is deliberately NOT bit-exact to the
    // LibTorch einsum sequence below.  It never engages outside the exact-K3
    // expanded T=1 MPS decode, and a failed one-shot canary qualification
    // falls back to the stock path.  At T=1 the kernel writes the pre-gate
    // [1,1,12288] row directly (the transpose below is a pure view then).
    if (positions == 1 &&
        cache.representation_ == MlaCacheRepresentation::ExpandedExact &&
        shape.is_exact_k3() && hidden.device().is_mps() &&
        query_states.is_contiguous() && mla_attn_metal_decode_ready()) {
      attention = at::empty({1, 1, value_width},
                            hidden.options().dtype(at::kFloat));
      mla_attn_metal_decode_f32(attention, query_states, key_states,
                                value_states);
    }
#endif
    if (!attention.defined()) {
      at::Tensor scores;
      if (cache.representation_ == MlaCacheRepresentation::ExpandedExact) {
        scores = at::einsum("bhqd,bhkd->bhqk",
                            {query_states, key_states});
      } else {
        const at::Tensor query_nope =
            query_states.narrow(-1, 0, shape.qk_nope_head_dim);
        const at::Tensor query_rope = query_states.narrow(
            -1, shape.qk_nope_head_dim, shape.qk_rope_head_dim);
        const at::Tensor query_latent = at::einsum(
            "bhqn,hnr->bhqr", {query_nope, absorbed_key_value->key_up});
        scores =
            at::einsum("bhqr,bkr->bhqk",
                       {query_latent, key_states.squeeze(1)}) +
            at::einsum("bhqp,bkp->bhqk",
                       {query_rope, value_states.squeeze(1)});
      }
      scores = scores * scaling;
      if (positions > 1) {
        scores = scores + causal_block_mask(scores, cache.length_, positions);
      }
      const at::Tensor probabilities = at::softmax(scores, -1, at::kFloat)
                                           .to(query_states.scalar_type());
      if (cache.representation_ == MlaCacheRepresentation::ExpandedExact) {
        attention = at::einsum(
            "bhqk,bhkd->bhqd", {probabilities, value_states});
      } else {
        const at::Tensor latent_output = at::einsum(
            "bhqk,bkr->bhqr", {probabilities, key_states.squeeze(1)});
        attention = at::einsum(
            "bhqr,hvr->bhqv",
            {latent_output, absorbed_key_value->value_up});
      }
      attention = attention.transpose(1, 2).contiguous();
      attention =
          attention.reshape({1, positions, value_width}).contiguous();
    }
    if (!raw_output_gate.defined()) {
      trace(MlaExecutionStage::OutputGate);
      raw_output_gate = linear(hidden, weights.output_gate);
    }
    const at::Tensor gate = at::sigmoid(raw_output_gate);
    attention = attention * gate;
    trace(MlaExecutionStage::Output);
    const at::Tensor output = linear(attention, weights.output);

    MlaPreparedDecode prepared;
    prepared.output = output;
    if (grow) {
      prepared.grown_key_storage = std::move(key_storage);
      prepared.grown_value_storage = std::move(value_storage);
    }
    prepared.nonce = nonce;
    prepared.expected_version = cache.version_;
    prepared.expected_length = cache.length_;
    prepared.next_length = needed;
    prepared.next_capacity = capacity;
    prepared.position_count = static_cast<std::size_t>(positions);
    prepared.owner = &cache;
    prepared.uses_grown_storage = grow;
    return prepared;
  } catch (...) {
    if (cache.pending_nonce_ == nonce) {
      cache.pending_nonce_ = 0;
    }
    throw;
  }
}

MlaPreparedDecode prepare_mla_decode(const at::Tensor& hidden,
                                     const MlaWeights& weights,
                                     MlaCache& cache,
                                     const bool allow_exact_query_alias,
                                     const MlaInputBundle* input_bundle) {
  if (!hidden.defined() || hidden.dim() != 3 || hidden.size(1) != 1) {
    throw std::invalid_argument(
        "MLA decode requires exactly one [1,1,hidden] position");
  }
  if (cache.representation() != MlaCacheRepresentation::ExpandedExact) {
    throw std::invalid_argument(
        "single-position legacy MLA decode is expanded-only; compact callers must provide absorbed metadata");
  }
  return prepare_mla_positions(hidden, weights, cache, nullptr,
                               allow_exact_query_alias, input_bundle);
}

namespace {

void validate_k3_mla_entry(const MlaWeights& weights, const MlaCache& cache,
                           const MlaInputBundle* input_bundle) {
  if (!cache.shape().is_exact_k3()) {
    throw std::invalid_argument(
        "production MLA requires the exact K3 model contract");
  }
  const MlaLinearEncoding encoding = weights.query_a.encoding;
  if (encoding != MlaLinearEncoding::DenseF32 &&
      encoding != MlaLinearEncoding::RowI8F32Scale &&
      encoding != MlaLinearEncoding::OriginalBf16) {
    throw std::invalid_argument(
        "production MLA decode has an unknown projection encoding");
  }
  for (const MlaLinearWeight* weight :
       {&weights.query_a, &weights.query_b, &weights.key_value_a,
        &weights.key_value_b, &weights.output_gate, &weights.output}) {
    if (weight->encoding != encoding) {
      throw std::invalid_argument(
          "production MLA decode requires one uniform original-BF16 or row-int8 projection representation");
    }
  }
  if (input_bundle != nullptr && input_bundle->projection.encoding != encoding) {
    throw std::invalid_argument(
        "production MLA same-input bundle encoding disagrees with its projections");
  }
}

}  // namespace

MlaPreparedDecode prepare_k3_mla_decode(const at::Tensor& hidden,
                                        const MlaWeights& weights,
                                        MlaCache& cache,
                                        const MlaInputBundle* input_bundle) {
  validate_k3_mla_entry(weights, cache, input_bundle);
  if (cache.representation() != MlaCacheRepresentation::ExpandedExact) {
    throw std::invalid_argument(
        "production decode remains expanded-exact until compact MLA qualification");
  }
  return prepare_mla_decode(hidden, weights, cache, true, input_bundle);
}

MlaPreparedDecode prepare_k3_mla_positions(
    const at::Tensor& hidden, const MlaWeights& weights, MlaCache& cache,
    const MlaAbsorbedKeyValue* absorbed_key_value,
    const MlaInputBundle* input_bundle) {
  validate_k3_mla_entry(weights, cache, input_bundle);
  if (cache.representation() != MlaCacheRepresentation::ExpandedExact ||
      absorbed_key_value != nullptr) {
    throw std::invalid_argument(
        "production K3 MLA remains expanded-exact: the compact absorbed "
        "contractions reassociate fp32 reductions and are not bit-exact");
  }
  return prepare_mla_positions(hidden, weights, cache, absorbed_key_value,
                               true, input_bundle);
}

MlaDecodeShell prepare_k3_mla_decode_shell(
    const at::Tensor& hidden, MlaCache& cache) {
  const MlaShape& shape = cache.shape_;
  shape.validate();
  if (!shape.is_exact_k3() ||
      cache.representation_ != MlaCacheRepresentation::ExpandedExact) {
    throw std::invalid_argument(
        "MLA decode shell requires the exact-K3 expanded cache");
  }
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != 1 ||
      hidden.size(2) != shape.hidden_size || hidden.device().is_meta()) {
    throw std::invalid_argument(
        "MLA decode shell requires fp32 [1,1,hidden] input");
  }
  if (!cache.can_append(1)) {
    throw std::invalid_argument(
        "MLA position exceeds the cache context/storage admission bound");
  }
  if (cache.pending_nonce_ != 0) {
    throw std::runtime_error(
        "MLA cache already has an unfinished prepared decode");
  }
  if (cache.next_nonce_ == 0 ||
      cache.next_nonce_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("MLA cache transaction nonce is exhausted");
  }
  const std::uint64_t nonce = cache.next_nonce_++;
  cache.pending_nonce_ = nonce;
  try {
    const std::int64_t needed = cache.length_ + 1;
    const bool grow = cache.capacity_ < needed;
    const std::int64_t capacity =
        grow ? grown_capacity(cache, needed) : cache.capacity_;
    at::Tensor key_storage = cache.key_storage_;
    at::Tensor value_storage = cache.value_storage_;
    if (grow) {
      const auto options = hidden.options().dtype(at::kFloat);
      const auto [key_shape, value_shape] =
          cache_storage_shapes(shape, cache.representation_, capacity);
      key_storage = at::empty(key_shape, options);
      value_storage = at::empty(value_shape, options);
      if (cache.length_ != 0) {
        key_storage.narrow(2, 0, cache.length_)
            .copy_(cache.committed_keys());
        value_storage.narrow(2, 0, cache.length_)
            .copy_(cache.committed_values());
      }
    } else if (!key_storage.defined() || !value_storage.defined()) {
      throw std::logic_error("MLA cache capacity exists without storage");
    }

    MlaDecodeShell shell;
    shell.key_states = key_storage.narrow(2, 0, needed);
    shell.value_states = value_storage.narrow(2, 0, needed);
    if (grow) {
      shell.prepared.grown_key_storage = std::move(key_storage);
      shell.prepared.grown_value_storage = std::move(value_storage);
    }
    shell.prepared.nonce = nonce;
    shell.prepared.expected_version = cache.version_;
    shell.prepared.expected_length = cache.length_;
    shell.prepared.next_length = needed;
    shell.prepared.next_capacity = capacity;
    shell.prepared.position_count = 1;
    shell.prepared.owner = &cache;
    shell.prepared.uses_grown_storage = grow;
    return shell;
  } catch (...) {
    if (cache.pending_nonce_ == nonce) {
      cache.pending_nonce_ = 0;
    }
    throw;
  }
}

void commit_mla_decode(MlaCache& cache, MlaPreparedDecode& prepared) {
  if (prepared.owner != &cache || prepared.finalized || prepared.nonce == 0) {
    throw std::invalid_argument(
        "MLA prepared decode is finalized or belongs to another cache");
  }
  if (cache.pending_nonce_ != prepared.nonce ||
      cache.version_ != prepared.expected_version ||
      cache.length_ != prepared.expected_length) {
    throw std::invalid_argument("MLA prepared decode is stale");
  }
  if (prepared.position_count == 0 ||
      prepared.position_count > static_cast<std::size_t>(
                                    std::numeric_limits<std::int64_t>::max()) ||
      !prepared.output.defined() || prepared.output.dim() != 3 ||
      prepared.output.size(0) != 1 ||
      prepared.output.size(1) !=
          static_cast<std::int64_t>(prepared.position_count) ||
      prepared.output.size(2) != cache.shape_.hidden_size ||
      prepared.next_length !=
          cache.length_ + static_cast<std::int64_t>(prepared.position_count) ||
      prepared.next_capacity < prepared.next_length ||
      prepared.next_capacity > cache.admitted_max_context_) {
    throw std::invalid_argument("MLA prepared decode has invalid staged state");
  }
  if (prepared.position_count >
      std::numeric_limits<std::uint64_t>::max() - cache.version_) {
    throw std::runtime_error("MLA cache version is exhausted");
  }
  if (prepared.uses_grown_storage) {
    const auto [key_shape, value_shape] = cache_storage_shapes(
        cache.shape_, cache.representation_, prepared.next_capacity);
    if (!prepared.grown_key_storage.defined() ||
        !prepared.grown_value_storage.defined() ||
        prepared.grown_key_storage.sizes() != at::IntArrayRef(key_shape) ||
        prepared.grown_value_storage.sizes() != at::IntArrayRef(value_shape)) {
      throw std::invalid_argument(
          "MLA prepared decode has invalid grown storage");
    }
    cache.key_storage_ = std::move(prepared.grown_key_storage);
    cache.value_storage_ = std::move(prepared.grown_value_storage);
    cache.capacity_ = prepared.next_capacity;
  } else if (prepared.next_capacity != cache.capacity_) {
    throw std::invalid_argument(
        "MLA in-place prepared decode changed cache capacity");
  }
  cache.length_ = prepared.next_length;
  cache.version_ += prepared.position_count;
  cache.pending_nonce_ = 0;
  prepared.finalized = true;
  prepared.owner = nullptr;
}

void cancel_mla_decode(MlaCache& cache, MlaPreparedDecode& prepared) {
  if (prepared.owner != &cache || prepared.finalized || prepared.nonce == 0) {
    throw std::invalid_argument(
        "MLA prepared decode is finalized or belongs to another cache");
  }
  if (cache.pending_nonce_ != prepared.nonce ||
      cache.version_ != prepared.expected_version ||
      cache.length_ != prepared.expected_length) {
    throw std::invalid_argument("MLA prepared decode is stale");
  }
  cache.pending_nonce_ = 0;
  prepared.grown_key_storage = at::Tensor();
  prepared.grown_value_storage = at::Tensor();
  prepared.output = at::Tensor();
  prepared.position_count = 0;
  prepared.finalized = true;
  prepared.owner = nullptr;
}

}  // namespace deltafin::provider_internal
