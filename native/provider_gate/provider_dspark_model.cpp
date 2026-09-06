#include "provider_dspark_model.h"

#include <ATen/ops/cat.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/topk.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

// K3_DSPARK_DEBUG=1: diagnostic-only dump of the draft model's own top-5
// ids+logits per proposed row, to stderr. Off by default; checked once and
// cached (env is immutable for the process lifetime). See DIRECTIVE STEP 2
// (DSpark day) in K3-ENGINE-STATUS.md.
bool dspark_debug_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_DSPARK_DEBUG");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

// Prints "tag row=<r> pos=<p> top5=[id:logit, ...]" for a 1-D logits row.
// CPU-materializes only the requested top-k slice, never the full tensor.
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

void require_bf16(const at::Tensor& tensor, const at::Device& device,
                  const at::IntArrayRef shape, const char* name) {
  if (!tensor.defined() || tensor.device() != device ||
      tensor.scalar_type() != at::kBFloat16 || !tensor.is_contiguous() ||
      tensor.sizes() != shape) {
    throw std::invalid_argument(std::string("DSpark model ") + name +
                                " violates its contiguous BF16 contract");
  }
}

void require_linear(const at::Tensor& tensor, const at::Device& device,
                    const std::int64_t rows, const std::int64_t columns,
                    const char* name) {
  require_bf16(tensor, device, {rows, columns}, name);
}

at::Tensor dense_linear(const at::Tensor& input, const at::Tensor& weight) {
  if (input.dim() != 2 || weight.dim() != 2 ||
      input.size(1) != weight.size(1)) {
    throw std::invalid_argument("DSpark model linear dimensions disagree");
  }
  return at::linear(input, weight, std::nullopt);
}

void validate_layer(const DSparkDecoderWeights& layer,
                    const DSparkShape& shape, const at::Device& device) {
  const std::int64_t query_width =
      shape.num_heads * shape.query_head_dim();
  const std::int64_t compressed =
      shape.kv_lora_rank + shape.qk_rope_head_dim;
  const std::int64_t expanded =
      shape.num_heads * (shape.qk_nope_head_dim + shape.value_head_dim);
  require_bf16(layer.input_norm, device, {shape.hidden_size},
               "input norm");
  require_linear(layer.attention.query_a, device, shape.q_lora_rank,
                 shape.hidden_size, "query-a weight");
  require_bf16(layer.attention.query_a_norm, device, {shape.q_lora_rank},
               "query-a norm");
  require_linear(layer.attention.query_b, device, query_width,
                 shape.q_lora_rank, "query-b weight");
  require_linear(layer.attention.key_value_a, device, compressed,
                 shape.hidden_size, "key/value-a weight");
  require_bf16(layer.attention.key_value_a_norm, device,
               {shape.kv_lora_rank}, "key/value-a norm");
  require_linear(layer.attention.key_value_b, device, expanded,
                 shape.kv_lora_rank, "key/value-b weight");
  require_linear(layer.attention.output, device, shape.hidden_size,
                 shape.num_heads * shape.value_head_dim, "attention output");
  require_bf16(layer.post_attention_norm, device, {shape.hidden_size},
               "post-attention norm");
  require_linear(layer.mlp.gate, device, shape.intermediate_size,
                 shape.hidden_size, "MLP gate");
  require_linear(layer.mlp.up, device, shape.intermediate_size,
                 shape.hidden_size, "MLP up");
  require_linear(layer.mlp.down, device, shape.hidden_size,
                 shape.intermediate_size, "MLP down");
}

bool same_storage(
    const std::array<at::Tensor, kDSparkLayers>& left,
    const std::array<at::Tensor, kDSparkLayers>& right) noexcept {
  for (std::size_t index = 0; index < kDSparkLayers; ++index) {
    if (left[index].unsafeGetTensorImpl() !=
        right[index].unsafeGetTensorImpl()) {
      return false;
    }
  }
  return true;
}

void require_positions(const at::Tensor& positions, const at::Device& device,
                       const std::int64_t start, const std::int64_t rows,
                       const std::int64_t maximum, const char* name) {
  if (!positions.defined() || positions.device() != device ||
      positions.scalar_type() != at::kLong || !positions.is_contiguous() ||
      positions.sizes() != at::IntArrayRef({rows})) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " positions violate their int64 row contract");
  }
  if (start < 0 || rows < 1 || start > maximum - rows) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " exceeds the context limit");
  }
  const at::Tensor expected =
      at::arange(start, start + rows, positions.options());
  if (!(positions == expected).all().item<bool>()) {
    throw std::invalid_argument(std::string("DSpark ") + name +
                                " must extend the exact cache boundary");
  }
}

void validate_target_io(const DSparkTargetIo& io, const DSparkShape& shape,
                        const at::Device& device) {
  if (io.embedding.quantized.defined() &&
      (io.embedding.quantized.device() != device ||
      io.embedding.quantized.scalar_type() != at::kChar ||
      !io.embedding.quantized.is_contiguous() ||
      io.embedding.quantized.sizes() !=
          at::IntArrayRef({shape.vocabulary_size, shape.hidden_size}) ||
      !io.embedding.row_scales.defined() ||
      io.embedding.row_scales.device() != device ||
      io.embedding.row_scales.scalar_type() != at::kFloat ||
      !io.embedding.row_scales.is_contiguous() ||
      io.embedding.row_scales.sizes() !=
          at::IntArrayRef({shape.vocabulary_size}))) {
    throw std::invalid_argument(
        "DSpark shared target embedding has an invalid resident shape");
  }
  if (io.head_packed_int8_qualified) {
    const at::Tensor& head = io.language_model_head.quantized;
    if (!head.defined() || head.device() != device ||
        head.scalar_type() != at::kChar || !head.is_contiguous() ||
        head.sizes() !=
            at::IntArrayRef({shape.vocabulary_size, shape.hidden_size}) ||
        io.language_model_head.dense_f32.defined() ||
        io.language_model_head.original_bf16.defined() ||
        !io.language_model_head.row_scales.defined() ||
       io.language_model_head.row_scales.device() != device ||
       io.language_model_head.row_scales.scalar_type() != at::kFloat ||
       !io.language_model_head.row_scales.is_contiguous() ||
       io.language_model_head.row_scales.sizes() !=
            at::IntArrayRef({shape.vocabulary_size})) {
      throw std::invalid_argument(
          "DSpark shared packed target head has invalid storage");
    }
  } else {
    const bool dense = io.language_model_head.dense_f32.defined();
    const bool original = io.language_model_head.original_bf16.defined();
    if (dense == original || io.language_model_head.quantized.defined() ||
        io.language_model_head.row_scales.defined() ||
        (dense &&
         (io.language_model_head.dense_f32.device() != device ||
          io.language_model_head.dense_f32.scalar_type() != at::kFloat ||
          !io.language_model_head.dense_f32.is_contiguous() ||
          io.language_model_head.dense_f32.sizes() !=
              at::IntArrayRef(
                  {shape.vocabulary_size, shape.hidden_size}))) ||
        (original &&
         !original_bf16_matrix_matches(
             io.language_model_head.original_bf16, device,
             static_cast<std::size_t>(shape.vocabulary_size),
             static_cast<std::size_t>(shape.hidden_size)))) {
      throw std::invalid_argument(
          "DSpark shared original target head has invalid storage");
    }
  }
  if (io.exact_k3 && !shape.is_exact_k3()) {
    throw std::invalid_argument(
        "DSpark exact target I/O requires exact K3 DSpark geometry");
  }
}

}  // namespace

DSparkModelWeights prepare_dspark_model_weights(const DSparkShape& shape,
                                                DSparkModelWeights weights) {
  const c10::InferenceMode inference_guard;
  shape.validate();
  if (!weights.context_projection.defined() ||
      weights.context_projection.device().is_meta()) {
    throw std::invalid_argument(
        "DSpark context projection must be materialized before binding");
  }
  const at::Device device = weights.context_projection.device();
  require_linear(weights.context_projection, device, shape.hidden_size,
                 shape.target_context_width(), "context projection");
  require_bf16(weights.context_norm, device, {shape.hidden_size},
               "context norm");
  std::vector<at::Tensor> context_weights;
  context_weights.reserve(kDSparkLayers);
  for (const DSparkDecoderWeights& layer : weights.layers) {
    validate_layer(layer, shape, device);
    context_weights.push_back(layer.attention.key_value_a);
  }
  require_bf16(weights.final_norm, device, {shape.hidden_size}, "final norm");
  require_linear(weights.markov_embedding, device, shape.vocabulary_size,
                 shape.markov_rank, "Markov embedding");
  require_linear(weights.markov_output, device, shape.vocabulary_size,
                 shape.markov_rank, "Markov output");
  require_linear(weights.confidence_weight, device, 1,
                 shape.hidden_size + shape.markov_rank,
                 "confidence weight");
  require_bf16(weights.confidence_bias, device, {1}, "confidence bias");
  weights.fused_context_projection =
      at::cat(context_weights, 0).contiguous();
  require_linear(weights.fused_context_projection, device,
                 static_cast<std::int64_t>(kDSparkLayers) *
                     (shape.kv_lora_rank + shape.qk_rope_head_dim),
                 shape.hidden_size, "fused context projection");
  return weights;
}

DSparkLatentContext DSparkCacheSnapshot::context(
    const std::size_t layer) const {
  if (layer >= kDSparkLayers || length < 0 || capacity < length) {
    throw std::invalid_argument("DSpark cache snapshot view is invalid");
  }
  return DSparkLatentContext{
      .latent = latent_storage[layer].narrow(0, 0, length),
      .positional = positional_storage[layer].narrow(0, 0, length),
  };
}

DSparkCache::DSparkCache(DSparkShape shape, at::Device device)
    : shape_(std::move(shape)), device_(device) {
  shape_.validate();
  if (device_.is_meta()) {
    throw std::invalid_argument("DSpark cache requires a concrete device");
  }
  const auto options =
      at::TensorOptions().dtype(at::kBFloat16).device(device_);
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    latent_storage_[layer] = at::empty({0, shape_.kv_lora_rank}, options);
    positional_storage_[layer] =
        at::empty({0, shape_.qk_rope_head_dim}, options);
  }
}

std::int64_t DSparkCache::length() const noexcept { return length_; }
std::int64_t DSparkCache::capacity() const noexcept { return capacity_; }
std::uint64_t DSparkCache::generation() const noexcept { return generation_; }
bool DSparkCache::requires_fork() const noexcept { return requires_fork_; }

DSparkLatentContext DSparkCache::context(const std::size_t layer) const {
  if (layer >= kDSparkLayers) {
    throw std::out_of_range("DSpark cache layer index is out of range");
  }
  return DSparkLatentContext{
      .latent = latent_storage_[layer].narrow(0, 0, length_),
      .positional = positional_storage_[layer].narrow(0, 0, length_),
  };
}

DSparkCacheSnapshot DSparkCache::snapshot() const {
  return DSparkCacheSnapshot{
      .owner = this,
      .generation = generation_,
      .length = length_,
      .capacity = capacity_,
      .latent_storage = latent_storage_,
      .positional_storage = positional_storage_,
  };
}

void DSparkCache::restore(const DSparkCacheSnapshot& snapshot) {
  if (snapshot.owner != this || snapshot.length < 0 ||
      snapshot.capacity < snapshot.length ||
      snapshot.capacity > shape_.max_position) {
    throw std::invalid_argument("DSpark cache snapshot is unowned or corrupt");
  }
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    require_bf16(snapshot.latent_storage[layer], device_,
                 {snapshot.capacity, shape_.kv_lora_rank},
                 "snapshot latent slab");
    require_bf16(snapshot.positional_storage[layer], device_,
                 {snapshot.capacity, shape_.qk_rope_head_dim},
                 "snapshot positional slab");
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max() ||
      snapshot.generation == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("DSpark cache generation is exhausted");
  }
  const bool same_boundary =
      length_ == snapshot.length && capacity_ == snapshot.capacity &&
      same_storage(latent_storage_, snapshot.latent_storage) &&
      same_storage(positional_storage_, snapshot.positional_storage);
  latent_storage_ = snapshot.latent_storage;
  positional_storage_ = snapshot.positional_storage;
  length_ = snapshot.length;
  capacity_ = snapshot.capacity;
  if (!same_boundary) {
    requires_fork_ = true;
  }
  generation_ = std::max(generation_, snapshot.generation) + 1;
}

void DSparkCache::clear() {
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("DSpark cache generation is exhausted");
  }
  const auto options =
      at::TensorOptions().dtype(at::kBFloat16).device(device_);
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    latent_storage_[layer] = at::empty({0, shape_.kv_lora_rank}, options);
    positional_storage_[layer] =
        at::empty({0, shape_.qk_rope_head_dim}, options);
  }
  length_ = 0;
  capacity_ = 0;
  requires_fork_ = false;
  ++generation_;
}

void DSparkCache::ensure_capacity(const std::int64_t required) {
  if (required <= capacity_ && !requires_fork_) {
    return;
  }
  if (required < 1 || required > shape_.max_position) {
    throw std::invalid_argument("DSpark cache capacity request is invalid");
  }
  std::int64_t next = std::max<std::int64_t>(8, capacity_);
  while (next < required) {
    if (next > shape_.max_position / 2) {
      next = shape_.max_position;
      break;
    }
    next *= 2;
  }
  next = std::min(next, shape_.max_position);
  const auto options =
      at::TensorOptions().dtype(at::kBFloat16).device(device_);
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    at::Tensor latent = at::empty({next, shape_.kv_lora_rank}, options);
    at::Tensor positional =
        at::empty({next, shape_.qk_rope_head_dim}, options);
    if (length_ != 0) {
      latent.narrow(0, 0, length_)
          .copy_(latent_storage_[layer].narrow(0, 0, length_));
      positional.narrow(0, 0, length_)
          .copy_(positional_storage_[layer].narrow(0, 0, length_));
    }
    latent_storage_[layer] = std::move(latent);
    positional_storage_[layer] = std::move(positional);
  }
  capacity_ = next;
  requires_fork_ = false;
}

void DSparkCache::append(
    const std::array<DSparkLatentContext, kDSparkLayers>& rows) {
  const std::int64_t count = rows[0].latent.size(0);
  if (count < 1 || count > shape_.max_position - length_) {
    throw std::invalid_argument("DSpark cache append row count is invalid");
  }
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    require_bf16(rows[layer].latent, device_,
                 {count, shape_.kv_lora_rank}, "appended latent rows");
    require_bf16(rows[layer].positional, device_,
                 {count, shape_.qk_rope_head_dim},
                 "appended positional rows");
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("DSpark cache generation is exhausted");
  }
  ensure_capacity(length_ + count);
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    latent_storage_[layer].narrow(0, length_, count).copy_(rows[layer].latent);
    positional_storage_[layer]
        .narrow(0, length_, count)
        .copy_(rows[layer].positional);
  }
  length_ += count;
  ++generation_;
}

DSparkModel::DSparkModel(DSparkShape shape, DSparkModelWeights weights,
                         DSparkTargetIo target_io)
    : shape_(std::move(shape)),
      weights_(prepare_dspark_model_weights(shape_, std::move(weights))),
      target_io_(std::move(target_io)),
      cache_(shape_, weights_.context_projection.device()) {
  validate_target_io(target_io_, shape_, weights_.context_projection.device());
}

const DSparkShape& DSparkModel::shape() const noexcept { return shape_; }
DSparkCache& DSparkModel::cache() noexcept { return cache_; }
const DSparkCache& DSparkModel::cache() const noexcept { return cache_; }

void DSparkModel::append_target_context(
    const at::Tensor& combined_target_hidden, const at::Tensor& positions) {
  const c10::InferenceMode inference_guard;
  if (!combined_target_hidden.defined() ||
      combined_target_hidden.scalar_type() != at::kBFloat16 ||
      !combined_target_hidden.is_contiguous() ||
      combined_target_hidden.device() != weights_.context_projection.device() ||
      combined_target_hidden.dim() != 2 ||
      combined_target_hidden.size(0) < 1 ||
      combined_target_hidden.size(1) != shape_.target_context_width()) {
    throw std::invalid_argument(
        "DSpark combined target context must be contiguous BF16 [T,5*H]");
  }
  const std::int64_t rows = combined_target_hidden.size(0);
  require_positions(positions, combined_target_hidden.device(), cache_.length(),
                    rows, shape_.max_position, "target context");
  const at::Tensor context = dspark_rms_norm_bf16(
      dense_linear(combined_target_hidden, weights_.context_projection),
      weights_.context_norm, shape_.rms_epsilon);
  const at::Tensor stacked =
      dense_linear(context, weights_.fused_context_projection);
  const std::int64_t compressed =
      shape_.kv_lora_rank + shape_.qk_rope_head_dim;
  std::array<DSparkLatentContext, kDSparkLayers> projected;
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    const at::Tensor slice =
        stacked.narrow(-1, static_cast<std::int64_t>(layer) * compressed,
                       compressed);
    projected[layer].latent = dspark_rms_norm_bf16(
        slice.narrow(-1, 0, shape_.kv_lora_rank).contiguous(),
        weights_.layers[layer].attention.key_value_a_norm,
        shape_.rms_epsilon);
    projected[layer].positional = dspark_apply_yarn_rotary_bf16(
        slice
            .narrow(-1, shape_.kv_lora_rank, shape_.qk_rope_head_dim)
            .contiguous(),
        positions, shape_,
        // Context appends carry the full committed prefix — a chat prefill
        // is arbitrarily many rows; positions were already range-checked
        // against the cache above. The trained 7-row ceiling applies only
        // to draft query/backbone forwards, not to appended context.
        shape_.max_position);
  }
  cache_.append(projected);
}

at::Tensor DSparkModel::forward_backbone(
    const at::Tensor& query_token_ids, const at::Tensor& positions) const {
  const c10::InferenceMode inference_guard;
  if (!query_token_ids.defined() ||
      query_token_ids.scalar_type() != at::kLong ||
      !query_token_ids.is_contiguous() || query_token_ids.dim() != 1 ||
      query_token_ids.size(0) < 1 ||
      query_token_ids.size(0) > kDSparkTrainedQueryRows ||
      query_token_ids.device() != weights_.context_projection.device()) {
    throw std::invalid_argument(
        "DSpark backbone IDs must be contiguous int64 [1..7]");
  }
  require_positions(positions, query_token_ids.device(), cache_.length(),
                    query_token_ids.size(0), shape_.max_position,
                    "backbone query");
  at::Tensor hidden = target_embedding_rows(
                          query_token_ids, target_io_.embedding,
                          target_io_.exact_k3)
                          .to(at::kBFloat16);
  return forward_backbone_embeddings(hidden, positions);
}

at::Tensor DSparkModel::forward_backbone_embeddings(
    const at::Tensor& query_embeddings, const at::Tensor& positions) const {
  const c10::InferenceMode inference_guard;
  if (!query_embeddings.defined() ||
      query_embeddings.scalar_type() != at::kBFloat16 ||
      !query_embeddings.is_contiguous() || query_embeddings.dim() != 2 ||
      query_embeddings.size(0) < 1 ||
      query_embeddings.size(0) > kDSparkTrainedQueryRows ||
      query_embeddings.size(1) != shape_.hidden_size ||
      query_embeddings.device() != weights_.context_projection.device()) {
    throw std::invalid_argument(
        "DSpark backbone embeddings must be contiguous BF16 [1..7,H]");
  }
  require_positions(positions, query_embeddings.device(), cache_.length(),
                    query_embeddings.size(0), shape_.max_position,
                    "backbone embedding query");
  at::Tensor hidden = query_embeddings;
  at::Tensor residual;
  for (std::size_t layer = 0; layer < kDSparkLayers; ++layer) {
    DSparkDecoderOutput output = run_dspark_decoder_layer(
        hidden, residual, positions, cache_.context(layer),
        weights_.layers[layer], shape_);
    hidden = std::move(output.hidden);
    residual = std::move(output.residual);
  }
  return dspark_rms_norm_bf16(residual + hidden, weights_.final_norm,
                              shape_.rms_epsilon);
}

DSparkProposalScores DSparkModel::propose(const std::int64_t anchor_token_id,
                                          const std::int64_t score_rows) const {
  const c10::InferenceMode inference_guard;
  if (anchor_token_id < 0 || anchor_token_id >= shape_.vocabulary_size) {
    throw std::invalid_argument("DSpark proposal anchor is out of range");
  }
  if (score_rows < 1 || score_rows > kDSparkTrainedQueryRows) {
    throw std::invalid_argument("DSpark scored prefix must contain 1..7 rows");
  }
  if (cache_.length() > shape_.max_position - kDSparkTrainedQueryRows) {
    throw std::invalid_argument(
        "DSpark trained query would exceed the context limit");
  }
  const at::Device device = weights_.context_projection.device();
  at::Tensor query_ids = at::full(
      {kDSparkTrainedQueryRows}, shape_.mask_token_id,
      at::TensorOptions().dtype(at::kLong).device(device));
  query_ids.index_put_({0}, anchor_token_id);
  const at::Tensor embeddings = target_embedding_rows(
                                    query_ids, target_io_.embedding,
                                    target_io_.exact_k3)
                                    .to(at::kBFloat16);
  return propose_from_embeddings(anchor_token_id, score_rows, embeddings);
}

DSparkProposalScores DSparkModel::propose_from_embeddings(
    const std::int64_t anchor_token_id, const std::int64_t score_rows,
    const at::Tensor& trained_query_embeddings) const {
  const c10::InferenceMode inference_guard;
  if (anchor_token_id < 0 || anchor_token_id >= shape_.vocabulary_size) {
    throw std::invalid_argument("DSpark proposal anchor is out of range");
  }
  if (score_rows < 1 || score_rows > kDSparkTrainedQueryRows) {
    throw std::invalid_argument("DSpark scored prefix must contain 1..7 rows");
  }
  if (!trained_query_embeddings.defined() ||
      trained_query_embeddings.scalar_type() != at::kBFloat16 ||
      !trained_query_embeddings.is_contiguous() ||
      trained_query_embeddings.sizes() !=
          at::IntArrayRef({kDSparkTrainedQueryRows, shape_.hidden_size}) ||
      trained_query_embeddings.device() !=
          weights_.context_projection.device()) {
    throw std::invalid_argument(
        "DSpark proposal requires exactly seven BF16 K3 embedding rows");
  }
  if (cache_.length() > shape_.max_position - kDSparkTrainedQueryRows) {
    throw std::invalid_argument(
        "DSpark trained query would exceed the context limit");
  }
  const at::Device device = weights_.context_projection.device();
  const at::Tensor positions =
      at::arange(cache_.length(), cache_.length() + kDSparkTrainedQueryRows,
                 at::TensorOptions().dtype(at::kLong).device(device));
  const at::Tensor hidden =
      forward_backbone_embeddings(trained_query_embeddings, positions);
  const at::Tensor scored_hidden = hidden.narrow(0, 0, score_rows).contiguous();
  const at::Tensor logits = target_language_model_head_rows(
      scored_hidden, target_io_.language_model_head,
      target_io_.head_packed_int8_qualified, target_io_.exact_k3);

  at::Tensor previous = at::tensor(
      {anchor_token_id}, at::TensorOptions().dtype(at::kLong).device(device));
  std::vector<at::Tensor> proposed;
  std::vector<at::Tensor> previous_embeddings;
  proposed.reserve(static_cast<std::size_t>(score_rows));
  previous_embeddings.reserve(static_cast<std::size_t>(score_rows));
  const bool debug = dspark_debug_enabled();
  const std::int64_t anchor_position = cache_.length();
  if (debug) {
    std::cerr << "K3_DSPARK_DEBUG draft_anchor anchor_token_id=" << anchor_token_id
              << " anchor_position=" << anchor_position
              << " cache_generation=" << cache_.generation() << "\n";
  }
  for (std::int64_t row = 0; row < score_rows; ++row) {
    const at::Tensor embedding =
        weights_.markov_embedding.index_select(0, previous);
    previous_embeddings.push_back(embedding);
    const at::Tensor bias =
        dense_linear(embedding, weights_.markov_output).squeeze(0);
    const at::Tensor combined = logits[row] + bias.to(at::kFloat);
    if (debug) {
      // "raw" = the parallel multi-token head alone; "combined" = raw plus
      // the sequential Markov correction that actually decides the token
      // (hypothesis-adjacent: if raw and combined top-1 disagree often, the
      // Markov correction is dominating/destabilizing the head's own signal).
      debug_print_topk("draft_raw", row, anchor_position + row, logits[row]);
      debug_print_topk("draft_combined", row, anchor_position + row, combined);
    }
    previous = at::argmax(combined).reshape({1});
    proposed.push_back(previous);
  }
  const at::Tensor token_ids = at::cat(proposed, 0).contiguous();
  const at::Tensor markov = at::cat(previous_embeddings, 0);
  const at::Tensor features = at::cat({scored_hidden, markov}, -1);
  const at::Tensor confidence =
      (dense_linear(features, weights_.confidence_weight) +
       weights_.confidence_bias)
          .squeeze(-1)
          .to(at::kFloat)
          .contiguous();
  return DSparkProposalScores{
      .token_ids = token_ids,
      .confidence_logits = confidence,
      .anchor_token_id = anchor_token_id,
      .anchor_position = cache_.length(),
      .cache_generation = cache_.generation(),
  };
}

}  // namespace deltafin::provider_internal
