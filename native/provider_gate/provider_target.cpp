#include "provider_target.h"
#include "provider_cuda_moe.h"

#include <ATen/ops/_weight_int8pack_mm.h>
#include <ATen/ops/add.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/div.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tanh.h>
#include <c10/core/InferenceMode.h>

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kK3Hidden = 7168;
constexpr std::int64_t kK3DenseIntermediate = 33792;
constexpr std::int64_t kK3Vocabulary = 163840;
constexpr std::uint32_t kK3Layers = 93;
constexpr std::uint32_t kResidualBlock = 12;
constexpr float kRmsEpsilon = 1.0e-5F;
constexpr float kSituBeta = 4.0F;
constexpr float kSituLinearBeta = 25.0F;
constexpr std::array<std::int64_t, 1> kLastDimension{-1};

void require_f32(const at::Tensor &tensor, const at::Device &device,
                 const at::IntArrayRef shape, const char *name) {
  if (!tensor.defined() || tensor.scalar_type() != at::kFloat ||
      tensor.device() != device || !tensor.is_contiguous() ||
      tensor.sizes() != shape) {
    throw std::invalid_argument(
        std::string(name) +
        " violates its contiguous fp32 shape/device contract");
  }
}

void require_linear(const MoeRowInt8Matrix &matrix, const at::Device &device,
                    const std::int64_t rows, const std::int64_t columns,
                    const bool packed, const char *name) {
  if (rows <= 0 || columns <= 0) {
    throw std::invalid_argument(std::string(name) +
                                " has invalid matrix dimensions");
  }
  if (packed) {
    if (!matrix.quantized.defined() ||
        matrix.quantized.scalar_type() != at::kChar ||
        matrix.quantized.device() != device ||
        !matrix.quantized.is_contiguous() ||
        matrix.quantized.sizes() != at::IntArrayRef({rows, columns}) ||
        !matrix.row_scales.defined() ||
        matrix.row_scales.scalar_type() != at::kFloat ||
        matrix.row_scales.device() != device ||
        !matrix.row_scales.is_contiguous() ||
        matrix.row_scales.sizes() != at::IntArrayRef({rows})) {
      throw std::invalid_argument(
          std::string(name) + " violates its row-int8 shape/device contract");
    }
    return;
  }
  const bool dense = matrix.dense_f32.defined();
  const bool original = matrix.original_bf16.defined();
  if (dense == original) {
    throw std::invalid_argument(
        std::string(name) +
        " needs exactly one dense-fp32 or original-BF16 representation");
  }
  if (dense &&
      (matrix.dense_f32.scalar_type() != at::kFloat ||
       matrix.dense_f32.device() != device ||
       !matrix.dense_f32.is_contiguous() ||
       matrix.dense_f32.sizes() != at::IntArrayRef({rows, columns}))) {
    throw std::invalid_argument(std::string(name) +
                                " has an invalid dense fp32 weight");
  }
  if (original &&
      !original_bf16_matrix_matches(
          matrix.original_bf16, device, static_cast<std::size_t>(rows),
          static_cast<std::size_t>(columns))) {
    throw std::invalid_argument(std::string(name) +
                                " has an invalid original-BF16 weight");
  }
}

std::int64_t linear_rows(const MoeRowInt8Matrix &matrix) {
  if (matrix.original_bf16.defined()) {
    return static_cast<std::int64_t>(matrix.original_bf16.rows);
  }
  const at::Tensor &storage =
      matrix.quantized.defined() ? matrix.quantized : matrix.dense_f32;
  return storage.dim() == 2 ? storage.size(0) : 0;
}

at::Tensor linear(const at::Tensor &input, const MoeRowInt8Matrix &weight,
                  const bool packed) {
  if (packed) {
    return deltafin::provider_internal::cuda_int8_linear(
        input, weight.quantized, weight.row_scales);
  }
  if (weight.original_bf16.defined()) {
    return original_bf16_linear(input, weight.original_bf16);
  }
  // Dense fallback creation belongs to weight binding, never the token path.
  return at::linear(input, weight.dense_f32, std::nullopt);
}

at::Tensor language_model_head_linear(
    const at::Tensor &input, const MoeRowInt8Matrix &weight,
    const bool packed) {
  if (packed || weight.original_bf16.defined()) {
    return linear(input, weight, packed);
  }
  // The established runtime deliberately uses hidden @ head.T here. Layer
  // projections above are nn.Linear/F.linear and retain at::linear instead.
  return at::matmul(input, weight.dense_f32.transpose(0, 1));
}

at::Tensor rms_norm(const at::Tensor &input, const at::Tensor &weight) {
  const at::Tensor variance = at::mean(at::pow(input, 2), kLastDimension, true);
  const at::Tensor normalized =
      at::mul(input, at::rsqrt(at::add(variance, kRmsEpsilon)));
  // KimiRMSNorm spells this as weight * normalized; retain that order.
  return at::mul(weight, normalized);
}

at::Tensor situ(const at::Tensor &gate, const at::Tensor &up) {
  const at::Tensor bounded_gate =
      at::mul(at::mul(at::tanh(at::div(gate, kSituBeta)), kSituBeta),
              at::sigmoid(gate));
  const at::Tensor bounded_up =
      at::mul(at::tanh(at::div(up, kSituLinearBeta)), kSituLinearBeta);
  return at::mul(bounded_gate, bounded_up);
}

at::Tensor apply_attention_residual(const at::Tensor &prefix_sum,
                                    const at::Tensor &anchors,
                                    const at::Tensor &score_weight) {
  const at::Tensor values = at::cat({anchors, prefix_sum.unsqueeze(1)}, 1);
  const at::Tensor variance =
      at::mean(at::pow(values, 2), kLastDimension, true);
  const at::Tensor keys =
      at::mul(values, at::rsqrt(at::add(variance, kRmsEpsilon)));
  const at::Tensor scores =
      at::sum(at::mul(keys, score_weight), kLastDimension, false);
  const at::Tensor probabilities =
      at::softmax(scores, -1, at::kFloat).unsqueeze(1);
  return at::matmul(probabilities, values).squeeze(1);
}

void validate_residual_weights(const TargetResidualWeights &weights,
                               const at::Device &device,
                               const std::int64_t hidden) {
  require_f32(weights.input_norm, device, {hidden}, "input norm");
  require_f32(weights.self_attention_res_norm, device, {hidden},
              "attention-residual norm");
  require_f32(weights.self_attention_res_projection, device, {1, hidden},
              "attention-residual projection");
  require_f32(weights.post_attention_norm, device, {hidden},
              "post-attention norm");
  require_f32(weights.mlp_res_norm, device, {hidden}, "MLP-residual norm");
  require_f32(weights.mlp_res_projection, device, {1, hidden},
              "MLP-residual projection");
  const bool has_attention_score =
      weights.self_attention_score_weight.defined();
  const bool has_mlp_score = weights.mlp_score_weight.defined();
  if (has_attention_score != has_mlp_score) {
    throw std::invalid_argument(
        "target residual bind supplied only one prepared score weight");
  }
  if (has_attention_score) {
    require_f32(weights.self_attention_score_weight, device, {hidden},
                "prepared attention-residual score weight");
    require_f32(weights.mlp_score_weight, device, {hidden},
                "prepared MLP-residual score weight");
  }
}

const at::Tensor &residual_score_weight(const at::Tensor &prepared,
                                        const at::Tensor &norm,
                                        const at::Tensor &projection,
                                        at::Tensor &temporary) {
  if (prepared.defined()) {
    return prepared;
  }
  temporary = at::mul(norm, projection.squeeze(0));
  return temporary;
}

void validate_hidden(const at::Tensor &hidden, const std::int64_t width,
                     const char *name) {
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      !hidden.is_contiguous() ||
      hidden.sizes() != at::IntArrayRef({1, width})) {
    throw std::invalid_argument(std::string(name) +
                                " must be contiguous fp32 [1,width]");
  }
}

void validate_hidden_rows(const at::Tensor &hidden, const std::int64_t width,
                          const char *name) {
  if (!hidden.defined() || hidden.scalar_type() != at::kFloat ||
      !hidden.is_contiguous() || hidden.dim() != 2 || hidden.size(0) < 1 ||
      hidden.size(0) > 64 || hidden.size(1) != width) {
    throw std::invalid_argument(std::string(name) +
                                " must be contiguous fp32 [1..64,width]");
  }
}

void validate_anchor_rows(const at::Tensor &anchors,
                          const at::Device &device,
                          const std::int64_t rows,
                          const std::int64_t hidden) {
  if (!anchors.defined() || anchors.scalar_type() != at::kFloat ||
      anchors.device() != device || !anchors.is_contiguous() ||
      anchors.dim() != 3 || anchors.size(0) != rows ||
      anchors.size(2) != hidden || anchors.size(1) > 8) {
    throw std::invalid_argument(
        "target residual rows violate [T,0..8,hidden] fp32 contract");
  }
}

void validate_hidden_on_device(const at::Tensor &hidden,
                               const std::int64_t width,
                               const at::Device &device, const char *name) {
  validate_hidden(hidden, width, name);
  if (hidden.device() != device) {
    throw std::invalid_argument(std::string(name) +
                                " is on the wrong provider device");
  }
}

void validate_anchors(const at::Tensor &anchors, const at::Device &device,
                      const std::int64_t hidden) {
  if (!anchors.defined() || anchors.scalar_type() != at::kFloat ||
      anchors.device() != device || !anchors.is_contiguous() ||
      anchors.dim() != 3 || anchors.size(0) != 1 || anchors.size(2) != hidden ||
      anchors.size(1) > 8) {
    throw std::invalid_argument(
        "target block residual violates [1,0..8,hidden] fp32 contract");
  }
}

constexpr std::int64_t
expected_anchor_count_before(const std::uint32_t layer_index) {
  // Layer zero sees the empty tape. Every prior multiple of twelve has
  // appended exactly one authoritative pre-attention hidden row.
  return static_cast<std::int64_t>((layer_index + kResidualBlock - 1) /
                                   kResidualBlock);
}

constexpr std::int64_t
expected_anchor_count_after(const std::uint32_t layer_index) {
  return static_cast<std::int64_t>(layer_index / kResidualBlock + 1);
}

void require_exact_anchor_count(const at::Tensor &anchors,
                                const std::int64_t expected,
                                const char *phase) {
  if (anchors.size(1) != expected) {
    throw std::invalid_argument(std::string("target residual tape has ") +
                                std::to_string(anchors.size(1)) +
                                " anchors during " + phase + ", expected " +
                                std::to_string(expected));
  }
}

} // namespace

TargetResidualWeights
precompute_target_residual_score_weights(TargetResidualWeights weights) {
  const c10::InferenceMode inference_guard;
  const std::int64_t hidden =
      weights.input_norm.dim() == 1 ? weights.input_norm.size(0) : 0;
  const at::Device device = weights.input_norm.device();
  validate_residual_weights(weights, device, hidden);
  weights.self_attention_score_weight =
      at::mul(weights.self_attention_res_norm,
              weights.self_attention_res_projection.squeeze(0))
          .contiguous();
  weights.mlp_score_weight =
      at::mul(weights.mlp_res_norm, weights.mlp_res_projection.squeeze(0))
          .contiguous();
  return weights;
}

TargetDenseWeights bundle_target_dense_gate_up(TargetDenseWeights weights,
                                               const bool qualified,
                                               const bool enable_experimental) {
  const c10::InferenceMode inference_guard;
  const std::int64_t rows =
      weights.gate.quantized.dim() == 2 ? weights.gate.quantized.size(0) : 0;
  const std::int64_t columns =
      weights.gate.quantized.dim() == 2 ? weights.gate.quantized.size(1) : 0;
  const at::Device device = weights.gate.quantized.device();
  require_linear(weights.gate, device, rows, columns,
                 weights.packed_int8_qualified, "dense gate");
  require_linear(weights.up, device, rows, columns,
                 weights.packed_int8_qualified, "dense up");
  if (weights.gate.quantized.sizes() != weights.up.quantized.sizes()) {
    throw std::invalid_argument(
        "dense gate/up bundle requires identical matrix shapes");
  }
  if (qualified && !weights.packed_int8_qualified) {
    throw std::invalid_argument(
        "dense gate/up one-call qualification requires packed int8");
  }

  MoeRowInt8Matrix bundle{
      at::cat({weights.gate.quantized, weights.up.quantized}, 0).contiguous(),
      at::cat({weights.gate.row_scales, weights.up.row_scales}, 0).contiguous(),
      at::Tensor(), {}};
  MoeRowInt8Matrix gate_view{bundle.quantized.slice(0, 0, rows),
                             bundle.row_scales.slice(0, 0, rows),
                             std::move(weights.gate.dense_f32), {}};
  MoeRowInt8Matrix up_view{bundle.quantized.slice(0, rows, rows * 2),
                           bundle.row_scales.slice(0, rows, rows * 2),
                           std::move(weights.up.dense_f32), {}};
  if (!gate_view.quantized.is_contiguous() ||
      !gate_view.row_scales.is_contiguous() ||
      !up_view.quantized.is_contiguous() ||
      !up_view.row_scales.is_contiguous()) {
    throw std::runtime_error(
        "dense gate/up bundle did not produce contiguous zero-copy views");
  }
  weights.gate = std::move(gate_view);
  weights.up = std::move(up_view);
  weights.gate_up = std::move(bundle);
  weights.bundled_gate_up_qualified = qualified;
  weights.bundled_gate_up_enabled =
      qualified && enable_experimental;
  return weights;
}

TargetTailWeights
precompute_target_tail_score_weight(TargetTailWeights weights) {
  const c10::InferenceMode inference_guard;
  const std::int64_t hidden =
      weights.output_res_norm.dim() == 1 ? weights.output_res_norm.size(0) : 0;
  const at::Device device = weights.output_res_norm.device();
  require_f32(weights.output_res_norm, device, {hidden},
              "output-residual norm");
  require_f32(weights.output_res_projection, device, {1, hidden},
              "output-residual projection");
  weights.output_score_weight =
      at::mul(weights.output_res_norm, weights.output_res_projection.squeeze(0))
          .contiguous();
  return weights;
}

TargetBlockResidual
empty_target_block_residual(const at::Device &device,
                            const std::int64_t hidden_size) {
  if (hidden_size <= 0) {
    throw std::invalid_argument("target hidden size must be positive");
  }
  return TargetBlockResidual{
      at::empty({1, 0, hidden_size},
                at::TensorOptions().dtype(at::kFloat).device(device))};
}

TargetAttentionInput
prepare_target_attention(const at::Tensor &hidden,
                         const TargetBlockResidual &residual,
                         const TargetResidualWeights &weights,
                         const std::uint32_t layer_index, const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = hidden.dim() == 2 ? hidden.size(1) : 0;
  validate_hidden(hidden, width, "target layer hidden");
  validate_anchors(residual.anchors, hidden.device(), width);
  validate_residual_weights(weights, hidden.device(), width);
  if (layer_index >= kK3Layers || (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target attention preparation refuses a non-K3 layer contract");
  }
  if (exact_k3) {
    require_exact_anchor_count(residual.anchors,
                               expected_anchor_count_before(layer_index),
                               "attention preparation");
  }

  at::Tensor attention_input = hidden;
  if (residual.anchors.size(1) != 0) {
    at::Tensor temporary_score_weight;
    attention_input = apply_attention_residual(
        hidden, residual.anchors,
        residual_score_weight(weights.self_attention_score_weight,
                              weights.self_attention_res_norm,
                              weights.self_attention_res_projection,
                              temporary_score_weight));
  }

  at::Tensor prefix_sum = hidden;
  at::Tensor next_anchors = residual.anchors;
  if (layer_index % kResidualBlock == 0) {
    next_anchors =
        at::cat({residual.anchors, hidden.unsqueeze(1)}, 1).contiguous();
    prefix_sum = at::Tensor();
  }
  return TargetAttentionInput{
      rms_norm(attention_input, weights.input_norm).contiguous(),
      std::move(prefix_sum), std::move(next_anchors), layer_index};
}

TargetAttentionRowsInput prepare_target_attention_rows(
    const at::Tensor &hidden_rows, const at::Tensor &residual_anchor_rows,
    const TargetResidualWeights &weights, const std::uint32_t layer_index,
    const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width =
      hidden_rows.dim() == 2 ? hidden_rows.size(1) : 0;
  validate_hidden_rows(hidden_rows, width, "target layer hidden rows");
  validate_anchor_rows(residual_anchor_rows, hidden_rows.device(),
                       hidden_rows.size(0), width);
  validate_residual_weights(weights, hidden_rows.device(), width);
  if (layer_index >= kK3Layers || (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target attention row preparation refuses a non-K3 layer contract");
  }
  if (exact_k3) {
    require_exact_anchor_count(residual_anchor_rows,
                               expected_anchor_count_before(layer_index),
                               "row attention preparation");
  }

  at::Tensor attention_input = hidden_rows;
  if (residual_anchor_rows.size(1) != 0) {
    at::Tensor temporary_score_weight;
    attention_input = apply_attention_residual(
        hidden_rows, residual_anchor_rows,
        residual_score_weight(weights.self_attention_score_weight,
                              weights.self_attention_res_norm,
                              weights.self_attention_res_projection,
                              temporary_score_weight));
  }

  at::Tensor prefix_sum = hidden_rows;
  at::Tensor next_anchors = residual_anchor_rows;
  if (layer_index % kResidualBlock == 0) {
    next_anchors = at::cat(
        {residual_anchor_rows, hidden_rows.unsqueeze(1)}, 1).contiguous();
    prefix_sum = at::Tensor();
  }
  return TargetAttentionRowsInput{
      rms_norm(attention_input, weights.input_norm).contiguous(),
      std::move(prefix_sum), std::move(next_anchors), layer_index};
}

TargetMlpInput prepare_target_mlp(const TargetAttentionInput &prepared,
                                  const at::Tensor &attention_output,
                                  const TargetResidualWeights &weights,
                                  const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width =
      attention_output.dim() == 2 ? attention_output.size(1) : 0;
  validate_hidden(attention_output, width, "target attention output");
  validate_anchors(prepared.next_anchors, attention_output.device(), width);
  validate_residual_weights(weights, attention_output.device(), width);
  if (prepared.layer_index >= kK3Layers || (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target MLP preparation refuses a non-K3 layer contract");
  }
  validate_hidden_on_device(prepared.normalized, width,
                            attention_output.device(),
                            "prepared attention input");
  const bool boundary = prepared.layer_index % kResidualBlock == 0;
  if (exact_k3) {
    require_exact_anchor_count(
        prepared.next_anchors,
        expected_anchor_count_after(prepared.layer_index), "MLP preparation");
    if (prepared.prefix_sum.defined() == boundary) {
      throw std::invalid_argument(
          "target attention prefix state disagrees with its layer boundary");
    }
  }
  if (prepared.prefix_sum.defined()) {
    validate_hidden_on_device(prepared.prefix_sum, width,
                              attention_output.device(), "target prefix sum");
  }
  const at::Tensor prefix = prepared.prefix_sum.defined()
                                ? at::add(prepared.prefix_sum, attention_output)
                                : attention_output;
  at::Tensor temporary_score_weight;
  const at::Tensor mixed = apply_attention_residual(
      prefix, prepared.next_anchors,
      residual_score_weight(weights.mlp_score_weight, weights.mlp_res_norm,
                            weights.mlp_res_projection,
                            temporary_score_weight));
  return TargetMlpInput{
      rms_norm(mixed, weights.post_attention_norm).contiguous(), mixed,
      prefix, prepared.next_anchors, prepared.layer_index};
}

TargetMlpRowsInput prepare_target_mlp_rows(
    const TargetAttentionRowsInput &prepared,
    const at::Tensor &attention_output_rows,
    const TargetResidualWeights &weights, const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = attention_output_rows.dim() == 2
                                 ? attention_output_rows.size(1)
                                 : 0;
  validate_hidden_rows(attention_output_rows, width,
                       "target attention output rows");
  const std::int64_t positions = attention_output_rows.size(0);
  validate_anchor_rows(prepared.next_anchors,
                       attention_output_rows.device(), positions, width);
  validate_residual_weights(weights, attention_output_rows.device(), width);
  if (prepared.layer_index >= kK3Layers ||
      (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target MLP row preparation refuses a non-K3 layer contract");
  }
  if (!prepared.normalized.defined() ||
      prepared.normalized.device() != attention_output_rows.device() ||
      prepared.normalized.scalar_type() != at::kFloat ||
      !prepared.normalized.is_contiguous() ||
      prepared.normalized.sizes() != attention_output_rows.sizes()) {
    throw std::invalid_argument(
        "prepared attention rows disagree with their output geometry");
  }
  const bool boundary = prepared.layer_index % kResidualBlock == 0;
  if (exact_k3) {
    require_exact_anchor_count(
        prepared.next_anchors,
        expected_anchor_count_after(prepared.layer_index),
        "row MLP preparation");
    if (prepared.prefix_sum.defined() == boundary) {
      throw std::invalid_argument(
          "target attention row prefix state disagrees with its layer boundary");
    }
  }
  if (prepared.prefix_sum.defined() &&
      (prepared.prefix_sum.device() != attention_output_rows.device() ||
       prepared.prefix_sum.scalar_type() != at::kFloat ||
       !prepared.prefix_sum.is_contiguous() ||
       prepared.prefix_sum.sizes() != attention_output_rows.sizes())) {
    throw std::invalid_argument(
        "target prefix rows violate their fp32 geometry");
  }
  const at::Tensor prefix = prepared.prefix_sum.defined()
                                ? at::add(prepared.prefix_sum,
                                          attention_output_rows)
                                : attention_output_rows;
  at::Tensor temporary_score_weight;
  const at::Tensor mixed = apply_attention_residual(
      prefix, prepared.next_anchors,
      residual_score_weight(weights.mlp_score_weight, weights.mlp_res_norm,
                            weights.mlp_res_projection,
                            temporary_score_weight));
  return TargetMlpRowsInput{
      rms_norm(mixed, weights.post_attention_norm).contiguous(), mixed,
      prefix, prepared.next_anchors, prepared.layer_index};
}

at::Tensor complete_target_layer(const TargetMlpInput &prepared,
                                 const at::Tensor &mlp_output,
                                 const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = mlp_output.dim() == 2 ? mlp_output.size(1) : 0;
  validate_hidden(mlp_output, width, "target MLP output");
  if (prepared.layer_index >= kK3Layers || (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target layer completion refuses a non-K3 layer contract");
  }
  validate_hidden_on_device(prepared.normalized, width, mlp_output.device(),
                            "prepared MLP input");
  validate_hidden_on_device(prepared.lookahead_source, width,
                            mlp_output.device(),
                            "prepared MLP lookahead source");
  validate_anchors(prepared.next_anchors, mlp_output.device(), width);
  if (!prepared.prefix_sum.defined()) {
    throw std::invalid_argument(
        "target layer completion is missing its authoritative prefix sum");
  }
  validate_hidden_on_device(prepared.prefix_sum, width, mlp_output.device(),
                            "target prefix sum");
  if (exact_k3) {
    require_exact_anchor_count(
        prepared.next_anchors,
        expected_anchor_count_after(prepared.layer_index), "layer completion");
  }
  return at::add(prepared.prefix_sum, mlp_output).contiguous();
}

at::Tensor complete_target_layer_rows(
    const TargetMlpRowsInput &prepared, const at::Tensor &mlp_output_rows,
    const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = mlp_output_rows.dim() == 2
                                 ? mlp_output_rows.size(1)
                                 : 0;
  validate_hidden_rows(mlp_output_rows, width, "target MLP output rows");
  const std::int64_t positions = mlp_output_rows.size(0);
  if (prepared.layer_index >= kK3Layers ||
      (exact_k3 && width != kK3Hidden)) {
    throw std::invalid_argument(
        "target layer row completion refuses a non-K3 layer contract");
  }
  for (const at::Tensor *tensor :
       {&prepared.normalized, &prepared.lookahead_source,
        &prepared.prefix_sum}) {
    if (!tensor->defined() || tensor->device() != mlp_output_rows.device() ||
        tensor->scalar_type() != at::kFloat || !tensor->is_contiguous() ||
        tensor->sizes() != mlp_output_rows.sizes()) {
      throw std::invalid_argument(
          "prepared MLP rows disagree with completion geometry");
    }
  }
  validate_anchor_rows(prepared.next_anchors, mlp_output_rows.device(),
                       positions, width);
  if (exact_k3) {
    require_exact_anchor_count(
        prepared.next_anchors,
        expected_anchor_count_after(prepared.layer_index),
        "row layer completion");
  }
  return at::add(prepared.prefix_sum, mlp_output_rows).contiguous();
}

at::Tensor run_target_dense(const at::Tensor &hidden,
                            const TargetDenseWeights &weights,
                            const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = hidden.dim() == 2 ? hidden.size(1) : 0;
  validate_hidden(hidden, width, "dense-layer input");
  const std::int64_t intermediate = linear_rows(weights.gate);
  if (exact_k3 &&
      (width != kK3Hidden || intermediate != kK3DenseIntermediate)) {
    throw std::invalid_argument(
        "dense-layer tape refuses a non-K3 dimension contract");
  }
  require_linear(weights.gate, hidden.device(), intermediate, width,
                 weights.packed_int8_qualified, "dense gate");
  require_linear(weights.up, hidden.device(), intermediate, width,
                 weights.packed_int8_qualified, "dense up");
  require_linear(weights.down, hidden.device(), width, intermediate,
                 weights.packed_int8_qualified, "dense down");
  if (weights.bundled_gate_up_enabled &&
      !weights.bundled_gate_up_qualified) {
    throw std::invalid_argument(
        "dense gate/up experiment was enabled without qualification");
  }
  at::Tensor gate;
  at::Tensor up;
  if (weights.bundled_gate_up_qualified &&
      weights.bundled_gate_up_enabled) {
    require_linear(weights.gate_up, hidden.device(), intermediate * 2, width,
                   weights.packed_int8_qualified,
                   "bundled dense gate/up");
    const at::Tensor gate_up = linear(
        hidden, weights.gate_up, weights.packed_int8_qualified);
    gate = gate_up.slice(1, 0, intermediate);
    up = gate_up.slice(1, intermediate, intermediate * 2);
  } else {
    gate = linear(hidden, weights.gate, weights.packed_int8_qualified);
    up = linear(hidden, weights.up, weights.packed_int8_qualified);
  }
  return linear(situ(gate, up), weights.down, weights.packed_int8_qualified)
      .contiguous();
}

at::Tensor run_target_dense_rows(const at::Tensor &hidden_rows,
                                 const TargetDenseWeights &weights,
                                 const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width =
      hidden_rows.dim() == 2 ? hidden_rows.size(1) : 0;
  validate_hidden_rows(hidden_rows, width, "dense-layer rows");
  const std::int64_t intermediate = linear_rows(weights.gate);
  if (exact_k3 &&
      (width != kK3Hidden || intermediate != kK3DenseIntermediate)) {
    throw std::invalid_argument(
        "dense-layer row tape refuses a non-K3 dimension contract");
  }
  require_linear(weights.gate, hidden_rows.device(), intermediate, width,
                 weights.packed_int8_qualified, "dense gate");
  require_linear(weights.up, hidden_rows.device(), intermediate, width,
                 weights.packed_int8_qualified, "dense up");
  require_linear(weights.down, hidden_rows.device(), width, intermediate,
                 weights.packed_int8_qualified, "dense down");
  if (weights.bundled_gate_up_enabled &&
      !weights.bundled_gate_up_qualified) {
    throw std::invalid_argument(
        "dense gate/up row experiment was enabled without qualification");
  }
  at::Tensor gate;
  at::Tensor up;
  if (weights.bundled_gate_up_qualified &&
      weights.bundled_gate_up_enabled) {
    require_linear(weights.gate_up, hidden_rows.device(), intermediate * 2,
                   width, weights.packed_int8_qualified,
                   "bundled dense gate/up");
    const at::Tensor gate_up = linear(
        hidden_rows, weights.gate_up, weights.packed_int8_qualified);
    gate = gate_up.slice(1, 0, intermediate);
    up = gate_up.slice(1, intermediate, intermediate * 2);
  } else {
    gate = linear(hidden_rows, weights.gate,
                  weights.packed_int8_qualified);
    up = linear(hidden_rows, weights.up, weights.packed_int8_qualified);
  }
  return linear(situ(gate, up), weights.down,
                weights.packed_int8_qualified)
      .contiguous();
}

at::Tensor finish_target_tail(const at::Tensor &hidden,
                              const TargetBlockResidual &residual,
                              const TargetTailWeights &weights,
                              const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width = hidden.dim() == 2 ? hidden.size(1) : 0;
  validate_hidden(hidden, width, "target tail hidden");
  validate_anchors(residual.anchors, hidden.device(), width);
  require_f32(weights.output_res_norm, hidden.device(), {width},
              "output-residual norm");
  require_f32(weights.output_res_projection, hidden.device(), {1, width},
              "output-residual projection");
  require_f32(weights.final_norm, hidden.device(), {width}, "final norm");
  if (weights.output_score_weight.defined()) {
    require_f32(weights.output_score_weight, hidden.device(), {width},
                "prepared output-residual score weight");
  }
  const std::int64_t vocabulary = linear_rows(weights.language_model_head);
  if (exact_k3 && (width != kK3Hidden || vocabulary != kK3Vocabulary ||
                   residual.anchors.size(1) != 8)) {
    throw std::invalid_argument(
        "target tail refuses a non-K3 dimension/residual contract");
  }
  require_linear(weights.language_model_head, hidden.device(), vocabulary,
                 width, weights.packed_int8_qualified, "language-model head");
  at::Tensor temporary_score_weight;
  const at::Tensor mixed = apply_attention_residual(
      hidden, residual.anchors,
      residual_score_weight(
          weights.output_score_weight, weights.output_res_norm,
          weights.output_res_projection, temporary_score_weight));
  const at::Tensor normalized = rms_norm(mixed, weights.final_norm);
  return language_model_head_linear(
             normalized, weights.language_model_head,
             weights.packed_int8_qualified)
      .contiguous();
}

at::Tensor finish_target_tail_rows(const at::Tensor &hidden_rows,
                                   const at::Tensor &residual_anchor_rows,
                                   const TargetTailWeights &weights,
                                   const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t width =
      hidden_rows.dim() == 2 ? hidden_rows.size(1) : 0;
  validate_hidden_rows(hidden_rows, width, "target tail rows");
  validate_anchor_rows(residual_anchor_rows, hidden_rows.device(),
                       hidden_rows.size(0), width);
  require_f32(weights.output_res_norm, hidden_rows.device(), {width},
              "output-residual norm");
  require_f32(weights.output_res_projection, hidden_rows.device(), {1, width},
              "output-residual projection");
  require_f32(weights.final_norm, hidden_rows.device(), {width},
              "final norm");
  if (weights.output_score_weight.defined()) {
    require_f32(weights.output_score_weight, hidden_rows.device(), {width},
                "prepared output-residual score weight");
  }
  const std::int64_t vocabulary = linear_rows(weights.language_model_head);
  if (exact_k3 && (width != kK3Hidden || vocabulary != kK3Vocabulary ||
                   residual_anchor_rows.size(1) != 8)) {
    throw std::invalid_argument(
        "target tail rows refuse a non-K3 dimension/residual contract");
  }
  require_linear(weights.language_model_head, hidden_rows.device(), vocabulary,
                 width, weights.packed_int8_qualified,
                 "language-model head");
  at::Tensor temporary_score_weight;
  const at::Tensor mixed = apply_attention_residual(
      hidden_rows, residual_anchor_rows,
      residual_score_weight(weights.output_score_weight,
                            weights.output_res_norm,
                            weights.output_res_projection,
                            temporary_score_weight));
  const at::Tensor normalized = rms_norm(mixed, weights.final_norm);
  return language_model_head_linear(
             normalized, weights.language_model_head,
             weights.packed_int8_qualified)
      .contiguous();
}

at::Tensor target_embedding_row(const std::uint32_t token_id,
                                const MoeRowInt8Matrix &embedding,
                                const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t vocabulary =
      embedding.quantized.dim() == 2 ? embedding.quantized.size(0) : 0;
  const std::int64_t width =
      embedding.quantized.dim() == 2 ? embedding.quantized.size(1) : 0;
  if (exact_k3 && (vocabulary != kK3Vocabulary || width != kK3Hidden)) {
    throw std::invalid_argument(
        "target embedding refuses a non-K3 dimension contract");
  }
  if (token_id >= static_cast<std::uint64_t>(vocabulary)) {
    throw std::invalid_argument("target embedding token ID is out of range");
  }
  require_linear(embedding, embedding.quantized.device(), vocabulary, width,
                 true, "token embedding");
  // A row lookup does not need a GEMM and preserves the exact int8*row-scale
  // representation used by the authoritative resident embedding.
  return at::mul(embedding.quantized[static_cast<std::int64_t>(token_id)].to(
                     at::kFloat),
                 embedding.row_scales[static_cast<std::int64_t>(token_id)])
      .reshape({1, width})
      .contiguous();
}

at::Tensor target_embedding_rows(const at::Tensor &token_ids,
                                 const MoeRowInt8Matrix &embedding,
                                 const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  const std::int64_t vocabulary =
      embedding.quantized.dim() == 2 ? embedding.quantized.size(0) : 0;
  const std::int64_t width =
      embedding.quantized.dim() == 2 ? embedding.quantized.size(1) : 0;
  if (exact_k3 && (vocabulary != kK3Vocabulary || width != kK3Hidden)) {
    throw std::invalid_argument(
        "target embedding rows refuse a non-K3 dimension contract");
  }
  require_linear(embedding, embedding.quantized.device(), vocabulary, width,
                 true, "token embedding rows");
  if (!token_ids.defined() || token_ids.scalar_type() != at::kLong ||
      token_ids.device() != embedding.quantized.device() ||
      !token_ids.is_contiguous() || token_ids.dim() != 1 ||
      token_ids.size(0) < 1 || token_ids.size(0) > 7) {
    throw std::invalid_argument(
        "target embedding row IDs must be contiguous int64 [1..7]");
  }
  if ((token_ids < 0).any().item<bool>() ||
      (token_ids >= vocabulary).any().item<bool>()) {
    throw std::invalid_argument("target embedding row ID is out of range");
  }
  const at::Tensor quantized =
      embedding.quantized.index_select(0, token_ids).to(at::kFloat);
  const at::Tensor scales =
      embedding.row_scales.index_select(0, token_ids).unsqueeze(1);
  return at::mul(quantized, scales).contiguous();
}

at::Tensor target_language_model_head_rows(
    const at::Tensor &hidden_rows,
    const MoeRowInt8Matrix &language_model_head,
    const bool packed_int8_qualified, const bool exact_k3) {
  const c10::InferenceMode inference_guard;
  if (!hidden_rows.defined() || !hidden_rows.is_contiguous() ||
      (hidden_rows.scalar_type() != at::kFloat &&
       hidden_rows.scalar_type() != at::kBFloat16) ||
      hidden_rows.dim() != 2 || hidden_rows.size(0) < 1 ||
      hidden_rows.size(0) > 7) {
    throw std::invalid_argument(
        "shared language-model head input must be contiguous fp32/BF16 [1..7,H]");
  }
  const std::int64_t width = hidden_rows.size(1);
  const std::int64_t vocabulary = linear_rows(language_model_head);
  if (exact_k3 && (width != kK3Hidden || vocabulary != kK3Vocabulary)) {
    throw std::invalid_argument(
        "shared language-model head refuses a non-K3 dimension contract");
  }
  require_linear(language_model_head, hidden_rows.device(), vocabulary, width,
                 packed_int8_qualified, "shared language-model head");
  const at::Tensor fp32 = hidden_rows.scalar_type() == at::kFloat
                              ? hidden_rows
                              : hidden_rows.to(at::kFloat);
  return language_model_head_linear(
             fp32, language_model_head, packed_int8_qualified)
      .contiguous();
}

} // namespace deltafin::provider_internal
