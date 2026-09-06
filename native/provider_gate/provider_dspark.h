#ifndef DELTAFIN_PROVIDER_DSPARK_H
#define DELTAFIN_PROVIDER_DSPARK_H

#include <ATen/ATen.h>

#include <cstdint>

namespace deltafin::provider_internal {

/*
 * Native arithmetic for the K3 DSpark proposal model.  This surface cannot
 * verify, accept, or emit token IDs: it only transforms already-bound tensors.
 * Full K3 remains the sole output authority.
 */
struct DSparkShape {
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t num_heads = 0;
  std::int64_t q_lora_rank = 0;
  std::int64_t kv_lora_rank = 0;
  std::int64_t qk_nope_head_dim = 0;
  std::int64_t qk_rope_head_dim = 0;
  std::int64_t value_head_dim = 0;
  std::int64_t max_position = 0;
  std::int64_t vocabulary_size = 0;
  std::int64_t target_hidden_size = 0;
  std::int64_t markov_rank = 0;
  std::int64_t mask_token_id = 0;
  double rms_epsilon = 0.0;
  double rope_theta = 0.0;
  double rope_factor = 0.0;
  std::int64_t rope_original_max_position = 0;
  double rope_beta_fast = 0.0;
  double rope_beta_slow = 0.0;
  double rope_mscale = 0.0;
  double rope_mscale_all_dim = 0.0;

  [[nodiscard]] static DSparkShape k3();
  [[nodiscard]] static DSparkShape small_canary();
  void validate() const;
  [[nodiscard]] bool is_exact_k3() const;
  [[nodiscard]] std::int64_t query_head_dim() const;
  [[nodiscard]] std::int64_t target_context_width() const;
};

struct DSparkMlaWeights {
  at::Tensor query_a;
  at::Tensor query_a_norm;
  at::Tensor query_b;
  at::Tensor key_value_a;
  at::Tensor key_value_a_norm;
  at::Tensor key_value_b;
  at::Tensor output;
};

struct DSparkMlpWeights {
  at::Tensor gate;
  at::Tensor up;
  at::Tensor down;
};

struct DSparkDecoderWeights {
  at::Tensor input_norm;
  DSparkMlaWeights attention;
  at::Tensor post_attention_norm;
  DSparkMlpWeights mlp;
};

// One layer's compact committed prefix. Empty prefixes are [0,L] and [0,R].
struct DSparkLatentContext {
  at::Tensor latent;
  at::Tensor positional;
};

struct DSparkMlaOutput {
  at::Tensor hidden;
  // The caller may append these rows only if its wider proposal transaction
  // commits.  The arithmetic primitive never mutates committed state.
  DSparkLatentContext query_context;
};

struct DSparkDecoderOutput {
  at::Tensor hidden;
  at::Tensor residual;
  DSparkLatentContext query_context;
};

[[nodiscard]] at::Tensor dspark_rms_norm_bf16(
    const at::Tensor& value, const at::Tensor& weight, double epsilon);

[[nodiscard]] at::Tensor dspark_yarn_inverse_frequencies(
    const DSparkShape& shape, const at::Device& device);

// `maximum_rows` bounds value.size(0): the draft query/backbone paths keep
// the trained 7-row ceiling (the default); the context-append path passes
// its own bound, since a chat prefill legitimately appends arbitrarily many
// target rows in one call (positions are range-checked separately).
[[nodiscard]] at::Tensor dspark_apply_yarn_rotary_bf16(
    const at::Tensor& value, const at::Tensor& positions,
    const DSparkShape& shape, std::int64_t maximum_rows = 7);

[[nodiscard]] DSparkMlaOutput run_dspark_mla(
    const at::Tensor& hidden, const at::Tensor& positions,
    const DSparkLatentContext& context, const DSparkMlaWeights& weights,
    const DSparkShape& shape);

[[nodiscard]] at::Tensor run_dspark_mlp(
    const at::Tensor& hidden, const DSparkMlpWeights& weights,
    const DSparkShape& shape);

[[nodiscard]] DSparkDecoderOutput run_dspark_decoder_layer(
    const at::Tensor& hidden, const at::Tensor& residual,
    const at::Tensor& positions, const DSparkLatentContext& context,
    const DSparkDecoderWeights& weights, const DSparkShape& shape);

}  // namespace deltafin::provider_internal

#endif
