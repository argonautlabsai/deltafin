#include <metal_stdlib>
using namespace metal;

/*
 * Bespoke decode-loop scaffold kernels.
 *
 * These are deliberately trivial: they exist so the LoopEncoder's owned
 * serial command queue, shared-event pool, embedded metallib, and 192-byte
 * shared-storage mailboxes can be qualified end-to-end (byte-exact against a
 * host reference) before any real math runs on the loop queue.  The
 * validated k3-proto kernel suite (int8-spine GEMV, RMSNorm, AttnRes mix,
 * fused KDA decode, router top-16, SiTU/shared expert, KV append) replaces
 * them in worklist steps 3-5.  Compiled with -fno-fast-math like the other
 * exact-contract provider shaders so that policy does not change when the
 * real fp32 kernels land.
 */

/* Byte-identical twin of the host RouteMailboxT1 (ids at 0, fp32 weight bit
 * patterns at 128, 192 bytes total), reused verbatim from the route
 * mailbox contract. */
struct DeltafinLoopMailboxT1 {
  long expert_ids[16];
  uint weight_bits[16];
};

/* Canary dispatch A ("route"): copy ids and weight bits through the loop
 * queue unchanged. */
kernel void deltafin_loop_canary_route_v1(
    device const long* expert_ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device DeltafinLoopMailboxT1* mailbox [[buffer(2)]],
    uint edge [[thread_position_in_grid]]) {
  if (edge < 16) {
    mailbox->expert_ids[edge] = expert_ids[edge];
    mailbox->weight_bits[edge] = as_type<uint>(weights[edge]);
  }
}

/* Canary dispatch B ("pilot"): a distinct deterministic transform so a
 * second command buffer on the same serial queue is provably executing this
 * pipeline, not dispatch A.  Both arithmetic steps are exact in fp32
 * (power-of-two scale, then one correctly rounded add), so the host fp32
 * reference is byte-identical with or without contraction. */
kernel void deltafin_loop_canary_pilot_v1(
    device const long* expert_ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device DeltafinLoopMailboxT1* mailbox [[buffer(2)]],
    uint edge [[thread_position_in_grid]]) {
  if (edge < 16) {
    mailbox->expert_ids[edge] = expert_ids[edge] + long(edge);
    mailbox->weight_bits[edge] = as_type<uint>(weights[edge] * 2.0f + 1.0f);
  }
}

/* ==========================================================================
 * Router side-queue chain — ported verbatim (math-identical) from the
 * validated k3-proto loopkernels suite (45/45 checks <=1e-6 relL2; router
 * harness includes engineered tie blocks and a full 896-way tie).  Three
 * kernels cover the per-layer T=1 route: RMSNorm on the post-attention
 * hidden, the int8 router GEMV (N=896, K=7168 in production; kernel is
 * generic over K%16==0, N%4==0), and the fused sigmoid+bias top-16 select
 * that writes the RouteMailboxT1 byte layout directly.
 *
 * ORDER CONTRACT (blueprint §5 risk 2): stock routing preserves whatever
 * at::topk(sorted=false) emitted on MPS — implementation-defined.  This
 * chain instead uses a fully deterministic tie-break (score-descending,
 * index-ascending), applied identically in the strided scan and at every
 * tree-combine step, so results are independent of thread scheduling.  The
 * documented divergence-vs-stock is edge ORDER only; the §3.3 gate compares
 * route multisets + weights, and the integration decides order policy.
 * ======================================================================== */

inline float deltafin_loop_sigmoid_p(float v) {
  return 1.0f / (1.0f + precise::exp(-v));
}

/* Reduce one float per thread across an nsg-simdgroup threadgroup (nsg<=32).
 * Leading barrier makes `buf` safe to reuse immediately after a prior call
 * on the same buffer completes. */
inline float deltafin_loop_tg_reduce_sum(float v, threadgroup float* buf,
                                         uint lane, uint sg, uint nsg) {
  const float s = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) buf[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0u; i < nsg; ++i) total += buf[i];
  return total;
}

struct DeltafinLoopGemv8DimsV1 {
  uint N;
  uint K;
};

inline float deltafin_loop_dot16_i8(uint4 w, thread const float4* xv) {
  float s = 0.0f;
  s += dot(float4(as_type<char4>(w.x)), xv[0]);
  s += dot(float4(as_type<char4>(w.y)), xv[1]);
  s += dot(float4(as_type<char4>(w.z)), xv[2]);
  s += dot(float4(as_type<char4>(w.w)), xv[3]);
  return s;
}

/* out[n] = scale[n] * sum_k x[k] * (float)W[n*K + k].  128 threads = 4
 * simdgroups x 4 consecutive rows; grid = ceil(N/16) threadgroups.
 * Requires K%16==0, N%4==0. */
kernel void deltafin_loop_gemv_i8_bundle_v1(
    device const char* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant DeltafinLoopGemv8DimsV1& D [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint r0 = (tg * 4u + sg) * 4u;
  if (r0 >= D.N) return;
  const uint NG = D.K / 16u;
  device const uint4* w4 = (device const uint4*)(W + (ulong)r0 * D.K);
  device const float4* x4 = (device const float4*)x;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (uint g = lane; g < NG; g += 32u) {
    float4 xv[4];
    device const float4* xg = x4 + g * 4u;
    xv[0] = xg[0];
    xv[1] = xg[1];
    xv[2] = xg[2];
    xv[3] = xg[3];
    a0 += deltafin_loop_dot16_i8(w4[0u * NG + g], xv);
    a1 += deltafin_loop_dot16_i8(w4[1u * NG + g], xv);
    a2 += deltafin_loop_dot16_i8(w4[2u * NG + g], xv);
    a3 += deltafin_loop_dot16_i8(w4[3u * NG + g], xv);
  }
  a0 = simd_sum(a0);
  a1 = simd_sum(a1);
  a2 = simd_sum(a2);
  a3 = simd_sum(a3);
  if (lane == 0u) {
    out[r0 + 0u] = a0 * scales[r0 + 0u];
    out[r0 + 1u] = a1 * scales[r0 + 1u];
    out[r0 + 2u] = a2 * scales[r0 + 2u];
    out[r0 + 3u] = a3 * scales[r0 + 3u];
  }
}

/* BF16-weight GEMV, math ported VERBATIM from the stock spine kernel
 * (deltafin_spine_bf16_gemv_rows4_t1_v1): identical bf16-bit expansion,
 * identical fma order, identical simd reduction — logits are bit-identical
 * to the stock path when given the same weight bits and input.  One SIMD
 * group owns four adjacent rows; four SIMD groups per 128-thread group. */
struct DeltafinLoopBf16GemvDimsV1 {
  uint rows;
  uint columns;
  uint reserved0;
  uint reserved1;
};

inline float4 deltafin_loop_decode_bf16x4_v1(const ushort4 bits) {
  return as_type<float4>(uint4(bits) << 16);
}

kernel void deltafin_loop_gemv_bf16_rows4_v1(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DeltafinLoopBf16GemvDimsV1& dims [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint row0 = (tg * 4 + simdgroup) * 4;
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  for (uint column = lane * 4; column < dims.columns; column += 128) {
    const float4 x = *((device const float4*)(input + column));
    if (row0 + 0 < dims.rows) {
      const float4 w = deltafin_loop_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 0) * dims.columns + column)));
      a0 = fma(w.x, x.x, a0);
      a0 = fma(w.y, x.y, a0);
      a0 = fma(w.z, x.z, a0);
      a0 = fma(w.w, x.w, a0);
    }
    if (row0 + 1 < dims.rows) {
      const float4 w = deltafin_loop_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 1) * dims.columns + column)));
      a1 = fma(w.x, x.x, a1);
      a1 = fma(w.y, x.y, a1);
      a1 = fma(w.z, x.z, a1);
      a1 = fma(w.w, x.w, a1);
    }
    if (row0 + 2 < dims.rows) {
      const float4 w = deltafin_loop_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 2) * dims.columns + column)));
      a2 = fma(w.x, x.x, a2);
      a2 = fma(w.y, x.y, a2);
      a2 = fma(w.z, x.z, a2);
      a2 = fma(w.w, x.w, a2);
    }
    if (row0 + 3 < dims.rows) {
      const float4 w = deltafin_loop_decode_bf16x4_v1(
          *((device const ushort4*)(weight +
              (row0 + 3) * dims.columns + column)));
      a3 = fma(w.x, x.x, a3);
      a3 = fma(w.y, x.y, a3);
      a3 = fma(w.z, x.z, a3);
      a3 = fma(w.w, x.w, a3);
    }
  }
  a0 = simd_sum(a0);
  a1 = simd_sum(a1);
  a2 = simd_sum(a2);
  a3 = simd_sum(a3);
  if (lane == 0) {
    if (row0 + 0 < dims.rows) output[row0 + 0] = a0;
    if (row0 + 1 < dims.rows) output[row0 + 1] = a1;
    if (row0 + 2 < dims.rows) output[row0 + 2] = a2;
    if (row0 + 3 < dims.rows) output[row0 + 3] = a3;
  }
}

/* AttnRes anchor mix — ported verbatim from the validated k3-proto suite:
 * values = [anchors ; hidden] (R = A+1 rows, A <= 8); per-row RMS (eps
 * 1e-5, no weight) folded into scores = rms_r * sum(values_r *
 * score_weight); fp32 softmax over R; out = p @ values.  One 256-thread
 * threadgroup handles the whole op. */
inline float2 deltafin_loop_tg_reduce_sum2(float2 v, threadgroup float2* buf,
                                           uint lane, uint sg, uint nsg) {
  const float2 s = float2(simd_sum(v.x), simd_sum(v.y));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) buf[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 total = float2(0.0f);
  for (uint i = 0u; i < nsg; ++i) total += buf[i];
  return total;
}

/* SiTU shared-expert activation, ported verbatim from the validated
 * k3-proto epilogue suite (beta = 4, linear beta = 25; matches Kimi's
 * SituAndMul.forward op order, fp32, precise:: transcendentals). */
struct DeltafinLoopSituDimsV1 {
  uint N; /* 6144 in production */
};

kernel void deltafin_loop_situ_v1(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* h_out [[buffer(2)]],
    constant DeltafinLoopSituDimsV1& D [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= D.N) return;
  float g = gate[i], u = up[i];
  float a = 4.0f * precise::tanh(g * 0.25f) * deltafin_loop_sigmoid_p(g);
  float b = 25.0f * precise::tanh(u * 0.04f);
  h_out[i] = a * b;
}

/* Elementwise residual add, bounds-checked grid-stride. */
kernel void deltafin_loop_add_v1(device const float* a [[buffer(0)]],
                                 device const float* b [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint& elements [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i < elements) {
    out[i] = a[i] + b[i];
  }
}

struct DeltafinLoopAttnResDimsV1 {
  uint A; /* anchor rows, R = A+1 <= 9 */
  uint H; /* 7168 */
};

kernel void deltafin_loop_attn_res_mix_v1(
    device const float* anchors [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device const float* score_weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant DeltafinLoopAttnResDimsV1& D [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_position_in_threadgroup]]) {
  const uint R = D.A + 1u;
  threadgroup float2 red2[8];
  threadgroup float scoreArr[9];
  threadgroup float pArr[9];

  for (uint r = 0u; r < R; ++r) {
    device const float* row =
        (r < D.A) ? (anchors + (ulong)r * D.H) : hidden;
    float2 acc = float2(0.0f);
    for (uint i = tid; i < D.H; i += 256u) {
      float v = row[i];
      acc += float2(v * v, v * score_weight[i]);
    }
    float2 tot = deltafin_loop_tg_reduce_sum2(acc, red2, lane, sg, 8u);
    if (tid == 0u) {
      float rms = 1.0f / precise::sqrt(tot.x / (float)D.H + 1e-5f);
      scoreArr[r] = tot.y * rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    float m = scoreArr[0];
    for (uint r = 1u; r < R; ++r) m = max(m, scoreArr[r]);
    float sum = 0.0f;
    for (uint r = 0u; r < R; ++r) {
      float e = precise::exp(scoreArr[r] - m);
      pArr[r] = e;
      sum += e;
    }
    for (uint r = 0u; r < R; ++r) pArr[r] /= sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < D.H; i += 256u) {
    float acc = 0.0f;
    for (uint r = 0u; r < R; ++r) {
      device const float* row =
          (r < D.A) ? (anchors + (ulong)r * D.H) : hidden;
      acc += pArr[r] * row[i];
    }
    out[i] = acc;
  }
}

/* Dense-fp32 variant of the rows4 GEMV, for router matrices materialized as
 * fp32 (MoeRowInt8Matrix::dense_f32).  Same blocking and reduction as the
 * bf16 kernel; the weight is read as float4 directly. */
kernel void deltafin_loop_gemv_f32_rows4_v1(
    device const float* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant DeltafinLoopBf16GemvDimsV1& dims [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  const uint row0 = (tg * 4 + simdgroup) * 4;
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  for (uint column = lane * 4; column < dims.columns; column += 128) {
    const float4 x = *((device const float4*)(input + column));
    if (row0 + 0 < dims.rows) {
      const float4 w = *((device const float4*)(weight +
          (row0 + 0) * dims.columns + column));
      a0 = fma(w.x, x.x, a0);
      a0 = fma(w.y, x.y, a0);
      a0 = fma(w.z, x.z, a0);
      a0 = fma(w.w, x.w, a0);
    }
    if (row0 + 1 < dims.rows) {
      const float4 w = *((device const float4*)(weight +
          (row0 + 1) * dims.columns + column));
      a1 = fma(w.x, x.x, a1);
      a1 = fma(w.y, x.y, a1);
      a1 = fma(w.z, x.z, a1);
      a1 = fma(w.w, x.w, a1);
    }
    if (row0 + 2 < dims.rows) {
      const float4 w = *((device const float4*)(weight +
          (row0 + 2) * dims.columns + column));
      a2 = fma(w.x, x.x, a2);
      a2 = fma(w.y, x.y, a2);
      a2 = fma(w.z, x.z, a2);
      a2 = fma(w.w, x.w, a2);
    }
    if (row0 + 3 < dims.rows) {
      const float4 w = *((device const float4*)(weight +
          (row0 + 3) * dims.columns + column));
      a3 = fma(w.x, x.x, a3);
      a3 = fma(w.y, x.y, a3);
      a3 = fma(w.z, x.z, a3);
      a3 = fma(w.w, x.w, a3);
    }
  }
  a0 = simd_sum(a0);
  a1 = simd_sum(a1);
  a2 = simd_sum(a2);
  a3 = simd_sum(a3);
  if (lane == 0) {
    if (row0 + 0 < dims.rows) output[row0 + 0] = a0;
    if (row0 + 1 < dims.rows) output[row0 + 1] = a1;
    if (row0 + 2 < dims.rows) output[row0 + 2] = a2;
    if (row0 + 3 < dims.rows) output[row0 + 3] = a3;
  }
}

struct DeltafinLoopRmsDimsV1 {
  uint N;
  float eps;
};

/* weight * (x * rsqrt(mean(x^2,-1) + eps)), fp32.  One threadgroup, 256
 * threads (8 simdgroups); each thread strides over N. */
kernel void deltafin_loop_rmsnorm_v1(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant DeltafinLoopRmsDimsV1& D [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float red[8];
  float ss = 0.0f;
  for (uint i = tid; i < D.N; i += 256u) {
    float v = x[i];
    ss += v * v;
  }
  float total = deltafin_loop_tg_reduce_sum(ss, red, lane, sg, 8u);
  float rms = 1.0f / precise::sqrt(total / (float)D.N + D.eps);
  for (uint i = tid; i < D.N; i += 256u) out[i] = weight[i] * x[i] * rms;
}

/* ==========================================================================
 * Fused KDA decode core — ported VERBATIM (math-identical) from the
 * validated k3-proto kda_kernels.metal kda_core (6 configs x 6 sub-op
 * comparisons vs fp64 at 5e-6; 0.90 ms/layer measured).  Covers, in one
 * dispatch (96 threadgroups x 128 threads, thread t owns v-column t of
 * S[h]): depthwise conv4+SiLU on q/k/v with window-state write-out,
 * per-head L2 normalization (+1/sqrt(128) on q), decay
 * exp(-5*sigmoid(exp(A_log)*(f_b+dt_bias))), beta sigmoid, the full delta
 * rule with S ping-pong, and per-head RMS x o_norm x output gate.  The
 * projections around it (bundled input GEMV, f_b, o_proj) are separate
 * GEMV dispatches.  All state fp32, precise:: transcendentals.
 * ======================================================================== */

constant uint kDeltafinLoopKdaProj = 12288u;              /* 96 x 128 */
constant uint kDeltafinLoopKdaOffG = 3u * 12288u;         /* g slice */
constant uint kDeltafinLoopKdaOffBeta = 4u * 12288u + 128u; /* beta slice */

/* Sum of one float per thread across the 128-thread (4 simdgroup) group. */
inline float deltafin_loop_tg_sum128(float v, threadgroup float* buf,
                                     uint lane, uint sg) {
  const float s = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) buf[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return buf[0] + buf[1] + buf[2] + buf[3];
}

kernel void deltafin_loop_kda_core_v1(
    device const float* proj [[buffer(0)]],      /* [49376] q|k|v|g|fa|beta */
    device const float* fb [[buffer(1)]],        /* [12288] */
    device const float* convw_q [[buffer(2)]],   /* [12288, 4] */
    device const float* convw_k [[buffer(3)]],   /* [12288, 4] */
    device const float* convw_v [[buffer(4)]],   /* [12288, 4] */
    device const float* convin_q [[buffer(5)]],  /* [12288, 4] */
    device const float* convin_k [[buffer(6)]],  /* [12288, 4] */
    device const float* convin_v [[buffer(7)]],  /* [12288, 4] */
    device float* convout_q [[buffer(8)]],       /* [12288, 4] */
    device float* convout_k [[buffer(9)]],       /* [12288, 4] */
    device float* convout_v [[buffer(10)]],      /* [12288, 4] */
    device const float* a_log [[buffer(11)]],    /* [128] */
    device const float* dt_bias [[buffer(12)]],  /* [12288] */
    device const float* o_norm [[buffer(13)]],   /* [128] */
    device const float* S_in [[buffer(14)]],     /* [96,128,128] */
    device float* S_out [[buffer(15)]],          /* [96,128,128] */
    device float* out [[buffer(16)]],            /* [12288] */
    uint h [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint c = h * 128u + t;

  /* Deltafin stores the q/k/v conv taps and window states as three separate
   * [12288,4] tensors; the fused math is identical to the validated
   * concatenated-buffer prototype — only the buffer indexing differs. */
  device const float* convw_all[3] = {convw_q, convw_k, convw_v};
  device const float* convin_all[3] = {convin_q, convin_k, convin_v};
  device float* convout_all[3] = {convout_q, convout_k, convout_v};

  float conv_res[3];
  for (uint m = 0u; m < 3u; ++m) {
    device const float* st = convin_all[m] + (ulong)c * 4u;
    device const float* wk = convw_all[m] + (ulong)c * 4u;
    const float s0 = st[1], s1 = st[2], s2 = st[3];
    const float s3 = proj[m * kDeltafinLoopKdaProj + c];
    const float acc = s0 * wk[0] + s1 * wk[1] + s2 * wk[2] + s3 * wk[3];
    device float* so = convout_all[m] + (ulong)c * 4u;
    so[0] = s0;
    so[1] = s1;
    so[2] = s2;
    so[3] = s3;
    conv_res[m] = acc * deltafin_loop_sigmoid_p(acc);
  }

  threadgroup float red[4];
  float qv = conv_res[0], kv = conv_res[1];
  const float vv = conv_res[2];
  const float qss = deltafin_loop_tg_sum128(qv * qv, red, lane, sg);
  const float kss = deltafin_loop_tg_sum128(kv * kv, red, lane, sg);
  qv /= max(precise::sqrt(qss), 1e-12f);
  kv /= max(precise::sqrt(kss), 1e-12f);
  const float q_tok = qv * 0.08838834764831845f;

  const float a = precise::exp(a_log[t]);
  const float rg = fb[c] + dt_bias[c];
  const float decay = precise::exp(-5.0f * deltafin_loop_sigmoid_p(a * rg));

  const float beta =
      deltafin_loop_sigmoid_p(proj[kDeltafinLoopKdaOffBeta + h]);

  threadgroup float k_sh[128], q_sh[128], d_sh[128];
  k_sh[t] = kv;
  q_sh[t] = q_tok;
  d_sh[t] = decay;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const float* Sh = S_in + (ulong)h * (128u * 128u);
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
  for (uint k = 0u; k < 128u; k += 4u) {
    acc0 = fma(k_sh[k + 0u] * d_sh[k + 0u], Sh[(k + 0u) * 128u + t], acc0);
    acc1 = fma(k_sh[k + 1u] * d_sh[k + 1u], Sh[(k + 1u) * 128u + t], acc1);
    acc2 = fma(k_sh[k + 2u] * d_sh[k + 2u], Sh[(k + 2u) * 128u + t], acc2);
    acc3 = fma(k_sh[k + 3u] * d_sh[k + 3u], Sh[(k + 3u) * 128u + t], acc3);
  }
  const float delta = vv - ((acc0 + acc1) + (acc2 + acc3));

  device float* Sho = S_out + (ulong)h * (128u * 128u);
  const float bd = beta * delta;
  float ov0 = 0.0f, ov1 = 0.0f, ov2 = 0.0f, ov3 = 0.0f;
  for (uint k = 0u; k < 128u; k += 4u) {
    const float sn0 =
        fma(Sh[(k + 0u) * 128u + t], d_sh[k + 0u], k_sh[k + 0u] * bd);
    const float sn1 =
        fma(Sh[(k + 1u) * 128u + t], d_sh[k + 1u], k_sh[k + 1u] * bd);
    const float sn2 =
        fma(Sh[(k + 2u) * 128u + t], d_sh[k + 2u], k_sh[k + 2u] * bd);
    const float sn3 =
        fma(Sh[(k + 3u) * 128u + t], d_sh[k + 3u], k_sh[k + 3u] * bd);
    Sho[(k + 0u) * 128u + t] = sn0;
    Sho[(k + 1u) * 128u + t] = sn1;
    Sho[(k + 2u) * 128u + t] = sn2;
    Sho[(k + 3u) * 128u + t] = sn3;
    ov0 = fma(q_sh[k + 0u], sn0, ov0);
    ov1 = fma(q_sh[k + 1u], sn1, ov1);
    ov2 = fma(q_sh[k + 2u], sn2, ov2);
    ov3 = fma(q_sh[k + 3u], sn3, ov3);
  }
  float ov = (ov0 + ov1) + (ov2 + ov3);

  const float oss = deltafin_loop_tg_sum128(ov * ov, red, lane, sg);
  ov *= 1.0f / precise::sqrt(oss * (1.0f / 128.0f) + 1e-5f);
  ov *= o_norm[t];
  ov *= deltafin_loop_sigmoid_p(proj[kDeltafinLoopKdaOffG + c]);
  out[c] = ov;
}

struct DeltafinLoopRouteDimsV1 {
  uint N; /* 896 in production */
};

/* sigmoid(logits) -> +bias -> deterministic top-16 (score-desc, index-asc
 * ties) -> gather weights from untouched sigmoid scores -> renormalize by
 * (sum + 1e-20) -> write RouteMailboxT1 layout (ids as split 32-bit words,
 * fp32 weight bits at byte 128).  One threadgroup, 256 threads. */
kernel void deltafin_loop_router_top16_v1(
    device const float* logits [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* scores_out [[buffer(2)]],
    device uint* mailbox [[buffer(3)]],
    constant DeltafinLoopRouteDimsV1& D [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float tg_choice[896];
  threadgroup float tg_score[896];

  for (uint i = tid; i < D.N; i += 256u) {
    float sc = deltafin_loop_sigmoid_p(logits[i]);
    tg_score[i] = sc;
    tg_choice[i] = sc + bias[i];
    scores_out[i] = sc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float red_val[256];
  threadgroup uint red_idx[256];

  for (uint k = 0u; k < 16u; ++k) {
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < D.N; i += 256u) {
      float v = tg_choice[i];
      if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
      }
    }
    red_val[tid] = bv;
    red_idx[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
      if (tid < s) {
        float ov = red_val[tid + s];
        uint oi = red_idx[tid + s];
        if (ov > red_val[tid] || (ov == red_val[tid] && oi < red_idx[tid])) {
          red_val[tid] = ov;
          red_idx[tid] = oi;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
      uint id = red_idx[0];
      float w = tg_score[id];
      mailbox[k * 2u + 0u] = id;
      mailbox[k * 2u + 1u] = 0u;
      mailbox[32u + k] = as_type<uint>(w);
      tg_choice[id] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) {
    float sum = 0.0f;
    for (uint k = 0u; k < 16u; ++k) sum += as_type<float>(mailbox[32u + k]);
    sum += 1e-20f;
    for (uint k = 0u; k < 16u; ++k) {
      float w = as_type<float>(mailbox[32u + k]) / sum;
      mailbox[32u + k] = as_type<uint>(w);
    }
  }
}

/*
 * Fused MLA decode attention core, shared verbatim with the standalone
 * provider_mla_attn_metal library (see the header). Bringing it into the
 * loop metallib is the Step-0 move for the MLA takeover: the 24 MLA layers
 * are the last ATen-resident layers in the decode path (measured 2026-08-24:
 * attention = 32.4s of a 113.8s rung, 28% of the token), and the loop
 * already owns every other kernel an MLA layer needs — int8 GEMV for the
 * projections, RMSNorm, AttnRes mix, residual add.
 */
#include "provider_mla_attn_core.metalh"

/* ---- MLA takeover: the two primitives the chain still needed --------------
 * Everything else an MLA layer does already had a loop kernel (int8 GEMV for
 * the four projections, RMSNorm, the attention core above). These two close
 * the gap.
 */

/* Pack one decode token's K/V rows out of the kv_b projection.
 *
 * kv_b writes [num_heads, qk_nope + value_head_dim] contiguously per head
 * (K3: 96 x 256). The key row a head stores is its 128-wide nope half
 * followed by the SHARED 64-wide rope vector — shared because
 * provider_mla.cpp expands one [1,1,rope] row across every head rather than
 * projecting per-head positional dims. The value row is the head's second
 * 128-wide half. Writing both here avoids two strided blits and a second
 * encoder.
 */
struct DeltafinLoopMlaPackDimsV1 {
  uint heads;       // 96
  uint nope_dim;    // 128
  uint rope_dim;    // 64
  uint value_dim;   // 128
};

kernel void deltafin_loop_mla_pack_kv_v1(
    device const float* expanded [[buffer(0)]],
    device const float* key_rope [[buffer(1)]],
    device float* new_key [[buffer(2)]],
    device float* new_value [[buffer(3)]],
    constant DeltafinLoopMlaPackDimsV1& D [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint key_width = D.nope_dim + D.rope_dim;
  const uint packed = D.nope_dim + D.value_dim;
  const uint key_total = D.heads * key_width;
  if (i < key_total) {
    const uint head = i / key_width;
    const uint d = i - head * key_width;
    new_key[i] = (d < D.nope_dim) ? expanded[head * packed + d]
                                  : key_rope[d - D.nope_dim];
  }
  const uint value_total = D.heads * D.value_dim;
  if (i < value_total) {
    const uint head = i / D.value_dim;
    const uint d = i - head * D.value_dim;
    new_value[i] = expanded[head * packed + D.nope_dim + d];
  }
}

/* attention * sigmoid(raw_gate), elementwise. K3's MLA output gate is a plain
 * logistic gate over two DIFFERENT tensors, so deltafin_loop_situ_v1 (the
 * tanh-based MoE activation) does not apply. Uses the same precise sigmoid
 * helper so the arithmetic matches the loop's other gated paths. */
kernel void deltafin_loop_mla_gate_v1(
    device const float* attention [[buffer(0)]],
    device const float* raw_gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& elements [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= elements) return;
  out[i] = attention[i] * deltafin_loop_sigmoid_p(raw_gate[i]);
}


/* ---- Wide-tile fused KDA (drafted verify tiles, 2..16 positions) ----------
 * K3_KDA_WIDE_FUSED=1. One threadgroup per head (96 x 128 threads); thread
 * t owns state column v = t and channel c = h*128 + t. The recurrent state
 * stays in registers across positions; per-position boundary states are
 * written for verify rollback. The per-position math mirrors
 * deltafin_loop_kda_core_v1 exactly (same association, same precise::
 * transcendentals), so the wide path lands on the numerics the
 * single-position loop path already qualified. Encoded on the ATen stream
 * between the batched projections and the output projection, replacing
 * ~140 ATen ops per layer pass. */
struct DeltafinLoopKdaWideDimsV1 {
  uint positions;    /* 2..16 */
  uint source_width; /* 3 + positions */
  uint retain;       /* 1: write per-position boundary states */
  uint reserved;
};

kernel void deltafin_loop_kda_wide_v1(
    device const float* src_q [[buffer(0)]],    /* [12288, source_width] */
    device const float* src_k [[buffer(1)]],
    device const float* src_v [[buffer(2)]],
    device const float* convw_q [[buffer(3)]],  /* [12288, 4] */
    device const float* convw_k [[buffer(4)]],
    device const float* convw_v [[buffer(5)]],
    device const float* a_log [[buffer(6)]],    /* [128] */
    device const float* dt_bias [[buffer(7)]],  /* [12288] */
    device const float* fb [[buffer(8)]],       /* [positions, 12288] */
    device const float* beta_raw [[buffer(9)]], /* [positions, 96] */
    device const float* gate [[buffer(10)]],    /* [positions, 12288] */
    device const float* o_norm [[buffer(11)]],  /* [128] */
    device const float* S_in [[buffer(12)]],    /* [96,128,128] */
    device float* S_out [[buffer(13)]],         /* [96,128,128] */
    device float* S_bound [[buffer(14)]],       /* [positions,96,128,128] */
    device float* out [[buffer(15)]],           /* [positions, 12288] */
    constant DeltafinLoopKdaWideDimsV1& dims [[buffer(16)]],
    uint h [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint c = h * 128u + t;
  const uint W = dims.source_width;
  const uint P = dims.positions;

  /* The state lives in device memory with the core kernel's coalesced
   * [k*128 + t] access (thread t owns column t): S_in feeds position 0;
   * each position writes its boundary slot (retain) or S_out in place, and
   * the next position reads what this thread wrote. */
  device const float* Sh = S_in + (ulong)h * 16384u;

  threadgroup float red[4];
  threadgroup float k_sh[128], q_sh[128], d_sh[128];
  const float a = precise::exp(a_log[t]);

  device const float* srcs[3] = {src_q, src_k, src_v};
  device const float* wks[3] = {convw_q, convw_k, convw_v};

  for (uint p = 0u; p < P; ++p) {
    float conv_res[3];
    for (uint m = 0u; m < 3u; ++m) {
      device const float* st = srcs[m] + (ulong)c * W + p;
      device const float* wk = wks[m] + (ulong)c * 4u;
      const float s0 = st[0], s1 = st[1], s2 = st[2], s3 = st[3];
      const float acc = s0 * wk[0] + s1 * wk[1] + s2 * wk[2] + s3 * wk[3];
      conv_res[m] = acc * deltafin_loop_sigmoid_p(acc);
    }
    float qv = conv_res[0], kv = conv_res[1];
    const float vv = conv_res[2];
    const float qss = deltafin_loop_tg_sum128(qv * qv, red, lane, sg);
    const float kss = deltafin_loop_tg_sum128(kv * kv, red, lane, sg);
    qv /= max(precise::sqrt(qss), 1e-12f);
    kv /= max(precise::sqrt(kss), 1e-12f);
    const float q_tok = qv * 0.08838834764831845f;

    const float rg = fb[(ulong)p * 12288u + c] + dt_bias[c];
    const float decay = precise::exp(-5.0f * deltafin_loop_sigmoid_p(a * rg));
    const float beta = deltafin_loop_sigmoid_p(beta_raw[p * 96u + h]);

    /* The previous position's readers of k_sh/q_sh/d_sh must be done. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    k_sh[t] = kv;
    q_sh[t] = q_tok;
    d_sh[t] = decay;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint k = 0u; k < 128u; k += 4u) {
      acc0 = fma(k_sh[k + 0u] * d_sh[k + 0u], Sh[(k + 0u) * 128u + t], acc0);
      acc1 = fma(k_sh[k + 1u] * d_sh[k + 1u], Sh[(k + 1u) * 128u + t], acc1);
      acc2 = fma(k_sh[k + 2u] * d_sh[k + 2u], Sh[(k + 2u) * 128u + t], acc2);
      acc3 = fma(k_sh[k + 3u] * d_sh[k + 3u], Sh[(k + 3u) * 128u + t], acc3);
    }
    const float delta = vv - ((acc0 + acc1) + (acc2 + acc3));

    device float* Sho = (dims.retain != 0u ? S_bound + ((ulong)p * 96u + h) * 16384u
                                           : S_out + (ulong)h * 16384u);
    const float bd = beta * delta;
    float ov0 = 0.0f, ov1 = 0.0f, ov2 = 0.0f, ov3 = 0.0f;
    for (uint k = 0u; k < 128u; k += 4u) {
      const float sn0 = fma(Sh[(k + 0u) * 128u + t], d_sh[k + 0u], k_sh[k + 0u] * bd);
      const float sn1 = fma(Sh[(k + 1u) * 128u + t], d_sh[k + 1u], k_sh[k + 1u] * bd);
      const float sn2 = fma(Sh[(k + 2u) * 128u + t], d_sh[k + 2u], k_sh[k + 2u] * bd);
      const float sn3 = fma(Sh[(k + 3u) * 128u + t], d_sh[k + 3u], k_sh[k + 3u] * bd);
      Sho[(k + 0u) * 128u + t] = sn0;
      Sho[(k + 1u) * 128u + t] = sn1;
      Sho[(k + 2u) * 128u + t] = sn2;
      Sho[(k + 3u) * 128u + t] = sn3;
      ov0 = fma(q_sh[k + 0u], sn0, ov0);
      ov1 = fma(q_sh[k + 1u], sn1, ov1);
      ov2 = fma(q_sh[k + 2u], sn2, ov2);
      ov3 = fma(q_sh[k + 3u], sn3, ov3);
    }
    float ov = (ov0 + ov1) + (ov2 + ov3);
    Sh = Sho;

    const float oss = deltafin_loop_tg_sum128(ov * ov, red, lane, sg);
    ov *= 1.0f / precise::sqrt(oss * (1.0f / 128.0f) + 1e-5f);
    ov *= o_norm[t];
    ov *= deltafin_loop_sigmoid_p(gate[(ulong)p * 12288u + c]);
    out[(ulong)p * 12288u + c] = ov;
  }
  if (dims.retain != 0u) {
    /* The final state is the last boundary; copy this thread's column. */
    device float* So = S_out + (ulong)h * 16384u;
    for (uint k = 0u; k < 128u; ++k) So[k * 128u + t] = Sh[k * 128u + t];
  }
}
