#ifndef DELTAFIN_PROVIDER_MLA_ATTN_METAL_H
#define DELTAFIN_PROVIDER_MLA_ATTN_METAL_H

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace deltafin::provider_internal {

constexpr std::uint32_t kMlaAttnMetalAbiV1 = 1;
constexpr std::uint32_t kMlaAttnMetalEmbeddedLibraryV1 = 1U << 0;
constexpr std::uint32_t kMlaAttnMetalFp32KvSlabV1 = 1U << 1;
constexpr std::uint32_t kMlaAttnMetalCurrentStreamV1 = 1U << 2;
constexpr std::uint32_t kMlaAttnMetalAsyncDispatchV1 = 1U << 3;
constexpr std::uint32_t kMlaAttnMetalStorageOffsetV1 = 1U << 4;
constexpr std::uint32_t kMlaAttnMetalRequiredCapabilitiesV1 =
    kMlaAttnMetalEmbeddedLibraryV1 | kMlaAttnMetalFp32KvSlabV1 |
    kMlaAttnMetalCurrentStreamV1 | kMlaAttnMetalAsyncDispatchV1 |
    kMlaAttnMetalStorageOffsetV1;

struct MlaAttnMetalCapabilities {
  std::uint32_t abi_version = 0;
  std::uint32_t flags = 0;
  std::uint32_t threads_per_threadgroup = 0;
  std::uint32_t reserved = 0;
};

struct MlaAttnMetalCanaryReport {
  std::uint32_t heads = 0;
  std::uint32_t kv_length = 0;
  std::uint32_t capacity = 0;
  std::uint32_t partitions = 0;
  std::uint32_t compared_elements = 0;
  std::uint32_t close_elements = 0;
  std::uint32_t nonfinite = 0;
  std::uint32_t query_offset_elements = 0;
  std::uint32_t key_offset_elements = 0;
  std::uint32_t value_offset_elements = 0;
  std::uint32_t destination_offset_elements = 0;
  std::uint32_t reserved = 0;
  double max_absolute_error = 0.0;
  double relative_l2_error = 0.0;
};

/* Tolerances the canary and its consumers agree on: the kernel reassociates
 * fp32 reductions (simd_sum trees, online softmax), so it is compared against
 * an fp64 host reference rather than for bit equality. */
constexpr double kMlaAttnMetalCanaryRtol = 2.0e-5;
constexpr double kMlaAttnMetalCanaryAtol = 2.0e-6;
constexpr double kMlaAttnMetalCanaryMaxRelL2 = 1.0e-5;

[[nodiscard]] bool mla_attn_metal_canary_passes(
    const MlaAttnMetalCanaryReport& report);

#if defined(__APPLE__)

/* Load and validate only the embedded fused-attention pipelines. */
[[nodiscard]] MlaAttnMetalCapabilities mla_attn_metal_capabilities_v1();

/*
 * Encode the fused T=1 MLA attention core on PyTorch's current MPS stream:
 *
 *   destination[0,0,h*128+d] =
 *       softmax_t(q[0,h,0,:] . key_states[0,h,t,:] * 192^-0.5)
 *           . value_states[0,h,:,d]
 *
 * destination is the contiguous fp32 pre-gate row [1,1,12288]; query is the
 * contiguous fp32 [1,96,1,192] head view; key_states/value_states are the
 * provider's live narrowed fp32 cache-slab views [1,96,S,192]/[1,96,S,128]
 * whose position pitch (slab capacity) is re-derived from stride(1) on every
 * call — MTLBuffers are never cached across tokens because cache growth swaps
 * slabs.  Non-zero 16-byte-aligned storage offsets are part of the contract.
 * Both dispatches (partition + combine) are encoded inside one
 * dispatch_sync_with_rethrow block with a buffer memory barrier between them;
 * the call is asynchronous and performs no command-buffer commit or host wait.
 *
 * NOT bit-exact to the LibTorch einsum/softmax sequence: fp32 reductions are
 * reassociated.  Production use is therefore an explicit env opt-in
 * (K3_MLA_METAL_ATTENTION=1) behind mla_attn_metal_decode_ready().
 */
void mla_attn_metal_decode_f32(const at::Tensor& destination,
                               const at::Tensor& query,
                               const at::Tensor& key_states,
                               const at::Tensor& value_states);

/* Test/canary surface: identical contract with an explicit flash-decoding
 * partition count in [1,32] (0 selects the production heuristic). */
void mla_attn_metal_decode_f32_with_partitions(
    const at::Tensor& destination, const at::Tensor& query,
    const at::Tensor& key_states, const at::Tensor& value_states,
    std::uint32_t partitions);

/* Deterministic S=257/capacity=300 qualification against an fp64 host
 * reference, including non-zero storage offsets, empty flash partitions, and
 * NaN poison beyond S to catch capacity-pitch regressions. */
[[nodiscard]] MlaAttnMetalCanaryReport mla_attn_metal_canary_v1();

/* True only when K3_MLA_METAL_ATTENTION=1 AND the session-wide one-shot
 * canary qualification passed on this process's MPS device. */
[[nodiscard]] bool mla_attn_metal_decode_ready();

#endif  // defined(__APPLE__)

}  // namespace deltafin::provider_internal

#endif
