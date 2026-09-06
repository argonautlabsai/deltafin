#if !defined(__APPLE__)
#error "provider_loop.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
#error "bespoke decode-loop scaffold requires its explicit production capability"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "bespoke decode-loop scaffold requires an embedded precompiled metallib"
#endif

#include "provider_loop.h"
#include "provider_prep_timer.h"

#include <ATen/ATen.h>
#include <unordered_set>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "bespoke decode-loop scaffold requires Objective-C ARC"
#endif

#include "deltafin_embedded_loop_kernels_metallib.h"

#include "../../tools/metal_moe_abi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace deltafin::provider_internal {
namespace {

/* Attention-timer split (provider_prep_timer.h). The previous MoE tail's
 * completion stamp (host media clock, ns), written by the tail's done
 * listener: a prepare-side wait that began before this stamp was queued
 * behind the previous layer's tail — drain, not this layer's own work. */
std::atomic<std::uint64_t> g_prep_tail_done_ns{0};

std::uint64_t prep_media_ns() {
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    return info;
  }();
  return static_cast<std::uint64_t>(mach_absolute_time()) * timebase.numer /
         timebase.denom;
}

/* A decode-thread semaphore wait on a loop shared-event listener, reported
 * to the open prepare/hint phase scope (plain wait outside one). `command`
 * is the CB carrying the awaited signal; the part of the wait before its
 * GPUStartTime — or before the previous tail's done stamp — is drain. */
long prep_timed_semaphore_wait(dispatch_semaphore_t semaphore,
                               const dispatch_time_t timeout,
                               id<MTLCommandBuffer> command) {
  if (!prep_phase_active()) {
    return dispatch_semaphore_wait(semaphore, timeout);
  }
  const std::uint64_t started = prep_media_ns();
  const long result = dispatch_semaphore_wait(semaphore, timeout);
  const std::uint64_t ended = prep_media_ns();
  const std::uint64_t waited = ended - started;
  std::uint64_t drain = 0;
  bool timestamped = false;
  if (command != nil) {
    const double gpu_start = command.GPUStartTime;
    if (gpu_start > 0.0) {
      timestamped = true;
      const auto gpu_start_ns = static_cast<std::uint64_t>(gpu_start * 1e9);
      if (gpu_start_ns > started) {
        drain = std::min(gpu_start_ns - started, waited);
      }
    }
  }
  const std::uint64_t tail_done =
      g_prep_tail_done_ns.load(std::memory_order_relaxed);
  if (tail_done > started) {
    drain = std::max(drain, std::min(tail_done - started, waited));
  }
  prep_note_wait(waited, drain, timestamped);
  return result;
}

/* The loop reuses the route-mailbox host layout verbatim: any drift here is
 * a readback-protocol break, so it is a compile error. */
static_assert(offsetof(RouteMailboxT1, expert_ids) == 0);
static_assert(offsetof(RouteMailboxT1, weight_bits) == 128);
static_assert(sizeof(RouteMailboxT1) == 192);

constexpr std::uint32_t kTopK =
    static_cast<std::uint32_t>(kRouteMailboxTopK);

/* Named slots of the shared-event pool (blueprint command-buffer plan). */
constexpr std::size_t kLoopEventRoute = 0;
constexpr std::size_t kLoopEventAtenBoundary = 1;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
  if (!condition) fail(message);
}

dispatch_data_t copy_embedded_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedLoopKernelsMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedLoopKernelsMetallib,
              kDeltafinEmbeddedLoopKernelsMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedLoopKernelsMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}

/* One cached device view of a spine tensor.  The strong MTLBuffer reference
 * is scaffold-only bookkeeping: entries are generation-stamped and evicted on
 * rebind, and the at::Tensor storages remain authoritative for lifetime. */
struct WeightsTableEntry {
  std::uint64_t generation = 0;
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0;
  std::size_t bytes = 0;
};

struct LoopEncoderState {
  std::mutex mutex;
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> route_pipeline = nil;
  id<MTLComputePipelineState> pilot_pipeline = nil;
  id<MTLComputePipelineState> gemv_pipeline = nil;
  id<MTLComputePipelineState> gemv_bf16_pipeline = nil;
  id<MTLComputePipelineState> gemv_f32_pipeline = nil;
  id<MTLComputePipelineState> router_top16_pipeline = nil;
  id<MTLComputePipelineState> kda_core_pipeline = nil;
  id<MTLComputePipelineState> attn_res_mix_pipeline = nil;
  id<MTLComputePipelineState> rmsnorm_pipeline = nil;
  id<MTLComputePipelineState> add_pipeline = nil;
  id<MTLComputePipelineState> situ_pipeline = nil;
  /* MLA takeover (plan Step 0): the fused decode-attention core, shared
   * verbatim with the standalone library via provider_mla_attn_core.metalh.
   * The 24 MLA layers are the last ATen-resident layers in the decode path
   * (measured 2026-08-24: attention 8.7ms of a 30.6ms layer, 28% of the
   * token). Building the pipelines here fails loudly if the kernels are
   * missing from the loop metallib. */
  id<MTLComputePipelineState> mla_part_pipeline = nil;
  id<MTLComputePipelineState> mla_combine_pipeline = nil;
  id<MTLComputePipelineState> mla_pack_pipeline = nil;
  id<MTLComputePipelineState> mla_gate_pipeline = nil;
  id<MTLCommandQueue> queue = nil;
  MTLSharedEventListener* listener = nil;
  std::array<id<MTLSharedEvent>, kLoopMetalSharedEventPoolSizeV1> events = {};
  std::array<std::uint64_t, kLoopMetalSharedEventPoolSizeV1> next_values = {};
  id<MTLBuffer> route_mailbox = nil;
  id<MTLBuffer> pilot_mailbox = nil;
  std::map<std::pair<std::uint32_t, std::uint32_t>, WeightsTableEntry> table;
  LoopWeightsTableStats table_stats;
  // Router shadow (K3-SIDEQUEUE-PLAN.md Step 2): persistent 896-float
  // logits/scores scratch plus the single in-flight shadow ticket (decode
  // is layer-serial, so at most one shadow is ever pending).
  id<MTLBuffer> shadow_logits = nil;
  id<MTLBuffer> shadow_scores = nil;
  // KDA parity scratch (plan Step 3a): projection bundle output, f_b
  // output, fused-core output, final o_proj output, staged conv windows and
  // recurrent state — all shared-storage so the host compares directly
  // after the completion event (~7.2 MB total).
  id<MTLBuffer> kda_proj = nil;
  id<MTLBuffer> kda_fb = nil;
  id<MTLBuffer> kda_core_out = nil;
  id<MTLBuffer> kda_final_out = nil;
  id<MTLBuffer> kda_conv_out = nil;
  id<MTLBuffer> kda_s_out = nil;
  id<MTLBuffer> kda_mixed = nil;
  // CB_tail scratch (Step 5): expert reduce output, routed norm/up chain,
  // shared-expert epilogue, and the merge — all shared-storage (~150 KB).
  id<MTLBuffer> moe_routed_out = nil;
  id<MTLBuffer> moe_normed = nil;
  id<MTLBuffer> moe_routed_full = nil;
  id<MTLBuffer> moe_gate_up = nil;
  id<MTLBuffer> moe_situ = nil;
  id<MTLBuffer> moe_shared_out = nil;
  id<MTLBuffer> moe_mlp_out = nil;
  // MLA parity scratch (Step 2c(ii)): bundle output, normed slices, q_b /
  // kv_b outputs, packed new row, attention partials + pre-gate row, gated
  // row, final o_proj output — all shared-storage (~2.1 MB). The private
  // KV scratch slab is the one variable-size piece: allocated lazily at
  // parity time with position pitch = the parity S, regrown geometrically
  // (never the live cache slab — parity must not write it).
  id<MTLBuffer> mla_bundle_out = nil;   // 14400
  id<MTLBuffer> mla_qa_norm = nil;      // 1536
  id<MTLBuffer> mla_query = nil;        // 18432
  id<MTLBuffer> mla_latent_norm = nil;  // 512
  id<MTLBuffer> mla_expanded = nil;     // 24576
  id<MTLBuffer> mla_new_key = nil;      // 96*192
  id<MTLBuffer> mla_new_value = nil;    // 96*128
  id<MTLBuffer> mla_attn_out = nil;     // 12288
  id<MTLBuffer> mla_gated = nil;        // 12288
  id<MTLBuffer> mla_final_out = nil;    // 7168
  id<MTLBuffer> mla_partials = nil;     // 96*32*132
  id<MTLBuffer> mla_mixed = nil;        // 7168 (chained res-mix output)
  id<MTLBuffer> mla_kv_keys = nil;
  id<MTLBuffer> mla_kv_values = nil;
  std::int64_t mla_kv_capacity = 0;
  LoopMlaParityStats mla_parity_stats;
  LoopMlaTakeoverStats mla_takeover_stats;
  LoopMlaChainStats mla_chain_stats;
  LoopKdaParityStats kda_parity_stats;
  bool shadow_active = false;
  /* Layer whose route CB is in flight on the side queue; collect refuses a
   * mismatched layer (fail-soft to the stock chain). 0xFFFFFFFF = none.
   * Needed once routes can be CHAINED at pre-commit time: an abandoned
   * chain must never satisfy a later layer's collect. */
  std::uint32_t route_pending_layer = 0xFFFFFFFFu;
  /* The done semaphore is one-shot; the true-route hint peek may consume
   * the wait before collect does (single decode thread coordinates). */
  bool shadow_ready_consumed = false;
  dispatch_semaphore_t shadow_ready = nil;
  id<MTLCommandBuffer> shadow_command = nil;
  LoopRouterShadowStats shadow_stats;
};

LoopEncoderState& loop_state() {
  static LoopEncoderState state;
  return state;
}

id<MTLBuffer> tensor_buffer(const at::Tensor& tensor) {
  // The same reviewed MPS storage representation used by the route mailbox
  // and the landed MLA attention kernel.
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

NSUInteger checked_byte_offset(const at::Tensor& tensor,
                               const std::size_t required_bytes,
                               const char* name) {
  require(tensor.storage_offset() >= 0,
          std::string(name) + " has a negative storage offset");
  const std::uint64_t elements =
      static_cast<std::uint64_t>(tensor.storage_offset());
  const std::uint64_t width =
      static_cast<std::uint64_t>(tensor.element_size());
  require(elements <= std::numeric_limits<std::uint64_t>::max() / width,
          std::string(name) + " storage offset overflows uint64");
  const std::uint64_t raw = elements * width;
  require(raw <= std::numeric_limits<NSUInteger>::max(),
          std::string(name) + " storage offset exceeds NSUInteger");
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  require(buffer != nil, std::string(name) + " has no MTLBuffer");
  const NSUInteger offset = static_cast<NSUInteger>(raw);
  require(offset <= buffer.length &&
              required_bytes <= buffer.length - offset,
          std::string(name) + " span exceeds its MTLBuffer");
  return offset;
}

id<MTLComputePipelineState> build_pipeline(id<MTLDevice> device,
                                           id<MTLLibrary> library,
                                           NSString* name,
                                           const NSUInteger minimum_threads) {
  NSError* error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  require(function != nil,
          std::string("embedded loop-kernels metallib is missing ") +
              name.UTF8String);
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  require(pipeline != nil,
          std::string("create loop-kernel Metal pipeline failed: ") +
              (error == nil ? "unknown"
                            : error.localizedDescription.UTF8String));
  require(pipeline.maxTotalThreadsPerThreadgroup >= minimum_threads,
          "loop-kernel Metal pipeline cannot admit its threadgroup");
  return pipeline;
}

void ensure_resources(LoopEncoderState& state, id<MTLDevice> device) {
  require(device != nil, "bespoke-loop Metal device is unavailable");
  if (state.device == device && state.queue != nil &&
      state.route_pipeline != nil && state.pilot_pipeline != nil &&
      state.gemv_pipeline != nil && state.gemv_bf16_pipeline != nil &&
      state.gemv_f32_pipeline != nil &&
      state.router_top16_pipeline != nil &&
      state.kda_core_pipeline != nil &&
      state.attn_res_mix_pipeline != nil && state.rmsnorm_pipeline != nil &&
      state.add_pipeline != nil && state.situ_pipeline != nil &&
      state.mla_part_pipeline != nil && state.mla_combine_pipeline != nil &&
      state.mla_pack_pipeline != nil && state.mla_gate_pipeline != nil &&
      state.listener != nil &&
      state.route_mailbox != nil && state.pilot_mailbox != nil &&
      state.events.front() != nil) {
    return;
  }

  state.device = device;
  state.library = nil;
  state.route_pipeline = nil;
  state.pilot_pipeline = nil;
  state.gemv_pipeline = nil;
  state.gemv_bf16_pipeline = nil;
  state.gemv_f32_pipeline = nil;
  state.router_top16_pipeline = nil;
  state.kda_core_pipeline = nil;
  state.attn_res_mix_pipeline = nil;
  state.rmsnorm_pipeline = nil;
  state.add_pipeline = nil;
  state.situ_pipeline = nil;
  state.mla_part_pipeline = nil;
  state.mla_combine_pipeline = nil;
  state.mla_pack_pipeline = nil;
  state.mla_gate_pipeline = nil;
  state.queue = nil;
  state.listener = nil;
  state.events = {};
  state.next_values = {};
  state.route_mailbox = nil;
  state.pilot_mailbox = nil;
  state.shadow_logits = nil;
  state.shadow_scores = nil;
  state.shadow_active = false;
  state.shadow_ready = nil;
  state.shadow_command = nil;
  state.kda_proj = nil;
  state.kda_fb = nil;
  state.kda_core_out = nil;
  state.kda_final_out = nil;
  state.kda_conv_out = nil;
  state.kda_s_out = nil;
  state.kda_mixed = nil;
  state.moe_routed_out = nil;
  state.moe_normed = nil;
  state.moe_routed_full = nil;
  state.moe_gate_up = nil;
  state.moe_situ = nil;
  state.moe_shared_out = nil;
  state.moe_mlp_out = nil;

  dispatch_data_t data = copy_embedded_metallib();
  require(data != nullptr, "wrap embedded loop-kernels metallib failed");
  NSError* error = nil;
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  require(library != nil,
          std::string("load embedded loop-kernels metallib failed: ") +
              (error == nil ? "unknown"
                            : error.localizedDescription.UTF8String));
  state.library = library;
  state.route_pipeline =
      build_pipeline(device, library, @"deltafin_loop_canary_route_v1", kTopK);
  state.pilot_pipeline =
      build_pipeline(device, library, @"deltafin_loop_canary_pilot_v1", kTopK);
  // Router side-queue chain (K3-SIDEQUEUE-PLAN.md): the GEMV runs 128
  // threads per threadgroup, the fused top-16 select runs 256.
  state.gemv_pipeline = build_pipeline(
      device, library, @"deltafin_loop_gemv_i8_bundle_v1", 128);
  state.gemv_bf16_pipeline = build_pipeline(
      device, library, @"deltafin_loop_gemv_bf16_rows4_v1", 128);
  state.gemv_f32_pipeline = build_pipeline(
      device, library, @"deltafin_loop_gemv_f32_rows4_v1", 128);
  state.router_top16_pipeline = build_pipeline(
      device, library, @"deltafin_loop_router_top16_v1", 256);
  state.kda_core_pipeline = build_pipeline(
      device, library, @"deltafin_loop_kda_core_v1", 128);
  state.attn_res_mix_pipeline = build_pipeline(
      device, library, @"deltafin_loop_attn_res_mix_v1", 256);
  state.rmsnorm_pipeline =
      build_pipeline(device, library, @"deltafin_loop_rmsnorm_v1", 256);
  state.add_pipeline =
      build_pipeline(device, library, @"deltafin_loop_add_v1", 256);
  state.situ_pipeline =
      build_pipeline(device, library, @"deltafin_loop_situ_v1", 256);
  // 256 threads = 8 simdgroups per threadgroup for the partial pass; the
  // combine pass merges P partials per head with a single simdgroup.
  state.mla_part_pipeline =
      build_pipeline(device, library, @"deltafin_mla_attn_part_f32_v1", 256);
  state.mla_combine_pipeline = build_pipeline(
      device, library, @"deltafin_mla_attn_combine_f32_v1", 32);
  state.mla_pack_pipeline = build_pipeline(
      device, library, @"deltafin_loop_mla_pack_kv_v1", 256);
  state.mla_gate_pipeline =
      build_pipeline(device, library, @"deltafin_loop_mla_gate_v1", 256);

  // The loop-owned serial queue: whole-layer command buffers execute in
  // commit order, independent of the ATen stream's queue.
  state.queue = [device newCommandQueue];
  loop_residency_attach_queue((__bridge void*)state.queue);
  require(state.queue != nil, "bespoke-loop command queue is unavailable");
  state.queue.label = @"deltafin.bespoke-loop";

  dispatch_queue_t listener_queue = dispatch_queue_create(
      "deltafin.bespoke-loop.events", DISPATCH_QUEUE_SERIAL);
  state.listener = [[MTLSharedEventListener alloc]
      initWithDispatchQueue:listener_queue];
  require(state.listener != nil,
          "bespoke-loop event listener is unavailable");

  for (std::size_t index = 0; index < state.events.size(); ++index) {
    state.events[index] = [device newSharedEvent];
    require(state.events[index] != nil,
            "bespoke-loop shared event is unavailable");
    state.next_values[index] = 1;
  }

  // 192-byte shared-storage mailboxes, the same allocation contract as the
  // route mailbox (host-visible contents, no readback blit).
  for (id<MTLBuffer> __strong* mailbox :
       {&state.route_mailbox, &state.pilot_mailbox}) {
    *mailbox = [device
        newBufferWithLength:sizeof(RouteMailboxT1)
                    options:MTLResourceStorageModeShared |
                            MTLResourceCPUCacheModeDefaultCache];
    require(*mailbox != nil, "bespoke-loop mailbox allocation failed");
  }

  for (id<MTLBuffer> __strong* scratch :
       {&state.shadow_logits, &state.shadow_scores}) {
    *scratch = [device
        newBufferWithLength:896 * sizeof(float)
                    options:MTLResourceStorageModeShared |
                            MTLResourceCPUCacheModeDefaultCache];
    require(*scratch != nil, "bespoke-loop shadow scratch allocation failed");
  }

  const auto kda_scratch = [&](id<MTLBuffer> __strong* buffer,
                               const std::size_t elements,
                               const char* name) {
    *buffer = [device
        newBufferWithLength:elements * sizeof(float)
                    options:MTLResourceStorageModeShared |
                            MTLResourceCPUCacheModeDefaultCache];
    require(*buffer != nil,
            std::string("bespoke-loop KDA scratch allocation failed: ") +
                name);
  };
  kda_scratch(&state.kda_proj, 49376, "proj");
  kda_scratch(&state.kda_fb, 12288, "fb");
  kda_scratch(&state.kda_core_out, 12288, "core_out");
  kda_scratch(&state.kda_final_out, 7168, "final_out");
  kda_scratch(&state.kda_conv_out, static_cast<std::size_t>(3) * 12288 * 4,
              "conv_out");
  kda_scratch(&state.kda_s_out,
              static_cast<std::size_t>(96) * 128 * 128, "s_out");
  kda_scratch(&state.kda_mixed, 7168, "mixed");
  kda_scratch(&state.moe_routed_out, 3584, "moe routed_out");
  kda_scratch(&state.moe_normed, 3584, "moe normed");
  kda_scratch(&state.moe_routed_full, 7168, "moe routed_full");
  kda_scratch(&state.moe_gate_up, 12288, "moe gate_up");
  kda_scratch(&state.moe_situ, 6144, "moe situ");
  kda_scratch(&state.moe_shared_out, 7168, "moe shared_out");
  kda_scratch(&state.moe_mlp_out, 7168, "moe mlp_out");
  kda_scratch(&state.mla_bundle_out, 14400, "mla bundle_out");
  kda_scratch(&state.mla_qa_norm, 1536, "mla qa_norm");
  kda_scratch(&state.mla_query, 18432, "mla query");
  kda_scratch(&state.mla_latent_norm, 512, "mla latent_norm");
  kda_scratch(&state.mla_expanded, 24576, "mla expanded");
  kda_scratch(&state.mla_new_key,
              static_cast<std::size_t>(96) * 192, "mla new_key");
  kda_scratch(&state.mla_new_value,
              static_cast<std::size_t>(96) * 128, "mla new_value");
  kda_scratch(&state.mla_attn_out, 12288, "mla attn_out");
  kda_scratch(&state.mla_gated, 12288, "mla gated");
  kda_scratch(&state.mla_final_out, 7168, "mla final_out");
  kda_scratch(&state.mla_partials,
              static_cast<std::size_t>(96) * 32 * 132, "mla partials");
  kda_scratch(&state.mla_mixed, 7168, "mla mixed");
  // KV scratch is lazily sized at parity time; a device rebuild must not
  // leave a stale-capacity buffer from the old device behind.
  state.mla_kv_keys = nil;
  state.mla_kv_values = nil;
  state.mla_kv_capacity = 0;
  state.shadow_active = false;
  state.shadow_ready = nil;
  state.shadow_command = nil;
}

std::uint64_t take_event_value(LoopEncoderState& state,
                               const std::size_t event_index) {
  require(event_index < state.next_values.size(),
          "bespoke-loop event index is outside the pool");
  require(state.next_values[event_index] !=
              std::numeric_limits<std::uint64_t>::max(),
          "bespoke-loop event counter exhausted");
  return state.next_values[event_index]++;
}

bool tensor_qualifies_for_table(const at::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_mps() &&
         (tensor.scalar_type() == at::kFloat ||
          tensor.scalar_type() == at::kChar) &&
         tensor.is_contiguous() && !tensor.requires_grad() &&
         tensor.numel() > 0 && tensor.storage_offset() >= 0;
}

}  // namespace

bool loop_metal_canary_passes(const LoopMetalCanaryReport& report) {
  return report.top_k == kTopK && report.route_ids_matched == kTopK &&
         report.route_weights_matched == kTopK &&
         report.pilot_ids_matched == kTopK &&
         report.pilot_weights_matched == kTopK &&
         report.command_buffers == 2 && report.events_signaled == 2 &&
         report.weights_table_hits == 2 &&
         report.weights_table_invalidations == 1 &&
         report.id_offset_elements != 0 &&
         report.weight_offset_elements != 0 && report.reserved == 0 &&
         std::isfinite(report.max_absolute_error) &&
         std::isfinite(report.relative_l2_error) &&
         report.relative_l2_error <= kLoopMetalCanaryMaxRelL2 &&
         report.mla_heads_checked == 96 &&
         std::isfinite(report.mla_attention_relative_l2) &&
         report.mla_attention_relative_l2 <= kLoopMlaAttentionMaxRelL2;
}

LoopMetalCapabilities loop_metal_capabilities_v1() {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  ensure_resources(state, stream->device());
  return LoopMetalCapabilities{
      .abi_version = kLoopMetalAbiV1,
      .flags = kLoopMetalRequiredCapabilitiesV1,
      .event_pool_size = kLoopMetalSharedEventPoolSizeV1,
      .reserved = 0,
  };
}

bool loop_weights_table_store(const std::uint32_t layer_index,
                              const std::uint64_t generation,
                              const std::uint32_t slot,
                              const at::Tensor& tensor) {
  try {
    if (!tensor_qualifies_for_table(tensor)) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(tensor.numel()) *
                              static_cast<std::size_t>(tensor.element_size());
    const NSUInteger offset =
        checked_byte_offset(tensor, bytes, "bespoke-loop weights-table tensor");
    id<MTLBuffer> buffer = tensor_buffer(tensor);
    LoopEncoderState& state = loop_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    WeightsTableEntry& entry = state.table[{layer_index, slot}];
    entry.generation = generation;
    entry.buffer = buffer;
    entry.offset = offset;
    entry.bytes = bytes;
    state.table_stats.entries =
        static_cast<std::uint64_t>(state.table.size());
    return true;
  } catch (...) {
    return false;
  }
}

bool loop_weights_table_hit(const std::uint32_t layer_index,
                            const std::uint64_t generation,
                            const std::uint32_t slot) {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto found = state.table.find({layer_index, slot});
  if (found == state.table.end()) {
    ++state.table_stats.misses;
    return false;
  }
  if (found->second.generation != generation) {
    // A spine generation change means the layer was rebound onto fresh
    // storages; the cached device view is stale and must be dropped.
    state.table.erase(found);
    state.table_stats.entries =
        static_cast<std::uint64_t>(state.table.size());
    ++state.table_stats.generation_invalidations;
    ++state.table_stats.misses;
    return false;
  }
  ++state.table_stats.hits;
  return true;
}

LoopWeightsTableStats loop_weights_table_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.table_stats;
}

void loop_weights_table_reset() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.table.clear();
  state.table_stats = LoopWeightsTableStats{};
}

namespace {

/* Exercises the public weights-table API end to end: store, a
 * generation-matched hit, a generation-mismatched hit (which evicts the
 * stale entry and counts as an invalidation), a re-store, and a re-hit.
 * Called only after loop_metal_canary_v1's mutex scope has released, since
 * every one of these calls takes the same state mutex itself. */
LoopMetalCanaryReport finish_canary_with_table(LoopMetalCanaryReport report,
                                               const at::Tensor& tensor) {
  constexpr std::uint32_t kCanaryLayer = 0;
  constexpr std::uint32_t kCanarySlot = 0;
  constexpr std::uint64_t kCanaryGenerationA = 1;
  constexpr std::uint64_t kCanaryGenerationB = 2;

  loop_weights_table_reset();
  require(loop_weights_table_store(kCanaryLayer, kCanaryGenerationA,
                                   kCanarySlot, tensor),
          "bespoke-loop weights-table canary store failed");
  require(loop_weights_table_hit(kCanaryLayer, kCanaryGenerationA,
                                 kCanarySlot),
          "bespoke-loop weights-table canary hit failed");
  require(!loop_weights_table_hit(kCanaryLayer, kCanaryGenerationB,
                                  kCanarySlot),
          "bespoke-loop weights-table canary failed to reject a stale "
          "generation");
  require(loop_weights_table_store(kCanaryLayer, kCanaryGenerationB,
                                   kCanarySlot, tensor),
          "bespoke-loop weights-table canary re-store failed");
  require(loop_weights_table_hit(kCanaryLayer, kCanaryGenerationB,
                                 kCanarySlot),
          "bespoke-loop weights-table canary re-hit failed");

  const LoopWeightsTableStats stats = loop_weights_table_stats();
  report.weights_table_hits = static_cast<std::uint32_t>(stats.hits);
  report.weights_table_invalidations =
      static_cast<std::uint32_t>(stats.generation_invalidations);
  loop_weights_table_reset();
  return report;
}

}  // namespace

/*
 * MLA attention through the loop encoder, checked against a double-precision
 * reference on identical inputs. Production head geometry (96 heads, 192-wide
 * keys, 128-wide values); a short KV prefix keeps the canary cheap while
 * exercising the real indexing — the slab pitch is the cache CAPACITY, not the
 * live length, which is the indexing mistake this check exists to catch.
 */
void run_loop_mla_attention_canary(LoopMetalCanaryReport& report) noexcept {
  constexpr std::uint32_t kHeads = 96;
  constexpr std::uint32_t kKeyDim = 192;
  constexpr std::uint32_t kValueDim = 128;
  constexpr std::uint32_t kLength = 5;     // committed prefix + staged token
  constexpr std::uint32_t kCapacity = 8;   // slab pitch > length on purpose
  constexpr std::uint32_t kPartitions = 2;
  constexpr std::uint32_t kPartialStride = 132;

  try {
    LoopEncoderState& state = loop_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream == nullptr) {
      return;
    }
    id<MTLDevice> device = stream->device();
    if (device == nil) {
      return;
    }
    ensure_resources(state, device);

    const auto make_buffer = [&](const std::size_t elements,
                                 const char* name) {
      id<MTLBuffer> buffer = [device
          newBufferWithLength:elements * sizeof(float)
                      options:MTLResourceStorageModeShared |
                              MTLResourceCPUCacheModeDefaultCache];
      require(buffer != nil,
              std::string("MLA canary buffer allocation failed: ") + name);
      return buffer;
    };

    id<MTLBuffer> q = make_buffer(kHeads * kKeyDim, "mla q");
    id<MTLBuffer> keys =
        make_buffer(kHeads * kCapacity * kKeyDim, "mla keys");
    id<MTLBuffer> values = make_buffer(kHeads * kCapacity * kValueDim, "mla values");
    id<MTLBuffer> partials = make_buffer(kHeads * kPartitions * kPartialStride, "mla partials");
    id<MTLBuffer> out =
        make_buffer(kHeads * kValueDim, "mla out");

    std::uint32_t lcg = 0x9E37'79B9U;
    const auto next_value = [&lcg]() {
      lcg = lcg * 1664525U + 1013904223U;
      return static_cast<float>((lcg >> 8) & 0xFFFF) / 32768.0F - 1.0F;
    };
    auto* q_host = static_cast<float*>(q.contents);
    auto* k_host = static_cast<float*>(keys.contents);
    auto* v_host = static_cast<float*>(values.contents);
    for (std::uint32_t i = 0; i < kHeads * kKeyDim; ++i) {
      q_host[i] = next_value();
    }
    // Fill the whole slab, including positions beyond kLength: the kernel
    // must ignore them. If it indexed by length instead of capacity, or read
    // past the live prefix, the reference below would disagree.
    for (std::uint32_t i = 0; i < kHeads * kCapacity * kKeyDim; ++i) {
      k_host[i] = next_value();
    }
    for (std::uint32_t i = 0; i < kHeads * kCapacity * kValueDim; ++i) {
      v_host[i] = next_value();
    }
    std::memset(partials.contents, 0,
                kHeads * kPartitions * kPartialStride * sizeof(float));
    std::memset(out.contents, 0, kHeads * kValueDim * sizeof(float));

    struct Dims {
      std::uint32_t kv_length;
      std::uint32_t capacity;
      std::uint32_t partitions;
      std::uint32_t chunk;
      float scale;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
      std::uint32_t reserved2;
    } dims{kLength, kCapacity, kPartitions,
           (kLength + kPartitions - 1) / kPartitions,
           static_cast<float>(1.0 / std::sqrt(static_cast<double>(kKeyDim))),
           0, 0, 0};

    id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "MLA canary command buffer is unavailable");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA canary encoder is unavailable");
      [encoder setComputePipelineState:state.mla_part_pipeline];
      [encoder setBuffer:q offset:0 atIndex:0];
      [encoder setBuffer:keys offset:0 atIndex:1];
      [encoder setBuffer:values offset:0 atIndex:2];
      [encoder setBuffer:partials offset:0 atIndex:3];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads * kPartitions, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder setComputePipelineState:state.mla_combine_pipeline];
      [encoder setBuffer:partials offset:0 atIndex:0];
      [encoder setBuffer:out offset:0 atIndex:1];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [encoder endEncoding];
      [command commit];
    }
    [command waitUntilCompleted];
    if (command.error != nil) {
      return;
    }

    // fp64 reference: softmax over the live prefix, weighted value sum.
    const auto* got = static_cast<const float*>(out.contents);
    double difference_squares = 0.0;
    double expected_squares = 0.0;
    std::vector<double> scores(kLength);
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const float* qh = q_host + head * kKeyDim;
      const float* kh = k_host + static_cast<std::size_t>(head) * kCapacity * kKeyDim;
      const float* vh = v_host + static_cast<std::size_t>(head) * kCapacity * kValueDim;
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::uint32_t t = 0; t < kLength; ++t) {
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kKeyDim; ++d) {
          dot += static_cast<double>(qh[d]) *
                 static_cast<double>(kh[static_cast<std::size_t>(t) * kKeyDim + d]);
        }
        scores[t] = dot * static_cast<double>(dims.scale);
        maximum = std::max(maximum, scores[t]);
      }
      double total = 0.0;
      for (std::uint32_t t = 0; t < kLength; ++t) {
        scores[t] = std::exp(scores[t] - maximum);
        total += scores[t];
      }
      for (std::uint32_t d = 0; d < kValueDim; ++d) {
        double accumulator = 0.0;
        for (std::uint32_t t = 0; t < kLength; ++t) {
          accumulator +=
              scores[t] *
              static_cast<double>(vh[static_cast<std::size_t>(t) * kValueDim + d]);
        }
        const double want = accumulator / total;
        const double difference =
            static_cast<double>(got[head * kValueDim + d]) - want;
        difference_squares += difference * difference;
        expected_squares += want * want;
      }
    }
    report.mla_heads_checked = kHeads;
    report.mla_attention_relative_l2 =
        expected_squares > 0.0
            ? std::sqrt(difference_squares / expected_squares)
            : std::sqrt(difference_squares);
  } catch (...) {
    // Qualification-only: a failure leaves mla_heads_checked at 0, which the
    // pass predicate treats as a failed check.
  }
}

LoopMetalCanaryReport loop_metal_canary_v1() {
  constexpr std::int64_t kIdOffsetElements = 4;
  constexpr std::int64_t kWeightOffsetElements = 8;
  constexpr std::int64_t kTailElements = 8;

  // Deterministic inputs, same LCG family as the MLA attention canary.
  std::uint32_t lcg = 0x3C6E'F35FU;
  const auto next_value = [&lcg]() {
    lcg = lcg * 1664525U + 1013904223U;
    return static_cast<float>((lcg >> 8) & 0xFFFF) / 32768.0F - 1.0F;
  };
  std::array<std::int64_t, kTopK> id_values = {};
  std::array<float, kTopK> weight_values = {};
  for (std::uint32_t edge = 0; edge < kTopK; ++edge) {
    id_values[edge] = 896 - 37 * static_cast<std::int64_t>(edge);
    weight_values[edge] = next_value();
  }

  const auto mps_long =
      at::TensorOptions().dtype(at::kLong).device(at::kMPS);
  const auto mps_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
  const auto cpu_long = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  const auto cpu_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor id_storage = at::zeros(
      {kIdOffsetElements + static_cast<std::int64_t>(kTopK) + kTailElements},
      mps_long);
  at::Tensor weight_storage = at::zeros(
      {kWeightOffsetElements + static_cast<std::int64_t>(kTopK) +
       kTailElements},
      mps_float);
  at::Tensor expert_ids = id_storage.narrow(
      0, kIdOffsetElements, static_cast<std::int64_t>(kTopK));
  at::Tensor weights = weight_storage.narrow(
      0, kWeightOffsetElements, static_cast<std::int64_t>(kTopK));
  expert_ids.copy_(
      at::from_blob(id_values.data(),
                    {static_cast<std::int64_t>(kTopK)}, cpu_long),
      false);
  weights.copy_(
      at::from_blob(weight_values.data(),
                    {static_cast<std::int64_t>(kTopK)}, cpu_float),
      false);

  // ATen wrote the inputs on its own stream; the loop queue may only consume
  // them behind an ATen-side boundary.  The canary uses the host-wait
  // flavor; production hooks will use the commit+signal fence instead.
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

  const std::size_t id_bytes = kTopK * sizeof(std::int64_t);
  const std::size_t weight_bytes = kTopK * sizeof(float);
  const NSUInteger id_offset =
      checked_byte_offset(expert_ids, id_bytes, "bespoke-loop canary ids");
  const NSUInteger weight_offset = checked_byte_offset(
      weights, weight_bytes, "bespoke-loop canary weights");

  LoopEncoderState& state = loop_state();
  // route_output/pilot_output outlive the mutex scope below (the report they
  // feed is built after it releases), so they are declared here.
  RouteMailboxT1 route_output;
  RouteMailboxT1 pilot_output;
  // Scoped so the lock releases before the weights-table round trip further
  // down, which calls the public table API below and takes this same mutex
  // itself (non-recursive; must not still be held).
  {
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  require(tensor_buffer(expert_ids).device == device,
          "bespoke-loop canary tensors belong to another MPS device");
  ensure_resources(state, device);

  std::memset(&route_output, 0xA5, sizeof(route_output));
  std::memset(&pilot_output, 0xA5, sizeof(pilot_output));
  {
    // Poison both mailboxes so a silently skipped dispatch cannot pass.
    std::memcpy(state.route_mailbox.contents, &route_output,
                sizeof(route_output));
    std::memcpy(state.pilot_mailbox.contents, &pilot_output,
                sizeof(pilot_output));
  }

  const std::uint64_t route_value = take_event_value(state, kLoopEventRoute);
  const std::uint64_t boundary_value =
      take_event_value(state, kLoopEventAtenBoundary);
  dispatch_semaphore_t route_ready = dispatch_semaphore_create(0);
  dispatch_semaphore_t boundary_ready = dispatch_semaphore_create(0);
  [state.events[kLoopEventRoute]
      notifyListener:state.listener
             atValue:route_value
               block:^(id<MTLSharedEvent>, std::uint64_t) {
                 dispatch_semaphore_signal(route_ready);
               }];
  [state.events[kLoopEventAtenBoundary]
      notifyListener:state.listener
             atValue:boundary_value
               block:^(id<MTLSharedEvent>, std::uint64_t) {
                 dispatch_semaphore_signal(boundary_ready);
               }];

  __block id<MTLCommandBuffer> route_command = nil;
  __block id<MTLCommandBuffer> pilot_command = nil;
  @autoreleasepool {
    // CB_a analogue: one command buffer on the loop queue ending in a
    // mailbox write and an event signal the host waits on.
    route_command = [state.queue commandBuffer];
    require(route_command != nil,
            "bespoke-loop route command buffer is unavailable");
    id<MTLComputeCommandEncoder> encoder =
        [route_command computeCommandEncoder];
    require(encoder != nil, "bespoke-loop route encoder is unavailable");
    [encoder setComputePipelineState:state.route_pipeline];
    [encoder setBuffer:tensor_buffer(expert_ids)
                offset:id_offset
               atIndex:0];
    [encoder setBuffer:tensor_buffer(weights)
                offset:weight_offset
               atIndex:1];
    [encoder setBuffer:state.route_mailbox offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(kTopK, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kTopK, 1, 1)];
    [encoder endEncoding];
    [route_command encodeSignalEvent:state.events[kLoopEventRoute]
                               value:route_value];
    [route_command commit];

    // CB_b analogue: a second command buffer committed to the same serial
    // queue without any host wait in between.
    pilot_command = [state.queue commandBuffer];
    require(pilot_command != nil,
            "bespoke-loop pilot command buffer is unavailable");
    id<MTLComputeCommandEncoder> pilot_encoder =
        [pilot_command computeCommandEncoder];
    require(pilot_encoder != nil,
            "bespoke-loop pilot encoder is unavailable");
    [pilot_encoder setComputePipelineState:state.pilot_pipeline];
    [pilot_encoder setBuffer:tensor_buffer(expert_ids)
                      offset:id_offset
                     atIndex:0];
    [pilot_encoder setBuffer:tensor_buffer(weights)
                      offset:weight_offset
                     atIndex:1];
    [pilot_encoder setBuffer:state.pilot_mailbox offset:0 atIndex:2];
    [pilot_encoder dispatchThreads:MTLSizeMake(kTopK, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kTopK, 1, 1)];
    [pilot_encoder endEncoding];
    [pilot_command encodeSignalEvent:state.events[kLoopEventAtenBoundary]
                               value:boundary_value];
    [pilot_command commit];
  }

  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long route_wait_result = dispatch_semaphore_wait(
      route_ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  require(route_wait_result == 0, "bespoke-loop route event timed out");
  const long boundary_wait_result = dispatch_semaphore_wait(
      boundary_ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  require(boundary_wait_result == 0, "bespoke-loop pilot event timed out");
  // NOTE: require() takes its message by value, so any argument building a
  // failure string from route_command.error/pilot_command.error must only
  // do so once the corresponding error is known non-nil -- unconditionally
  // sending -localizedDescription.UTF8String to a nil NSError is itself safe
  // (nil-messaging), but the resulting null `const char*` handed to
  // std::string's operator+ is not: it is immediately strlen'd.
  if (route_command.error != nil) {
    fail(std::string("bespoke-loop route command failed: ") +
         route_command.error.localizedDescription.UTF8String);
  }
  if (pilot_command.error != nil) {
    fail(std::string("bespoke-loop pilot command failed: ") +
         pilot_command.error.localizedDescription.UTF8String);
  }
  std::memcpy(&route_output, state.route_mailbox.contents,
              sizeof(route_output));
  std::memcpy(&pilot_output, state.pilot_mailbox.contents,
              sizeof(pilot_output));
  }  // end of state.mutex scope

  LoopMetalCanaryReport report{
      .top_k = kTopK,
      .route_ids_matched = 0,
      .route_weights_matched = 0,
      .pilot_ids_matched = 0,
      .pilot_weights_matched = 0,
      .command_buffers = 2,
      .events_signaled = 2,
      .weights_table_hits = 0,
      .weights_table_invalidations = 0,
      .id_offset_elements = static_cast<std::uint32_t>(kIdOffsetElements),
      .weight_offset_elements =
          static_cast<std::uint32_t>(kWeightOffsetElements),
      .reserved = 0,
      .max_absolute_error = 0.0,
      .relative_l2_error = 0.0,
  };
  double difference_squares = 0.0;
  double expected_squares = 0.0;
  for (std::uint32_t edge = 0; edge < kTopK; ++edge) {
    std::uint32_t expected_bits = 0;
    std::memcpy(&expected_bits, &weight_values[edge], sizeof(expected_bits));
    if (route_output.expert_ids[edge] == id_values[edge]) {
      ++report.route_ids_matched;
    }
    if (route_output.weight_bits[edge] == expected_bits) {
      ++report.route_weights_matched;
    }
    const std::int64_t pilot_id =
        id_values[edge] + static_cast<std::int64_t>(edge);
    const float pilot_weight = weight_values[edge] * 2.0F + 1.0F;
    std::uint32_t pilot_bits = 0;
    std::memcpy(&pilot_bits, &pilot_weight, sizeof(pilot_bits));
    if (pilot_output.expert_ids[edge] == pilot_id) {
      ++report.pilot_ids_matched;
    }
    if (pilot_output.weight_bits[edge] == pilot_bits) {
      ++report.pilot_weights_matched;
    }
    float actual_weight = 0.0F;
    std::memcpy(&actual_weight, &pilot_output.weight_bits[edge],
                sizeof(actual_weight));
    const double got = static_cast<double>(actual_weight);
    const double want = static_cast<double>(pilot_weight);
    const double difference = std::abs(got - want);
    report.max_absolute_error =
        std::max(report.max_absolute_error, difference);
    difference_squares += difference * difference;
    expected_squares += want * want;
  }
  report.relative_l2_error =
      expected_squares > 0.0
          ? std::sqrt(difference_squares / expected_squares)
          : std::sqrt(difference_squares);

  // Weights-table round trip below exercises the public table API (store, a
  // generation-matched hit, a generation-mismatched hit/invalidation, a
  // re-store, and a re-hit) now that the mutex scope above has released;
  // every one of those calls takes the same state mutex itself.
  // MLA takeover Step 1: drive the ported attention core through the LOOP's
  // own encoder at production head geometry. Step 0 proved the kernels are in
  // the metallib; this proves this file's binding/dispatch path drives them
  // correctly, which the standalone library's separate encoder does not cover.
  run_loop_mla_attention_canary(report);

  return finish_canary_with_table(report, weight_storage);
}

namespace {

constexpr std::uint32_t kRouterN = 896;
constexpr std::uint32_t kRouterK = 7168;

struct LoopGemvDimsHost {
  std::uint32_t N;
  std::uint32_t K;
};
struct LoopRouteDimsHost {
  std::uint32_t N;
};

/* Replays the kernel's selection and weight math bit-for-bit on the
 * kernel's OWN published fp32 sigmoid scores (scores_out readback): choice
 * = score + bias in fp32; 16 extractions under the documented tie rule
 * (score-descending, index-ascending); weights gathered from the untouched
 * scores and renormalized by the same sequential fp32 sum + 1e-20f.  Every
 * operation is on identical fp32 bit patterns, so ids must match exactly
 * and weights must match bitwise. */
struct RouterHostReference {
  std::array<std::int64_t, kTopK> ids = {};
  std::array<std::uint32_t, kTopK> weight_bits = {};
};

RouterHostReference reference_router_from_scores(const float* scores,
                                                 const float* bias) {
  RouterHostReference reference;
  std::array<float, kRouterN> choice = {};
  for (std::uint32_t i = 0; i < kRouterN; ++i) {
    choice[i] = scores[i] + bias[i];
  }
  std::array<float, kTopK> gathered = {};
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    float best_value = -std::numeric_limits<float>::infinity();
    std::uint32_t best_index = 0xFFFFFFFFU;
    for (std::uint32_t i = 0; i < kRouterN; ++i) {
      const float value = choice[i];
      if (value > best_value || (value == best_value && i < best_index)) {
        best_value = value;
        best_index = i;
      }
    }
    reference.ids[k] = static_cast<std::int64_t>(best_index);
    gathered[k] = scores[best_index];
    choice[best_index] = -std::numeric_limits<float>::infinity();
  }
  float sum = 0.0F;
  for (std::uint32_t k = 0; k < kTopK; ++k) sum += gathered[k];
  sum += 1e-20F;
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    const float weight = gathered[k] / sum;
    std::memcpy(&reference.weight_bits[k], &weight, sizeof(std::uint32_t));
  }
  return reference;
}

id<MTLBuffer> shared_float_buffer(id<MTLDevice> device,
                                  const std::size_t elements,
                                  const char* name) {
  id<MTLBuffer> buffer = [device
      newBufferWithLength:elements * sizeof(float)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeDefaultCache];
  require(buffer != nil,
          std::string("router-canary buffer allocation failed: ") + name);
  return buffer;
}



/* ---- K3_GPU_TIMELINE=1: per-class GPU interval timeline -----------------
 * Every live-path loop CB (and the captured ATen boundary root) registers a
 * completion handler recording its GPU execution interval. The drain, at the
 * existing 69-layer stats cadence, reports per-class busy plus the union
 * coverage of the window: span - union = GPU idle while the decode loop was
 * active — the number that separates a kernel war from a choreography war.
 * Diagnostic only; inert unless K3_GPU_TIMELINE=1. */
struct GpuTimelineInterval {
  const char* klass;
  double start;
  double end;
};
int gpu_timeline_level() {
  static const int level = [] {
    const char* value = std::getenv("K3_GPU_TIMELINE");
    if (value == nullptr) {
      return 0;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 0;
  }();
  return level;
}
bool gpu_timeline_enabled() { return gpu_timeline_level() > 0; }
std::mutex& gpu_timeline_mutex() {
  static std::mutex* mutex = new std::mutex;
  return *mutex;
}
std::vector<GpuTimelineInterval>& gpu_timeline_intervals() {
  static std::vector<GpuTimelineInterval>* intervals =
      new std::vector<GpuTimelineInterval>;
  return *intervals;
}
void gpu_timeline_note(id<MTLCommandBuffer> command, const char* klass) {
  if (!gpu_timeline_enabled() || command == nil) {
    return;
  }
  [command addCompletedHandler:^(id<MTLCommandBuffer> done) {
    const double start = done.GPUStartTime;
    const double end = done.GPUEndTime;
    if (!(end > start)) {
      return;
    }
    std::lock_guard<std::mutex> lock(gpu_timeline_mutex());
    gpu_timeline_intervals().push_back({klass, start, end});
  }];
}
void gpu_timeline_drain_and_print() {
  if (!gpu_timeline_enabled()) {
    return;
  }
  std::vector<GpuTimelineInterval> intervals;
  {
    std::lock_guard<std::mutex> lock(gpu_timeline_mutex());
    intervals.swap(gpu_timeline_intervals());
  }
  if (intervals.size() < 8) {
    return;
  }
  std::sort(intervals.begin(), intervals.end(),
            [](const GpuTimelineInterval& a, const GpuTimelineInterval& b) {
              return a.start < b.start;
            });
  const double span_start = intervals.front().start;
  double span_end = span_start;
  double union_busy = 0.0;
  double cover_end = span_start;
  std::map<std::string, std::pair<std::uint64_t, double>> classes;
  for (const auto& interval : intervals) {
    span_end = std::max(span_end, interval.end);
    if (interval.start > cover_end) {
      union_busy += interval.end - interval.start;
      cover_end = interval.end;
    } else if (interval.end > cover_end) {
      union_busy += interval.end - cover_end;
      cover_end = interval.end;
    }
    auto& slot = classes[interval.klass];
    slot.first += 1;
    slot.second += interval.end - interval.start;
  }
  const double span_ms = (span_end - span_start) * 1e3;
  const double busy_ms = union_busy * 1e3;
  std::fprintf(stderr,
               "[gpu-timeline] cbs=%zu span=%.1fms union_busy=%.1fms "
               "idle=%.1fms (%.0f%%)\n",
               intervals.size(), span_ms, busy_ms, span_ms - busy_ms,
               span_ms > 0.0 ? (span_ms - busy_ms) / span_ms * 100.0 : 0.0);
  for (const auto& [name, slot] : classes) {
    std::fprintf(stderr,
                 "[gpu-timeline]   %-11s n=%-4llu busy=%.2fms mean=%.3fms\n",
                 name.c_str(), static_cast<unsigned long long>(slot.first),
                 slot.second * 1e3, slot.second * 1e3 / slot.first);
  }
  // Level 2: dump the raw start-ordered sequence once — relative starts,
  // durations, and the gap since the previous CB ended. The gaps ARE the
  // choreography: kda->tail gap = expert claim/feed window, tail->next-kda
  // gap = boundary handoff + prepare.
  // Level N>=2 dumps the (N-1)-th flushed window, so level 4 shows a steady
  // decode window instead of the prompt; levels >= 3 dump up to 160 CBs.
  static bool sequence_dumped = false;
  static int windows_flushed = 0;
  ++windows_flushed;
  if (gpu_timeline_level() >= 2 && !sequence_dumped &&
      windows_flushed >= gpu_timeline_level() - 1) {
    sequence_dumped = true;
    double previous_end = span_start;
    const std::size_t limit = std::min<std::size_t>(
        intervals.size(), gpu_timeline_level() >= 3 ? 160 : 40);
    for (std::size_t index = 0; index < limit; ++index) {
      const auto& interval = intervals[index];
      std::fprintf(stderr,
                   "[gpu-seq] %5.1fms %-11s dur=%6.3fms gap=%6.3fms\n",
                   (interval.start - span_start) * 1e3, interval.klass,
                   (interval.end - interval.start) * 1e3,
                   (interval.start - previous_end) * 1e3);
      previous_end = std::max(previous_end, interval.end);
    }
  }
}

/* Encodes one router_top16 dispatch (plus, when `weights` is non-nil, a
 * preceding GEMV producing the logits) on the loop queue, waits on a pool
 * event through the listener, and reads the mailbox + scores back.  The
 * caller holds state.mutex. */
void run_router_dispatch(LoopEncoderState& state,
                         id<MTLBuffer> weights_i8, id<MTLBuffer> scales,
                         id<MTLBuffer> x, id<MTLBuffer> logits,
                         id<MTLBuffer> bias, id<MTLBuffer> scores,
                         RouteMailboxT1& mailbox_output,
                         std::uint32_t& command_buffers) {
  RouteMailboxT1 poison;
  std::memset(&poison, 0xA5, sizeof(poison));
  std::memcpy(state.route_mailbox.contents, &poison, sizeof(poison));

  const std::uint64_t event_value = take_event_value(state, kLoopEventRoute);
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  [state.events[kLoopEventRoute]
      notifyListener:state.listener
             atValue:event_value
               block:^(id<MTLSharedEvent>, std::uint64_t) {
                 dispatch_semaphore_signal(ready);
               }];

  __block id<MTLCommandBuffer> command = nil;
  @autoreleasepool {
    command = [state.queue commandBuffer];
    require(command != nil, "router-canary command buffer is unavailable");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "router-canary encoder is unavailable");
    if (weights_i8 != nil) {
      const LoopGemvDimsHost gemv_dims{kRouterN, kRouterK};
      [encoder setComputePipelineState:state.gemv_pipeline];
      [encoder setBuffer:weights_i8 offset:0 atIndex:0];
      [encoder setBuffer:x offset:0 atIndex:1];
      [encoder setBuffer:scales offset:0 atIndex:2];
      [encoder setBuffer:logits offset:0 atIndex:3];
      [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake((kRouterN + 15) / 16, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      // Same-encoder sequential dispatches on one serial queue: the router
      // select below reads `logits` only after the GEMV wrote it.  Metal
      // tracks the hazard for untracked-free shared buffers within one
      // encoder via memory barriers; be explicit for clarity.
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    const LoopRouteDimsHost route_dims{kRouterN};
    [encoder setComputePipelineState:state.router_top16_pipeline];
    [encoder setBuffer:logits offset:0 atIndex:0];
    [encoder setBuffer:bias offset:0 atIndex:1];
    [encoder setBuffer:scores offset:0 atIndex:2];
    [encoder setBuffer:state.route_mailbox offset:0 atIndex:3];
    [encoder setBytes:&route_dims length:sizeof(route_dims) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command encodeSignalEvent:state.events[kLoopEventRoute]
                         value:event_value];
    gpu_timeline_note(command, "route-side");
    [command commit];
  }
  ++command_buffers;

  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = dispatch_semaphore_wait(
      ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  require(wait_result == 0, "router-canary event timed out");
  if (command.error != nil) {
    fail(std::string("router-canary command failed: ") +
         command.error.localizedDescription.UTF8String);
  }
  std::memcpy(&mailbox_output, state.route_mailbox.contents,
              sizeof(mailbox_output));
}

/* Compares one completed dispatch against the bitwise host reference and
 * (when expected ids are pinned by the case design) the documented tie
 * rule's exact outcome. */
bool router_case_pair_exact(const RouteMailboxT1& mailbox,
                            const float* scores, const float* bias,
                            const std::int64_t* pinned_ids,
                            double& max_weight_abs_error) {
  const RouterHostReference reference =
      reference_router_from_scores(scores, bias);
  bool exact = true;
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    if (mailbox.expert_ids[k] != reference.ids[k]) exact = false;
    if (mailbox.weight_bits[k] != reference.weight_bits[k]) exact = false;
    if (pinned_ids != nullptr && mailbox.expert_ids[k] != pinned_ids[k]) {
      exact = false;
    }
    float got = 0.0F;
    float want = 0.0F;
    std::memcpy(&got, &mailbox.weight_bits[k], sizeof(got));
    std::memcpy(&want, &reference.weight_bits[k], sizeof(want));
    max_weight_abs_error =
        std::max(max_weight_abs_error,
                 std::abs(static_cast<double>(got) -
                          static_cast<double>(want)));
  }
  return exact;
}

}  // namespace

bool loop_router_canary_passes(const LoopRouterCanaryReport& report) {
  return report.cases_run != 0 &&
         report.cases_pair_exact == report.cases_run &&
         report.gemv_rows_compared == kRouterN &&
         std::isfinite(report.gemv_relative_l2_error) &&
         report.gemv_relative_l2_error <= kLoopRouterCanaryMaxGemvRelL2 &&
         report.max_weight_absolute_error == 0.0 && report.reserved == 0;
}

LoopRouterCanaryReport loop_router_canary_v1() {
  LoopRouterCanaryReport report;
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  ensure_resources(state, device);

  id<MTLBuffer> logits = shared_float_buffer(device, kRouterN, "logits");
  id<MTLBuffer> bias = shared_float_buffer(device, kRouterN, "bias");
  id<MTLBuffer> scores = shared_float_buffer(device, kRouterN, "scores");
  float* logits_host = static_cast<float*>(logits.contents);
  float* bias_host = static_cast<float*>(bias.contents);
  const float* scores_host = static_cast<const float*>(scores.contents);

  std::uint32_t lcg = 0x1234'5678U;
  const auto next_value = [&lcg]() {
    lcg = lcg * 1664525U + 1013904223U;
    return static_cast<float>((lcg >> 8) & 0xFFFF) / 16384.0F - 2.0F;
  };

  RouteMailboxT1 mailbox;
  const auto run_controlled_case =
      [&](const std::int64_t* pinned_ids) {
        run_router_dispatch(state, nil, nil, nil, logits, bias, scores,
                            mailbox, report.command_buffers);
        ++report.cases_run;
        if (router_case_pair_exact(mailbox, scores_host, bias_host,
                                   pinned_ids,
                                   report.max_weight_absolute_error)) {
          ++report.cases_pair_exact;
        }
      };

  // Case 1: random logits, zero bias.
  for (std::uint32_t i = 0; i < kRouterN; ++i) {
    logits_host[i] = next_value();
    bias_host[i] = 0.0F;
  }
  run_controlled_case(nullptr);

  // Case 2: engineered 20-way exact tie at [200,220): everything else far
  // below.  The documented rule must select the LOWEST 16 indices of the
  // tie block, ascending.
  {
    for (std::uint32_t i = 0; i < kRouterN; ++i) logits_host[i] = -20.0F;
    for (std::uint32_t i = 200; i < 220; ++i) logits_host[i] = 0.0F;
    std::array<std::int64_t, kTopK> pinned = {};
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      pinned[k] = static_cast<std::int64_t>(200 + k);
    }
    run_controlled_case(pinned.data());
  }

  // Case 3: full 896-way tie -> ids 0..15 ascending.
  {
    for (std::uint32_t i = 0; i < kRouterN; ++i) logits_host[i] = 0.25F;
    std::array<std::int64_t, kTopK> pinned = {};
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      pinned[k] = static_cast<std::int64_t>(k);
    }
    run_controlled_case(pinned.data());
  }

  // Case 4: very negative logits — renormalization stability (+1e-20).
  for (std::uint32_t i = 0; i < kRouterN; ++i) logits_host[i] = -80.0F;
  run_controlled_case(nullptr);

  // Case 5: bias affects selection only — a large bias boost must select
  // index 500, but its weight must come from the un-boosted sigmoid score.
  for (std::uint32_t i = 0; i < kRouterN; ++i) {
    logits_host[i] = next_value();
    bias_host[i] = (i == 500) ? 10.0F : 0.0F;
  }
  run_router_dispatch(state, nil, nil, nil, logits, bias, scores, mailbox,
                      report.command_buffers);
  ++report.cases_run;
  {
    bool found_500 = false;
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      if (mailbox.expert_ids[k] == 500) found_500 = true;
    }
    if (found_500 &&
        router_case_pair_exact(mailbox, scores_host, bias_host, nullptr,
                               report.max_weight_absolute_error)) {
      ++report.cases_pair_exact;
    }
  }

  // Case 6: the full chain at the production shape — int8 GEMV into the
  // router select in one command buffer, GEMV logits held to the validated
  // fp64 relL2 bound, selection re-checked pair-exact on the GPU's own
  // outputs.
  {
    id<MTLBuffer> weights_i8 = [device
        newBufferWithLength:static_cast<NSUInteger>(kRouterN) * kRouterK
                    options:MTLResourceStorageModeShared |
                            MTLResourceCPUCacheModeDefaultCache];
    require(weights_i8 != nil, "router-canary int8 weight buffer failed");
    id<MTLBuffer> scales = shared_float_buffer(device, kRouterN, "scales");
    id<MTLBuffer> x = shared_float_buffer(device, kRouterK, "x");
    auto* w_host = static_cast<std::int8_t*>(weights_i8.contents);
    auto* scales_host = static_cast<float*>(scales.contents);
    auto* x_host = static_cast<float*>(x.contents);
    for (std::size_t i = 0; i < static_cast<std::size_t>(kRouterN) * kRouterK;
         ++i) {
      lcg = lcg * 1664525U + 1013904223U;
      w_host[i] = static_cast<std::int8_t>(
          static_cast<std::int32_t>((lcg >> 8) & 0xFF) - 128);
    }
    for (std::uint32_t n = 0; n < kRouterN; ++n) {
      lcg = lcg * 1664525U + 1013904223U;
      scales_host[n] =
          1.0e-3F + static_cast<float>((lcg >> 8) & 0xFFF) / 1.0e6F;
    }
    for (std::uint32_t k = 0; k < kRouterK; ++k) x_host[k] = next_value();
    for (std::uint32_t i = 0; i < kRouterN; ++i) bias_host[i] = 0.0F;

    run_router_dispatch(state, weights_i8, scales, x, logits, bias, scores,
                        mailbox, report.command_buffers);
    ++report.cases_run;

    // fp64 GEMV reference against the GPU's fp32 logits.
    double difference_squares = 0.0;
    double expected_squares = 0.0;
    for (std::uint32_t n = 0; n < kRouterN; ++n) {
      double accumulator = 0.0;
      const std::int8_t* row =
          w_host + static_cast<std::size_t>(n) * kRouterK;
      for (std::uint32_t k = 0; k < kRouterK; ++k) {
        accumulator += static_cast<double>(row[k]) *
                       static_cast<double>(x_host[k]);
      }
      const double expected =
          accumulator * static_cast<double>(scales_host[n]);
      const double got = static_cast<double>(logits_host[n]);
      const double difference = got - expected;
      difference_squares += difference * difference;
      expected_squares += expected * expected;
      ++report.gemv_rows_compared;
    }
    report.gemv_relative_l2_error =
        expected_squares > 0.0
            ? std::sqrt(difference_squares / expected_squares)
            : std::sqrt(difference_squares);

    if (router_case_pair_exact(mailbox, scores_host, bias_host, nullptr,
                               report.max_weight_absolute_error)) {
      ++report.cases_pair_exact;
    }
  }

  return report;
}

int route_sidequeue_mode() {
  static const int mode = [] {
    const char* value = std::getenv("K3_ROUTE_SIDEQUEUE");
    if (value == nullptr) return 0;
    if (std::strcmp(value, "shadow") == 0) return 1;
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 2;
    }
    return 0;
  }();
  return mode;
}

namespace {

constexpr std::size_t kLoopEventShadowBoundary = 2;
constexpr std::size_t kLoopEventShadowDone = 3;

bool shadow_tensor_ok(const at::Tensor& tensor, const at::ScalarType type,
                      const std::int64_t elements) {
  return tensor.defined() && tensor.device().is_mps() &&
         tensor.scalar_type() == type && tensor.is_contiguous() &&
         !tensor.requires_grad() && tensor.numel() == elements;
}

/* Drops an unpaired pending shadow (compare was skipped by an error path):
 * wait out its command buffer so the scratch and mailbox are quiescent
 * before reuse.  Caller holds state.mutex. */
void shadow_drain_locked(LoopEncoderState& state) {
  if (!state.shadow_active) {
    return;
  }
  if (state.shadow_ready != nil) {
    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    static_cast<void>(dispatch_semaphore_wait(
        state.shadow_ready,
        dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds)));
  }
  state.shadow_active = false;
  state.shadow_ready = nil;
  state.shadow_command = nil;
  ++state.shadow_stats.skipped;
}

}  // namespace

bool loop_router_shadow_begin(const std::uint32_t layer_index,
                              const at::Tensor& hidden,
                              const at::Tensor& quantized,
                              const at::Tensor& row_scales,
                              const at::Tensor& bias) {
  static_cast<void>(layer_index);
  constexpr std::int64_t kHidden = 7168;
  constexpr std::int64_t kExperts = 896;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[route-shadow] first skip: %s\n", reason);
    }
  };
  if (!shadow_tensor_ok(hidden, at::kFloat, kHidden) ||
      !shadow_tensor_ok(quantized, at::kChar, kExperts * kHidden) ||
      !shadow_tensor_ok(row_scales, at::kFloat, kExperts) ||
      !shadow_tensor_ok(bias, at::kFloat, kExperts)) {
    report_skip_once("input tensors do not qualify");
    LoopEncoderState& state = loop_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.shadow_stats.skipped;
    return false;
  }
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return false;
  }

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    ++state.shadow_stats.skipped;
    return false;
  }
  try {
    ensure_resources(state, device);
    shadow_drain_locked(state);

    const NSUInteger hidden_offset = checked_byte_offset(
        hidden, kHidden * sizeof(float), "shadow router hidden");
    const NSUInteger quantized_offset = checked_byte_offset(
        quantized, static_cast<std::size_t>(kExperts) * kHidden,
        "shadow router weights");
    const NSUInteger scales_offset = checked_byte_offset(
        row_scales, kExperts * sizeof(float), "shadow router scales");
    const NSUInteger bias_offset = checked_byte_offset(
        bias, kExperts * sizeof(float), "shadow router bias");

    RouteMailboxT1 poison;
    std::memset(&poison, 0xA5, sizeof(poison));
    std::memcpy(state.pilot_mailbox.contents, &poison, sizeof(poison));

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    // ATen-side half of the fence: signal on the current root command
    // buffer, then release it with the same non-blocking commit the route
    // path uses, so the side queue's wait can satisfy without a host stall.
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "shadow router MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "shadow router root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "shadow router boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "shadow router command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "shadow router encoder is unavailable");
      const LoopGemvDimsHost gemv_dims{
          static_cast<std::uint32_t>(kExperts),
          static_cast<std::uint32_t>(kHidden)};
      [encoder setComputePipelineState:state.gemv_pipeline];
      [encoder setBuffer:tensor_buffer(quantized)
                  offset:quantized_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(hidden)
                  offset:hidden_offset
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(row_scales)
                  offset:scales_offset
                 atIndex:2];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:3];
      [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake((kExperts + 15) / 16, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const LoopRouteDimsHost route_dims{
          static_cast<std::uint32_t>(kExperts)};
      [encoder setComputePipelineState:state.router_top16_pipeline];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(bias) offset:bias_offset atIndex:1];
      [encoder setBuffer:state.shadow_scores offset:0 atIndex:2];
      [encoder setBuffer:state.pilot_mailbox offset:0 atIndex:3];
      [encoder setBytes:&route_dims length:sizeof(route_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    state.shadow_active = true;
    state.route_pending_layer = layer_index;
    state.shadow_ready_consumed = false;
    state.shadow_ready = ready;
    state.shadow_command = command;
    return true;
  } catch (const std::exception& error) {
    // Shadow is diagnostic-only: any failure counts a skip and leaves the
    // stock path untouched.
    report_skip_once(error.what());
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  } catch (...) {
    report_skip_once("unknown exception");
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  }
}

bool loop_router_shadow_begin_bf16(const std::uint32_t layer_index,
                                   const at::Tensor& hidden,
                                   const at::Tensor& weight_bits,
                                   const std::size_t element_offset,
                                   const at::Tensor& bias) {
  static_cast<void>(layer_index);
  constexpr std::int64_t kHidden = 7168;
  constexpr std::int64_t kExperts = 896;
  constexpr std::int64_t kWeightElements = kExperts * kHidden;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[route-shadow] first bf16 skip: %s\n", reason);
    }
  };
  const bool bits_ok = weight_bits.defined() &&
      weight_bits.device().is_mps() &&
      (weight_bits.scalar_type() == at::kBFloat16 ||
       weight_bits.scalar_type() == at::kShort) &&
      weight_bits.is_contiguous() && !weight_bits.requires_grad() &&
      element_offset <= std::numeric_limits<std::size_t>::max() -
                            static_cast<std::size_t>(kWeightElements) &&
      weight_bits.numel() >= 0 &&
      static_cast<std::size_t>(weight_bits.numel()) >=
          element_offset + static_cast<std::size_t>(kWeightElements);
  if (!shadow_tensor_ok(hidden, at::kFloat, kHidden) || !bits_ok ||
      !shadow_tensor_ok(bias, at::kFloat, kExperts)) {
    report_skip_once("bf16 input tensors do not qualify");
    LoopEncoderState& state = loop_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.shadow_stats.skipped;
    return false;
  }
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return false;
  }

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    ++state.shadow_stats.skipped;
    return false;
  }
  try {
    ensure_resources(state, device);
    shadow_drain_locked(state);

    const NSUInteger hidden_offset = checked_byte_offset(
        hidden, kHidden * sizeof(float), "shadow bf16 router hidden");
    // Validate the whole prefix span [storage_offset, +element_offset+N*K)
    // against the slab's MTLBuffer, then step to the router's origin.
    const NSUInteger slab_offset = checked_byte_offset(
        weight_bits,
        (element_offset + static_cast<std::size_t>(kWeightElements)) *
            sizeof(std::uint16_t),
        "shadow bf16 router weights");
    const NSUInteger weight_offset =
        slab_offset +
        static_cast<NSUInteger>(element_offset * sizeof(std::uint16_t));
    const NSUInteger bias_offset = checked_byte_offset(
        bias, kExperts * sizeof(float), "shadow bf16 router bias");

    RouteMailboxT1 poison;
    std::memset(&poison, 0xA5, sizeof(poison));
    std::memcpy(state.pilot_mailbox.contents, &poison, sizeof(poison));

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "shadow bf16 router MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "shadow bf16 router root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "shadow bf16 router boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil,
              "shadow bf16 router command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "shadow bf16 router encoder is unavailable");
      struct LoopBf16GemvDimsHost {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t reserved0;
        std::uint32_t reserved1;
      };
      const LoopBf16GemvDimsHost gemv_dims{
          static_cast<std::uint32_t>(kExperts),
          static_cast<std::uint32_t>(kHidden), 0, 0};
      [encoder setComputePipelineState:state.gemv_bf16_pipeline];
      [encoder setBuffer:tensor_buffer(weight_bits)
                  offset:weight_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(hidden)
                  offset:hidden_offset
                 atIndex:1];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:2];
      [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake((kExperts + 15) / 16, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const LoopRouteDimsHost route_dims{
          static_cast<std::uint32_t>(kExperts)};
      [encoder setComputePipelineState:state.router_top16_pipeline];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(bias) offset:bias_offset atIndex:1];
      [encoder setBuffer:state.shadow_scores offset:0 atIndex:2];
      [encoder setBuffer:state.pilot_mailbox offset:0 atIndex:3];
      [encoder setBytes:&route_dims length:sizeof(route_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    state.shadow_active = true;
    state.route_pending_layer = layer_index;
    state.shadow_ready_consumed = false;
    state.shadow_ready = ready;
    state.shadow_command = command;
    return true;
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  } catch (...) {
    report_skip_once("unknown exception");
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  }
}

bool loop_router_shadow_begin_f32(const std::uint32_t layer_index,
                                  const at::Tensor& hidden,
                                  const at::Tensor& dense_f32,
                                  const at::Tensor& bias) {
  static_cast<void>(layer_index);
  constexpr std::int64_t kHidden = 7168;
  constexpr std::int64_t kExperts = 896;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[route-shadow] first f32 skip: %s\n", reason);
    }
  };
  if (!shadow_tensor_ok(hidden, at::kFloat, kHidden) ||
      !shadow_tensor_ok(dense_f32, at::kFloat, kExperts * kHidden) ||
      !shadow_tensor_ok(bias, at::kFloat, kExperts)) {
    report_skip_once("f32 input tensors do not qualify");
    LoopEncoderState& state = loop_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.shadow_stats.skipped;
    return false;
  }
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return false;
  }

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    ++state.shadow_stats.skipped;
    return false;
  }
  try {
    ensure_resources(state, device);
    shadow_drain_locked(state);

    const NSUInteger hidden_offset = checked_byte_offset(
        hidden, kHidden * sizeof(float), "shadow f32 router hidden");
    const NSUInteger weight_offset = checked_byte_offset(
        dense_f32,
        static_cast<std::size_t>(kExperts) * kHidden * sizeof(float),
        "shadow f32 router weights");
    const NSUInteger bias_offset = checked_byte_offset(
        bias, kExperts * sizeof(float), "shadow f32 router bias");

    RouteMailboxT1 poison;
    std::memset(&poison, 0xA5, sizeof(poison));
    std::memcpy(state.pilot_mailbox.contents, &poison, sizeof(poison));

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "shadow f32 router MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "shadow f32 router root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "shadow f32 router boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil,
              "shadow f32 router command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "shadow f32 router encoder is unavailable");
      struct LoopBf16GemvDimsHost {
        std::uint32_t rows;
        std::uint32_t columns;
        std::uint32_t reserved0;
        std::uint32_t reserved1;
      };
      const LoopBf16GemvDimsHost gemv_dims{
          static_cast<std::uint32_t>(kExperts),
          static_cast<std::uint32_t>(kHidden), 0, 0};
      [encoder setComputePipelineState:state.gemv_f32_pipeline];
      [encoder setBuffer:tensor_buffer(dense_f32)
                  offset:weight_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(hidden)
                  offset:hidden_offset
                 atIndex:1];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:2];
      [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake((kExperts + 15) / 16, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const LoopRouteDimsHost route_dims{
          static_cast<std::uint32_t>(kExperts)};
      [encoder setComputePipelineState:state.router_top16_pipeline];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(bias) offset:bias_offset atIndex:1];
      [encoder setBuffer:state.shadow_scores offset:0 atIndex:2];
      [encoder setBuffer:state.pilot_mailbox offset:0 atIndex:3];
      [encoder setBytes:&route_dims length:sizeof(route_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    state.shadow_active = true;
    state.route_pending_layer = layer_index;
    state.shadow_ready_consumed = false;
    state.shadow_ready = ready;
    state.shadow_command = command;
    return true;
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  } catch (...) {
    report_skip_once("unknown exception");
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return false;
  }
}

void loop_router_shadow_compare(const std::uint32_t layer_index,
                                const std::uint16_t* expert_ids,
                                const std::uint32_t* weight_bits) {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.shadow_active) {
    ++state.shadow_stats.skipped;
    return;
  }
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = dispatch_semaphore_wait(
      state.shadow_ready,
      dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  const bool command_failed =
      wait_result != 0 ||
      (state.shadow_command != nil && state.shadow_command.error != nil);
  if (command_failed) {
    static std::atomic<bool> failure_reported{false};
    if (!failure_reported.exchange(true)) {
      const char* description =
          (state.shadow_command != nil && state.shadow_command.error != nil)
              ? state.shadow_command.error.localizedDescription.UTF8String
              : "event wait timed out";
      std::fprintf(stderr, "[route-shadow] first failure: %s\n",
                   description == nullptr ? "unknown" : description);
    }
    state.shadow_active = false;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    ++state.shadow_stats.skipped;
    return;
  }
  state.shadow_active = false;
  state.shadow_ready = nil;
  state.shadow_command = nil;

  RouteMailboxT1 side;
  std::memcpy(&side, state.pilot_mailbox.contents, sizeof(side));

  std::array<std::uint16_t, kTopK> stock_sorted = {};
  std::array<std::uint16_t, kTopK> side_sorted = {};
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    stock_sorted[k] = expert_ids[k];
    side_sorted[k] = static_cast<std::uint16_t>(side.expert_ids[k]);
  }
  std::sort(stock_sorted.begin(), stock_sorted.end());
  std::sort(side_sorted.begin(), side_sorted.end());
  const bool ids_match = stock_sorted == side_sorted;

  double max_weight_abs = 0.0;
  if (ids_match) {
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      const std::uint16_t id = expert_ids[k];
      for (std::uint32_t j = 0; j < kTopK; ++j) {
        if (static_cast<std::uint16_t>(side.expert_ids[j]) != id) {
          continue;
        }
        float stock_weight = 0.0F;
        float side_weight = 0.0F;
        std::memcpy(&stock_weight, &weight_bits[k], sizeof(stock_weight));
        std::memcpy(&side_weight, &side.weight_bits[j], sizeof(side_weight));
        max_weight_abs = std::max(
            max_weight_abs, std::abs(static_cast<double>(stock_weight) -
                                     static_cast<double>(side_weight)));
        break;
      }
    }
  }

  ++state.shadow_stats.compared;
  state.shadow_stats.max_weight_abs_error =
      std::max(state.shadow_stats.max_weight_abs_error, max_weight_abs);
  if (!ids_match) {
    ++state.shadow_stats.id_set_mismatches;
    std::fprintf(stderr,
                 "[route-shadow] ID-SET MISMATCH layer=%u compared=%llu\n",
                 layer_index,
                 static_cast<unsigned long long>(state.shadow_stats.compared));
  }
  if (state.shadow_stats.compared == 1 ||
      state.shadow_stats.compared % 920 == 0) {
    std::fprintf(
        stderr,
        "[route-shadow] compared=%llu id_set_mismatches=%llu "
        "max_weight_abs=%.3e skipped=%llu\n",
        static_cast<unsigned long long>(state.shadow_stats.compared),
        static_cast<unsigned long long>(state.shadow_stats.id_set_mismatches),
        state.shadow_stats.max_weight_abs_error,
        static_cast<unsigned long long>(state.shadow_stats.skipped));
  }
}

LoopRouterShadowStats loop_router_shadow_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.shadow_stats;
}

bool loop_router_collect(const std::uint32_t layer_index,
                         std::uint16_t* expert_ids_out,
                         std::uint32_t* weight_bits_out) {
  static const bool probing = [] {
    const char* value = std::getenv("K3_ROUTE_SYNC_PROBE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.shadow_active ||
      state.route_pending_layer != layer_index) {
    ++state.shadow_stats.skipped;
    return false;
  }
  const auto wait_started = std::chrono::steady_clock::now();
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  long wait_result = 0;
  if (!state.shadow_ready_consumed) {
    wait_result = prep_timed_semaphore_wait(
        state.shadow_ready,
        dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
        state.shadow_command);
    state.shadow_ready_consumed = true;
  }
  const double wait_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - wait_started)
                             .count();
  const bool command_failed =
      wait_result != 0 ||
      (state.shadow_command != nil && state.shadow_command.error != nil);
  state.shadow_active = false;
  state.route_pending_layer = 0xFFFFFFFFu;
  state.shadow_ready = nil;
  state.shadow_command = nil;
  if (command_failed) {
    static std::atomic<bool> failure_reported{false};
    if (!failure_reported.exchange(true)) {
      std::fprintf(stderr,
                   "[route-sidequeue] first collect failure (%s)\n",
                   wait_result != 0 ? "event wait timed out"
                                    : "command buffer error");
    }
    ++state.shadow_stats.skipped;
    return false;
  }

  RouteMailboxT1 side;
  std::memcpy(&side, state.pilot_mailbox.contents, sizeof(side));
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    const std::int64_t id = side.expert_ids[k];
    if (id < 0 || id >= 896) {
      static std::atomic<bool> range_reported{false};
      if (!range_reported.exchange(true)) {
        std::fprintf(stderr,
                     "[route-sidequeue] first out-of-range id %lld\n",
                     static_cast<long long>(id));
      }
      ++state.shadow_stats.skipped;
      return false;
    }
    expert_ids_out[k] = static_cast<std::uint16_t>(id);
    weight_bits_out[k] = side.weight_bits[k];
  }
  ++state.shadow_stats.compared;

  if (probing) {
    static double wait_accumulator_ms = 0.0;
    static std::uint64_t wait_calls = 0;
    wait_accumulator_ms += wait_ms;
    ++wait_calls;
    if (wait_calls % 92 == 0) {
      std::fprintf(stderr,
                   "[route-probe-sidequeue] calls=%llu wait_mean=%.3fms\n",
                   static_cast<unsigned long long>(wait_calls),
                   wait_accumulator_ms / static_cast<double>(wait_calls));
      wait_accumulator_ms = 0.0;
      wait_calls = 0;
    }
  }
  return true;
}

namespace {

constexpr std::uint32_t kKdaHeads = 96;
constexpr std::uint32_t kKdaWidth = 128;
constexpr std::uint32_t kKdaProj = kKdaHeads * kKdaWidth;  // 12288
constexpr std::uint32_t kKdaBundle = 4 * kKdaProj + 128 + kKdaHeads;  // 49376
constexpr std::uint32_t kKdaConvChannels = 3 * kKdaProj;
constexpr std::uint32_t kKdaState = kKdaHeads * kKdaWidth * kKdaWidth;

double kda_sigmoid_ref(const double v) { return 1.0 / (1.0 + std::exp(-v)); }

/* fp64 reference of the fused core, structured exactly like the kernel. */
void kda_core_reference(const float* proj, const float* fb,
                        const float* convw, const float* convin,
                        const float* a_log, const float* dt_bias,
                        const float* o_norm, const float* s_in,
                        std::vector<double>& conv_out,
                        std::vector<double>& s_out,
                        std::vector<double>& out) {
  conv_out.assign(static_cast<std::size_t>(kKdaConvChannels) * 4, 0.0);
  s_out.assign(kKdaState, 0.0);
  out.assign(kKdaProj, 0.0);
  std::vector<double> conv_res(static_cast<std::size_t>(3) * kKdaProj, 0.0);
  for (std::uint32_t m = 0; m < 3; ++m) {
    for (std::uint32_t c = 0; c < kKdaProj; ++c) {
      const std::size_t ch = static_cast<std::size_t>(m) * kKdaProj + c;
      const float* st = convin + ch * 4;
      const float* wk = convw + ch * 4;
      const double s0 = st[1], s1 = st[2], s2 = st[3];
      const double s3 = proj[m * kKdaProj + c];
      const double acc =
          s0 * wk[0] + s1 * wk[1] + s2 * wk[2] + s3 * wk[3];
      conv_out[ch * 4 + 0] = s0;
      conv_out[ch * 4 + 1] = s1;
      conv_out[ch * 4 + 2] = s2;
      conv_out[ch * 4 + 3] = s3;
      conv_res[ch] = acc * kda_sigmoid_ref(acc);
    }
  }
  for (std::uint32_t h = 0; h < kKdaHeads; ++h) {
    double qss = 0.0;
    double kss = 0.0;
    for (std::uint32_t t = 0; t < kKdaWidth; ++t) {
      const std::uint32_t c = h * kKdaWidth + t;
      qss += conv_res[c] * conv_res[c];
      kss += conv_res[kKdaProj + c] * conv_res[kKdaProj + c];
    }
    const double q_norm = std::max(std::sqrt(qss), 1e-12);
    const double k_norm = std::max(std::sqrt(kss), 1e-12);
    const double beta =
        kda_sigmoid_ref(proj[4 * kKdaProj + 128 + h]);
    std::array<double, 128> qv{};
    std::array<double, 128> kv{};
    std::array<double, 128> dv{};
    for (std::uint32_t t = 0; t < kKdaWidth; ++t) {
      const std::uint32_t c = h * kKdaWidth + t;
      qv[t] = conv_res[c] / q_norm * 0.08838834764831845;
      kv[t] = conv_res[kKdaProj + c] / k_norm;
      const double a = std::exp(static_cast<double>(a_log[t]));
      const double rg = static_cast<double>(fb[c]) + dt_bias[c];
      dv[t] = std::exp(-5.0 * kda_sigmoid_ref(a * rg));
    }
    const float* sh = s_in + static_cast<std::size_t>(h) * kKdaWidth * kKdaWidth;
    double* sho = s_out.data() +
                  static_cast<std::size_t>(h) * kKdaWidth * kKdaWidth;
    for (std::uint32_t t = 0; t < kKdaWidth; ++t) {
      const std::uint32_t c = h * kKdaWidth + t;
      double acc = 0.0;
      for (std::uint32_t k = 0; k < kKdaWidth; ++k) {
        acc += kv[k] * dv[k] * sh[k * kKdaWidth + t];
      }
      const double delta = conv_res[2 * kKdaProj + c] - acc;
      const double bd = beta * delta;
      double ov = 0.0;
      for (std::uint32_t k = 0; k < kKdaWidth; ++k) {
        const double sn = sh[k * kKdaWidth + t] * dv[k] + kv[k] * bd;
        sho[k * kKdaWidth + t] = sn;
        ov += qv[k] * sn;
      }
      out[c] = ov;
    }
    double oss = 0.0;
    for (std::uint32_t t = 0; t < kKdaWidth; ++t) {
      oss += out[h * kKdaWidth + t] * out[h * kKdaWidth + t];
    }
    const double scale = 1.0 / std::sqrt(oss / 128.0 + 1e-5);
    for (std::uint32_t t = 0; t < kKdaWidth; ++t) {
      const std::uint32_t c = h * kKdaWidth + t;
      out[c] *= scale;
      out[c] *= o_norm[t];
      out[c] *= kda_sigmoid_ref(proj[3 * kKdaProj + c]);
    }
  }
}

double kda_rel_l2(const float* got, const double* want,
                  const std::size_t count) {
  double difference_squares = 0.0;
  double expected_squares = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double difference = static_cast<double>(got[i]) - want[i];
    difference_squares += difference * difference;
    expected_squares += want[i] * want[i];
  }
  return expected_squares > 0.0
             ? std::sqrt(difference_squares / expected_squares)
             : std::sqrt(difference_squares);
}

}  // namespace

bool loop_kda_canary_passes(const LoopKdaCanaryReport& report) {
  return report.cases_run != 0 && report.cases_passed == report.cases_run &&
         std::isfinite(report.max_out_relative_l2) &&
         std::isfinite(report.max_state_relative_l2) &&
         std::isfinite(report.max_conv_relative_l2) &&
         report.max_out_relative_l2 <= kLoopKdaCanaryMaxRelL2 &&
         report.max_state_relative_l2 <= kLoopKdaCanaryMaxRelL2 &&
         report.max_conv_relative_l2 <= kLoopKdaCanaryMaxRelL2 &&
         report.reserved == 0;
}

LoopKdaCanaryReport loop_kda_canary_v1() {
  LoopKdaCanaryReport report;
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  ensure_resources(state, device);

  const auto make_buffer = [&](const std::size_t elements,
                               const char* name) {
    return shared_float_buffer(device, elements, name);
  };
  id<MTLBuffer> proj = make_buffer(kKdaBundle, "kda proj");
  id<MTLBuffer> fb = make_buffer(kKdaProj, "kda fb");
  id<MTLBuffer> convw = make_buffer(
      static_cast<std::size_t>(kKdaConvChannels) * 4, "kda convw");
  id<MTLBuffer> convin = make_buffer(
      static_cast<std::size_t>(kKdaConvChannels) * 4, "kda convin");
  id<MTLBuffer> convout = make_buffer(
      static_cast<std::size_t>(kKdaConvChannels) * 4, "kda convout");
  id<MTLBuffer> a_log = make_buffer(kKdaWidth, "kda a_log");
  id<MTLBuffer> dt_bias = make_buffer(kKdaProj, "kda dt_bias");
  id<MTLBuffer> o_norm = make_buffer(kKdaWidth, "kda o_norm");
  id<MTLBuffer> s_in = make_buffer(kKdaState, "kda S_in");
  id<MTLBuffer> s_out = make_buffer(kKdaState, "kda S_out");
  id<MTLBuffer> out = make_buffer(kKdaProj, "kda out");

  std::uint32_t lcg = 0x0DDB'A11FU;
  const auto next_value = [&lcg]() {
    lcg = lcg * 1664525U + 1013904223U;
    return static_cast<float>((lcg >> 8) & 0xFFFF) / 32768.0F - 1.0F;
  };
  const auto fill = [&](id<MTLBuffer> buffer, const std::size_t elements,
                        const float scale, const float offset) {
    float* host = static_cast<float*>(buffer.contents);
    for (std::size_t i = 0; i < elements; ++i) {
      host[i] = next_value() * scale + offset;
    }
  };

  const auto run_case = [&](const bool zero_state,
                            const bool saturated_decay) {
    fill(proj, kKdaBundle, 1.0F, 0.0F);
    fill(fb, kKdaProj, saturated_decay ? 0.0F : 1.0F,
         saturated_decay ? 40.0F : 0.0F);
    fill(convw, static_cast<std::size_t>(kKdaConvChannels) * 4, 1.0F, 0.0F);
    fill(convin, static_cast<std::size_t>(kKdaConvChannels) * 4, 1.0F, 0.0F);
    fill(a_log, kKdaWidth, 0.5F, 0.0F);
    fill(dt_bias, kKdaProj, saturated_decay ? 0.0F : 1.0F,
         saturated_decay ? 40.0F : 0.0F);
    fill(o_norm, kKdaWidth, 0.5F, 1.0F);
    if (zero_state) {
      std::memset(s_in.contents, 0,
                  static_cast<std::size_t>(kKdaState) * sizeof(float));
    } else {
      fill(s_in, kKdaState, 0.5F, 0.0F);
    }
    std::memset(s_out.contents, 0xA5,
                static_cast<std::size_t>(kKdaState) * sizeof(float));
    std::memset(out.contents, 0xA5, kKdaProj * sizeof(float));

    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);
    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];
    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "kda-canary command buffer is unavailable");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "kda-canary encoder is unavailable");
      [encoder setComputePipelineState:state.kda_core_pipeline];
      const NSUInteger conv_channel_bytes =
          static_cast<NSUInteger>(kKdaProj) * 4 * sizeof(float);
      [encoder setBuffer:proj offset:0 atIndex:0];
      [encoder setBuffer:fb offset:0 atIndex:1];
      [encoder setBuffer:convw offset:0 atIndex:2];
      [encoder setBuffer:convw offset:conv_channel_bytes atIndex:3];
      [encoder setBuffer:convw offset:2 * conv_channel_bytes atIndex:4];
      [encoder setBuffer:convin offset:0 atIndex:5];
      [encoder setBuffer:convin offset:conv_channel_bytes atIndex:6];
      [encoder setBuffer:convin offset:2 * conv_channel_bytes atIndex:7];
      [encoder setBuffer:convout offset:0 atIndex:8];
      [encoder setBuffer:convout offset:conv_channel_bytes atIndex:9];
      [encoder setBuffer:convout offset:2 * conv_channel_bytes atIndex:10];
      [encoder setBuffer:a_log offset:0 atIndex:11];
      [encoder setBuffer:dt_bias offset:0 atIndex:12];
      [encoder setBuffer:o_norm offset:0 atIndex:13];
      [encoder setBuffer:s_in offset:0 atIndex:14];
      [encoder setBuffer:s_out offset:0 atIndex:15];
      [encoder setBuffer:out offset:0 atIndex:16];
      [encoder dispatchThreadgroups:MTLSizeMake(kKdaHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kKdaWidth, 1, 1)];
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }
    ++report.command_buffers;
    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    const long wait_result = dispatch_semaphore_wait(
        ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
    require(wait_result == 0, "kda-canary event timed out");
    if (command.error != nil) {
      fail(std::string("kda-canary command failed: ") +
           command.error.localizedDescription.UTF8String);
    }

    std::vector<double> ref_conv;
    std::vector<double> ref_state;
    std::vector<double> ref_out;
    kda_core_reference(static_cast<const float*>(proj.contents),
                       static_cast<const float*>(fb.contents),
                       static_cast<const float*>(convw.contents),
                       static_cast<const float*>(convin.contents),
                       static_cast<const float*>(a_log.contents),
                       static_cast<const float*>(dt_bias.contents),
                       static_cast<const float*>(o_norm.contents),
                       static_cast<const float*>(s_in.contents), ref_conv,
                       ref_state, ref_out);
    const double out_rel = kda_rel_l2(
        static_cast<const float*>(out.contents), ref_out.data(), kKdaProj);
    const double state_rel =
        kda_rel_l2(static_cast<const float*>(s_out.contents),
                   ref_state.data(), kKdaState);
    const double conv_rel = kda_rel_l2(
        static_cast<const float*>(convout.contents), ref_conv.data(),
        static_cast<std::size_t>(kKdaConvChannels) * 4);
    report.max_out_relative_l2 =
        std::max(report.max_out_relative_l2, out_rel);
    report.max_state_relative_l2 =
        std::max(report.max_state_relative_l2, state_rel);
    report.max_conv_relative_l2 =
        std::max(report.max_conv_relative_l2, conv_rel);
    ++report.cases_run;
    if (out_rel <= kLoopKdaCanaryMaxRelL2 &&
        state_rel <= kLoopKdaCanaryMaxRelL2 &&
        conv_rel <= kLoopKdaCanaryMaxRelL2) {
      ++report.cases_passed;
    }
  };

  run_case(false, false);
  run_case(true, false);
  run_case(false, true);
  return report;
}

int kda_loop_mode() {
  static const int mode = [] {
    const char* value = std::getenv("K3_KDA_LOOP");
    if (value == nullptr) return 0;
    if (std::strcmp(value, "parity") == 0) return 1;
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 2;
    }
    return 0;
  }();
  return mode;
}

namespace {

std::atomic<bool> g_loop_chain_safe{false};

bool shared_boundary_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_KDA_SHARED_BOUNDARY");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

/* Host clock in the MTLCommandBuffer kernel/GPU timestamp timebase. */
double host_media_seconds() noexcept {
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return info;
  }();
  return static_cast<double>(mach_absolute_time()) * timebase.numer /
         timebase.denom / 1e9;
}

}  // namespace

void loop_set_chain_safe(const bool safe) noexcept {
  g_loop_chain_safe.store(safe, std::memory_order_release);
}

namespace {
std::atomic<bool> g_fresh_anchor_cat{false};
/* Separate from the anchor-cat marker: a fresh fp32-arena dequant encode.
 * int8-form loop CBs read no arena, so they must neither fence for this
 * NOR consume it (a later fp32 reader still needs the ordering). */
std::atomic<bool> g_fresh_dequant{false};
}  // namespace

void loop_note_fresh_anchor_cat() noexcept {
  g_fresh_anchor_cat.store(true, std::memory_order_release);
}

void loop_note_fresh_dequant() noexcept {
  g_fresh_dequant.store(true, std::memory_order_release);
}

void loop_drain_aten_stream() noexcept {
  try {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream != nullptr) {
      const std::uint64_t drain_started = prep_media_ns();
      stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
      prep_note_wait(prep_media_ns() - drain_started, 0, false);
    }
  } catch (...) {
  }
}

bool loop_consume_fresh_marker() noexcept {
  return g_fresh_dequant.exchange(false, std::memory_order_acq_rel);
}

at::Tensor loop_host_alias_of_mps(const at::Tensor& tensor) {
  require(tensor.defined() && tensor.device().is_mps(),
          "host alias needs an MPS tensor");
  require(tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
          "host alias needs a contiguous fp32 tensor");
  const std::size_t bytes =
      static_cast<std::size_t>(tensor.numel()) * tensor.element_size();
  const NSUInteger offset =
      checked_byte_offset(tensor, bytes, "routed-input host alias");
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  require(buffer.storageMode == MTLStorageModeShared,
          "host alias needs shared-mode MTLBuffer storage");
  // Everything ATen enqueued for this tensor must have completed before the
  // CPU (and the Metal MoE staging memcpy) reads the bytes.
  loop_drain_aten_stream();
  require(!loop_tail_poisoned(), "host alias after a poisoned loop tail");
  auto* keepalive = new at::Tensor(tensor);
  void* pointer = static_cast<std::uint8_t*>(buffer.contents) + offset;
  return at::from_blob(
      pointer, tensor.sizes(),
      [keepalive](void*) { delete keepalive; },
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

namespace {

/* Encodes one projection GEMV on the loop queue, dispatching by the live
 * KdaProjection form (dense fp32 / row-int8 / original-BF16) so the encoder
 * is host-independent.  Throws (caught by the parity wrapper) on any shape
 * or form mismatch.  All three kernels share the 16-rows-per-threadgroup,
 * 128-thread blocking. */
void encode_kda_projection(id<MTLComputeCommandEncoder> encoder,
                           LoopEncoderState& state,
                           const KdaProjection& projection,
                           const std::uint32_t rows,
                           const std::uint32_t columns,
                           id<MTLBuffer> x_buffer, const NSUInteger x_offset,
                           id<MTLBuffer> y_buffer,
                           const NSUInteger y_offset, const char* name) {
  const std::size_t weight_elements =
      static_cast<std::size_t>(rows) * columns;
  if (projection.original_bf16.defined()) {
    const OriginalBf16Matrix& matrix = projection.original_bf16;
    require(matrix.is_owned() && matrix.owned_storage != nullptr &&
                matrix.owned_storage->tensor.defined() &&
                matrix.rows == rows && matrix.columns == columns,
            std::string("KDA loop bf16 projection form mismatch: ") + name);
    const at::Tensor& bits = matrix.owned_storage->tensor;
    require(bits.device().is_mps() && bits.is_contiguous() &&
                (bits.scalar_type() == at::kBFloat16 ||
                 bits.scalar_type() == at::kShort) &&
                static_cast<std::size_t>(bits.numel()) >=
                    matrix.owned_element_offset + weight_elements,
            std::string("KDA loop bf16 slab does not qualify: ") + name);
    const NSUInteger slab_offset = checked_byte_offset(
        bits,
        (matrix.owned_element_offset + weight_elements) *
            sizeof(std::uint16_t),
        name);
    struct Dims {
      std::uint32_t rows;
      std::uint32_t columns;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
    };
    const Dims dims{rows, columns, 0, 0};
    [encoder setComputePipelineState:state.gemv_bf16_pipeline];
    [encoder setBuffer:tensor_buffer(bits)
                offset:slab_offset + static_cast<NSUInteger>(
                                         matrix.owned_element_offset *
                                         sizeof(std::uint16_t))
               atIndex:0];
    [encoder setBuffer:x_buffer offset:x_offset atIndex:1];
    [encoder setBuffer:y_buffer offset:y_offset atIndex:2];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 15) / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return;
  }
  require(projection.weight.defined() &&
              projection.weight.device().is_mps() &&
              projection.weight.is_contiguous() &&
              static_cast<std::size_t>(projection.weight.numel()) ==
                  weight_elements,
          std::string("KDA loop projection weight does not qualify: ") +
              name);
  if (projection.scale.defined()) {
    require(projection.weight.scalar_type() == at::kChar &&
                projection.scale.device().is_mps() &&
                projection.scale.is_contiguous() &&
                projection.scale.scalar_type() == at::kFloat &&
                static_cast<std::size_t>(projection.scale.numel()) == rows,
            std::string("KDA loop int8 projection does not qualify: ") +
                name);
    const NSUInteger weight_offset =
        checked_byte_offset(projection.weight, weight_elements, name);
    const NSUInteger scale_offset = checked_byte_offset(
        projection.scale, rows * sizeof(float), name);
    struct Dims {
      std::uint32_t N;
      std::uint32_t K;
    };
    const Dims dims{rows, columns};
    [encoder setComputePipelineState:state.gemv_pipeline];
    [encoder setBuffer:tensor_buffer(projection.weight)
                offset:weight_offset
               atIndex:0];
    [encoder setBuffer:x_buffer offset:x_offset atIndex:1];
    [encoder setBuffer:tensor_buffer(projection.scale)
                offset:scale_offset
               atIndex:2];
    [encoder setBuffer:y_buffer offset:y_offset atIndex:3];
    [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 15) / 16, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    return;
  }
  require(projection.weight.scalar_type() == at::kFloat,
          std::string("KDA loop dense projection is not fp32: ") + name);
  const NSUInteger weight_offset = checked_byte_offset(
      projection.weight, weight_elements * sizeof(float), name);
  struct Dims {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
  };
  const Dims dims{rows, columns, 0, 0};
  [encoder setComputePipelineState:state.gemv_f32_pipeline];
  [encoder setBuffer:tensor_buffer(projection.weight)
              offset:weight_offset
             atIndex:0];
  [encoder setBuffer:x_buffer offset:x_offset atIndex:1];
  [encoder setBuffer:y_buffer offset:y_offset atIndex:2];
  [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake((rows + 15) / 16, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

double kda_parity_rel_l2_tensor(const float* got, const at::Tensor& want) {
  const at::Tensor host =
      want.to(at::kCPU, at::kFloat).contiguous();
  const float* reference = host.const_data_ptr<float>();
  double difference_squares = 0.0;
  double expected_squares = 0.0;
  const std::size_t count = static_cast<std::size_t>(host.numel());
  for (std::size_t i = 0; i < count; ++i) {
    const double difference =
        static_cast<double>(got[i]) - static_cast<double>(reference[i]);
    difference_squares += difference * difference;
    expected_squares +=
        static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
  }
  return expected_squares > 0.0
             ? std::sqrt(difference_squares / expected_squares)
             : std::sqrt(difference_squares);
}

/* Int8 row-scale GEMV encode for an MLA projection. MlaLinearWeight's
 * RowI8F32Scale layout (kChar [rows, columns] + kFloat [rows]) is exactly
 * what deltafin_loop_gemv_i8_bundle_v1 indexes (Step 2b), so this is the
 * KdaProjection int8 branch minus the form dispatch. */
void encode_mla_gemv(id<MTLComputeCommandEncoder> encoder,
                     LoopEncoderState& state, const MlaLinearWeight& weight,
                     const std::uint32_t rows, const std::uint32_t columns,
                     id<MTLBuffer> x_buffer, const NSUInteger x_offset,
                     id<MTLBuffer> y_buffer, const NSUInteger y_offset,
                     const char* name) {
  require(weight.encoding == MlaLinearEncoding::RowI8F32Scale,
          std::string("MLA parity projection is not row-int8: ") + name);
  const std::int64_t expected_rows = static_cast<std::int64_t>(rows);
  const std::int64_t expected_columns = static_cast<std::int64_t>(columns);
  require(weight.data.defined() && weight.data.device().is_mps() &&
              weight.data.scalar_type() == at::kChar &&
              weight.data.is_contiguous() && weight.data.dim() == 2 &&
              weight.data.size(0) == expected_rows &&
              weight.data.size(1) == expected_columns,
          std::string("MLA parity int8 payload does not qualify: ") + name);
  require(weight.row_scale.defined() && weight.row_scale.device().is_mps() &&
              weight.row_scale.scalar_type() == at::kFloat &&
              weight.row_scale.is_contiguous() &&
              weight.row_scale.numel() == expected_rows,
          std::string("MLA parity row scales do not qualify: ") + name);
  const NSUInteger data_offset = checked_byte_offset(
      weight.data,
      static_cast<std::size_t>(rows) * columns * sizeof(std::int8_t), name);
  const NSUInteger scale_offset = checked_byte_offset(
      weight.row_scale, static_cast<std::size_t>(rows) * sizeof(float), name);
  const LoopGemvDimsHost dims{rows, columns};
  [encoder setComputePipelineState:state.gemv_pipeline];
  [encoder setBuffer:tensor_buffer(weight.data) offset:data_offset atIndex:0];
  [encoder setBuffer:x_buffer offset:x_offset atIndex:1];
  [encoder setBuffer:tensor_buffer(weight.row_scale)
              offset:scale_offset
             atIndex:2];
  [encoder setBuffer:y_buffer offset:y_offset atIndex:3];
  [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
  [encoder dispatchThreadgroups:MTLSizeMake((rows + 15) / 16, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

}  // namespace

void loop_kda_parity_compare(const std::uint32_t layer_index,
                             const at::Tensor& normalized,
                             const KdaWeights& weights,
                             const KdaState& state_in,
                             const KdaDecodeResult& stock) {
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kProjWidth = 12288;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[kda-parity] first skip: %s\n", reason);
    }
  };

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    ++state.kda_parity_stats.skipped;
    return;
  }
  try {
    ensure_resources(state, device);

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("KDA parity tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    const NSUInteger hidden_offset =
        plain(normalized, kHiddenWidth, "normalized");
    const NSUInteger a_log_offset = plain(weights.a_log, 128, "a_log");
    const NSUInteger dt_bias_offset =
        plain(weights.dt_bias, kProjWidth, "dt_bias");
    const NSUInteger o_norm_offset =
        plain(weights.output_norm, 128, "output_norm");
    const NSUInteger convw_q_offset = plain(
        weights.query_convolution, kProjWidth * 4, "query_convolution");
    const NSUInteger convw_k_offset =
        plain(weights.key_convolution, kProjWidth * 4, "key_convolution");
    const NSUInteger convw_v_offset = plain(
        weights.value_convolution, kProjWidth * 4, "value_convolution");
    const NSUInteger convin_q_offset = plain(
        state_in.query_convolution, kProjWidth * 4, "query conv state");
    const NSUInteger convin_k_offset =
        plain(state_in.key_convolution, kProjWidth * 4, "key conv state");
    const NSUInteger convin_v_offset = plain(
        state_in.value_convolution, kProjWidth * 4, "value conv state");
    const NSUInteger s_in_offset =
        plain(state_in.recurrent,
              static_cast<std::int64_t>(96) * 128 * 128, "recurrent state");

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "KDA parity MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "KDA parity root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "KDA parity boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "KDA parity command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "KDA parity encoder is unavailable");

      // Projection bundle into proj[49376]: q|k|v|g @0/12288/24576/36864,
      // f_a @49152, beta @49280 — all reading the normalized hidden.
      encode_kda_projection(encoder, state, weights.query_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, 0, "query_projection");
      encode_kda_projection(encoder, state, weights.key_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, kProjWidth * sizeof(float),
                            "key_projection");
      encode_kda_projection(encoder, state, weights.value_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, 2 * kProjWidth * sizeof(float),
                            "value_projection");
      encode_kda_projection(encoder, state,
                            weights.recurrent_gate_projection, kProjWidth,
                            kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            3 * kProjWidth * sizeof(float),
                            "recurrent_gate_projection");
      encode_kda_projection(encoder, state, weights.feature_a_projection,
                            128, kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            4 * kProjWidth * sizeof(float),
                            "feature_a_projection");
      encode_kda_projection(encoder, state, weights.beta_projection, 96,
                            kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            (4 * kProjWidth + 128) * sizeof(float),
                            "beta_projection");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // f_b on the f_a slice.
      encode_kda_projection(encoder, state, weights.feature_b_projection,
                            kProjWidth, 128, state.kda_proj,
                            4 * kProjWidth * sizeof(float), state.kda_fb, 0,
                            "feature_b_projection");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      const NSUInteger conv_channel_bytes =
          static_cast<NSUInteger>(kProjWidth) * 4 * sizeof(float);
      [encoder setComputePipelineState:state.kda_core_pipeline];
      [encoder setBuffer:state.kda_proj offset:0 atIndex:0];
      [encoder setBuffer:state.kda_fb offset:0 atIndex:1];
      [encoder setBuffer:tensor_buffer(weights.query_convolution)
                  offset:convw_q_offset
                 atIndex:2];
      [encoder setBuffer:tensor_buffer(weights.key_convolution)
                  offset:convw_k_offset
                 atIndex:3];
      [encoder setBuffer:tensor_buffer(weights.value_convolution)
                  offset:convw_v_offset
                 atIndex:4];
      [encoder setBuffer:tensor_buffer(state_in.query_convolution)
                  offset:convin_q_offset
                 atIndex:5];
      [encoder setBuffer:tensor_buffer(state_in.key_convolution)
                  offset:convin_k_offset
                 atIndex:6];
      [encoder setBuffer:tensor_buffer(state_in.value_convolution)
                  offset:convin_v_offset
                 atIndex:7];
      [encoder setBuffer:state.kda_conv_out offset:0 atIndex:8];
      [encoder setBuffer:state.kda_conv_out
                  offset:conv_channel_bytes
                 atIndex:9];
      [encoder setBuffer:state.kda_conv_out
                  offset:2 * conv_channel_bytes
                 atIndex:10];
      [encoder setBuffer:tensor_buffer(weights.a_log)
                  offset:a_log_offset
                 atIndex:11];
      [encoder setBuffer:tensor_buffer(weights.dt_bias)
                  offset:dt_bias_offset
                 atIndex:12];
      [encoder setBuffer:tensor_buffer(weights.output_norm)
                  offset:o_norm_offset
                 atIndex:13];
      [encoder setBuffer:tensor_buffer(state_in.recurrent)
                  offset:s_in_offset
                 atIndex:14];
      [encoder setBuffer:state.kda_s_out offset:0 atIndex:15];
      [encoder setBuffer:state.kda_core_out offset:0 atIndex:16];
      [encoder dispatchThreadgroups:MTLSizeMake(96, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      encode_kda_projection(encoder, state, weights.output_projection,
                            kHiddenWidth, kProjWidth, state.kda_core_out, 0,
                            state.kda_final_out, 0, "output_projection");
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    const long wait_result = dispatch_semaphore_wait(
        ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
    require(wait_result == 0, "KDA parity event timed out");
    if (command.error != nil) {
      fail(std::string("KDA parity command failed: ") +
           command.error.localizedDescription.UTF8String);
    }

    const double out_rel = kda_parity_rel_l2_tensor(
        static_cast<const float*>(state.kda_final_out.contents),
        stock.output);
    double state_rel = kda_parity_rel_l2_tensor(
        static_cast<const float*>(state.kda_s_out.contents),
        stock.next_state.recurrent);
    const float* conv_host =
        static_cast<const float*>(state.kda_conv_out.contents);
    state_rel = std::max(
        state_rel,
        kda_parity_rel_l2_tensor(conv_host,
                                 stock.next_state.query_convolution));
    state_rel = std::max(
        state_rel,
        kda_parity_rel_l2_tensor(conv_host + kProjWidth * 4,
                                 stock.next_state.key_convolution));
    state_rel = std::max(
        state_rel,
        kda_parity_rel_l2_tensor(conv_host + 2 * kProjWidth * 4,
                                 stock.next_state.value_convolution));

    ++state.kda_parity_stats.compared;
    state.kda_parity_stats.max_output_relative_l2 =
        std::max(state.kda_parity_stats.max_output_relative_l2, out_rel);
    state.kda_parity_stats.max_state_relative_l2 =
        std::max(state.kda_parity_stats.max_state_relative_l2, state_rel);
    const bool failed = !(out_rel <= 1.0e-5 && state_rel <= 1.0e-5) ||
                        !std::isfinite(out_rel) || !std::isfinite(state_rel);
    if (failed) {
      ++state.kda_parity_stats.failures;
      std::fprintf(stderr,
                   "[kda-parity] FAILURE layer=%u out_rel=%.3e "
                   "state_rel=%.3e\n",
                   layer_index, out_rel, state_rel);
    }
    if (state.kda_parity_stats.compared == 1 ||
        state.kda_parity_stats.compared % 690 == 0) {
      std::fprintf(
          stderr,
          "[kda-parity] compared=%llu failures=%llu out_rel_max=%.3e "
          "state_rel_max=%.3e skipped=%llu\n",
          static_cast<unsigned long long>(state.kda_parity_stats.compared),
          static_cast<unsigned long long>(state.kda_parity_stats.failures),
          state.kda_parity_stats.max_output_relative_l2,
          state.kda_parity_stats.max_state_relative_l2,
          static_cast<unsigned long long>(state.kda_parity_stats.skipped));
    }
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    ++state.kda_parity_stats.skipped;
  } catch (...) {
    report_skip_once("unknown exception");
    ++state.kda_parity_stats.skipped;
  }
}

LoopKdaParityStats loop_kda_parity_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.kda_parity_stats;
}

int mla_loop_mode() {
  static const int mode = [] {
    const char* value = std::getenv("K3_MLA_LOOP");
    if (value == nullptr) {
      return 0;
    }
    if (std::strcmp(value, "parity") == 0) {
      return 1;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 2;
    }
    if (std::strcmp(value, "chain") == 0) {
      return 3;
    }
    return 0;
  }();
  return mode;
}

void loop_mla_parity_compare(const std::uint32_t layer_index,
                             const at::Tensor& hidden,
                             const MlaWeights& weights,
                             const MlaInputBundle* input_bundle,
                             const MlaCache& cache,
                             const MlaPreparedDecode& stock) {
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kBundleRows = 14400;
  constexpr std::uint32_t kQueryARows = 1536;
  constexpr std::uint32_t kLatentWidth = 512;
  constexpr std::uint32_t kRopeWidth = 64;
  constexpr std::uint32_t kGateWidth = 12288;
  constexpr std::uint32_t kQueryBRows = 18432;   // 96 * 192
  constexpr std::uint32_t kKeyValueBRows = 24576;  // 96 * 256
  constexpr std::uint32_t kHeads = 96;
  constexpr std::uint32_t kKeyDim = 192;
  constexpr std::uint32_t kValueDim = 128;
  constexpr std::uint32_t kMaxPartitions = 32;
  constexpr float kRmsEpsilon = 1.0e-5F;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[mla-parity] first skip: %s\n", reason);
    }
  };

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    ++state.mla_parity_stats.skipped;
    return;
  }
  try {
    ensure_resources(state, device);

    require(input_bundle != nullptr &&
                input_bundle->query_a_rows == kQueryARows &&
                input_bundle->key_value_a_rows == kLatentWidth + kRopeWidth &&
                input_bundle->output_gate_rows == kGateWidth,
            "MLA parity requires the exact-K3 input bundle");
    require(stock.position_count == 1 && stock.output.defined(),
            "MLA parity requires a T=1 prepared decode");

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("MLA parity tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    const NSUInteger hidden_offset = plain(hidden, kHiddenWidth, "hidden");
    const NSUInteger qa_norm_offset =
        plain(weights.query_a_norm, kQueryARows, "query_a_norm");
    const NSUInteger kv_norm_offset =
        plain(weights.key_value_a_norm, kLatentWidth, "key_value_a_norm");

    // The committed prefix from the storage the prepare actually chose:
    // growth reallocates, and the cache's own slab is stale in that case.
    // length has NOT advanced yet (commit runs after parity), so these are
    // exactly the rows the stock attention consumed — plus its staged new
    // row at index `length`, which parity ignores (it packs its own).
    const std::int64_t length = cache.length();
    require(length >= 0 && length == stock.expected_length,
            "MLA parity cache length does not match the prepared decode");
    const std::int64_t parity_length = length + 1;
    const at::Tensor key_prefix =
        stock.uses_grown_storage
            ? stock.grown_key_storage.narrow(2, 0, length)
            : cache.committed_keys();
    const at::Tensor value_prefix =
        stock.uses_grown_storage
            ? stock.grown_value_storage.narrow(2, 0, length)
            : cache.committed_values();
    const auto slab_view = [&](const at::Tensor& tensor,
                               const std::int64_t width, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
                  tensor.size(0) == 1 && tensor.size(1) == kHeads &&
                  tensor.size(2) == length && tensor.size(3) == width &&
                  tensor.stride(3) == 1 && tensor.stride(2) == width &&
                  tensor.stride(1) >= length * width,
              std::string("MLA parity slab view does not qualify: ") + name);
      const std::size_t span =
          length > 0 ? (static_cast<std::size_t>(tensor.stride(1)) *
                            (kHeads - 1) +
                        static_cast<std::size_t>(length) * width) *
                           sizeof(float)
                     : 0;
      return std::pair<NSUInteger, NSUInteger>(
          checked_byte_offset(tensor, span, name),
          static_cast<NSUInteger>(tensor.stride(1) * sizeof(float)));
    };
    const auto [key_base, key_head_stride] =
        slab_view(key_prefix, kKeyDim, "key prefix");
    const auto [value_base, value_head_stride] =
        slab_view(value_prefix, kValueDim, "value prefix");

    // Private KV scratch slab, pitch = parity_length. Geometric regrowth;
    // never the live slab (both paths run during parity).
    if (state.mla_kv_keys == nil || state.mla_kv_values == nil ||
        state.mla_kv_capacity < parity_length) {
      // +256 headroom: parity S grows by one per token, and exact-fit
      // regrowth would reallocate both buffers on every call.
      const std::int64_t grown = std::max<std::int64_t>(
          parity_length + 256, (state.mla_kv_capacity * 3) / 2);
      // Reset BEFORE allocating: a partially successful pair (keys fresh,
      // values nil) with a stale non-zero capacity would pass the guard on
      // a later shorter sequence and hand Metal a nil buffer — an ObjC
      // exception the C++ catch blocks cannot make fail-soft.
      state.mla_kv_keys = nil;
      state.mla_kv_values = nil;
      state.mla_kv_capacity = 0;
      id<MTLBuffer> keys = [device
          newBufferWithLength:static_cast<NSUInteger>(grown) * kHeads *
                              kKeyDim * sizeof(float)
                      options:MTLResourceStorageModeShared |
                              MTLResourceCPUCacheModeDefaultCache];
      id<MTLBuffer> values = [device
          newBufferWithLength:static_cast<NSUInteger>(grown) * kHeads *
                              kValueDim * sizeof(float)
                      options:MTLResourceStorageModeShared |
                              MTLResourceCPUCacheModeDefaultCache];
      require(keys != nil && values != nil,
              "MLA parity KV scratch allocation failed");
      state.mla_kv_keys = keys;
      state.mla_kv_values = values;
      state.mla_kv_capacity = grown;
    }
    const NSUInteger scratch_key_pitch =
        static_cast<NSUInteger>(parity_length) * kKeyDim * sizeof(float);
    const NSUInteger scratch_value_pitch =
        static_cast<NSUInteger>(parity_length) * kValueDim * sizeof(float);

    const std::uint32_t partitions = std::min<std::uint32_t>(
        kMaxPartitions,
        std::max<std::uint32_t>(
            1, (static_cast<std::uint32_t>(parity_length) + 255) / 256));
    struct AttnDims {
      std::uint32_t kv_length;
      std::uint32_t capacity;
      std::uint32_t partitions;
      std::uint32_t chunk;
      float scale;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
      std::uint32_t reserved2;
    } attn_dims{
        static_cast<std::uint32_t>(parity_length),
        static_cast<std::uint32_t>(parity_length),
        partitions,
        (static_cast<std::uint32_t>(parity_length) + partitions - 1) /
            partitions,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(kKeyDim))),
        0, 0, 0};

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "MLA parity MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "MLA parity root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "MLA parity boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "MLA parity command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];

      if (length > 0) {
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "MLA parity prefix blit encoder unavailable");
        for (std::uint32_t head = 0; head < kHeads; ++head) {
          [blit copyFromBuffer:tensor_buffer(key_prefix)
                  sourceOffset:key_base + head * key_head_stride
                      toBuffer:state.mla_kv_keys
             destinationOffset:head * scratch_key_pitch
                          size:static_cast<NSUInteger>(length) * kKeyDim *
                               sizeof(float)];
          [blit copyFromBuffer:tensor_buffer(value_prefix)
                  sourceOffset:value_base + head * value_head_stride
                      toBuffer:state.mla_kv_values
             destinationOffset:head * scratch_value_pitch
                          size:static_cast<NSUInteger>(length) * kValueDim *
                               sizeof(float)];
        }
        [blit endEncoding];
      }

      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA parity encoder is unavailable");
      // Op 1: bundle GEMV — q_a[0,1536) | kv_a latent[1536,2048) |
      // key_rope[2048,2112) | gate[2112,14400).
      encode_mla_gemv(encoder, state, input_bundle->projection, kBundleRows,
                      kHiddenWidth, tensor_buffer(hidden), hidden_offset,
                      state.mla_bundle_out, 0, "input_bundle");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // Ops 2 + 4: the two rmsnorms, both reading bundle output slices.
      struct RmsDimsHost {
        std::uint32_t N;
        float eps;
      };
      const RmsDimsHost qa_dims{kQueryARows, kRmsEpsilon};
      [encoder setComputePipelineState:state.rmsnorm_pipeline];
      [encoder setBuffer:state.mla_bundle_out offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.query_a_norm)
                  offset:qa_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_qa_norm offset:0 atIndex:2];
      [encoder setBytes:&qa_dims length:sizeof(qa_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      const RmsDimsHost latent_dims{kLatentWidth, kRmsEpsilon};
      [encoder setBuffer:state.mla_bundle_out
                  offset:kQueryARows * sizeof(float)
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.key_value_a_norm)
                  offset:kv_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_latent_norm offset:0 atIndex:2];
      [encoder setBytes:&latent_dims length:sizeof(latent_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // Ops 3 + 5: q_b and kv_b GEMVs on the normed slices.
      encode_mla_gemv(encoder, state, weights.query_b, kQueryBRows,
                      kQueryARows, state.mla_qa_norm, 0, state.mla_query, 0,
                      "query_b");
      encode_mla_gemv(encoder, state, weights.key_value_b, kKeyValueBRows,
                      kLatentWidth, state.mla_latent_norm, 0,
                      state.mla_expanded, 0, "key_value_b");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // Op 6: pack the new key/value row (shared rope tail per head).
      struct PackDimsHost {
        std::uint32_t heads;
        std::uint32_t nope_dim;
        std::uint32_t rope_dim;
        std::uint32_t value_dim;
      };
      const PackDimsHost pack_dims{kHeads, 128, kRopeWidth, kValueDim};
      [encoder setComputePipelineState:state.mla_pack_pipeline];
      [encoder setBuffer:state.mla_expanded offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth) * sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_new_key offset:0 atIndex:2];
      [encoder setBuffer:state.mla_new_value offset:0 atIndex:3];
      [encoder setBytes:&pack_dims length:sizeof(pack_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(kHeads * kKeyDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];

      // Op 7: place the packed row at scratch index `length` per head.
      id<MTLBlitCommandEncoder> place = [command blitCommandEncoder];
      require(place != nil, "MLA parity placement blit encoder unavailable");
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        [place copyFromBuffer:state.mla_new_key
                 sourceOffset:head * kKeyDim * sizeof(float)
                     toBuffer:state.mla_kv_keys
            destinationOffset:head * scratch_key_pitch +
                              static_cast<NSUInteger>(length) * kKeyDim *
                                  sizeof(float)
                         size:kKeyDim * sizeof(float)];
        [place copyFromBuffer:state.mla_new_value
                 sourceOffset:head * kValueDim * sizeof(float)
                     toBuffer:state.mla_kv_values
            destinationOffset:head * scratch_value_pitch +
                              static_cast<NSUInteger>(length) * kValueDim *
                                  sizeof(float)
                         size:kValueDim * sizeof(float)];
      }
      [place endEncoding];

      encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA parity attention encoder is unavailable");
      // Op 8: flash attention part + combine on the private slab.
      [encoder setComputePipelineState:state.mla_part_pipeline];
      [encoder setBuffer:state.mla_query offset:0 atIndex:0];
      [encoder setBuffer:state.mla_kv_keys offset:0 atIndex:1];
      [encoder setBuffer:state.mla_kv_values offset:0 atIndex:2];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:3];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads * partitions, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:state.mla_combine_pipeline];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:0];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:1];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // Op 9: attention * sigmoid(raw gate slice of the bundle output).
      const std::uint32_t gate_elements = kGateWidth;
      [encoder setComputePipelineState:state.mla_gate_pipeline];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth + kRopeWidth) *
                         sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_gated offset:0 atIndex:2];
      [encoder setBytes:&gate_elements
                 length:sizeof(gate_elements)
                atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kGateWidth, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      // Op 10: o_proj into the final [7168] row.
      encode_mla_gemv(encoder, state, weights.output, kHiddenWidth,
                      kGateWidth, state.mla_gated, 0, state.mla_final_out, 0,
                      "output");
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    const long wait_result = dispatch_semaphore_wait(
        ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
    require(wait_result == 0, "MLA parity event timed out");
    if (command.error != nil) {
      fail(std::string("MLA parity command failed: ") +
           command.error.localizedDescription.UTF8String);
    }

    const double out_rel = kda_parity_rel_l2_tensor(
        static_cast<const float*>(state.mla_final_out.contents),
        stock.output);

    ++state.mla_parity_stats.compared;
    state.mla_parity_stats.max_output_relative_l2 =
        std::max(state.mla_parity_stats.max_output_relative_l2, out_rel);
    const bool failed = !(out_rel <= 5.0e-6) || !std::isfinite(out_rel);
    if (failed) {
      ++state.mla_parity_stats.failures;
      std::fprintf(stderr, "[mla-parity] FAILURE layer=%u out_rel=%.3e\n",
                   layer_index, out_rel);
    }
    if (state.mla_parity_stats.compared == 1 ||
        state.mla_parity_stats.compared % 240 == 0) {
      std::fprintf(
          stderr,
          "[mla-parity] compared=%llu failures=%llu out_rel_max=%.3e "
          "skipped=%llu\n",
          static_cast<unsigned long long>(state.mla_parity_stats.compared),
          static_cast<unsigned long long>(state.mla_parity_stats.failures),
          state.mla_parity_stats.max_output_relative_l2,
          static_cast<unsigned long long>(state.mla_parity_stats.skipped));
    }
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    ++state.mla_parity_stats.skipped;
  } catch (...) {
    report_skip_once("unknown exception");
    ++state.mla_parity_stats.skipped;
  }
}

LoopMlaParityStats loop_mla_parity_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.mla_parity_stats;
}

bool loop_mla_takeover_run(const std::uint32_t layer_index,
                           const at::Tensor& hidden,
                           const MlaWeights& weights,
                           const MlaInputBundle* input_bundle,
                           const at::Tensor& key_states,
                           const at::Tensor& value_states,
                           const at::Tensor& output) {
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kBundleRows = 14400;
  constexpr std::uint32_t kQueryARows = 1536;
  constexpr std::uint32_t kLatentWidth = 512;
  constexpr std::uint32_t kRopeWidth = 64;
  constexpr std::uint32_t kGateWidth = 12288;
  constexpr std::uint32_t kQueryBRows = 18432;
  constexpr std::uint32_t kKeyValueBRows = 24576;
  constexpr std::uint32_t kHeads = 96;
  constexpr std::uint32_t kKeyDim = 192;
  constexpr std::uint32_t kValueDim = 128;
  constexpr std::uint32_t kMaxPartitions = 32;
  constexpr float kRmsEpsilon = 1.0e-5F;
  static std::atomic<bool> fallback_reported{false};
  const auto report_fallback_once = [&](const char* reason) {
    if (!fallback_reported.exchange(true)) {
      std::fprintf(stderr, "[mla-loop] first fallback (layer %u): %s\n",
                   layer_index, reason);
    }
  };

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_fallback_once("current MPS stream unavailable");
    return false;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_fallback_once("current MPS device unavailable");
    ++state.mla_takeover_stats.fallbacks;
    return false;
  }
  try {
    ensure_resources(state, device);

    require(input_bundle != nullptr &&
                input_bundle->query_a_rows == kQueryARows &&
                input_bundle->key_value_a_rows == kLatentWidth + kRopeWidth &&
                input_bundle->output_gate_rows == kGateWidth,
            "MLA takeover requires the exact-K3 input bundle");

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("MLA takeover tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    const NSUInteger hidden_offset = plain(hidden, kHiddenWidth, "hidden");
    const NSUInteger qa_norm_offset =
        plain(weights.query_a_norm, kQueryARows, "query_a_norm");
    const NSUInteger kv_norm_offset =
        plain(weights.key_value_a_norm, kLatentWidth, "key_value_a_norm");
    const NSUInteger output_offset = plain(output, kHiddenWidth, "output");

    require(key_states.defined() && key_states.dim() == 4 &&
                key_states.size(2) >= 1,
            "MLA takeover key states view is invalid");
    const std::int64_t kv_length = key_states.size(2);
    const auto slab_view = [&](const at::Tensor& tensor,
                               const std::int64_t width, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
                  tensor.size(0) == 1 && tensor.size(1) == kHeads &&
                  tensor.size(2) == kv_length && tensor.size(3) == width &&
                  tensor.stride(3) == 1 && tensor.stride(2) == width &&
                  tensor.stride(1) >= kv_length * width &&
                  tensor.stride(1) % width == 0,
              std::string("MLA takeover slab view does not qualify: ") +
                  name);
      const std::size_t span =
          (static_cast<std::size_t>(tensor.stride(1)) * (kHeads - 1) +
           static_cast<std::size_t>(kv_length) * width) *
          sizeof(float);
      return std::pair<NSUInteger, NSUInteger>(
          checked_byte_offset(tensor, span, name),
          static_cast<NSUInteger>(tensor.stride(1) * sizeof(float)));
    };
    const auto [key_base, key_head_stride] =
        slab_view(key_states, kKeyDim, "key states");
    const auto [value_base, value_head_stride] =
        slab_view(value_states, kValueDim, "value states");
    // The attention kernel derives BOTH slab addresses from one
    // dims.capacity — the storages share their allocated capacity.
    const std::int64_t slab_capacity =
        key_states.stride(1) / static_cast<std::int64_t>(kKeyDim);
    require(value_states.stride(1) ==
                slab_capacity * static_cast<std::int64_t>(kValueDim),
            "MLA takeover key/value slab capacities disagree");

    const std::uint32_t partitions = std::min<std::uint32_t>(
        kMaxPartitions,
        std::max<std::uint32_t>(
            1, (static_cast<std::uint32_t>(kv_length) + 255) / 256));
    struct AttnDims {
      std::uint32_t kv_length;
      std::uint32_t capacity;
      std::uint32_t partitions;
      std::uint32_t chunk;
      float scale;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
      std::uint32_t reserved2;
    } attn_dims{
        static_cast<std::uint32_t>(kv_length),
        static_cast<std::uint32_t>(slab_capacity),
        partitions,
        (static_cast<std::uint32_t>(kv_length) + partitions - 1) /
            partitions,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(kKeyDim))),
        0, 0, 0};

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "MLA takeover MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "MLA takeover root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "MLA takeover boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "MLA takeover command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA takeover encoder is unavailable");
      encode_mla_gemv(encoder, state, input_bundle->projection, kBundleRows,
                      kHiddenWidth, tensor_buffer(hidden), hidden_offset,
                      state.mla_bundle_out, 0, "input_bundle");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      struct RmsDimsHost {
        std::uint32_t N;
        float eps;
      };
      const RmsDimsHost qa_dims{kQueryARows, kRmsEpsilon};
      [encoder setComputePipelineState:state.rmsnorm_pipeline];
      [encoder setBuffer:state.mla_bundle_out offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.query_a_norm)
                  offset:qa_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_qa_norm offset:0 atIndex:2];
      [encoder setBytes:&qa_dims length:sizeof(qa_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      const RmsDimsHost latent_dims{kLatentWidth, kRmsEpsilon};
      [encoder setBuffer:state.mla_bundle_out
                  offset:kQueryARows * sizeof(float)
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.key_value_a_norm)
                  offset:kv_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_latent_norm offset:0 atIndex:2];
      [encoder setBytes:&latent_dims length:sizeof(latent_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_mla_gemv(encoder, state, weights.query_b, kQueryBRows,
                      kQueryARows, state.mla_qa_norm, 0, state.mla_query, 0,
                      "query_b");
      encode_mla_gemv(encoder, state, weights.key_value_b, kKeyValueBRows,
                      kLatentWidth, state.mla_latent_norm, 0,
                      state.mla_expanded, 0, "key_value_b");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      struct PackDimsHost {
        std::uint32_t heads;
        std::uint32_t nope_dim;
        std::uint32_t rope_dim;
        std::uint32_t value_dim;
      };
      const PackDimsHost pack_dims{kHeads, 128, kRopeWidth, kValueDim};
      [encoder setComputePipelineState:state.mla_pack_pipeline];
      [encoder setBuffer:state.mla_expanded offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth) * sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_new_key offset:0 atIndex:2];
      [encoder setBuffer:state.mla_new_value offset:0 atIndex:3];
      [encoder setBytes:&pack_dims length:sizeof(pack_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(kHeads * kKeyDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];

      // Stage the packed row into the LIVE (or grown) slab at row S-1 —
      // the uncommitted staging slot, which is scratch until commit.
      id<MTLBlitCommandEncoder> place = [command blitCommandEncoder];
      require(place != nil, "MLA takeover staging blit encoder unavailable");
      const NSUInteger staging_row =
          static_cast<NSUInteger>(kv_length - 1);
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        [place copyFromBuffer:state.mla_new_key
                 sourceOffset:head * kKeyDim * sizeof(float)
                     toBuffer:tensor_buffer(key_states)
            destinationOffset:key_base + head * key_head_stride +
                              staging_row * kKeyDim * sizeof(float)
                         size:kKeyDim * sizeof(float)];
        [place copyFromBuffer:state.mla_new_value
                 sourceOffset:head * kValueDim * sizeof(float)
                     toBuffer:tensor_buffer(value_states)
            destinationOffset:value_base + head * value_head_stride +
                              staging_row * kValueDim * sizeof(float)
                         size:kValueDim * sizeof(float)];
      }
      [place endEncoding];

      encoder = [command computeCommandEncoder];
      require(encoder != nil,
              "MLA takeover attention encoder is unavailable");
      [encoder setComputePipelineState:state.mla_part_pipeline];
      [encoder setBuffer:state.mla_query offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(key_states)
                  offset:key_base
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(value_states)
                  offset:value_base
                 atIndex:2];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:3];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads * partitions, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:state.mla_combine_pipeline];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:0];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:1];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const std::uint32_t gate_elements = kGateWidth;
      [encoder setComputePipelineState:state.mla_gate_pipeline];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth + kRopeWidth) *
                         sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_gated offset:0 atIndex:2];
      [encoder setBytes:&gate_elements
                 length:sizeof(gate_elements)
                atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kGateWidth, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_mla_gemv(encoder, state, weights.output, kHiddenWidth,
                      kGateWidth, state.mla_gated, 0, tensor_buffer(output),
                      output_offset, "output");
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    const long wait_result = dispatch_semaphore_wait(
        ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
    if (wait_result != 0) {
      // The CB may still execute after a timeout that later resolves, and
      // unlike parity its writes hit the LIVE slab row (which the stock
      // fallback would rewrite and COMMIT) and PyTorch-pooled buffers the
      // fallback frees for immediate reuse. Fallback is only safe behind
      // a DEAD CB — drain it, accepting the stall (the MoE-tail timeout
      // paths make the same call for the same reason).
      [command waitUntilCompleted];
      fail("MLA takeover event timed out (command drained before fallback)");
    }
    if (command.error != nil) {
      fail(std::string("MLA takeover command failed: ") +
           command.error.localizedDescription.UTF8String);
    }

    ++state.mla_takeover_stats.taken;
    if (state.mla_takeover_stats.taken == 1 ||
        state.mla_takeover_stats.taken % 240 == 0) {
      std::fprintf(
          stderr, "[mla-loop] taken=%llu fallbacks=%llu\n",
          static_cast<unsigned long long>(state.mla_takeover_stats.taken),
          static_cast<unsigned long long>(
              state.mla_takeover_stats.fallbacks));
    }
    return true;
  } catch (const std::exception& error) {
    report_fallback_once(error.what());
    ++state.mla_takeover_stats.fallbacks;
    return false;
  } catch (...) {
    report_fallback_once("unknown exception");
    ++state.mla_takeover_stats.fallbacks;
    return false;
  }
}

LoopMlaTakeoverStats loop_mla_takeover_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.mla_takeover_stats;
}

namespace {
/* Step 4 chain-lite: everything a committed-but-unwaited MLA layer CB
 * needs for its later host wait and output publication. Owned
 * precommit -> collect/abandon; ARC keeps the ObjC handles alive. The
 * tensor members are STRONG REFS across the parked window — both the
 * CB's inputs (weights views, residual carriers, slab views incl. any
 * grown storage the shell allocated) and its fresh outputs. */
struct LoopMlaPendingState {
  std::uint32_t layer_index = 0;
  bool ready_consumed = false;
  id<MTLCommandBuffer> command = nil;
  dispatch_semaphore_t ready = nil;
  MlaWeights weights;
  MlaLinearWeight bundle_projection;
  at::Tensor hidden;
  at::Tensor anchors;
  at::Tensor score_weight;
  at::Tensor input_norm;
  at::Tensor key_states;
  at::Tensor value_states;
  at::Tensor normalized;
  at::Tensor output;
};

/* The chained CB writes the LIVE staging slot and caller-owned tensors,
 * so it must be provably DEAD before any fallback or teardown releases
 * them (the bd597d8 review lesson). Semaphore first; hard-drain on
 * timeout. */
void drain_mla_pending(LoopMlaPendingState& pending) noexcept {
  if (pending.ready != nil && !pending.ready_consumed) {
    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    if (dispatch_semaphore_wait(
            pending.ready,
            dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds)) != 0 &&
        pending.command != nil) {
      [pending.command waitUntilCompleted];
    }
    pending.ready_consumed = true;
  }
}
}  // namespace

struct LoopMlaPendingHandle {
  LoopMlaPendingState state;
};

LoopMlaPendingHandle* loop_mla_layer_precommit(
    const std::uint32_t layer_index, const at::Tensor& hidden,
    const at::Tensor& anchors, const at::Tensor& score_weight,
    const at::Tensor& input_norm, MlaWeights weights,
    const MlaInputBundle* input_bundle, const at::Tensor& key_states,
    const at::Tensor& value_states) {
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kBundleRows = 14400;
  constexpr std::uint32_t kQueryARows = 1536;
  constexpr std::uint32_t kLatentWidth = 512;
  constexpr std::uint32_t kRopeWidth = 64;
  constexpr std::uint32_t kGateWidth = 12288;
  constexpr std::uint32_t kQueryBRows = 18432;
  constexpr std::uint32_t kKeyValueBRows = 24576;
  constexpr std::uint32_t kHeads = 96;
  constexpr std::uint32_t kKeyDim = 192;
  constexpr std::uint32_t kValueDim = 128;
  constexpr std::uint32_t kMaxPartitions = 32;
  constexpr float kRmsEpsilon = 1.0e-5F;
  static std::atomic<bool> fallback_reported{false};
  const auto report_fallback_once = [&](const char* reason) {
    if (!fallback_reported.exchange(true)) {
      std::fprintf(stderr, "[mla-chain] first fallback (layer %u): %s\n",
                   layer_index, reason);
    }
  };

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_fallback_once("current MPS stream unavailable");
    return nullptr;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_fallback_once("current MPS device unavailable");
    ++state.mla_chain_stats.begin_fallbacks;
    return nullptr;
  }
  try {
    ensure_resources(state, device);

    require(input_bundle != nullptr &&
                input_bundle->query_a_rows == kQueryARows &&
                input_bundle->key_value_a_rows == kLatentWidth + kRopeWidth &&
                input_bundle->output_gate_rows == kGateWidth,
            "MLA chain requires the exact-K3 input bundle");

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("MLA chain tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    const NSUInteger hidden_offset = plain(hidden, kHiddenWidth, "hidden");
    const NSUInteger score_offset =
        plain(score_weight, kHiddenWidth, "score_weight");
    const NSUInteger input_norm_offset =
        plain(input_norm, kHiddenWidth, "input_norm");
    const NSUInteger qa_norm_offset =
        plain(weights.query_a_norm, kQueryARows, "query_a_norm");
    const NSUInteger kv_norm_offset =
        plain(weights.key_value_a_norm, kLatentWidth, "key_value_a_norm");

    std::uint32_t anchor_rows = 0;
    NSUInteger anchors_offset = 0;
    if (anchors.defined() && anchors.numel() != 0) {
      require(anchors.device().is_mps() &&
                  anchors.scalar_type() == at::kFloat &&
                  anchors.is_contiguous() && anchors.dim() == 3 &&
                  anchors.size(0) == 1 && anchors.size(1) >= 1 &&
                  anchors.size(1) <= 8 &&
                  anchors.size(2) == kHiddenWidth,
              "MLA chain anchors do not qualify");
      anchor_rows = static_cast<std::uint32_t>(anchors.size(1));
      anchors_offset = checked_byte_offset(
          anchors,
          static_cast<std::size_t>(anchor_rows) * kHiddenWidth *
              sizeof(float),
          "anchors");
    }

    require(key_states.defined() && key_states.dim() == 4 &&
                key_states.size(2) >= 1,
            "MLA chain key states view is invalid");
    const std::int64_t kv_length = key_states.size(2);
    const auto slab_view = [&](const at::Tensor& tensor,
                               const std::int64_t width, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat && tensor.dim() == 4 &&
                  tensor.size(0) == 1 && tensor.size(1) == kHeads &&
                  tensor.size(2) == kv_length && tensor.size(3) == width &&
                  tensor.stride(3) == 1 && tensor.stride(2) == width &&
                  tensor.stride(1) >= kv_length * width &&
                  tensor.stride(1) % width == 0,
              std::string("MLA chain slab view does not qualify: ") + name);
      const std::size_t span =
          (static_cast<std::size_t>(tensor.stride(1)) * (kHeads - 1) +
           static_cast<std::size_t>(kv_length) * width) *
          sizeof(float);
      return std::pair<NSUInteger, NSUInteger>(
          checked_byte_offset(tensor, span, name),
          static_cast<NSUInteger>(tensor.stride(1) * sizeof(float)));
    };
    const auto [key_base, key_head_stride] =
        slab_view(key_states, kKeyDim, "key states");
    const auto [value_base, value_head_stride] =
        slab_view(value_states, kValueDim, "value states");
    const std::int64_t slab_capacity =
        key_states.stride(1) / static_cast<std::int64_t>(kKeyDim);
    require(value_states.stride(1) ==
                slab_capacity * static_cast<std::int64_t>(kValueDim),
            "MLA chain key/value slab capacities disagree");

    at::Tensor normalized =
        at::empty({1, static_cast<std::int64_t>(kHiddenWidth)},
                  hidden.options());
    at::Tensor output =
        at::empty({1, 1, static_cast<std::int64_t>(kHiddenWidth)},
                  hidden.options());
    const NSUInteger normalized_offset =
        plain(normalized, kHiddenWidth, "normalized (fresh)");
    const NSUInteger output_offset = plain(output, kHiddenWidth, "output");

    const std::uint32_t partitions = std::min<std::uint32_t>(
        kMaxPartitions,
        std::max<std::uint32_t>(
            1, (static_cast<std::uint32_t>(kv_length) + 255) / 256));
    struct AttnDims {
      std::uint32_t kv_length;
      std::uint32_t capacity;
      std::uint32_t partitions;
      std::uint32_t chunk;
      float scale;
      std::uint32_t reserved0;
      std::uint32_t reserved1;
      std::uint32_t reserved2;
    } attn_dims{
        static_cast<std::uint32_t>(kv_length),
        static_cast<std::uint32_t>(slab_capacity),
        partitions,
        (static_cast<std::uint32_t>(kv_length) + partitions - 1) /
            partitions,
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(kKeyDim))),
        0, 0, 0};

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil, "MLA chain MPS command buffer unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil, "MLA chain root command buffer unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "MLA chain boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil, "MLA chain command buffer unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA chain encoder unavailable");

      // Attention head: residual mix (A>0) + input rmsnorm — the exact
      // encode loop_kda_layer_begin performs for KDA layers.
      id<MTLBuffer> norm_source = tensor_buffer(hidden);
      NSUInteger norm_source_offset = hidden_offset;
      if (anchor_rows != 0) {
        struct AttnResDimsHost {
          std::uint32_t A;
          std::uint32_t H;
        };
        const AttnResDimsHost mix_dims{anchor_rows, kHiddenWidth};
        [encoder setComputePipelineState:state.attn_res_mix_pipeline];
        [encoder setBuffer:tensor_buffer(anchors)
                    offset:anchors_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(hidden)
                    offset:hidden_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(score_weight)
                    offset:score_offset
                   atIndex:2];
        [encoder setBuffer:state.mla_mixed offset:0 atIndex:3];
        [encoder setBytes:&mix_dims length:sizeof(mix_dims) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        norm_source = state.mla_mixed;
        norm_source_offset = 0;
      }
      struct RmsDimsHost {
        std::uint32_t N;
        float eps;
      };
      const RmsDimsHost hidden_norm_dims{kHiddenWidth, kRmsEpsilon};
      [encoder setComputePipelineState:state.rmsnorm_pipeline];
      [encoder setBuffer:norm_source offset:norm_source_offset atIndex:0];
      [encoder setBuffer:tensor_buffer(input_norm)
                  offset:input_norm_offset
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(normalized)
                  offset:normalized_offset
                 atIndex:2];
      [encoder setBytes:&hidden_norm_dims
                 length:sizeof(hidden_norm_dims)
                atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // The validated ten-op chain (bd597d8), reading the CB-computed
      // normalized row. Kept textually in sync with loop_mla_takeover_run
      // — the chained-parity gate re-verifies the pairing end-to-end.
      encode_mla_gemv(encoder, state, input_bundle->projection, kBundleRows,
                      kHiddenWidth, tensor_buffer(normalized),
                      normalized_offset, state.mla_bundle_out, 0,
                      "input_bundle");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const RmsDimsHost qa_dims{kQueryARows, kRmsEpsilon};
      [encoder setComputePipelineState:state.rmsnorm_pipeline];
      [encoder setBuffer:state.mla_bundle_out offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.query_a_norm)
                  offset:qa_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_qa_norm offset:0 atIndex:2];
      [encoder setBytes:&qa_dims length:sizeof(qa_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      const RmsDimsHost latent_dims{kLatentWidth, kRmsEpsilon};
      [encoder setBuffer:state.mla_bundle_out
                  offset:kQueryARows * sizeof(float)
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights.key_value_a_norm)
                  offset:kv_norm_offset
                 atIndex:1];
      [encoder setBuffer:state.mla_latent_norm offset:0 atIndex:2];
      [encoder setBytes:&latent_dims length:sizeof(latent_dims) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_mla_gemv(encoder, state, weights.query_b, kQueryBRows,
                      kQueryARows, state.mla_qa_norm, 0, state.mla_query, 0,
                      "query_b");
      encode_mla_gemv(encoder, state, weights.key_value_b, kKeyValueBRows,
                      kLatentWidth, state.mla_latent_norm, 0,
                      state.mla_expanded, 0, "key_value_b");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      struct PackDimsHost {
        std::uint32_t heads;
        std::uint32_t nope_dim;
        std::uint32_t rope_dim;
        std::uint32_t value_dim;
      };
      const PackDimsHost pack_dims{kHeads, 128, kRopeWidth, kValueDim};
      [encoder setComputePipelineState:state.mla_pack_pipeline];
      [encoder setBuffer:state.mla_expanded offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth) * sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_new_key offset:0 atIndex:2];
      [encoder setBuffer:state.mla_new_value offset:0 atIndex:3];
      [encoder setBytes:&pack_dims length:sizeof(pack_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(kHeads * kKeyDim, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];

      id<MTLBlitCommandEncoder> place = [command blitCommandEncoder];
      require(place != nil, "MLA chain staging blit encoder unavailable");
      const NSUInteger staging_row = static_cast<NSUInteger>(kv_length - 1);
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        [place copyFromBuffer:state.mla_new_key
                 sourceOffset:head * kKeyDim * sizeof(float)
                     toBuffer:tensor_buffer(key_states)
            destinationOffset:key_base + head * key_head_stride +
                              staging_row * kKeyDim * sizeof(float)
                         size:kKeyDim * sizeof(float)];
        [place copyFromBuffer:state.mla_new_value
                 sourceOffset:head * kValueDim * sizeof(float)
                     toBuffer:tensor_buffer(value_states)
            destinationOffset:value_base + head * value_head_stride +
                              staging_row * kValueDim * sizeof(float)
                         size:kValueDim * sizeof(float)];
      }
      [place endEncoding];

      encoder = [command computeCommandEncoder];
      require(encoder != nil, "MLA chain attention encoder unavailable");
      [encoder setComputePipelineState:state.mla_part_pipeline];
      [encoder setBuffer:state.mla_query offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(key_states)
                  offset:key_base
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(value_states)
                  offset:value_base
                 atIndex:2];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:3];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads * partitions, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:state.mla_combine_pipeline];
      [encoder setBuffer:state.mla_partials offset:0 atIndex:0];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:1];
      [encoder setBytes:&attn_dims length:sizeof(attn_dims) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(kHeads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const std::uint32_t gate_elements = kGateWidth;
      [encoder setComputePipelineState:state.mla_gate_pipeline];
      [encoder setBuffer:state.mla_attn_out offset:0 atIndex:0];
      [encoder setBuffer:state.mla_bundle_out
                  offset:(kQueryARows + kLatentWidth + kRopeWidth) *
                         sizeof(float)
                 atIndex:1];
      [encoder setBuffer:state.mla_gated offset:0 atIndex:2];
      [encoder setBytes:&gate_elements
                 length:sizeof(gate_elements)
                atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(kGateWidth, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_mla_gemv(encoder, state, weights.output, kHiddenWidth,
                      kGateWidth, state.mla_gated, 0, tensor_buffer(output),
                      output_offset, "output");
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      [command commit];
    }

    auto handle = std::make_unique<LoopMlaPendingHandle>();
    handle->state.layer_index = layer_index;
    handle->state.command = command;
    handle->state.ready = ready;
    handle->state.weights = std::move(weights);
    handle->state.bundle_projection = input_bundle->projection;
    handle->state.hidden = hidden;
    handle->state.anchors = anchors;
    handle->state.score_weight = score_weight;
    handle->state.input_norm = input_norm;
    handle->state.key_states = key_states;
    handle->state.value_states = value_states;
    handle->state.normalized = std::move(normalized);
    handle->state.output = std::move(output);
    ++state.mla_chain_stats.chained;
    return handle.release();
  } catch (const std::exception& error) {
    report_fallback_once(error.what());
    ++state.mla_chain_stats.begin_fallbacks;
    return nullptr;
  } catch (...) {
    report_fallback_once("unknown exception");
    ++state.mla_chain_stats.begin_fallbacks;
    return nullptr;
  }
}

bool loop_mla_layer_collect(LoopMlaPendingHandle* handle,
                            const std::uint32_t layer_index,
                            at::Tensor& output_out) {
  if (handle == nullptr) {
    return false;
  }
  std::unique_ptr<LoopMlaPendingHandle> owned(handle);
  LoopEncoderState& state = loop_state();
  if (owned->state.layer_index != layer_index) {
    drain_mla_pending(owned->state);
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.mla_chain_stats.abandoned;
    return false;
  }
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = prep_timed_semaphore_wait(
      owned->state.ready,
      dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
      owned->state.command);
  if (wait_result != 0) {
    if (owned->state.command != nil) {
      [owned->state.command waitUntilCompleted];
    }
    owned->state.ready_consumed = true;
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.mla_chain_stats.abandoned;
    return false;
  }
  owned->state.ready_consumed = true;
  if (owned->state.command != nil && owned->state.command.error != nil) {
    std::lock_guard<std::mutex> lock(state.mutex);
    ++state.mla_chain_stats.abandoned;
    return false;
  }
  output_out = std::move(owned->state.output);
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.mla_chain_stats.collected;
  if (state.mla_chain_stats.collected == 1 ||
      state.mla_chain_stats.collected % 240 == 0) {
    std::fprintf(
        stderr,
        "[mla-chain] chained=%llu collected=%llu abandoned=%llu "
        "begin_fallbacks=%llu\n",
        static_cast<unsigned long long>(state.mla_chain_stats.chained),
        static_cast<unsigned long long>(state.mla_chain_stats.collected),
        static_cast<unsigned long long>(state.mla_chain_stats.abandoned),
        static_cast<unsigned long long>(
            state.mla_chain_stats.begin_fallbacks));
  }
  return true;
}

void loop_mla_layer_abandon(LoopMlaPendingHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }
  std::unique_ptr<LoopMlaPendingHandle> owned(handle);
  drain_mla_pending(owned->state);
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  ++state.mla_chain_stats.abandoned;
}

LoopMlaChainStats loop_mla_chain_stats() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.mla_chain_stats;
}

namespace {

/* Step 6 milestone (i): everything a committed-but-unwaited KDA layer CB
 * needs for its later host wait, probe accounting, and output publication.
 * Owned begin -> finish; ARC keeps the ObjC handles alive. */
struct LoopKdaPendingState {
  std::uint32_t layer_index = 0;
  bool shared_fence = false;
  bool prefix_defined = false;
  std::uint32_t mlp_anchor_rows = 0;
  double commit_host_ms = 0.0;
  double commit_media_time = 0.0;
  double loop_commit_media = 0.0;
  /* The done semaphore is a one-shot: whoever waits first consumes the
   * signal. The arena barrier and finish coordinate through this flag
   * (single decode thread). */
  bool ready_consumed = false;
  std::chrono::steady_clock::time_point wait_started{};
  id<MTLCommandBuffer> command = nil;
  id<MTLCommandBuffer> boundary_root = nil;
  dispatch_semaphore_t ready = nil;
  at::Tensor normalized;
  at::Tensor mlp_normalized;
  at::Tensor mlp_mixed;
  at::Tensor prefix_tensor;
  /// The published next-anchor slab: the caller's tensor, or the CB-built
  /// [anchors | hidden] concatenation under K3_LOOP_CAT boundary layers.
  at::Tensor next_anchors;
  at::Tensor output;
  at::Tensor next_conv_q;
  at::Tensor next_conv_k;
  at::Tensor next_conv_v;
  at::Tensor next_recurrent;
};

namespace {
/* K3_CB_FUSION=1 shared state: the just-encoded MoE tail CB stays OPEN so
 * the successor layer's KDA and route encoders can ride in the same CB
 * (one queue admission instead of three). Mid-CB shared-event signals keep
 * every host wait at its current stream position; loop_fused_flush()
 * (defined next to the deferred-tail machinery below) performs the single
 * commit and the synchronous tail wait inside the same finish ABI call, so
 * the expert-byte lease and every ATen consumer keep today's contract —
 * no Rust-side parking, unlike K3_TAIL_ASYNC. Touched only from the one
 * decode thread, like the rest of the loop statics. */
id<MTLCommandBuffer> g_fused_open_command = nil;
dispatch_semaphore_t g_fused_tail_ready = nil;
/* A mid-encode failure left an un-ended encoder on the open CB: it can
 * never legally be committed. Flush turns this into the sticky poison
 * error instead of committing a malformed CB. */
bool g_fused_broken = false;
}  // namespace

bool loop_kda_layer_begin(const std::uint32_t layer_index,
                          const at::Tensor& hidden, const at::Tensor& anchors,
                          const at::Tensor& score_weight,
                          const at::Tensor& input_norm,
                          const at::Tensor& prefix_sum,
                          const at::Tensor& next_anchors,
                          const at::Tensor& mlp_score_weight,
                          const at::Tensor& post_attention_norm,
                          const KdaWeights& weights, const KdaState& state_in,
                          const bool cat_next_anchors_in_cb,
                          LoopKdaPendingState& pending) {
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kProjWidth = 12288;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[kda-loop] first fallback: %s\n", reason);
    }
  };

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return false;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    return false;
  }
  // Set once this call starts encoding into the OPEN fused CB; a later
  // throw then marks the CB broken (see the catch blocks) instead of
  // leaving a half-encoded buffer for the flush to commit.
  bool fused_encode_started = false;
  try {
    ensure_resources(state, device);

    // Post-verify commits can publish conv states as non-contiguous views
    // (measured: exactly the verify-boundary tokens) — restore contiguity
    // with cheap ATen copies encoded before the fence signal. Any such
    // fresh copy makes the ATen fence load-bearing for this layer, so the
    // shared-boundary skip below must stand down.
    bool fresh_aten_copies = false;
    const auto contiguous_mps = [&](const at::Tensor& tensor) {
      if (tensor.is_contiguous()) {
        return tensor;
      }
      fresh_aten_copies = true;
      return tensor.contiguous();
    };
    const at::Tensor conv_q = contiguous_mps(state_in.query_convolution);
    const at::Tensor conv_k = contiguous_mps(state_in.key_convolution);
    const at::Tensor conv_v = contiguous_mps(state_in.value_convolution);
    const at::Tensor recurrent = contiguous_mps(state_in.recurrent);

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("KDA loop tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    // 3b head inputs: the raw hidden, the anchor slab (A = 0..8 rows), the
    // bind-time-cached score weight, and the input norm weight.
    const NSUInteger raw_hidden_offset =
        plain(hidden, kHiddenWidth, "hidden");
    require(anchors.defined() && anchors.device().is_mps() &&
                anchors.scalar_type() == at::kFloat &&
                anchors.is_contiguous() && anchors.dim() == 3 &&
                anchors.size(0) == 1 && anchors.size(2) == kHiddenWidth &&
                anchors.size(1) >= 0 && anchors.size(1) <= 8,
            "KDA loop anchors do not qualify");
    const std::uint32_t anchor_rows =
        static_cast<std::uint32_t>(anchors.size(1));
    const NSUInteger anchors_offset =
        anchor_rows != 0
            ? checked_byte_offset(
                  anchors,
                  static_cast<std::size_t>(anchor_rows) * kHiddenWidth *
                      sizeof(float),
                  "anchors")
            : 0;
    const NSUInteger score_weight_offset =
        plain(score_weight, kHiddenWidth, "score_weight");
    const NSUInteger input_norm_offset =
        plain(input_norm, kHiddenWidth, "input_norm");

    // The post-norm hidden is a loop OUTPUT the downstream MLP head
    // consumes, and simultaneously the in-CB input of the six projections.
    const auto mps_float_options =
        at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
    at::Tensor normalized = at::empty({1, kHiddenWidth}, mps_float_options);
    const NSUInteger hidden_offset =
        plain(normalized, kHiddenWidth, "normalized");

    // 3c MLP-head inputs and outputs.
    require(next_anchors.defined() && next_anchors.device().is_mps() &&
                next_anchors.scalar_type() == at::kFloat &&
                next_anchors.is_contiguous() && next_anchors.dim() == 3 &&
                next_anchors.size(0) == 1 &&
                next_anchors.size(2) == kHiddenWidth &&
                next_anchors.size(1) >= 0 &&
                next_anchors.size(1) <=
                    (cat_next_anchors_in_cb ? 7 : 8),
            "KDA loop next_anchors do not qualify");
    // K3_LOOP_CAT boundary layers: build [next_anchors | hidden] on the
    // loop queue itself (two blit copies ahead of the compute encoder).
    // The append source is this layer's own `hidden`, produced by the
    // tail CB the loop queue already ordered this CB behind — the ATen
    // at::cat, its private fence, and its uncommitted-fused-CB read
    // hazard all disappear.
    const std::uint32_t source_anchor_rows =
        static_cast<std::uint32_t>(next_anchors.size(1));
    at::Tensor cat_slab;
    if (cat_next_anchors_in_cb) {
      cat_slab = at::empty(
          {1, static_cast<std::int64_t>(source_anchor_rows) + 1,
           static_cast<std::int64_t>(kHiddenWidth)},
          at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
    }
    const at::Tensor& mlp_anchor_slab =
        cat_next_anchors_in_cb ? cat_slab : next_anchors;
    const std::uint32_t mlp_anchor_rows =
        cat_next_anchors_in_cb ? source_anchor_rows + 1
                               : source_anchor_rows;
    const NSUInteger source_anchors_offset =
        (cat_next_anchors_in_cb && source_anchor_rows != 0)
            ? checked_byte_offset(
                  next_anchors,
                  static_cast<std::size_t>(source_anchor_rows) *
                      kHiddenWidth * sizeof(float),
                  "cat source anchors")
            : 0;
    const NSUInteger next_anchors_offset =
        mlp_anchor_rows != 0
            ? checked_byte_offset(
                  mlp_anchor_slab,
                  static_cast<std::size_t>(mlp_anchor_rows) * kHiddenWidth *
                      sizeof(float),
                  "next_anchors")
            : 0;
    const NSUInteger mlp_score_weight_offset =
        plain(mlp_score_weight, kHiddenWidth, "mlp_score_weight");
    const NSUInteger post_norm_offset =
        plain(post_attention_norm, kHiddenWidth, "post_attention_norm");
    NSUInteger prefix_sum_offset = 0;
    if (prefix_sum.defined()) {
      prefix_sum_offset = plain(prefix_sum, kHiddenWidth, "prefix_sum");
    }
    at::Tensor mlp_mixed = at::empty({1, kHiddenWidth}, mps_float_options);
    at::Tensor mlp_normalized =
        at::empty({1, kHiddenWidth}, mps_float_options);
    const NSUInteger mlp_mixed_offset =
        plain(mlp_mixed, kHiddenWidth, "mlp mixed");
    const NSUInteger mlp_normalized_offset =
        plain(mlp_normalized, kHiddenWidth, "mlp normalized");
    at::Tensor prefix_tensor;
    NSUInteger prefix_offset = 0;
    if (prefix_sum.defined()) {
      prefix_tensor = at::empty({1, kHiddenWidth}, mps_float_options);
      prefix_offset = plain(prefix_tensor, kHiddenWidth, "prefix");
    }
    const NSUInteger a_log_offset = plain(weights.a_log, 128, "a_log");
    const NSUInteger dt_bias_offset =
        plain(weights.dt_bias, kProjWidth, "dt_bias");
    const NSUInteger o_norm_offset =
        plain(weights.output_norm, 128, "output_norm");
    const NSUInteger convw_q_offset = plain(
        weights.query_convolution, kProjWidth * 4, "query_convolution");
    const NSUInteger convw_k_offset =
        plain(weights.key_convolution, kProjWidth * 4, "key_convolution");
    const NSUInteger convw_v_offset = plain(
        weights.value_convolution, kProjWidth * 4, "value_convolution");
    const NSUInteger convin_q_offset =
        plain(conv_q, kProjWidth * 4, "query conv state");
    const NSUInteger convin_k_offset =
        plain(conv_k, kProjWidth * 4, "key conv state");
    const NSUInteger convin_v_offset =
        plain(conv_v, kProjWidth * 4, "value conv state");
    const NSUInteger s_in_offset =
        plain(recurrent, static_cast<std::int64_t>(96) * 128 * 128,
              "recurrent state");

    // Freshly allocated outputs: the loop queue writes them, the caller's
    // staging/commit contract consumes them exactly like stock results.
    const auto mps_float = at::TensorOptions()
                               .dtype(at::kFloat)
                               .device(at::kMPS);
    at::Tensor output = at::empty({1, kHiddenWidth}, mps_float);
    at::Tensor next_conv_q = at::empty({1, kProjWidth, 4}, mps_float);
    at::Tensor next_conv_k = at::empty({1, kProjWidth, 4}, mps_float);
    at::Tensor next_conv_v = at::empty({1, kProjWidth, 4}, mps_float);
    at::Tensor next_recurrent = at::empty({1, 96, 128, 128}, mps_float);
    const NSUInteger out_offset =
        plain(output, kHiddenWidth, "loop output");
    const NSUInteger next_q_offset =
        plain(next_conv_q, kProjWidth * 4, "next query conv");
    const NSUInteger next_k_offset =
        plain(next_conv_k, kProjWidth * 4, "next key conv");
    const NSUInteger next_v_offset =
        plain(next_conv_v, kProjWidth * 4, "next value conv");
    const NSUInteger next_s_offset =
        plain(next_recurrent, static_cast<std::int64_t>(96) * 128 * 128,
              "next recurrent");

    // Shared-boundary experiment: when every input of this CB is already the
    // product of a host-waited loop CB or long-completed ATen work (previous
    // MoE tail ran on the loop, and no FRESH anchor at::cat was encoded for
    // this layer), the ATen fence is vestigial — the sync-E drain at the
    // previous final_tile left the ATen stream empty. Fresh-cat layers
    // (1 in kLoopResidualBlock) and any layer after a stock completion keep
    // the private fence. Consume the one-shot marker unconditionally.
    const bool fresh_anchor_cat =
        g_fresh_anchor_cat.exchange(false, std::memory_order_acq_rel);
    // int8-form weights read no fp32 arena: the dequant marker is neither
    // an ordering requirement for this CB nor ours to consume — a later
    // fp32-form reader still needs it.
    const bool int8_form = weights.query_projection.scale.defined();
    const bool fresh_dequant =
        int8_form
            ? false
            : g_fresh_dequant.exchange(false, std::memory_order_acq_rel);
    const bool shared_fence = shared_boundary_enabled() && !fresh_anchor_cat &&
                              !fresh_dequant && !fresh_aten_copies &&
                              g_loop_chain_safe.load(std::memory_order_acquire);
    const std::uint64_t boundary_value =
        shared_fence ? 0 : take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    __block id<MTLCommandBuffer> boundary_root = nil;
    const auto commit_started = std::chrono::steady_clock::now();
    if (!shared_fence) {
      // This path encodes on the ATen stream, whose subsequent chunks may
      // consume the previous layer's output; restore the synchronous tail
      // contract first.
      loop_moe_tail_drain();
      at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
          stream->endKernelCoalescing();
          MPSCommandBuffer* command = stream->commandBuffer();
          require(command != nil,
                  "KDA loop MPS command buffer is unavailable");
          id<MTLCommandBuffer> root = command.rootCommandBuffer;
          require(root != nil, "KDA loop root command buffer is unavailable");
          [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];
          boundary_root = root;
          gpu_timeline_note(root, "aten-root");
        }
      });
      require(try_commit_mps_stream_for_route(),
              "KDA loop boundary commit failed");
    }
    const double commit_host_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - commit_started)
            .count();
    // Same timebase as MTLCommandBuffer kernel/GPU timestamps (host uptime
    // seconds): lets the probe split commit -> driver-sched -> GPU-start.
    const double commit_media_time = [] {
      static mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        return info;
      }();
      return static_cast<double>(mach_absolute_time()) * timebase.numer /
             timebase.denom / 1e9;
    }();

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    const auto wait_started = std::chrono::steady_clock::now();
    // K3_CB_FUSION=1 ride: encode this layer's KDA work into the still-open
    // fused tail CB (one admission for tail+KDA+route). Only legal on the
    // shared-fence path — every private-fence layer already flushed the
    // open CB through the loop_moe_tail_drain() above, so the stash is nil
    // here and the stock own-CB path runs.
    const bool fused_ride = loop_cb_fusion_enabled() && shared_fence &&
                            g_fused_open_command != nil && !g_fused_broken;
    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      if (fused_ride) {
        command = g_fused_open_command;
        fused_encode_started = true;
      } else {
        command = [state.queue commandBuffer];
      }
      require(command != nil, "KDA loop command buffer is unavailable");
      if (!shared_fence) {
        [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                              value:boundary_value];
      }
      if (cat_next_anchors_in_cb) {
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        require(blit != nil, "KDA loop cat blit encoder is unavailable");
        const NSUInteger row_bytes =
            static_cast<NSUInteger>(kHiddenWidth) * sizeof(float);
        if (source_anchor_rows != 0) {
          [blit copyFromBuffer:tensor_buffer(next_anchors)
                  sourceOffset:source_anchors_offset
                      toBuffer:tensor_buffer(cat_slab)
             destinationOffset:next_anchors_offset
                          size:static_cast<NSUInteger>(source_anchor_rows) *
                               row_bytes];
        }
        [blit copyFromBuffer:tensor_buffer(hidden)
                sourceOffset:raw_hidden_offset
                    toBuffer:tensor_buffer(cat_slab)
           destinationOffset:next_anchors_offset +
                             static_cast<NSUInteger>(source_anchor_rows) *
                                 row_bytes
                        size:row_bytes];
        [blit endEncoding];
      }
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "KDA loop encoder is unavailable");

      // 3b head: AttnRes anchor mix (when anchors exist) and the input
      // RMSNorm run in the same command buffer, ahead of the projections —
      // the host wait no longer covers any ATen-encoded prefix math.
      id<MTLBuffer> norm_source_buffer = tensor_buffer(hidden);
      NSUInteger norm_source_offset = raw_hidden_offset;
      if (anchor_rows != 0) {
        struct AttnResDimsHost {
          std::uint32_t A;
          std::uint32_t H;
        };
        const AttnResDimsHost mix_dims{anchor_rows, kHiddenWidth};
        [encoder setComputePipelineState:state.attn_res_mix_pipeline];
        [encoder setBuffer:tensor_buffer(anchors)
                    offset:anchors_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(hidden)
                    offset:raw_hidden_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(score_weight)
                    offset:score_weight_offset
                   atIndex:2];
        [encoder setBuffer:state.kda_mixed offset:0 atIndex:3];
        [encoder setBytes:&mix_dims length:sizeof(mix_dims) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        norm_source_buffer = state.kda_mixed;
        norm_source_offset = 0;
      }
      {
        struct RmsDimsHost {
          std::uint32_t N;
          float eps;
        };
        const RmsDimsHost rms_dims{kHiddenWidth, 1e-5F};
        [encoder setComputePipelineState:state.rmsnorm_pipeline];
        [encoder setBuffer:norm_source_buffer
                    offset:norm_source_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(input_norm)
                    offset:input_norm_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(normalized)
                    offset:hidden_offset
                   atIndex:2];
        [encoder setBytes:&rms_dims length:sizeof(rms_dims) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      encode_kda_projection(encoder, state, weights.query_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, 0, "query_projection");
      encode_kda_projection(encoder, state, weights.key_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, kProjWidth * sizeof(float),
                            "key_projection");
      encode_kda_projection(encoder, state, weights.value_projection,
                            kProjWidth, kHiddenWidth,
                            tensor_buffer(normalized), hidden_offset,
                            state.kda_proj, 2 * kProjWidth * sizeof(float),
                            "value_projection");
      encode_kda_projection(encoder, state,
                            weights.recurrent_gate_projection, kProjWidth,
                            kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            3 * kProjWidth * sizeof(float),
                            "recurrent_gate_projection");
      encode_kda_projection(encoder, state, weights.feature_a_projection,
                            128, kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            4 * kProjWidth * sizeof(float),
                            "feature_a_projection");
      encode_kda_projection(encoder, state, weights.beta_projection, 96,
                            kHiddenWidth, tensor_buffer(normalized),
                            hidden_offset, state.kda_proj,
                            (4 * kProjWidth + 128) * sizeof(float),
                            "beta_projection");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encode_kda_projection(encoder, state, weights.feature_b_projection,
                            kProjWidth, 128, state.kda_proj,
                            4 * kProjWidth * sizeof(float), state.kda_fb, 0,
                            "feature_b_projection");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      [encoder setComputePipelineState:state.kda_core_pipeline];
      [encoder setBuffer:state.kda_proj offset:0 atIndex:0];
      [encoder setBuffer:state.kda_fb offset:0 atIndex:1];
      [encoder setBuffer:tensor_buffer(weights.query_convolution)
                  offset:convw_q_offset
                 atIndex:2];
      [encoder setBuffer:tensor_buffer(weights.key_convolution)
                  offset:convw_k_offset
                 atIndex:3];
      [encoder setBuffer:tensor_buffer(weights.value_convolution)
                  offset:convw_v_offset
                 atIndex:4];
      [encoder setBuffer:tensor_buffer(conv_q)
                  offset:convin_q_offset
                 atIndex:5];
      [encoder setBuffer:tensor_buffer(conv_k)
                  offset:convin_k_offset
                 atIndex:6];
      [encoder setBuffer:tensor_buffer(conv_v)
                  offset:convin_v_offset
                 atIndex:7];
      [encoder setBuffer:tensor_buffer(next_conv_q)
                  offset:next_q_offset
                 atIndex:8];
      [encoder setBuffer:tensor_buffer(next_conv_k)
                  offset:next_k_offset
                 atIndex:9];
      [encoder setBuffer:tensor_buffer(next_conv_v)
                  offset:next_v_offset
                 atIndex:10];
      [encoder setBuffer:tensor_buffer(weights.a_log)
                  offset:a_log_offset
                 atIndex:11];
      [encoder setBuffer:tensor_buffer(weights.dt_bias)
                  offset:dt_bias_offset
                 atIndex:12];
      [encoder setBuffer:tensor_buffer(weights.output_norm)
                  offset:o_norm_offset
                 atIndex:13];
      [encoder setBuffer:tensor_buffer(recurrent)
                  offset:s_in_offset
                 atIndex:14];
      [encoder setBuffer:tensor_buffer(next_recurrent)
                  offset:next_s_offset
                 atIndex:15];
      [encoder setBuffer:state.kda_core_out offset:0 atIndex:16];
      [encoder dispatchThreadgroups:MTLSizeMake(96, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      encode_kda_projection(encoder, state, weights.output_projection,
                            kHiddenWidth, kProjWidth, state.kda_core_out, 0,
                            tensor_buffer(output), out_offset,
                            "output_projection");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

      // 3c MLP head in the same command buffer: residual add (skipped at
      // boundary layers — prefix aliases the attention output), the second
      // AttnRes mix over the post-append anchors, and the post-attention
      // RMSNorm producing the router input.
      id<MTLBuffer> prefix_buffer = tensor_buffer(output);
      NSUInteger prefix_read_offset = out_offset;
      if (prefix_sum.defined()) {
        const std::uint32_t add_elements = kHiddenWidth;
        [encoder setComputePipelineState:state.add_pipeline];
        [encoder setBuffer:tensor_buffer(prefix_sum)
                    offset:prefix_sum_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(output)
                    offset:out_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(prefix_tensor)
                    offset:prefix_offset
                   atIndex:2];
        [encoder setBytes:&add_elements
                   length:sizeof(add_elements)
                  atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(kHiddenWidth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        prefix_buffer = tensor_buffer(prefix_tensor);
        prefix_read_offset = prefix_offset;
      }
      id<MTLBuffer> mlp_norm_source = prefix_buffer;
      NSUInteger mlp_norm_source_offset = prefix_read_offset;
      if (mlp_anchor_rows != 0) {
        struct AttnResDimsHost {
          std::uint32_t A;
          std::uint32_t H;
        };
        const AttnResDimsHost mix_dims{mlp_anchor_rows, kHiddenWidth};
        [encoder setComputePipelineState:state.attn_res_mix_pipeline];
        [encoder setBuffer:tensor_buffer(mlp_anchor_slab)
                    offset:next_anchors_offset
                   atIndex:0];
        [encoder setBuffer:prefix_buffer
                    offset:prefix_read_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(mlp_score_weight)
                    offset:mlp_score_weight_offset
                   atIndex:2];
        [encoder setBuffer:tensor_buffer(mlp_mixed)
                    offset:mlp_mixed_offset
                   atIndex:3];
        [encoder setBytes:&mix_dims length:sizeof(mix_dims) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        mlp_norm_source = tensor_buffer(mlp_mixed);
        mlp_norm_source_offset = mlp_mixed_offset;
      }
      {
        struct RmsDimsHost {
          std::uint32_t N;
          float eps;
        };
        const RmsDimsHost rms_dims{kHiddenWidth, 1e-5F};
        [encoder setComputePipelineState:state.rmsnorm_pipeline];
        [encoder setBuffer:mlp_norm_source
                    offset:mlp_norm_source_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(post_attention_norm)
                    offset:post_norm_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(mlp_normalized)
                    offset:mlp_normalized_offset
                   atIndex:2];
        [encoder setBytes:&rms_dims length:sizeof(rms_dims) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      if (!fused_ride) {
        gpu_timeline_note(command, "kda");
        [command commit];
      }
      // Fused ride: no commit here — the open CB (already timeline-noted
      // as "fused") is committed once by loop_fused_flush(); the mid-CB
      // done signal above still fires this layer's ready semaphore at the
      // same stream position an own-CB commit would have.
    }
    const double loop_commit_media = host_media_seconds();

    pending.layer_index = layer_index;
    pending.shared_fence = shared_fence;
    pending.prefix_defined = prefix_sum.defined();
    pending.mlp_anchor_rows = mlp_anchor_rows;
    pending.commit_host_ms = commit_host_ms;
    pending.commit_media_time = commit_media_time;
    pending.loop_commit_media = loop_commit_media;
    pending.wait_started = wait_started;
    pending.command = command;
    pending.boundary_root = boundary_root;
    pending.ready = ready;
    pending.normalized = std::move(normalized);
    pending.mlp_normalized = std::move(mlp_normalized);
    pending.mlp_mixed = std::move(mlp_mixed);
    pending.prefix_tensor = std::move(prefix_tensor);
    pending.next_anchors =
        cat_next_anchors_in_cb ? std::move(cat_slab) : next_anchors;
    pending.output = std::move(output);
    pending.next_conv_q = std::move(next_conv_q);
    pending.next_conv_k = std::move(next_conv_k);
    pending.next_conv_v = std::move(next_conv_v);
    pending.next_recurrent = std::move(next_recurrent);
    return true;
  } catch (const std::exception& error) {
    if (fused_encode_started) {
      g_fused_broken = true;
    }
    report_skip_once(error.what());
    return false;
  } catch (...) {
    if (fused_encode_started) {
      g_fused_broken = true;
    }
    report_skip_once("unknown exception");
    return false;
  }
}

bool loop_kda_layer_finish(LoopKdaPendingState& pending,
                           at::Tensor& normalized_out,
                           at::Tensor& mlp_normalized_out,
                           at::Tensor& lookahead_out, at::Tensor& prefix_out,
                           KdaDecodeResult& result) {
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[kda-loop-finish] first fallback: %s\n", reason);
    }
  };
  static const bool probing = [] {
    const char* value = std::getenv("K3_ROUTE_SYNC_PROBE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  try {
    const std::uint32_t layer_index = pending.layer_index;
    const bool shared_fence = pending.shared_fence;
    const std::uint32_t mlp_anchor_rows = pending.mlp_anchor_rows;
    const double commit_host_ms = pending.commit_host_ms;
    const double commit_media_time = pending.commit_media_time;
    const double loop_commit_media = pending.loop_commit_media;
    const auto wait_started = pending.wait_started;
    id<MTLCommandBuffer> command = pending.command;
    id<MTLCommandBuffer> boundary_root = pending.boundary_root;
    dispatch_semaphore_t ready = pending.ready;
    at::Tensor normalized = std::move(pending.normalized);
    at::Tensor mlp_normalized = std::move(pending.mlp_normalized);
    at::Tensor mlp_mixed = std::move(pending.mlp_mixed);
    at::Tensor prefix_tensor = std::move(pending.prefix_tensor);
    at::Tensor output = std::move(pending.output);
    at::Tensor next_conv_q = std::move(pending.next_conv_q);
    at::Tensor next_conv_k = std::move(pending.next_conv_k);
    at::Tensor next_conv_v = std::move(pending.next_conv_v);
    at::Tensor next_recurrent = std::move(pending.next_recurrent);
    pending.command = nil;
    pending.boundary_root = nil;
    pending.ready = nil;

    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    long wait_result = 0;
    if (!pending.ready_consumed) {
      wait_result = prep_timed_semaphore_wait(
          ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
          command);
      pending.ready_consumed = true;
    }
    const double wait_done_media = host_media_seconds();
    require(wait_result == 0, "KDA loop event timed out");
    if (command.error != nil) {
      fail(std::string("KDA loop command failed: ") +
           command.error.localizedDescription.UTF8String);
    }
    if (probing) {
      const double wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - wait_started)
              .count();
      static double wait_accumulator_ms = 0.0;
      static std::uint64_t wait_calls = 0;
      static double wait_min_ms = 1e30;
      static double wait_max_ms = 0.0;
      static const bool tracing = [] {
        const char* value = std::getenv("K3_KDA_LOOP_TRACE");
        return value != nullptr && std::strcmp(value, "1") == 0;
      }();
      if (tracing && wait_calls < 92) {
        std::fprintf(stderr, "[kda-loop-trace] layer=%u wait=%.3fms\n",
                     layer_index, wait_ms);
      }
      wait_accumulator_ms += wait_ms;
      wait_min_ms = std::min(wait_min_ms, wait_ms);
      wait_max_ms = std::max(wait_max_ms, wait_ms);
      ++wait_calls;
      // Anatomy: GPU-domain segments of the boundary. root busy = the ATen
      // chunk carrying the signal actually executing; fence gap = scheduling
      // hole between its end and our loop CB starting; loop busy = the KDA
      // kernel chain itself. Large residual (wait - these) = host encode or
      // root scheduling delay behind earlier uncommitted work.
      static double commit_acc = 0.0, root_acc = 0.0, gap_acc = 0.0,
                    busy_acc = 0.0, sched_acc = 0.0, kstart_acc = 0.0;
      static std::uint64_t shared_count = 0;
      static double adm_acc = 0.0, sgpu_acc = 0.0, lst_acc = 0.0;
      if (shared_fence) {
        ++shared_count;
        [command waitUntilCompleted];
        adm_acc += (command.GPUStartTime - loop_commit_media) * 1e3;
        sgpu_acc += (command.GPUEndTime - command.GPUStartTime) * 1e3;
        lst_acc += (wait_done_media - command.GPUEndTime) * 1e3;
      }
      if (boundary_root != nil) {
        [command waitUntilCompleted];
        [boundary_root waitUntilCompleted];
        const double root_busy =
            (boundary_root.GPUEndTime - boundary_root.GPUStartTime) * 1e3;
        const double fence_gap =
            (command.GPUStartTime - boundary_root.GPUEndTime) * 1e3;
        const double loop_busy =
            (command.GPUEndTime - command.GPUStartTime) * 1e3;
        // commit -> driver began processing the root chunk (CPU timebase).
        const double sched_delay =
            (boundary_root.kernelStartTime - commit_media_time) * 1e3;
        // driver done -> GPU actually started executing the root chunk.
        const double kernel_to_gpu =
            (boundary_root.GPUStartTime - boundary_root.kernelEndTime) * 1e3;
        commit_acc += commit_host_ms;
        root_acc += root_busy;
        gap_acc += fence_gap;
        busy_acc += loop_busy;
        sched_acc += sched_delay;
        kstart_acc += kernel_to_gpu;
      }
      if (wait_calls % 69 == 0) {
        std::fprintf(
            stderr,
            "[kda-loop] calls=%llu wait_mean=%.3fms min=%.3f max=%.3f\n",
            static_cast<unsigned long long>(wait_calls),
            wait_accumulator_ms / static_cast<double>(wait_calls),
            wait_min_ms, wait_max_ms);
        std::fprintf(
            stderr,
            "[kda-anatomy] commit=%.3f sched=%.3f kern2gpu=%.3f "
            "root_busy=%.3f fence_gap=%.3f loop_busy=%.3f residual=%.3f "
            "shared=%llu/69 (ms mean/69)\n",
            commit_acc / 69.0, sched_acc / 69.0, kstart_acc / 69.0,
            root_acc / 69.0, gap_acc / 69.0, busy_acc / 69.0,
            (wait_accumulator_ms - root_acc - gap_acc - busy_acc) / 69.0,
            static_cast<unsigned long long>(shared_count));
        if (shared_count > 0) {
          const double n = static_cast<double>(shared_count);
          std::fprintf(stderr,
                       "[kda-shared] admission=%.3f gpu_busy=%.3f "
                       "listener_lag=%.3f (ms mean/%llu)\n",
                       adm_acc / n, sgpu_acc / n, lst_acc / n,
                       static_cast<unsigned long long>(shared_count));
        }
        gpu_timeline_drain_and_print();
        adm_acc = 0.0;
        sgpu_acc = 0.0;
        lst_acc = 0.0;
        wait_accumulator_ms = 0.0;
        wait_calls = 0;
        wait_min_ms = 1e30;
        wait_max_ms = 0.0;
        commit_acc = 0.0;
        root_acc = 0.0;
        gap_acc = 0.0;
        busy_acc = 0.0;
        sched_acc = 0.0;
        kstart_acc = 0.0;
        shared_count = 0;
      }
    }

    normalized_out = std::move(normalized);
    mlp_normalized_out = std::move(mlp_normalized);
    // At boundary layers the residual carrier aliases the attention output,
    // exactly like stock.  When there was no mix (A == 0), the lookahead
    // source equals the prefix — mirror stock's `mixed = prefix` handle.
    prefix_out = pending.prefix_defined ? prefix_tensor : output;
    lookahead_out = mlp_anchor_rows != 0 ? std::move(mlp_mixed) : prefix_out;
    result.output = std::move(output);
    result.next_state.query_convolution = std::move(next_conv_q);
    result.next_state.key_convolution = std::move(next_conv_k);
    result.next_state.value_convolution = std::move(next_conv_v);
    result.next_state.recurrent = std::move(next_recurrent);
    return true;
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    return false;
  } catch (...) {
    report_skip_once("unknown exception");
    return false;
  }
}

}  // namespace

bool loop_kda_layer(const std::uint32_t layer_index,
                    const at::Tensor& hidden, const at::Tensor& anchors,
                    const at::Tensor& score_weight,
                    const at::Tensor& input_norm,
                    const at::Tensor& prefix_sum,
                    const at::Tensor& next_anchors,
                    const at::Tensor& mlp_score_weight,
                    const at::Tensor& post_attention_norm,
                    const KdaWeights& weights, const KdaState& state_in,
                    at::Tensor& normalized_out,
                    at::Tensor& mlp_normalized_out,
                    at::Tensor& lookahead_out, at::Tensor& prefix_out,
                    KdaDecodeResult& result) {
  LoopKdaPendingState pending;
  if (!loop_kda_layer_begin(layer_index, hidden, anchors, score_weight,
                            input_norm, prefix_sum, next_anchors,
                            mlp_score_weight, post_attention_norm, weights,
                            state_in, /*cat_next_anchors_in_cb=*/false,
                            pending)) {
    return false;
  }
  return loop_kda_layer_finish(pending, normalized_out, mlp_normalized_out,
                               lookahead_out, prefix_out, result);
}

struct LoopKdaPendingHandle {
  LoopKdaPendingState state;
};

bool loop_cat_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_LOOP_CAT");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

int kda_precommit_mode() {
  static const int mode = [] {
    const char* value = std::getenv("K3_KDA_PRECOMMIT");
    if (value == nullptr) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "parity") == 0 || std::strcmp(value, "2") == 0) {
      return 2;
    }
    return 0;
  }();
  return mode;
}

bool kda_precommit_enabled() { return kda_precommit_mode() != 0; }

bool kda_int8_direct_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_KDA_INT8_DIRECT");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

namespace {
bool loop_router_chain_begin(std::uint32_t layer_index,
                             const at::Tensor& hidden,
                             const LoopMoeMatrixView& router,
                             const at::Tensor& bias) noexcept;
}  // namespace

LoopKdaPendingHandle* loop_kda_layer_precommit(
    const std::uint32_t layer_index, const at::Tensor& hidden,
    const at::Tensor& anchors, const at::Tensor& score_weight,
    const at::Tensor& input_norm, const at::Tensor& prefix_sum,
    const at::Tensor& next_anchors, const at::Tensor& mlp_score_weight,
    const at::Tensor& post_attention_norm, const KdaWeights& weights,
    const KdaState& state_in, const LoopMoeMatrixView& router,
    const at::Tensor& router_bias, const bool cat_next_anchors_in_cb) {
  auto handle = std::make_unique<LoopKdaPendingHandle>();
  if (!loop_kda_layer_begin(layer_index, hidden, anchors, score_weight,
                            input_norm, prefix_sum, next_anchors,
                            mlp_score_weight, post_attention_norm, weights,
                            state_in, cat_next_anchors_in_cb,
                            handle->state)) {
    return nullptr;
  }
  // Milestone (iii): chain the layer's route CB directly behind the KDA CB
  // (serial queue = ordering; the route reads the KDA CB's mlp_normalized
  // output buffer). Fail-soft: prepare_moe_t1 begins normally if absent.
  if ((router.dense_f32.defined() ||
       (router.int8_weight.defined() && router.row_scales.defined())) &&
      router_bias.defined() && handle->state.mlp_normalized.defined()) {
    static_cast<void>(loop_router_chain_begin(
        layer_index, handle->state.mlp_normalized, router, router_bias));
  }
  return handle.release();
}

bool loop_kda_layer_collect(LoopKdaPendingHandle* handle,
                            at::Tensor& normalized_out,
                            at::Tensor& mlp_normalized_out,
                            at::Tensor& lookahead_out, at::Tensor& prefix_out,
                            KdaDecodeResult& result,
                            at::Tensor& next_anchors_out) {
  if (handle == nullptr) {
    return false;
  }
  std::unique_ptr<LoopKdaPendingHandle> owned(handle);
  next_anchors_out = std::move(owned->state.next_anchors);
  return loop_kda_layer_finish(owned->state, normalized_out,
                               mlp_normalized_out, lookahead_out, prefix_out,
                               result);
}

void loop_kda_layer_abandon(LoopKdaPendingHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }
  std::unique_ptr<LoopKdaPendingHandle> owned(handle);
  // Let the committed CB drain before its tensors go away. The CB only
  // writes its own fresh output tensors, so completion is harmless; the
  // timeout mirrors the finish path and simply drops on the floor.
  if (owned->state.ready != nil && !owned->state.ready_consumed) {
    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    static_cast<void>(prep_timed_semaphore_wait(
        owned->state.ready,
        dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
        owned->state.command));
    owned->state.ready_consumed = true;
  }
}

void loop_kda_layer_wait(LoopKdaPendingHandle* handle) noexcept {
  if (handle == nullptr || handle->state.ready == nil ||
      handle->state.ready_consumed) {
    return;
  }
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  static_cast<void>(prep_timed_semaphore_wait(
      handle->state.ready,
      dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
      handle->state.command));
  handle->state.ready_consumed = true;
}

std::uint32_t loop_router_pending_layer() {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.shadow_active ? state.route_pending_layer : 0xFFFFFFFFu;
}

bool loop_router_peek_chained(const std::uint32_t layer_index,
                              std::uint16_t* expert_ids_out) noexcept {
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.shadow_active ||
      state.route_pending_layer != layer_index ||
      expert_ids_out == nullptr) {
    return false;
  }
  if (!state.shadow_ready_consumed) {
    constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
    const long wait_result = prep_timed_semaphore_wait(
        state.shadow_ready,
        dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds),
        state.shadow_command);
    state.shadow_ready_consumed = true;
    if (wait_result != 0) {
      return false;
    }
  }
  if (state.shadow_command != nil && state.shadow_command.error != nil) {
    return false;
  }
  RouteMailboxT1 side;
  std::memcpy(&side, state.pilot_mailbox.contents, sizeof(side));
  for (std::uint32_t k = 0; k < kTopK; ++k) {
    const std::int64_t id = side.expert_ids[k];
    if (id < 0 || id >= 896) {
      return false;
    }
    expert_ids_out[k] = static_cast<std::uint16_t>(id);
  }
  // The mailbox and shadow state stay intact: collect still consumes the
  // route for execution; this peek only feeds the prefetch hint.
  return true;
}

namespace {

/* Step 6 milestone (iii): encode + commit the layer's route CB directly
 * BEHIND the just-pre-committed KDA CB on the serial loop queue. No ATen
 * fence and no event wait: queue order guarantees the router input (the
 * KDA CB's mlp_normalized output buffer) is final, and the KDA CB's own
 * fence already ordered the fresh weight dequant. loop_router_collect
 * later consumes the mailbox exactly as with a prepare-time begin. */
bool loop_router_chain_begin(const std::uint32_t layer_index,
                             const at::Tensor& hidden,
                             const LoopMoeMatrixView& router,
                             const at::Tensor& bias) noexcept {
  constexpr std::int64_t kHidden = 7168;
  constexpr std::int64_t kExperts = 896;
  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  // Same contract as loop_kda_layer_begin: once encoding into the OPEN
  // fused CB has started, a throw marks it broken rather than committable.
  bool fused_encode_started = false;
  try {
    require(state.queue != nil, "chained route: loop queue unavailable");
    shadow_drain_locked(state);

    const bool int8_form =
        router.int8_weight.defined() && router.row_scales.defined();
    const at::Tensor& weight_tensor =
        int8_form ? router.int8_weight : router.dense_f32;
    const NSUInteger hidden_offset = checked_byte_offset(
        hidden, kHidden * sizeof(float), "chained router hidden");
    const NSUInteger weight_offset = checked_byte_offset(
        weight_tensor,
        static_cast<std::size_t>(kExperts) * kHidden *
            (int8_form ? sizeof(std::int8_t) : sizeof(float)),
        "chained router weights");
    const NSUInteger scales_offset =
        int8_form ? checked_byte_offset(router.row_scales,
                                        kExperts * sizeof(float),
                                        "chained router scales")
                  : 0;
    const NSUInteger bias_offset = checked_byte_offset(
        bias, kExperts * sizeof(float), "chained router bias");

    RouteMailboxT1 poison;
    std::memset(&poison, 0xA5, sizeof(poison));
    std::memcpy(state.pilot_mailbox.contents, &poison, sizeof(poison));

    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);
    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   dispatch_semaphore_signal(ready);
                 }];

    // K3_CB_FUSION=1 ride: the route encoders join the open fused CB right
    // behind the KDA encoders just recorded into it — encoder order within
    // one CB replaces the serial-queue ordering the chained pair relied on.
    const bool fused_ride = loop_cb_fusion_enabled() &&
                            g_fused_open_command != nil && !g_fused_broken;
    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      if (fused_ride) {
        command = g_fused_open_command;
        fused_encode_started = true;
      } else {
        command = [state.queue commandBuffer];
      }
      require(command != nil, "chained route command buffer unavailable");
      if (!fused_ride) {
        gpu_timeline_note(command, "route-chain");
      }
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "chained route encoder unavailable");
      if (int8_form) {
        const LoopGemvDimsHost gemv_dims{
            static_cast<std::uint32_t>(kExperts),
            static_cast<std::uint32_t>(kHidden)};
        [encoder setComputePipelineState:state.gemv_pipeline];
        [encoder setBuffer:tensor_buffer(weight_tensor)
                    offset:weight_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(hidden)
                    offset:hidden_offset
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(router.row_scales)
                    offset:scales_offset
                   atIndex:2];
        [encoder setBuffer:state.shadow_logits offset:0 atIndex:3];
        [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((kExperts + 15) / 16, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else {
        struct LoopBf16GemvDimsHost {
          std::uint32_t rows;
          std::uint32_t columns;
          std::uint32_t reserved0;
          std::uint32_t reserved1;
        };
        const LoopBf16GemvDimsHost gemv_dims{
            static_cast<std::uint32_t>(kExperts),
            static_cast<std::uint32_t>(kHidden), 0, 0};
        [encoder setComputePipelineState:state.gemv_f32_pipeline];
        [encoder setBuffer:tensor_buffer(weight_tensor)
                    offset:weight_offset
                   atIndex:0];
        [encoder setBuffer:tensor_buffer(hidden)
                    offset:hidden_offset
                   atIndex:1];
        [encoder setBuffer:state.shadow_logits offset:0 atIndex:2];
        [encoder setBytes:&gemv_dims length:sizeof(gemv_dims) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((kExperts + 15) / 16, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const LoopRouteDimsHost route_dims{
          static_cast<std::uint32_t>(kExperts)};
      [encoder setComputePipelineState:state.router_top16_pipeline];
      [encoder setBuffer:state.shadow_logits offset:0 atIndex:0];
      [encoder setBuffer:tensor_buffer(bias) offset:bias_offset atIndex:1];
      [encoder setBuffer:state.shadow_scores offset:0 atIndex:2];
      [encoder setBuffer:state.pilot_mailbox offset:0 atIndex:3];
      [encoder setBytes:&route_dims length:sizeof(route_dims) atIndex:4];
      [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      if (!fused_ride) {
        [command commit];
      }
      // Fused ride: the single commit happens in loop_fused_flush(); the
      // mid-CB done signal above fires the shadow semaphore at the same
      // stream position an own-CB commit would have.
    }

    state.shadow_active = true;
    state.route_pending_layer = layer_index;
    state.shadow_ready_consumed = false;
    state.shadow_ready = ready;
    state.shadow_command = command;
    static std::atomic<std::uint64_t> chained{0};
    const std::uint64_t n = chained.fetch_add(1) + 1;
    if (n == 1 || n % 690 == 0) {
      std::fprintf(stderr, "[route-chain] chained=%llu\n",
                   static_cast<unsigned long long>(n));
    }
    return true;
  } catch (const std::exception& error) {
    static std::atomic<bool> chain_fail_reported{false};
    if (!chain_fail_reported.exchange(true)) {
      std::fprintf(stderr, "[route-chain] first failure: %s\n",
                   error.what());
    }
    if (fused_encode_started) {
      g_fused_broken = true;
    }
    state.shadow_active = false;
    state.route_pending_layer = 0xFFFFFFFFu;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    return false;
  } catch (...) {
    if (fused_encode_started) {
      g_fused_broken = true;
    }
    state.shadow_active = false;
    state.route_pending_layer = 0xFFFFFFFFu;
    state.shadow_ready = nil;
    state.shadow_command = nil;
    return false;
  }
}

}  // namespace

namespace {

/* Form-agnostic GEMV encode over a LoopMoeMatrixView (dense-f32 / row-int8
 * / original-BF16), mirroring encode_kda_projection's dispatch.  Throws on
 * shape/form mismatch (caught by loop_moe_tail's fail-soft wrapper). */
void encode_loop_matrix(id<MTLComputeCommandEncoder> encoder,
                        LoopEncoderState& state,
                        const LoopMoeMatrixView& view,
                        const std::uint32_t rows,
                        const std::uint32_t columns, id<MTLBuffer> x_buffer,
                        const NSUInteger x_offset, id<MTLBuffer> y_buffer,
                        const NSUInteger y_offset, const char* name) {
  KdaProjection adapter;
  if (view.dense_f32.defined()) {
    adapter = KdaProjection{view.dense_f32, at::Tensor()};
  } else if (view.int8_weight.defined() && view.row_scales.defined()) {
    adapter = KdaProjection{view.int8_weight, view.row_scales};
  } else if (view.original_bf16 != nullptr &&
             view.original_bf16->defined()) {
    adapter = KdaProjection{at::Tensor(), at::Tensor(),
                            *view.original_bf16};
  } else {
    fail(std::string("loop MoE matrix has no usable form: ") + name);
  }
  encode_kda_projection(encoder, state, adapter, rows, columns, x_buffer,
                        x_offset, y_buffer, y_offset, name);
}

}  // namespace

bool loop_tail_async_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_TAIL_ASYNC");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

namespace {
/* The single in-flight deferred tail. Like the rest of the loop state this
 * is touched only from the engine's one decode thread (Rust call or a
 * provider finish hook on that same call stack), so plain statics match the
 * surrounding discipline. ARC retains both references. */
dispatch_semaphore_t g_pending_tail_ready = nil;
id<MTLCommandBuffer> g_pending_tail_command = nil;
/* Sticky: a deferred tail failed (timeout or CB error). Some drain callers
 * legitimately swallow exceptions (fail-soft hooks, tail fallback); this
 * flag lets the finish ABI turn the swallowed failure into the hard
 * sequence error the synchronous contract would have raised. */
std::atomic<bool> g_tail_poisoned{false};
}  // namespace

bool loop_tail_poisoned() noexcept {
  return g_tail_poisoned.load(std::memory_order_acquire);
}

bool loop_cb_fusion_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_CB_FUSION");
    const bool requested = value != nullptr && std::strcmp(value, "1") == 0;
    if (requested && loop_tail_async_enabled()) {
      std::fprintf(stderr,
                   "[cb-fusion] disabled: K3_TAIL_ASYNC=1 owns the tail "
                   "commit sites\n");
      return false;
    }
    return requested;
  }();
  return enabled;
}

void loop_fused_flush() {
  if (g_fused_open_command == nil && !g_fused_broken) {
    return;
  }
  id<MTLCommandBuffer> command = g_fused_open_command;
  dispatch_semaphore_t ready = g_fused_tail_ready;
  const bool broken = g_fused_broken;
  g_fused_open_command = nil;
  g_fused_tail_ready = nil;
  g_fused_broken = false;
  if (broken || command == nil || ready == nil) {
    // The un-committed CB is dropped, never run: its listeners are healed
    // by the next higher signal on the shared event, and the sticky poison
    // turns the swallowed drain failure into the hard sequence error the
    // synchronous contract would have raised.
    g_tail_poisoned.store(true, std::memory_order_release);
    fail("fused loop CB was broken by a mid-encode failure");
  }
  [command commit];
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = prep_timed_semaphore_wait(
      ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds), command);
  if (wait_result != 0) {
    g_tail_poisoned.store(true, std::memory_order_release);
    fail("fused loop CB tail wait timed out");
  }
  if (command.error != nil) {
    g_tail_poisoned.store(true, std::memory_order_release);
    fail(std::string("fused loop CB failed: ") +
         command.error.localizedDescription.UTF8String);
  }
}

void loop_moe_tail_drain() {
  // Fusion first: an open fused CB holds the tail encoders uncommitted, so
  // restoring the synchronous tail contract must commit + wait it here —
  // every existing drain site then covers K3_CB_FUSION=1 unchanged.
  loop_fused_flush();
  if (g_pending_tail_ready == nil) {
    return;
  }
  dispatch_semaphore_t ready = g_pending_tail_ready;
  id<MTLCommandBuffer> command = g_pending_tail_command;
  g_pending_tail_ready = nil;
  g_pending_tail_command = nil;
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  const long wait_result = prep_timed_semaphore_wait(
      ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds), command);
  if (wait_result != 0) {
    g_tail_poisoned.store(true, std::memory_order_release);
    fail("deferred loop MoE tail event timed out");
  }
  if (command.error != nil) {
    g_tail_poisoned.store(true, std::memory_order_release);
    fail(std::string("deferred loop MoE tail command failed: ") +
         command.error.localizedDescription.UTF8String);
  }
}

bool loop_moe_tail(const std::uint32_t layer_index,
                   const at::Tensor& routed_input,
                   const at::Tensor& identity, const at::Tensor& prefix_sum,
                   const std::uint16_t* expert_ids,
                   const std::uint32_t* weight_bits,
                   const std::uint8_t* const* expert_blobs,
                   const std::uint32_t expert_count,
                   const at::Tensor& routed_norm,
                   const LoopMoeMatrixView& routed_up,
                   const bool shared_combined,
                   const LoopMoeMatrixView& shared_gate_up,
                   const LoopMoeMatrixView& shared_gate,
                   const LoopMoeMatrixView& shared_up,
                   const LoopMoeMatrixView& shared_down,
                   at::Tensor& row_hidden_out) {
  static_cast<void>(layer_index);
  static_cast<void>(expert_ids);
  constexpr std::uint32_t kHiddenWidth = 7168;
  constexpr std::uint32_t kRoutedWidth = 3584;
  constexpr std::uint32_t kSharedWidth = 6144;
  static std::atomic<bool> skip_reported{false};
  const auto report_skip_once = [&](const char* reason) {
    if (!skip_reported.exchange(true)) {
      std::fprintf(stderr, "[moe-tail] first fallback: %s\n", reason);
    }
  };
  static const bool probing = [] {
    const char* value = std::getenv("K3_ROUTE_SYNC_PROBE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();

  LoopEncoderState& state = loop_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  if (stream == nullptr) {
    report_skip_once("current MPS stream unavailable");
    return false;
  }
  id<MTLDevice> device = stream->device();
  if (device == nil) {
    report_skip_once("current MPS device unavailable");
    return false;
  }
  try {
    ensure_resources(state, device);

    const auto plain = [&](const at::Tensor& tensor,
                           const std::int64_t elements, const char* name) {
      require(tensor.defined() && tensor.device().is_mps() &&
                  tensor.scalar_type() == at::kFloat &&
                  tensor.is_contiguous() && tensor.numel() == elements,
              std::string("loop MoE tail tensor does not qualify: ") + name);
      return checked_byte_offset(
          tensor, static_cast<std::size_t>(elements) * sizeof(float), name);
    };
    const NSUInteger routed_input_offset =
        plain(routed_input, kRoutedWidth, "routed_input");
    const NSUInteger identity_offset =
        plain(identity, kHiddenWidth, "identity");
    const NSUInteger prefix_offset =
        plain(prefix_sum, kHiddenWidth, "prefix_sum");
    const NSUInteger routed_norm_offset =
        plain(routed_norm, kRoutedWidth, "routed_norm");
    require(expert_blobs != nullptr && expert_count >= 1 &&
                expert_count <= 16,
            "loop MoE tail expert set does not qualify");

    at::Tensor row_hidden = at::empty(
        {1, kHiddenWidth},
        at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
    const NSUInteger row_hidden_offset =
        plain(row_hidden, kHiddenWidth, "row hidden");

    const std::uint64_t boundary_value =
        take_event_value(state, kLoopEventShadowBoundary);
    const std::uint64_t done_value =
        take_event_value(state, kLoopEventShadowDone);

    // The previous layer's deferred tail (if any) must complete before this
    // layer's staged expert bytes may recycle its arena generation. Cheap
    // no-op when nothing is pending or K3_TAIL_ASYNC is off.
    loop_moe_tail_drain();
    // ATen-side fence half: routed_input (and any pilot ops) were encoded
    // on the ATen stream after the KDA CB returned; commit + signal so the
    // tail CB may consume them.
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        stream->endKernelCoalescing();
        MPSCommandBuffer* command = stream->commandBuffer();
        require(command != nil,
                "loop MoE tail MPS command buffer is unavailable");
        id<MTLCommandBuffer> root = command.rootCommandBuffer;
        require(root != nil,
                "loop MoE tail root command buffer is unavailable");
        [root encodeSignalEvent:state.events[kLoopEventShadowBoundary]
                          value:boundary_value];
      }
    });
    require(try_commit_mps_stream_for_route(),
            "loop MoE tail boundary commit failed");

    dispatch_semaphore_t ready = dispatch_semaphore_create(0);
    [state.events[kLoopEventShadowDone]
        notifyListener:state.listener
               atValue:done_value
                 block:^(id<MTLSharedEvent>, std::uint64_t) {
                   g_prep_tail_done_ns.store(prep_media_ns(),
                                             std::memory_order_relaxed);
                   dispatch_semaphore_signal(ready);
                 }];

    const auto wait_started = std::chrono::steady_clock::now();
    __block id<MTLCommandBuffer> command = nil;
    @autoreleasepool {
      command = [state.queue commandBuffer];
      require(command != nil,
              "loop MoE tail command buffer is unavailable");
      [command encodeWaitForEvent:state.events[kLoopEventShadowBoundary]
                            value:boundary_value];

      // Expert stack (glu -> w2 -> reduce) via metal_moe's own pipelines
      // and wrap cache, encoded onto this CB.  Its encoder is closed inside
      // the call; encoder-to-encoder ordering on one CB is serial.
      std::array<float, 16> route_weights{};
      for (std::uint32_t k = 0; k < expert_count; ++k) {
        std::memcpy(&route_weights[k], &weight_bits[k], sizeof(float));
      }
      const int moe_status = k3_metal_moe_encode_t1_raw_v1(
          command, device, expert_blobs, static_cast<int>(expert_count),
          route_weights.data(), tensor_buffer(routed_input),
          routed_input_offset, state.moe_routed_out, 0);
      require(moe_status == 0,
              std::string("loop MoE expert-stack encode failed: status ") +
                  std::to_string(moe_status));

      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      require(encoder != nil, "loop MoE tail encoder is unavailable");
      {
        struct RmsDimsHost {
          std::uint32_t N;
          float eps;
        };
        const RmsDimsHost rms_dims{kRoutedWidth, 1e-5F};
        [encoder setComputePipelineState:state.rmsnorm_pipeline];
        [encoder setBuffer:state.moe_routed_out offset:0 atIndex:0];
        [encoder setBuffer:tensor_buffer(routed_norm)
                    offset:routed_norm_offset
                   atIndex:1];
        [encoder setBuffer:state.moe_normed offset:0 atIndex:2];
        [encoder setBytes:&rms_dims length:sizeof(rms_dims) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      encode_loop_matrix(encoder, state, routed_up, kHiddenWidth,
                         kRoutedWidth, state.moe_normed, 0,
                         state.moe_routed_full, 0, "routed_up");
      if (shared_combined) {
        encode_loop_matrix(encoder, state, shared_gate_up,
                           2 * kSharedWidth, kHiddenWidth,
                           tensor_buffer(identity), identity_offset,
                           state.moe_gate_up, 0, "shared_gate_up");
      } else {
        encode_loop_matrix(encoder, state, shared_gate, kSharedWidth,
                           kHiddenWidth, tensor_buffer(identity),
                           identity_offset, state.moe_gate_up, 0,
                           "shared_gate");
        encode_loop_matrix(encoder, state, shared_up, kSharedWidth,
                           kHiddenWidth, tensor_buffer(identity),
                           identity_offset, state.moe_gate_up,
                           kSharedWidth * sizeof(float), "shared_up");
      }
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      {
        struct SituDimsHost {
          std::uint32_t N;
        };
        const SituDimsHost situ_dims{kSharedWidth};
        [encoder setComputePipelineState:state.situ_pipeline];
        [encoder setBuffer:state.moe_gate_up offset:0 atIndex:0];
        [encoder setBuffer:state.moe_gate_up
                    offset:kSharedWidth * sizeof(float)
                   atIndex:1];
        [encoder setBuffer:state.moe_situ offset:0 atIndex:2];
        [encoder setBytes:&situ_dims length:sizeof(situ_dims) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(kSharedWidth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      encode_loop_matrix(encoder, state, shared_down, kHiddenWidth,
                         kSharedWidth, state.moe_situ, 0,
                         state.moe_shared_out, 0, "shared_down");
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      {
        const std::uint32_t add_elements = kHiddenWidth;
        [encoder setComputePipelineState:state.add_pipeline];
        [encoder setBuffer:state.moe_routed_full offset:0 atIndex:0];
        [encoder setBuffer:state.moe_shared_out offset:0 atIndex:1];
        [encoder setBuffer:state.moe_mlp_out offset:0 atIndex:2];
        [encoder setBytes:&add_elements
                   length:sizeof(add_elements)
                  atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(kHiddenWidth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:state.add_pipeline];
        [encoder setBuffer:tensor_buffer(prefix_sum)
                    offset:prefix_offset
                   atIndex:0];
        [encoder setBuffer:state.moe_mlp_out offset:0 atIndex:1];
        [encoder setBuffer:tensor_buffer(row_hidden)
                    offset:row_hidden_offset
                   atIndex:2];
        [encoder setBytes:&add_elements
                   length:sizeof(add_elements)
                  atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(kHiddenWidth, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [encoder endEncoding];
      [command encodeSignalEvent:state.events[kLoopEventShadowDone]
                           value:done_value];
      gpu_timeline_note(command, loop_cb_fusion_enabled() ? "fused"
                                                          : "moe-tail");
      if (!loop_cb_fusion_enabled()) {
        [command commit];
      }
    }

    if (loop_cb_fusion_enabled()) {
      // K3_CB_FUSION=1: leave the CB open — the precommit orchestration
      // hook rides the successor's KDA/route encoders into it, and
      // loop_fused_flush() (hook exit or any drain site) performs the
      // single commit plus this synchronous wait inside the same finish
      // ABI call. The mid-CB done signal above keeps the tail semaphore
      // firing at exactly today's stream position, so the expert-byte
      // lease and every ATen consumer keep the synchronous contract.
      g_fused_open_command = command;
      g_fused_tail_ready = ready;
    } else if (loop_tail_async_enabled()) {
      // K3_TAIL_ASYNC=1: stash instead of waiting. The next tail's entry
      // drain (or an eager loop_moe_tail_drain wherever synchronous
      // semantics are required back) performs the wait and error check;
      // the Rust lease park keeps the staged expert bytes alive until
      // that drain proves the GPU is done reading them.
      g_pending_tail_ready = ready;
      g_pending_tail_command = command;
    } else {
      constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
      const long wait_result = dispatch_semaphore_wait(
          ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
      require(wait_result == 0, "loop MoE tail event timed out");
      if (command.error != nil) {
        fail(std::string("loop MoE tail command failed: ") +
             command.error.localizedDescription.UTF8String);
      }
    }
    if (probing) {
      const double wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - wait_started)
              .count();
      static double wait_accumulator_ms = 0.0;
      static std::uint64_t wait_calls = 0;
      wait_accumulator_ms += wait_ms;
      ++wait_calls;
      if (wait_calls % 92 == 0) {
        std::fprintf(stderr,
                     "[moe-tail] calls=%llu wait_mean=%.3fms\n",
                     static_cast<unsigned long long>(wait_calls),
                     wait_accumulator_ms / static_cast<double>(wait_calls));
        wait_accumulator_ms = 0.0;
        wait_calls = 0;
      }
    }

    row_hidden_out = std::move(row_hidden);
    return true;
  } catch (const std::exception& error) {
    report_skip_once(error.what());
    return false;
  } catch (...) {
    report_skip_once("unknown exception");
    return false;
  }
}

bool bespoke_loop_decode_ready() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_BESPOKE_LOOP");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  if (!enabled) {
    return false;
  }
  static const bool qualified = [] {
    try {
      const LoopMetalCapabilities capabilities = loop_metal_capabilities_v1();
      if (capabilities.abi_version != kLoopMetalAbiV1 ||
          capabilities.flags != kLoopMetalRequiredCapabilitiesV1 ||
          capabilities.event_pool_size != kLoopMetalSharedEventPoolSizeV1 ||
          capabilities.reserved != 0) {
        std::fprintf(stderr, "[bespoke-loop] disabled: capability mismatch\n");
        return false;
      }
      const LoopMetalCanaryReport report = loop_metal_canary_v1();
      const bool passed = loop_metal_canary_passes(report);
      std::fprintf(stderr,
                   "[bespoke-loop] qualification %s: route=%u/%u pilot=%u/%u "
                   "table_hits=%u table_invalidations=%u max_abs=%.3e "
                   "rel_l2=%.3e\n",
                   passed ? "PASS" : "FAIL", report.route_ids_matched,
                   report.top_k, report.pilot_ids_matched, report.top_k,
                   report.weights_table_hits,
                   report.weights_table_invalidations,
                   report.max_absolute_error, report.relative_l2_error);
      return passed;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[bespoke-loop] disabled: %s\n", error.what());
      return false;
    } catch (...) {
      std::fprintf(stderr, "[bespoke-loop] disabled: unknown failure\n");
      return false;
    }
  }();
  return qualified;
}

void loop_gpu_timeline_note_aten(const char* klass) noexcept {
  if (!gpu_timeline_enabled()) {
    return;
  }
  try {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream == nullptr) {
      return;
    }
    __block id<MTLCommandBuffer> root = nil;
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        MPSCommandBuffer* command = stream->commandBuffer();
        root = command != nil ? command.rootCommandBuffer : nil;
      }
    });
    // Decode thread only. PyTorch keeps one root CB open across coalesced
    // ops, so consecutive notes usually see the same buffer.
    static id<MTLCommandBuffer> last_noted = nil;
    if (root == nil || root == last_noted) {
      return;
    }
    last_noted = root;
    gpu_timeline_note(root, klass);
  } catch (...) {
  }
}

void loop_gpu_timeline_note_cb(id<MTLCommandBuffer> command,
                               const char* klass) noexcept {
  gpu_timeline_note(command, klass);
}

/* K3_GPU_TIMELINE: decompose one ATen host sync (the wide tile's route
 * readback) into the queue latency before the root CB's GPUStartTime, its
 * GPU execution, and the completion latency after GPUEndTime. Host stamps
 * are CLOCK_UPTIME_RAW ns (the mach_absolute_time clock GPUStartTime uses).
 * Diagnostic only; every function is a no-op unless the timeline is on. */
namespace {
std::atomic<std::uint64_t> g_aten_sync_n{0};
std::atomic<std::uint64_t> g_aten_sync_pre_ns{0};
std::atomic<std::uint64_t> g_aten_sync_exec_ns{0};
std::atomic<std::uint64_t> g_aten_sync_post_ns{0};
std::atomic<std::uint64_t> g_aten_sync_total_ns{0};
std::atomic<std::uint64_t> g_aten_sync_untimed{0};
// Finer split of `pre`: host commit delay (sync begin -> driver kernelStartTime)
// vs driver/queue latency (kernelStartTime -> GPUStartTime).
std::atomic<std::uint64_t> g_aten_sync_commit_ns{0};
std::atomic<std::uint64_t> g_aten_sync_queue_ns{0};
// Host time spent in the two device ops that precede the readback
// (expert_ids.to(kFloat) and at::cat) — an implicit sync there shows as ms.
std::atomic<std::uint64_t> g_aten_sync_prep_ns{0};
std::atomic<std::uint64_t> g_aten_sync_second_ns{0};
}  // namespace

void loop_aten_sync_note_prep(const std::uint64_t ns) noexcept {
  g_aten_sync_prep_ns.fetch_add(ns, std::memory_order_relaxed);
}

void loop_aten_sync_note_second(const std::uint64_t ns) noexcept {
  g_aten_sync_second_ns.fetch_add(ns, std::memory_order_relaxed);
}

/* ---- K3_KDA_WIDE_FUSED: wide-tile fused KDA on the ATen stream ----------
 * Replaces the batched ATen short-conv + per-position recurrence + output
 * norm/gate (~140 tiny ops per layer pass) with one kernel encoded into the
 * current MPS stream between the batched projections and the output
 * projection. Default off. */
namespace {
struct KdaWideDimsV1 {
  std::uint32_t positions;
  std::uint32_t source_width;
  std::uint32_t retain;
  std::uint32_t reserved;
};
static_assert(sizeof(KdaWideDimsV1) == 16);
struct KdaWidePipeline {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pipeline = nil;
};
KdaWidePipeline& kda_wide_pipeline_for(id<MTLDevice> device) {
  static KdaWidePipeline cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  if (cache.device != device || cache.pipeline == nil) {
    dispatch_data_t data = copy_embedded_metallib();
    require(data != nullptr, "wrap embedded loop-kernels metallib failed (kda-wide)");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    require(library != nil,
            std::string("load embedded loop-kernels metallib failed (kda-wide): ") +
                (error == nil ? "unknown" : error.localizedDescription.UTF8String));
    cache = KdaWidePipeline{
        device, library,
        build_pipeline(device, library, @"deltafin_loop_kda_wide_v1", 128)};
  }
  return cache;
}
void require_kda_wide_f32(const at::Tensor& tensor, const std::int64_t numel,
                          const char* name) {
  require(tensor.defined() && tensor.device().is_mps() &&
              tensor.scalar_type() == at::kFloat && tensor.is_contiguous() &&
              tensor.numel() == numel,
          std::string("kda-wide ") + name +
              " violates its contiguous fp32 MPS contract");
}
}  // namespace


/* ---- K3_RESIDENCY_SET=1: pin spine allocations with an MTLResidencySet --
 * Every command buffer that binds a resource makes the driver validate its
 * residency; with ~1 GB of spine bound per layer that validation showed up
 * as 6-7 ms of driver CPU per attention command buffer, independent of the
 * op count. A residency set attached to the queues keeps the allocations
 * resident once, so per-CB validation is skipped. Heap-backed buffers
 * register their heap. Diagnostic; inert unless the knob is set. */
namespace {
struct ResidencyState {
  std::mutex mutex;
  id<MTLResidencySet> set = nil;
  std::unordered_set<void*> seen;
  std::unordered_set<void*> queues;
  std::uint64_t count = 0;
  std::uint64_t bytes = 0;
};
ResidencyState& residency_state() {
  static ResidencyState* state = new ResidencyState;
  return *state;
}
bool residency_ensure_set_locked(ResidencyState& state, id<MTLDevice> device) {
  if (@available(macOS 15.0, *)) {
    if (state.set == nil) {
      MTLResidencySetDescriptor* descriptor = [[MTLResidencySetDescriptor alloc] init];
      descriptor.initialCapacity = 4096;
      NSError* error = nil;
      state.set = [device newResidencySetWithDescriptor:descriptor error:&error];
      if (state.set == nil) {
        std::fprintf(stderr, "[residency] newResidencySet failed: %s\n",
                     error == nil ? "unknown" : error.localizedDescription.UTF8String);
        return false;
      }
      std::fprintf(stderr, "[residency] residency set created\n");
    }
    return true;
  }
  return false;
}
void residency_attach_queue_locked(ResidencyState& state, id<MTLCommandQueue> queue) {
  if (queue == nil || !residency_ensure_set_locked(state, queue.device)) {
    return;
  }
  void* key = (__bridge void*)queue;
  if (!state.queues.insert(key).second) {
    return;
  }
  if (@available(macOS 15.0, *)) {
    [queue addResidencySet:state.set];
  }
  std::fprintf(stderr, "[residency] queue attached (%zu queues)\n", state.queues.size());
}
}  // namespace

int loop_residency_set_level() noexcept {
  static const int level = [] {
    const char* value = std::getenv("K3_RESIDENCY_SET");
    if (value == nullptr) {
      return 0;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 0;
  }();
  return level;
}

bool loop_residency_set_enabled() noexcept {
  return loop_residency_set_level() >= 1;
}

// Level 1 pins the spine only (BF16 storages at retain, int8 matrices at
// first use): 64 heaps / 33 GiB, stable for the life of the process.
// Level 2 also pins the activation heaps seen by the Metal bridges. Those
// heaps churn with PyTorch's allocator, so the set grows with the run
// (RS_a: 128 allocations / 52.7 GiB at 200 tokens; with the fused KDA path
// W_a reached 512 / 68.7 GiB, host available fell to 7 GiB and the run
// aborted on MLA growth admission at token 117).
// Level 3 also pins the expert bridge's wrapped slabs and scratch buffers
// (RS60c: 1,024+ allocations / 70 GiB, no gain over the spine alone).
bool loop_residency_activations_enabled() noexcept {
  return loop_residency_set_level() >= 2;
}
bool loop_residency_expert_buffers_enabled() noexcept {
  return loop_residency_set_level() >= 3;
}

void loop_residency_attach_queue(void* queue) noexcept {
  if (!loop_residency_set_enabled() || queue == nullptr) {
    return;
  }
  try {
    ResidencyState& state = residency_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    residency_attach_queue_locked(state, (__bridge id<MTLCommandQueue>)queue);
  } catch (...) {
  }
}

namespace {
void residency_register_buffer(id<MTLBuffer> buffer) {
    if (buffer == nil) {
      return;
    }
    ResidencyState& state = residency_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!residency_ensure_set_locked(state, buffer.device)) {
      return;
    }
    // The ATen stream's queue attaches lazily on the first registration.
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream != nullptr) {
      residency_attach_queue_locked(state, stream->commandQueue());
    }
    id<MTLHeap> heap = buffer.heap;
    void* key = heap != nil ? (__bridge void*)heap : (__bridge void*)buffer;
    if (!state.seen.insert(key).second) {
      return;
    }
    if (@available(macOS 15.0, *)) {
      if (heap != nil) {
        [state.set addAllocation:heap];
      } else {
        [state.set addAllocation:buffer];
      }
      [state.set commit];
      [state.set requestResidency];
    }
    state.count += 1;
    state.bytes += heap != nil ? heap.size : buffer.length;
    if ((state.count & (state.count - 1)) == 0 || state.count % 512 == 0) {
      std::fprintf(stderr, "[residency] %llu allocations, %.2f GiB pinned\n",
                   static_cast<unsigned long long>(state.count),
                   static_cast<double>(state.bytes) / (1024.0 * 1024.0 * 1024.0));
    }
}
}  // namespace

void loop_residency_register_tensor(const at::Tensor& tensor) noexcept {
  if (!loop_residency_set_enabled() || !tensor.defined() ||
      !tensor.device().is_mps()) {
    return;
  }
  try {
    residency_register_buffer(tensor_buffer(tensor));
  } catch (...) {
  }
}

void loop_residency_register_activation(const at::Tensor& tensor) noexcept {
  if (loop_residency_activations_enabled()) {
    loop_residency_register_tensor(tensor);
  }
}

void loop_residency_register_allocation(void* buffer) noexcept {
  if (!loop_residency_set_enabled() || buffer == nullptr) {
    return;
  }
  try {
    residency_register_buffer((__bridge id<MTLBuffer>)buffer);
  } catch (...) {
  }
}

extern "C" int k3_metal_moe_encode_positions_flat_stream_v1(
    void* encoder_handle, void* device_handle,
    const std::uint8_t* const* expert_blobs, int n_edges,
    const int* position_offsets, int n_positions, const float* weights,
    void* x_handle, std::size_t x_off, void* out_handle, std::size_t out_off);

/* K3_MOE_ATEN_STREAM=1: run the expert bridge's flat position batch on the
 * current MPS stream with device-side x/out (no host copies, no per-wave
 * wait); `synchronize` drains the stream once, on the tile's final wave, so
 * the expert slabs may be recycled by the reader afterwards. */
int loop_moe_positions_flat_on_aten_stream(
    const at::Tensor& x, const at::Tensor& out,
    const std::uint8_t* const* expert_blobs, const int n_edges,
    const int* position_offsets, const int n_positions, const float* weights,
    const bool synchronize) {
  const std::int64_t width = 3584;
  const std::int64_t n = n_positions;
  require(x.defined() && x.device().is_mps() && x.scalar_type() == at::kFloat &&
              x.is_contiguous() && x.numel() == n * width,
          "moe-stream x violates its contiguous fp32 MPS contract");
  require(out.defined() && out.device().is_mps() && out.scalar_type() == at::kFloat &&
              out.is_contiguous() && out.numel() == n * width,
          "moe-stream out violates its contiguous fp32 MPS contract");
  id<MTLBuffer> xb = tensor_buffer(x);
  id<MTLBuffer> ob = tensor_buffer(out);
  require(xb != nil && ob != nil, "moe-stream tensors have no MTLBuffer");
  const NSUInteger xo = checked_byte_offset(x, static_cast<std::size_t>(n * width) * 4, "moe-stream x");
  const NSUInteger oo = checked_byte_offset(out, static_cast<std::size_t>(n * width) * 4, "moe-stream out");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "moe-stream: current MPS stream is unavailable");
  __block int rc = -1;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil, "moe-stream: current MPS stream has no encoder");
      rc = k3_metal_moe_encode_positions_flat_stream_v1(
          (__bridge void*)encoder, (__bridge void*)stream->device(),
          expert_blobs, n_edges, position_offsets, n_positions, weights,
          (__bridge void*)xb, static_cast<std::size_t>(xo),
          (__bridge void*)ob, static_cast<std::size_t>(oo));
    }
  });
  if (rc == 0 && synchronize) {
    const std::uint64_t started = prep_steady_ns();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    prep_note_wait(prep_steady_ns() - started, 0, false);
  }
  return rc;
}

/* K3_PREP_SPLIT_COMMITS=1 (diagnostic): commit the ATen stream at each
 * phase boundary of the wide prepare so libtorch's profiler reports the
 * driver time of each phase's command buffer separately. */
bool loop_prep_split_commits_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_PREP_SPLIT_COMMITS");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
void loop_aten_stream_commit_phase(const char* phase) noexcept {
  if (!loop_prep_split_commits_enabled()) {
    return;
  }
  try {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream != nullptr) {
      stream->synchronize(at::mps::SyncType::COMMIT);
      static std::atomic<std::uint64_t> commits{0};
      const std::uint64_t n = commits.fetch_add(1, std::memory_order_relaxed);
      if (n < 8) {
        std::fprintf(stderr, "[prep-split] commit after %s\n", phase);
      }
    }
  } catch (...) {
  }
}

bool loop_kda_wide_fused_enabled() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_KDA_WIDE_FUSED");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

void loop_kda_wide_fused_on_aten_stream(
    const at::Tensor& src_q, const at::Tensor& src_k, const at::Tensor& src_v,
    const at::Tensor& convw_q, const at::Tensor& convw_k,
    const at::Tensor& convw_v, const at::Tensor& a_log,
    const at::Tensor& dt_bias, const at::Tensor& fb, const at::Tensor& beta,
    const at::Tensor& gate, const at::Tensor& o_norm, const at::Tensor& s_in,
    const at::Tensor& s_out, const at::Tensor& s_bound, const at::Tensor& out,
    const std::uint32_t positions, const bool retain) {
  require(positions >= 2 && positions <= 16,
          "kda-wide supports 2..16 positions");
  const std::int64_t width = 3 + static_cast<std::int64_t>(positions);
  const std::int64_t p = positions;
  require_kda_wide_f32(src_q, 12288 * width, "source q");
  require_kda_wide_f32(src_k, 12288 * width, "source k");
  require_kda_wide_f32(src_v, 12288 * width, "source v");
  require_kda_wide_f32(convw_q, 12288 * 4, "conv weight q");
  require_kda_wide_f32(convw_k, 12288 * 4, "conv weight k");
  require_kda_wide_f32(convw_v, 12288 * 4, "conv weight v");
  require_kda_wide_f32(a_log, 128, "a_log");
  require_kda_wide_f32(dt_bias, 12288, "dt_bias");
  require_kda_wide_f32(fb, p * 12288, "feature b");
  require_kda_wide_f32(beta, p * 96, "beta");
  require_kda_wide_f32(gate, p * 12288, "output gate");
  require_kda_wide_f32(o_norm, 128, "output norm");
  require_kda_wide_f32(s_in, 96 * 128 * 128, "state in");
  require_kda_wide_f32(s_out, 96 * 128 * 128, "state out");
  if (retain) {
    require_kda_wide_f32(s_bound, p * 96 * 128 * 128, "boundary states");
  }
  require_kda_wide_f32(out, p * 12288, "output rows");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "kda-wide: current MPS stream is unavailable");
  KdaWidePipeline& pipe = kda_wide_pipeline_for(stream->device());
  const KdaWideDimsV1 dims{positions, static_cast<std::uint32_t>(width),
                          retain ? 1u : 0u, 0u};
  const std::array<const at::Tensor*, 16> tensors{
      &src_q, &src_k, &src_v, &convw_q, &convw_k, &convw_v, &a_log, &dt_bias,
      &fb, &beta, &gate, &o_norm, &s_in, &s_out, retain ? &s_bound : &s_out,
      &out};
  std::array<id<MTLBuffer>, 16> buffers{};
  std::array<NSUInteger, 16> offsets{};
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    buffers[index] = tensor_buffer(*tensors[index]);
    require(buffers[index] != nil, "kda-wide tensor has no MTLBuffer");
    offsets[index] = checked_byte_offset(
        *tensors[index], static_cast<std::size_t>(tensors[index]->numel()) * 4,
        "kda-wide tensor");
  }
  static std::atomic<bool> announced{false};
  if (!announced.exchange(true)) {
    std::fprintf(stderr, "[kda-wide] fused wide KDA active (positions=%u retain=%d)\n",
                 positions, retain ? 1 : 0);
  }
  id<MTLComputePipelineState> pipeline = pipe.pipeline;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil, "kda-wide: current MPS stream has no encoder");
      [encoder setComputePipelineState:pipeline];
      for (std::size_t index = 0; index < 16; ++index) {
        [encoder setBuffer:buffers[index] offset:offsets[index] atIndex:index];
      }
      [encoder setBytes:&dims length:sizeof(dims) atIndex:16];
      [encoder dispatchThreadgroups:MTLSizeMake(96, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
  });
}


std::uint64_t loop_aten_sync_now_ns() noexcept {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

void* loop_aten_sync_begin() noexcept {
  if (!gpu_timeline_enabled()) {
    return nullptr;
  }
  try {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream == nullptr) {
      return nullptr;
    }
    __block id<MTLCommandBuffer> root = nil;
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        MPSCommandBuffer* command = stream->commandBuffer();
        root = command != nil ? command.rootCommandBuffer : nil;
      }
    });
    if (root == nil) {
      return nullptr;
    }
    return const_cast<void*>(CFBridgingRetain(root));
  } catch (...) {
    return nullptr;
  }
}

void loop_aten_sync_end(void* token, const std::uint64_t began_ns,
                        const std::uint64_t ended_ns) noexcept {
  if (!gpu_timeline_enabled()) {
    if (token != nullptr) {
      CFBridgingRelease(token);
    }
    return;
  }
  const std::uint64_t waited = ended_ns > began_ns ? ended_ns - began_ns : 0;
  g_aten_sync_total_ns.fetch_add(waited, std::memory_order_relaxed);
  bool timed = false;
  if (token != nullptr) {
    id<MTLCommandBuffer> root = (__bridge_transfer id<MTLCommandBuffer>)token;
    const double start = root.GPUStartTime;
    const double end = root.GPUEndTime;
    if (root.status == MTLCommandBufferStatusCompleted && start > 0.0 &&
        end > start) {
      timed = true;
      const auto start_ns = static_cast<std::uint64_t>(start * 1e9);
      const auto end_ns = static_cast<std::uint64_t>(end * 1e9);
      const std::uint64_t pre =
          start_ns > began_ns ? std::min(start_ns - began_ns, waited) : 0;
      const double kernel_start = root.kernelStartTime;
      const auto kernel_start_ns = static_cast<std::uint64_t>(kernel_start * 1e9);
      const std::uint64_t commit_delay =
          kernel_start > 0.0 && kernel_start_ns > began_ns
              ? std::min(kernel_start_ns - began_ns, pre)
              : 0;
      g_aten_sync_commit_ns.fetch_add(commit_delay, std::memory_order_relaxed);
      g_aten_sync_queue_ns.fetch_add(pre - commit_delay, std::memory_order_relaxed);
      const std::uint64_t exec = std::min(end_ns - start_ns, waited - pre);
      const std::uint64_t post = ended_ns > end_ns ? ended_ns - end_ns : 0;
      g_aten_sync_pre_ns.fetch_add(pre, std::memory_order_relaxed);
      g_aten_sync_exec_ns.fetch_add(exec, std::memory_order_relaxed);
      g_aten_sync_post_ns.fetch_add(post, std::memory_order_relaxed);
    }
  }
  if (!timed) {
    g_aten_sync_untimed.fetch_add(1, std::memory_order_relaxed);
  }
  const std::uint64_t n = g_aten_sync_n.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % 930 == 0) {
    const std::uint64_t untimed = g_aten_sync_untimed.load(std::memory_order_relaxed);
    const double timed_n = static_cast<double>(n > untimed ? n - untimed : 1);
    std::fprintf(
        stderr,
        "[aten-sync] n=%llu wait=%.3fms/sync pre=%.3f (commit=%.3f queue=%.3f) "
        "exec=%.3f post=%.3f (ms per timed sync) prep_ops=%.3fms/sync second=%.3fms/sync untimed=%llu\n",
        static_cast<unsigned long long>(n),
        g_aten_sync_total_ns.load(std::memory_order_relaxed) * 1e-6 / n,
        g_aten_sync_pre_ns.load(std::memory_order_relaxed) * 1e-6 / timed_n,
        g_aten_sync_commit_ns.load(std::memory_order_relaxed) * 1e-6 / timed_n,
        g_aten_sync_queue_ns.load(std::memory_order_relaxed) * 1e-6 / timed_n,
        g_aten_sync_exec_ns.load(std::memory_order_relaxed) * 1e-6 / timed_n,
        g_aten_sync_post_ns.load(std::memory_order_relaxed) * 1e-6 / timed_n,
        g_aten_sync_prep_ns.load(std::memory_order_relaxed) * 1e-6 / n,
        g_aten_sync_second_ns.load(std::memory_order_relaxed) * 1e-6 / n,
        static_cast<unsigned long long>(untimed));
  }
}

void loop_gpu_timeline_flush() noexcept {
  try {
    gpu_timeline_drain_and_print();
  } catch (...) {
  }
}

}  // namespace deltafin::provider_internal
