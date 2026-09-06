#include "provider_route_mailbox.h"
#include "provider_prep_timer.h"

#if !defined(__APPLE__)
#error "provider_route_mailbox.mm is an Apple-only Objective-C++ source"
#endif

#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "Deltafin's route-mailbox Metal bridge requires Objective-C ARC"
#endif

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#include "deltafin_embedded_route_mailbox_metallib.h"
#else
#include "deltafin_embedded_route_mailbox_msl.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

namespace deltafin::provider_internal {
namespace {

/*
 * Env opt-in probe (K3_ROUTE_SYNC_PROBE=1): decompose the per-layer route
 * boundary into its host-encode and GPU-drain halves. One stderr summary per
 * 92 mailbox calls (one decode token), aggregated under the mailbox mutex, so
 * the probe adds no per-call I/O to the measured path.
 */
bool route_sync_probe_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_ROUTE_SYNC_PROBE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

struct RouteSyncProbe {
  std::uint64_t calls = 0;
  double encode_ms = 0.0;
  double commit_ms = 0.0;
  double wait_ms = 0.0;
  double wait_min_ms = std::numeric_limits<double>::infinity();
  double wait_max_ms = 0.0;
  // Decomposition of the wait via the root command buffer's GPU timestamps
  // (CACurrentMediaTime domain): gpu_busy is the buffer's own execution time
  // — under COMMIT_AND_CONTINUE that buffer holds the whole layer chain
  // since the previous boundary, so busy ≈ the layer's true GPU compute.
  // start_lag = time between the host starting to wait and the buffer
  // starting on the GPU (prior-boundary tail + scheduling bubble);
  // completion_lag = GPU-done to listener-notify latency.
  double gpu_busy_ms = 0.0;
  double gpu_start_lag_ms = 0.0;
  double completion_lag_ms = 0.0;

  void absorb(const double encode, const double commit, const double wait,
              const double gpu_busy, const double gpu_start_lag,
              const double completion_lag) {
    ++calls;
    encode_ms += encode;
    commit_ms += commit;
    wait_ms += wait;
    wait_min_ms = std::min(wait_min_ms, wait);
    wait_max_ms = std::max(wait_max_ms, wait);
    gpu_busy_ms += gpu_busy;
    gpu_start_lag_ms += gpu_start_lag;
    completion_lag_ms += completion_lag;
    constexpr std::uint64_t kRoutedLayersPerToken = 92;
    if (calls < kRoutedLayersPerToken) {
      return;
    }
    const double count = static_cast<double>(calls);
    std::fprintf(stderr,
                 "[route-probe] calls=%llu encode_mean=%.3fms "
                 "commit_mean=%.3fms wait_mean=%.3fms wait_min=%.3fms "
                 "wait_max=%.3fms total=%.1fms\n",
                 static_cast<unsigned long long>(calls), encode_ms / count,
                 commit_ms / count, wait_ms / count, wait_min_ms, wait_max_ms,
                 encode_ms + commit_ms + wait_ms);
    std::fprintf(stderr,
                 "[route-probe] wait decomposition: gpu_busy_mean=%.3fms "
                 "gpu_start_lag_mean=%.3fms completion_lag_mean=%.3fms "
                 "(busy %.0f%% of wait)\n",
                 gpu_busy_ms / count, gpu_start_lag_ms / count,
                 completion_lag_ms / count,
                 wait_ms > 0.0 ? 100.0 * gpu_busy_ms / wait_ms : 0.0);
    *this = RouteSyncProbe{};
  }
};

double elapsed_ms(const std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - since)
      .count();
}

// Seconds in the same mach-time domain as MTLCommandBuffer's GPUStartTime /
// GPUEndTime (CACurrentMediaTime equivalent, without linking QuartzCore).
double media_time_seconds() {
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    return info;
  }();
  return static_cast<double>(mach_absolute_time()) * timebase.numer /
         timebase.denom / 1e9;
}

static_assert(offsetof(RouteMailboxT1, expert_ids) == 0);
static_assert(offsetof(RouteMailboxT1, weight_bits) == 128);
static_assert(sizeof(RouteMailboxT1) == 192);

// Mirror of the MSL DeltafinPilotMailboxRows layout (header 4x uint32, then
// the id and score arrays at 8-aligned offsets).
static_assert(offsetof(PilotMailboxRows, expert_ids) == 16);
static_assert(offsetof(PilotMailboxRows, choice_scores) == 16400);
static_assert(sizeof(PilotMailboxRows) == 24592);

struct MailboxState {
  std::mutex mutex;
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> mailbox = nil;
  id<MTLSharedEvent> event = nil;
  MTLSharedEventListener* listener = nil;
  std::uint64_t next_event = 1;
  bool rejected = false;
  RouteSyncProbe probe;
  // Pilot-hint mailbox slot (sync-E removal). Poll-only: no listener — the
  // consumer checks signaledValue at hint time and never blocks.
  id<MTLComputePipelineState> pilot_pipeline = nil;
  id<MTLBuffer> pilot_mailbox = nil;
  id<MTLSharedEvent> pilot_event = nil;
  std::uint64_t pilot_next_event = 1;
  bool pilot_rejected = false;
};

MailboxState& mailbox_state() {
  static MailboxState value;
  return value;
}

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
dispatch_data_t copy_embedded_route_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedRouteMailboxMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedRouteMailboxMetallib,
              kDeltafinEmbeddedRouteMailboxMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedRouteMailboxMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}
#endif

id<MTLBuffer> tensor_buffer(const at::Tensor& tensor) {
  // This is the same reviewed MPS storage representation used by the existing
  // current-stream Metal bridges in this repository.
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

bool qualifies(const at::Tensor& expert_ids, const at::Tensor& weights) {
  return expert_ids.defined() && weights.defined() &&
         expert_ids.device().is_mps() && weights.device().is_mps() &&
         expert_ids.scalar_type() == at::kLong &&
         weights.scalar_type() == at::kFloat && expert_ids.is_contiguous() &&
         weights.is_contiguous() && !expert_ids.requires_grad() &&
         !weights.requires_grad() &&
         expert_ids.numel() == static_cast<std::int64_t>(kRouteMailboxTopK) &&
         weights.numel() == static_cast<std::int64_t>(kRouteMailboxTopK);
}

NSUInteger checked_byte_offset(const at::Tensor& tensor,
                               const NSUInteger bytes) {
  TORCH_CHECK(tensor.storage_offset() >= 0,
              "route mailbox tensor has a negative storage offset");
  const auto elements = static_cast<std::uint64_t>(tensor.storage_offset());
  const auto width = static_cast<std::uint64_t>(tensor.element_size());
  TORCH_CHECK(elements <= std::numeric_limits<std::uint64_t>::max() / width,
              "route mailbox tensor offset overflows uint64");
  const std::uint64_t raw = elements * width;
  TORCH_CHECK(raw <= std::numeric_limits<NSUInteger>::max(),
              "route mailbox tensor offset exceeds NSUInteger");
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  TORCH_CHECK(buffer != nil, "route mailbox tensor has no MTLBuffer");
  const NSUInteger offset = static_cast<NSUInteger>(raw);
  TORCH_CHECK(offset <= buffer.length && bytes <= buffer.length - offset,
              "route mailbox tensor span exceeds its MTLBuffer");
  return offset;
}

void ensure_resources(MailboxState& state, id<MTLDevice> device) {
  if (state.device == device && state.pipeline != nil &&
      state.mailbox != nil && state.event != nil && state.listener != nil &&
      state.pilot_pipeline != nil && state.pilot_mailbox != nil &&
      state.pilot_event != nil) {
    return;
  }

  state.device = device;
  state.pipeline = nil;
  state.mailbox = nil;
  state.event = nil;
  state.listener = nil;
  state.next_event = 1;
  state.pilot_pipeline = nil;
  state.pilot_mailbox = nil;
  state.pilot_event = nil;
  state.pilot_next_event = 1;

  NSError* error = nil;
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  dispatch_data_t data = copy_embedded_route_metallib();
  TORCH_CHECK(data != nullptr,
              "route mailbox embedded metallib wrapping failed");
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  TORCH_CHECK(library != nil, "route mailbox metallib loading failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
#else
  MTLCompileOptions* options = [MTLCompileOptions new];
  // The standalone development graph retains source compilation for its
  // shader gate. Cargo production builds define the precompiled branch above.
  id<MTLLibrary> library = [device
      newLibraryWithSource:[NSString
                               stringWithUTF8String:
                                   kDeltafinEmbeddedRouteMailboxMsl]
                   options:options
                     error:&error];
  TORCH_CHECK(library != nil, "route mailbox Metal compilation failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
#endif
  id<MTLFunction> function =
      [library newFunctionWithName:@"deltafin_route_mailbox_t1"];
  TORCH_CHECK(function != nil, "route mailbox Metal function is missing");
  state.pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(state.pipeline != nil,
              "route mailbox Metal pipeline creation failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
  state.mailbox = [device
      newBufferWithLength:sizeof(RouteMailboxT1)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeDefaultCache];
  TORCH_CHECK(state.mailbox != nil,
              "route mailbox shared allocation failed");
  state.event = [device newSharedEvent];
  TORCH_CHECK(state.event != nil,
              "route mailbox shared event is unavailable");
  dispatch_queue_t listener_queue = dispatch_queue_create(
      "deltafin.route-mailbox", DISPATCH_QUEUE_SERIAL);
  state.listener = [[MTLSharedEventListener alloc]
      initWithDispatchQueue:listener_queue];
  TORCH_CHECK(state.listener != nil,
              "route mailbox event listener is unavailable");

  id<MTLFunction> pilot_function =
      [library newFunctionWithName:@"deltafin_pilot_mailbox_rows"];
  TORCH_CHECK(pilot_function != nil,
              "pilot mailbox Metal function is missing");
  state.pilot_pipeline =
      [device newComputePipelineStateWithFunction:pilot_function error:&error];
  TORCH_CHECK(state.pilot_pipeline != nil,
              "pilot mailbox Metal pipeline creation failed: ",
              error == nil ? "unknown"
                           : error.localizedDescription.UTF8String);
  state.pilot_mailbox = [device
      newBufferWithLength:sizeof(PilotMailboxRows)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeDefaultCache];
  TORCH_CHECK(state.pilot_mailbox != nil,
              "pilot mailbox shared allocation failed");
  state.pilot_event = [device newSharedEvent];
  TORCH_CHECK(state.pilot_event != nil,
              "pilot mailbox shared event is unavailable");
}

void materialize(const at::Tensor& expert_ids, const at::Tensor& weights,
                 RouteMailboxT1& output) {
  TORCH_CHECK(qualifies(expert_ids, weights),
              "route mailbox inputs do not satisfy its T=1 contract");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "current MPS stream is unavailable");

  MailboxState& state = mailbox_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  TORCH_CHECK(!state.rejected, "route mailbox was rejected by an earlier gate");
  id<MTLDevice> device = stream->device();
  TORCH_CHECK(device != nil, "current MPS device is unavailable");
  ensure_resources(state, device);

  constexpr NSUInteger kIdBytes =
      kRouteMailboxTopK * sizeof(std::int64_t);
  constexpr NSUInteger kWeightBytes =
      kRouteMailboxTopK * sizeof(std::uint32_t);
  const NSUInteger id_offset = checked_byte_offset(expert_ids, kIdBytes);
  const NSUInteger weight_offset = checked_byte_offset(weights, kWeightBytes);
  TORCH_CHECK(state.next_event != std::numeric_limits<std::uint64_t>::max(),
              "route mailbox event counter exhausted");
  const std::uint64_t event_value = state.next_event++;
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  [state.event notifyListener:state.listener
                     atValue:event_value
                       block:^(id<MTLSharedEvent>, std::uint64_t) {
                         dispatch_semaphore_signal(ready);
                       }];

  const bool probing = route_sync_probe_enabled();
  const auto encode_started = std::chrono::steady_clock::now();
  __block id<MTLCommandBuffer> root_command = nil;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      TORCH_CHECK(encoder != nil,
                  "route mailbox current-stream encoder is unavailable");
      [encoder setComputePipelineState:state.pipeline];
      [encoder setBuffer:tensor_buffer(expert_ids)
                  offset:id_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(weights)
                  offset:weight_offset
                 atIndex:1];
      [encoder setBuffer:state.mailbox offset:0 atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(kRouteMailboxTopK, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kRouteMailboxTopK, 1, 1)];
      stream->endKernelCoalescing();
      MPSCommandBuffer* command = stream->commandBuffer();
      TORCH_CHECK(command != nil,
                  "route mailbox MPS command buffer is unavailable");
      root_command = command.rootCommandBuffer;
      TORCH_CHECK(root_command != nil,
                  "route mailbox root command buffer is unavailable");
      [root_command encodeSignalEvent:state.event value:event_value];
    }
  });

  // Commit precisely through the mailbox and let later MPS work begin on a
  // fresh command buffer.  Waiting on this event is narrower than a global MPS
  // synchronization and keeps both route tensors under one host boundary.
  const double encode_elapsed = probing ? elapsed_ms(encode_started) : 0.0;
  const auto commit_started = std::chrono::steady_clock::now();
  stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
  const double commit_elapsed = probing ? elapsed_ms(commit_started) : 0.0;
  const auto wait_started = std::chrono::steady_clock::now();
  const double wait_started_media = probing ? media_time_seconds() : 0.0;
  constexpr std::int64_t kTimeoutNanoseconds = 10LL * NSEC_PER_SEC;
  // Attention-timer split (provider_prep_timer.h): report this event wait
  // to the open prepare scope, with the pre-GPUStartTime part as drain.
  const bool prep_timed = deltafin::provider_internal::prep_phase_active();
  const double prep_wait_started = prep_timed ? media_time_seconds() : 0.0;
  const long wait_result = dispatch_semaphore_wait(
      ready, dispatch_time(DISPATCH_TIME_NOW, kTimeoutNanoseconds));
  if (prep_timed) {
    const double prep_wait_ended = media_time_seconds();
    const double prep_gpu_start = root_command.GPUStartTime;
    const double prep_waited =
        std::max(0.0, prep_wait_ended - prep_wait_started);
    const double prep_drain = prep_gpu_start > 0.0
        ? std::clamp(prep_gpu_start - prep_wait_started, 0.0, prep_waited)
        : 0.0;
    deltafin::provider_internal::prep_note_wait(
        static_cast<std::uint64_t>(prep_waited * 1e9),
        static_cast<std::uint64_t>(prep_drain * 1e9), prep_gpu_start > 0.0);
  }
  TORCH_CHECK(wait_result == 0, "route mailbox event timed out");
  if (probing) {
    const double wait_ended_media = media_time_seconds();
    const double gpu_start = root_command.GPUStartTime;
    const double gpu_end = root_command.GPUEndTime;
    // A buffer that reports no GPU timestamps (seen on the first boundary)
    // would poison the lag means with wall-clock-sized values; skip its
    // decomposition and keep only the plain wait.
    const bool timestamps_valid = gpu_start > 0.0 && gpu_end >= gpu_start;
    const double gpu_busy =
        timestamps_valid ? (gpu_end - gpu_start) * 1000.0 : 0.0;
    const double gpu_start_lag = timestamps_valid
        ? std::max(0.0, gpu_start - wait_started_media) * 1000.0
        : 0.0;
    const double completion_lag = timestamps_valid
        ? std::max(0.0, wait_ended_media - gpu_end) * 1000.0
        : 0.0;
    state.probe.absorb(encode_elapsed, commit_elapsed,
                       elapsed_ms(wait_started), gpu_busy, gpu_start_lag,
                       completion_lag);
  }
  TORCH_CHECK(root_command.error == nil, "route mailbox command failed: ",
              root_command.error.localizedDescription.UTF8String);
  const void* contents = state.mailbox.contents;
  TORCH_CHECK(contents != nullptr,
              "route mailbox shared storage is not CPU-visible");
  std::memcpy(&output, contents, sizeof(output));
}

bool pilot_qualifies(const at::Tensor& expert_ids,
                     const at::Tensor& choice_scores) {
  if (!expert_ids.defined() || !choice_scores.defined() ||
      !expert_ids.device().is_mps() || !choice_scores.device().is_mps() ||
      expert_ids.scalar_type() != at::kLong ||
      choice_scores.scalar_type() != at::kFloat ||
      !expert_ids.is_contiguous() || !choice_scores.is_contiguous() ||
      expert_ids.requires_grad() || choice_scores.requires_grad() ||
      expert_ids.dim() != 2 || choice_scores.dim() != 2 ||
      expert_ids.sizes() != choice_scores.sizes()) {
    return false;
  }
  const std::int64_t positions = expert_ids.size(0);
  const std::int64_t width = expert_ids.size(1);
  return positions >= 1 &&
         positions <= static_cast<std::int64_t>(kPilotMailboxMaxPositions) &&
         width >= 1 &&
         width <= static_cast<std::int64_t>(kPilotMailboxMaxWidth);
}

std::uint64_t pilot_publish(const at::Tensor& expert_ids,
                            const at::Tensor& choice_scores,
                            const std::uint32_t layer_index,
                            const std::uint32_t expert_count) {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr,
              "pilot mailbox: current MPS stream is unavailable");
  MailboxState& state = mailbox_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  TORCH_CHECK(!state.pilot_rejected,
              "pilot mailbox was rejected by an earlier gate");
  id<MTLDevice> device = stream->device();
  TORCH_CHECK(device != nil,
              "pilot mailbox: current MPS device is unavailable");
  ensure_resources(state, device);

  const auto positions = static_cast<std::uint32_t>(expert_ids.size(0));
  const auto width = static_cast<std::uint32_t>(expert_ids.size(1));
  const NSUInteger total = static_cast<NSUInteger>(positions) * width;
  const NSUInteger id_bytes = total * sizeof(std::int64_t);
  const NSUInteger score_bytes = total * sizeof(float);
  const NSUInteger id_offset = checked_byte_offset(expert_ids, id_bytes);
  const NSUInteger score_offset =
      checked_byte_offset(choice_scores, score_bytes);
  TORCH_CHECK(state.pilot_next_event !=
                  std::numeric_limits<std::uint64_t>::max(),
              "pilot mailbox event counter exhausted");
  const std::uint64_t generation = state.pilot_next_event++;
  struct {
    std::uint32_t layer_index;
    std::uint32_t expert_count;
    std::uint32_t position_count;
    std::uint32_t width;
  } header{layer_index, expert_count, positions, width};

  // Encode-and-signal only: no commit, no wait. The copy rides the stream's
  // next natural commit (per-layer loop fence or route boundary), so the
  // event is already signaled when the consumer polls at mailbox-open. A
  // consumer that polls too early gets a miss and returns an empty hint —
  // the true-route topup absorbs it.
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      TORCH_CHECK(encoder != nil,
                  "pilot mailbox current-stream encoder is unavailable");
      [encoder setComputePipelineState:state.pilot_pipeline];
      [encoder setBuffer:tensor_buffer(expert_ids)
                  offset:id_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(choice_scores)
                  offset:score_offset
                 atIndex:1];
      [encoder setBuffer:state.pilot_mailbox offset:0 atIndex:2];
      [encoder setBytes:&header length:sizeof(header) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(total < 64 ? total : 64, 1, 1)];
      stream->endKernelCoalescing();
      MPSCommandBuffer* command = stream->commandBuffer();
      TORCH_CHECK(command != nil,
                  "pilot mailbox MPS command buffer is unavailable");
      id<MTLCommandBuffer> root_command = command.rootCommandBuffer;
      TORCH_CHECK(root_command != nil,
                  "pilot mailbox root command buffer is unavailable");
      [root_command encodeSignalEvent:state.pilot_event value:generation];
    }
  });
  // v1.1: commit (never wait). v1 relied on the stream's next natural commit
  // and measured 95% consumer misses on the fused loop stack — the ATen
  // stream barely commits per layer there, so publishes sat encoded in
  // limbo while the read path lost its pilot lead (expert_demand_read
  // +80%, marginal +15%). A COMMIT_AND_CONTINUE here starts the copy
  // immediately and keeps the host free; the poll stays wait-free.
  stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
  return generation;
}

}  // namespace

extern "C" int deltafin_route_mailbox_metallib_cycle_test_v1(int cycles) {
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  if (cycles <= 0 || cycles > 256) return -1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return -2;
    for (int cycle = 0; cycle < cycles; ++cycle) {
      @autoreleasepool {
        dispatch_data_t data = copy_embedded_route_metallib();
        if (data == nullptr) return -3;
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithData:data error:&error];
        if (library == nil || error != nil) return -4;
      }
    }
  }
  return 0;
#else
  (void)cycles;
  return -5;
#endif
}

bool try_commit_mps_stream_for_route() noexcept {
  static std::atomic<bool> rejected{false};
  if (rejected.load(std::memory_order_relaxed)) {
    return false;
  }
  try {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    if (stream == nullptr) {
      rejected.store(true, std::memory_order_relaxed);
      return false;
    }
    // Non-blocking: commits the encoded prefix and continues on a fresh
    // command buffer, exactly the commit flavor the mailbox itself uses.
    stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    return true;
  } catch (...) {
    // A failed scheduling hint must never make target routing unavailable.
    rejected.store(true, std::memory_order_relaxed);
    return false;
  }
}

bool try_materialize_mps_route_t1(const at::Tensor& expert_ids,
                                  const at::Tensor& weights,
                                  RouteMailboxT1& output) noexcept {
  RouteMailboxT1 candidate;
  try {
    if (!qualifies(expert_ids, weights)) {
      return false;
    }
    materialize(expert_ids, weights, candidate);
  } catch (...) {
    // A private bridge must never make target routing unavailable.  Reject it
    // process-wide after any setup/runtime fault and let every caller use the
    // established LibTorch .to(CPU) path instead.
    try {
      MailboxState& state = mailbox_state();
      std::lock_guard<std::mutex> guard(state.mutex);
      state.rejected = true;
    } catch (...) {
      // Even failure bookkeeping is subordinate to preserving the fallback.
    }
    return false;
  }
  output = candidate;
  return true;
}

std::uint64_t try_publish_mps_pilot_rows(const at::Tensor& expert_ids,
                                         const at::Tensor& choice_scores,
                                         const std::uint32_t layer_index,
                                         const std::uint32_t expert_count)
    noexcept {
  try {
    if (!pilot_qualifies(expert_ids, choice_scores)) {
      static bool reported = false;
      if (!reported) {
        reported = true;
        std::fprintf(stderr,
                     "[pilot-mailbox] first qualify-fail: defined=%d/%d "
                     "mps=%d/%d dtype=%d/%d contig=%d/%d dim=%lld sizes_eq=%d\n",
                     expert_ids.defined(), choice_scores.defined(),
                     expert_ids.defined() && expert_ids.device().is_mps(),
                     choice_scores.defined() &&
                         choice_scores.device().is_mps(),
                     expert_ids.defined() &&
                         expert_ids.scalar_type() == at::kLong,
                     choice_scores.defined() &&
                         choice_scores.scalar_type() == at::kFloat,
                     expert_ids.defined() && expert_ids.is_contiguous(),
                     choice_scores.defined() && choice_scores.is_contiguous(),
                     expert_ids.defined()
                         ? static_cast<long long>(expert_ids.dim())
                         : -1,
                     expert_ids.defined() && choice_scores.defined() &&
                         expert_ids.sizes() == choice_scores.sizes());
      }
      return 0;
    }
    return pilot_publish(expert_ids, choice_scores, layer_index,
                         expert_count);
  } catch (const std::exception& error) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr, "[pilot-mailbox] first publish exception: %s\n",
                   error.what());
    }
    try {
      MailboxState& state = mailbox_state();
      std::lock_guard<std::mutex> guard(state.mutex);
      state.pilot_rejected = true;
    } catch (...) {
    }
    return 0;
  } catch (...) {
    // The pilot is advisory: any bridge fault latches this path off and the
    // caller keeps the established .to(kCPU) route for future layers.
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "[pilot-mailbox] first publish exception: non-std\n");
    }
    try {
      MailboxState& state = mailbox_state();
      std::lock_guard<std::mutex> guard(state.mutex);
      state.pilot_rejected = true;
    } catch (...) {
      // Failure bookkeeping is subordinate to preserving the fallback.
    }
    return 0;
  }
}

bool try_poll_mps_pilot_rows(const std::uint64_t generation,
                             PilotMailboxRows& output) noexcept {
  if (generation == 0) {
    return false;
  }
  try {
    MailboxState& state = mailbox_state();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.pilot_rejected || state.pilot_event == nil ||
        state.pilot_mailbox == nil) {
      return false;
    }
    if (state.pilot_event.signaledValue < generation) {
      return false;
    }
    const auto* contents =
        static_cast<const PilotMailboxRows*>(state.pilot_mailbox.contents);
    if (contents == nullptr) {
      return false;
    }
    // Header first, then only the populated spans — the slot arrays are
    // sized for the pilot maxima and rarely full.
    std::memcpy(&output, contents, offsetof(PilotMailboxRows, expert_ids));
    if (output.position_count < 1 ||
        output.position_count > kPilotMailboxMaxPositions ||
        output.width < 1 || output.width > kPilotMailboxMaxWidth) {
      return false;
    }
    const std::size_t slots =
        static_cast<std::size_t>(output.position_count) * output.width;
    std::memcpy(output.expert_ids.data(), contents->expert_ids.data(),
                slots * sizeof(std::int64_t));
    std::memcpy(output.choice_scores.data(), contents->choice_scores.data(),
                slots * sizeof(float));
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace deltafin::provider_internal
