#ifndef DELTAFIN_PROVIDER_ROUTE_MAILBOX_H
#define DELTAFIN_PROVIDER_ROUTE_MAILBOX_H

#include <ATen/core/Tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace deltafin::provider_internal {

constexpr std::size_t kRouteMailboxTopK = 16;

/*
 * Host-visible, byte-exact result of one current-stream MPS boundary.  Keep
 * weights as bits: routing is an exact target decision, so crossing the host
 * boundary must not perform a floating-point conversion or Python boxing.
 */
struct RouteMailboxT1 {
  std::array<std::int64_t, kRouteMailboxTopK> expert_ids = {};
  std::array<std::uint32_t, kRouteMailboxTopK> weight_bits = {};
};

/*
 * Pilot-hint mailbox (sync-E removal, K3_PILOT_MAILBOX=1). The pilot's
 * prediction tensors are copied GPU-side into this host-shared struct and a
 * shared event is signaled in stream order; the consumer polls the event and
 * reads the struct with no .to(kCPU) drain. Sized for the pilot maxima.
 */
constexpr std::size_t kPilotMailboxMaxPositions = 64;
constexpr std::size_t kPilotMailboxMaxWidth = 32;
constexpr std::size_t kPilotMailboxSlots =
    kPilotMailboxMaxPositions * kPilotMailboxMaxWidth;

struct PilotMailboxRows {
  std::uint32_t layer_index = 0;
  std::uint32_t expert_count = 0;
  std::uint32_t position_count = 0;
  std::uint32_t width = 0;
  std::array<std::int64_t, kPilotMailboxSlots> expert_ids = {};
  std::array<float, kPilotMailboxSlots> choice_scores = {};
};

#if defined(__APPLE__)
/*
 * Returns false without modifying `output` when the tensors do not qualify or
 * when Metal setup/execution fails.  Callers must retain their established
 * tensor-to-CPU path as the authoritative fallback.
 */
[[nodiscard]] bool try_materialize_mps_route_t1(
    const at::Tensor& expert_ids, const at::Tensor& weights,
    RouteMailboxT1& output) noexcept;

/*
 * Encode a GPU-side copy of one pilot prediction (ids [P,W] kLong, scores
 * [P,W] kFloat, both contiguous MPS) into the shared pilot mailbox and signal
 * its event in stream order. Never commits and never waits — the copy rides
 * the stream's next natural commit (per-layer loop fence or route boundary),
 * so by the time the host consumes the hint at mailbox-open the event has
 * already signaled. Returns the generation to poll, or 0 on any failure
 * (latches off process-wide; callers keep the .to(kCPU) path for that case).
 */
[[nodiscard]] std::uint64_t try_publish_mps_pilot_rows(
    const at::Tensor& expert_ids, const at::Tensor& choice_scores,
    std::uint32_t layer_index, std::uint32_t expert_count) noexcept;

/*
 * Non-blocking: fills `output` and returns true iff `generation` has
 * signaled. False means not-ready or rejected — the caller returns an EMPTY
 * hint (the true-route topup covers the misses); it must NOT fall back to a
 * stream drain, which would reintroduce exactly the stall this removes.
 */
[[nodiscard]] bool try_poll_mps_pilot_rows(std::uint64_t generation,
                                           PilotMailboxRows& output) noexcept;

/*
 * Commit everything encoded so far on the current MPS stream without waiting
 * (COMMIT_AND_CONTINUE), so the GPU starts executing while the host keeps
 * encoding. Purely a scheduling hint: op inputs, outputs, and order are
 * unchanged. Returns false (and latches off) on any failure; callers must
 * treat a failed flush as a no-op.
 */
bool try_commit_mps_stream_for_route() noexcept;
#endif

}  // namespace deltafin::provider_internal

#endif
