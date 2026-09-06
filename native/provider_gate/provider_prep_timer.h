#ifndef DELTAFIN_PROVIDER_PREP_TIMER_H
#define DELTAFIN_PROVIDER_PREP_TIMER_H

/*
 * Prepare-phase host-wait decomposition (K3-BRIEF-attention-timer-split,
 * 2026-09-02).
 *
 * Rust's `attention_resident` timer wraps exactly one FFI call
 * (deltafin_provider_target_sequence_prepare_v1) and its `expert_plan`
 * bucket is dominated by another (…_take_prefetch_hint_v1); both are
 * opaque to Rust. This unit splits each into
 *
 *   total  = wall of the ABI call on the decode thread
 *   wait   = host-BLOCKING waits inside it (loop semaphore waits on
 *            MTLSharedEvent listeners, the route-mailbox event wait, ATen
 *            device->host syncs)
 *   drain  = the part of `wait` spent before the GPU started the awaited
 *            command buffer, or before the previous layer's MoE tail
 *            completed — i.e. queued behind EARLIER work, not this layer's
 *   encode = total - wait (host CPU on the critical path: ATen dispatch,
 *            Metal encoding, allocation, bookkeeping — plus any host stall
 *            that is not a hooked wait site, so it is an upper bound)
 *
 * Prepare is bucketed by tile width because the two regimes take
 * different code: single-position tiles ride the bespoke loop / precommit
 * path, wide drafted-verify tiles the stock batched ATen path.
 *
 * Mechanism: the two ABI exports open a PrepPhaseScope on the calling
 * thread; every hooked wait site reports through prep_note_wait, a no-op
 * outside a scope. Two clock reads per wait, no behavior change, always
 * compiled. Counters are cumulative and monotonic so per-chunk
 * differencing works like [phases] always has; read them with
 * deltafin_provider_prep_timer_report_v1.
 *
 * Report word order (kPrepTimerReportWords):
 *   [0..6)   single-position prepare: calls, total_ns, wait_ns, drain_ns,
 *            waits (hooked wait events), untimed_syncs (waits with no
 *            GPU timestamp, e.g. ATen .to(kCPU) — drain unknown, 0)
 *   [6..12)  wide-tile prepare, same fields
 *   [12..18) prefetch hint, same fields
 *   [18..24) finish_experts, single-row tiles; [24..30) wide tiles
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace deltafin::provider_internal {

enum class PrepPhase : int { None = 0, Prepare = 1, Hint = 2, Finish = 3 };

constexpr std::size_t kPrepTimerBucketWords = 6;
constexpr std::size_t kPrepTimerSubRegions = 3;
/* Words [30..33): cumulative WALL ns of three sub-regions inside the finish
 * scope (any tile width): [30] the per-layer routed-input device->host
 * materialization (an untimed ATen sync), [31] execute_routed_moe_positions
 * (Metal MoE encode + CB wait + output handling), [32] the next layer's KDA
 * precommit hook (encode + fused flush / tail drain). finish total minus the
 * three = the remaining host work (row bookkeeping, residual/anchor ATen
 * ops). Sub-regions are counted only while a Finish scope is open. */
constexpr std::size_t kPrepTimerReportWords =
    5 * kPrepTimerBucketWords + kPrepTimerSubRegions;

struct PrepTimerBucket {
  std::atomic<std::uint64_t> calls{0};
  std::atomic<std::uint64_t> total_ns{0};
  std::atomic<std::uint64_t> wait_ns{0};
  std::atomic<std::uint64_t> drain_ns{0};
  std::atomic<std::uint64_t> waits{0};
  std::atomic<std::uint64_t> untimed_syncs{0};
};

struct PrepTimerCounters {
  /* [0] single-position tiles, [1] wide (drafted verify) tiles. */
  PrepTimerBucket prepare[2];
  PrepTimerBucket hint;
  /// finish_experts ABI call (Rust's expert_kernel bucket): [0] single-row
  /// tiles, [1] wide tiles. Waits = the Metal MoE CB waits (timestamped) and
  /// the routed-input device->host syncs (untimed).
  PrepTimerBucket finish[2];
  std::atomic<std::uint64_t> finish_sub_ns[kPrepTimerSubRegions]{};
};

inline PrepTimerCounters& prep_timer_counters() {
  static PrepTimerCounters counters;
  return counters;
}

struct PrepThreadState {
  PrepPhase phase = PrepPhase::None;
  int bucket = 0;
  std::uint64_t wait_ns = 0;
  std::uint64_t drain_ns = 0;
  std::uint64_t waits = 0;
  std::uint64_t untimed_syncs = 0;
};

inline PrepThreadState& prep_thread_state() {
  thread_local PrepThreadState state;
  return state;
}

inline std::uint64_t prep_steady_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

[[nodiscard]] inline bool prep_phase_active() {
  return prep_thread_state().phase != PrepPhase::None;
}

/* Report one host-blocking wait that just returned on this thread.
 * `drain_ns` is the portion attributable to earlier GPU work (clamped to
 * the wait); `timestamped` is false when the site had no command-buffer
 * timestamp to derive it from. No-op outside a phase scope. */
inline void prep_note_wait(const std::uint64_t wait_ns,
                           const std::uint64_t drain_ns,
                           const bool timestamped) {
  PrepThreadState& state = prep_thread_state();
  if (state.phase == PrepPhase::None) {
    return;
  }
  state.wait_ns += wait_ns;
  state.drain_ns += std::min(drain_ns, wait_ns);
  state.waits += 1;
  if (!timestamped) {
    state.untimed_syncs += 1;
  }
}

/* RAII wall timer for one finish sub-region; accumulates only while a
 * Finish scope is open on this thread (no-op elsewhere, e.g. prefill). */
class PrepSubScope {
 public:
  explicit PrepSubScope(const std::size_t region) noexcept
      : region_(region),
        active_(region < kPrepTimerSubRegions &&
                prep_thread_state().phase == PrepPhase::Finish),
        started_ns_(active_ ? prep_steady_ns() : 0) {}
  ~PrepSubScope() {
    if (active_) {
      prep_timer_counters().finish_sub_ns[region_].fetch_add(
          prep_steady_ns() - started_ns_, std::memory_order_relaxed);
    }
  }
  PrepSubScope(const PrepSubScope&) = delete;
  PrepSubScope& operator=(const PrepSubScope&) = delete;

 private:
  std::size_t region_;
  bool active_;
  std::uint64_t started_ns_;
};

/* RAII phase scope opened by the ABI export. A nested open (none exists
 * today) leaves the outer scope's accounting untouched. */
class PrepPhaseScope {
 public:
  PrepPhaseScope(const PrepPhase phase, const int bucket) noexcept
      : started_ns_(prep_steady_ns()) {
    PrepThreadState& state = prep_thread_state();
    if (state.phase != PrepPhase::None) {
      return;
    }
    active_ = true;
    state.phase = phase;
    state.bucket = bucket;
    state.wait_ns = 0;
    state.drain_ns = 0;
    state.waits = 0;
    state.untimed_syncs = 0;
  }

  ~PrepPhaseScope() {
    if (!active_) {
      return;
    }
    PrepThreadState& state = prep_thread_state();
    const std::uint64_t total_ns = prep_steady_ns() - started_ns_;
    PrepTimerCounters& counters = prep_timer_counters();
    PrepTimerBucket& bucket = state.phase == PrepPhase::Hint
        ? counters.hint
        : state.phase == PrepPhase::Finish
            ? counters.finish[state.bucket == 0 ? 0 : 1]
            : counters.prepare[state.bucket == 0 ? 0 : 1];
    bucket.calls.fetch_add(1, std::memory_order_relaxed);
    bucket.total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    bucket.wait_ns.fetch_add(state.wait_ns, std::memory_order_relaxed);
    bucket.drain_ns.fetch_add(state.drain_ns, std::memory_order_relaxed);
    bucket.waits.fetch_add(state.waits, std::memory_order_relaxed);
    bucket.untimed_syncs.fetch_add(state.untimed_syncs,
                                   std::memory_order_relaxed);
    state.phase = PrepPhase::None;
  }

  PrepPhaseScope(const PrepPhaseScope&) = delete;
  PrepPhaseScope& operator=(const PrepPhaseScope&) = delete;

 private:
  std::uint64_t started_ns_;
  bool active_ = false;
};

/* Copies up to `count` words of the cumulative report (order above);
 * words past kPrepTimerReportWords are zeroed. */
inline void prep_timer_report(std::uint64_t* values, const std::size_t count) {
  if (values == nullptr) {
    return;
  }
  const PrepTimerCounters& counters = prep_timer_counters();
  const PrepTimerBucket* buckets[5] = {&counters.prepare[0], &counters.prepare[1],
                                       &counters.hint, &counters.finish[0],
                                       &counters.finish[1]};
  for (std::size_t index = 0; index < count; ++index) {
    std::uint64_t value = 0;
    if (index >= 5 * kPrepTimerBucketWords && index < kPrepTimerReportWords) {
      value = counters.finish_sub_ns[index - 5 * kPrepTimerBucketWords].load(
          std::memory_order_relaxed);
    } else if (index < 5 * kPrepTimerBucketWords) {
      const PrepTimerBucket& bucket = *buckets[index / kPrepTimerBucketWords];
      switch (index % kPrepTimerBucketWords) {
        case 0: value = bucket.calls.load(std::memory_order_relaxed); break;
        case 1: value = bucket.total_ns.load(std::memory_order_relaxed); break;
        case 2: value = bucket.wait_ns.load(std::memory_order_relaxed); break;
        case 3: value = bucket.drain_ns.load(std::memory_order_relaxed); break;
        case 4: value = bucket.waits.load(std::memory_order_relaxed); break;
        default:
          value = bucket.untimed_syncs.load(std::memory_order_relaxed);
          break;
      }
    }
    values[index] = value;
  }
}

}  // namespace deltafin::provider_internal

#endif
