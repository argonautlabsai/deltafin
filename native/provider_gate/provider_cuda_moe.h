#ifndef DELTAFIN_PROVIDER_CUDA_MOE_H
#define DELTAFIN_PROVIDER_CUDA_MOE_H

#include "provider_moe.h"

#include <ATen/ATen.h>

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace deltafin::provider_internal {

enum class CudaMoeCacheReserveKind : std::uint32_t {
  Auto = 0,
  Bytes = 1,
  RatioPpm = 2,
};

struct CudaMoeCachePolicy {
  bool automatic_capacity = true;
  std::uint64_t capacity_experts = 0;
  CudaMoeCacheReserveKind reserve_kind = CudaMoeCacheReserveKind::Auto;
  std::uint64_t reserve_value = 0;
};

struct CudaMoeResidencyPlanReport {
  std::vector<std::uint16_t> missing_experts;
  std::size_t capacity_experts = 0;
  bool residency_enabled = false;
};

struct CudaMoeHostExpert {
  std::uint16_t expert = 0;
  at::Tensor bytes;
};

struct CudaMoeHostFallback {
  std::vector<std::uint16_t> canonical_experts;
  std::vector<CudaMoeHostExpert> resident_experts;
};

/* Outcome of one out-of-plan admission (startup seeding). The entry counts
 * are session totals after the call; `capacity_experts` is the frozen
 * budget in experts. */
struct CudaMoeCacheAdmitReport {
  std::size_t admitted = 0;
  std::size_t already_resident = 0;
  std::size_t skipped_full = 0;
  std::size_t pinned_entries = 0;
  std::size_t resident_entries = 0;
  std::size_t capacity_experts = 0;
};

struct CudaMoeCacheStats {
  std::size_t capacity_experts = 0;
  std::size_t resident_entries = 0;
  std::size_t pinned_entries = 0;
  bool residency_enabled = false;
};

/*
 * Live device/allocator accounting used by Rust's memory-admission policy.
 * Every qualified CUDA LibTorch build exposes this accounting, even when the
 * optional NVCC MXFP4 kernel is absent. A CPU-only build reports every field
 * unknown; it never substitutes host RAM for discrete VRAM.
 */
struct CudaProviderMemorySnapshot {
  bool active_valid = false;
  bool reserved_valid = false;
  bool total_valid = false;
  bool available_valid = false;
  bool cache_trimmed = false;
  std::uint64_t active_bytes = 0;
  std::uint64_t reserved_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t available_bytes = 0;
};

[[nodiscard]] CudaProviderMemorySnapshot cuda_provider_memory_snapshot(
    const at::Device& device, bool trim_unused);

/*
 * Emitted only after a classified allocator failure and a successful stream
 * drain. Callers must still require an automatic backend request before using
 * the pinned plan for an exact CPU reconstruction.
 */
class CudaMoeRecoverableError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/*
 * One provider session owns one CUDA expert cache.  Expert bytes enter through
 * the authenticated Rust reader, are copied into ATen-owned device storage,
 * and never leave an allocation or stream lifetime hidden behind a raw global
 * pointer.  The implementation is a fail-closed stub when this provider was
 * built without both CUDA LibTorch and NVCC.
 */
class CudaMoeExpertCache final {
 public:
  explicit CudaMoeExpertCache(const at::Device& device);
  ~CudaMoeExpertCache();

  CudaMoeExpertCache(const CudaMoeExpertCache&) = delete;
  CudaMoeExpertCache& operator=(const CudaMoeExpertCache&) = delete;
  CudaMoeExpertCache(CudaMoeExpertCache&&) = delete;
  CudaMoeExpertCache& operator=(CudaMoeExpertCache&&) = delete;

  /* Runs the ABI/shape/device and generated-weight known-answer gates once. */
  [[nodiscard]] bool available();
  [[nodiscard]] const std::string& detail() const;

  /* Must be called before capability/budget use; the policy is then frozen. */
  void configure(const CudaMoeCachePolicy& policy);

  /*
   * Pin every initial hit before any miss can be admitted. `plan_id` is owned
   * by the provider session and must be completed or cancelled exactly once.
   */
  [[nodiscard]] CudaMoeResidencyPlanReport plan(
      std::uint64_t plan_id, std::uint32_t layer_index,
      std::span<const std::uint16_t> canonical_experts);
  void cancel_plan(std::uint64_t plan_id) noexcept;

  /* Bounded D2H materialization of the plan's pinned hits only. */
  [[nodiscard]] CudaMoeHostFallback materialize_plan_for_cpu(
      std::uint64_t plan_id);

  /*
   * Admit experts outside any plan: startup seeding from the learned heat
   * histogram. A pinned entry never leaves through LRU eviction; an unpinned
   * seed joins the LRU as its coldest entry, so a live route always outranks
   * it. `expert_major_bytes` is the exact raw-v1 slab for `canonical_experts`
   * (ascending IDs, at most one position tile). Budget rules are those of a
   * miss admission; pinned seeds may evict LRU residents, never other pins.
   */
  [[nodiscard]] CudaMoeCacheAdmitReport admit(
      std::uint32_t layer_index,
      std::span<const std::uint16_t> canonical_experts,
      std::span<const std::uint8_t> expert_major_bytes, bool pinned);

  /* Live counts; capacity is zero until the budget was first established. */
  [[nodiscard]] CudaMoeCacheStats stats() const;

  /* Marks the adapter terminal; this operation is allocation-failure safe. */
  void poison_external(const char* failure) noexcept;

  [[nodiscard]] at::Tensor execute_t1(
      const PreparedMoeT1& prepared,
      const CanonicalExpertBatchT1& experts);

  [[nodiscard]] at::Tensor execute_positions_t1(
      std::span<const PreparedMoeT1* const> prepared_rows,
      const CanonicalExpertPositionTileT1& experts);

  [[nodiscard]] at::Tensor execute_positions_plan_t1(
      std::uint64_t plan_id,
      std::span<const PreparedMoeT1* const> prepared_rows,
      const CanonicalExpertPositionTileT1& missing_experts);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/* Compile-time capability only; `available()` is the runtime authority. */
[[nodiscard]] bool cuda_moe_compiled() noexcept;

}  // namespace deltafin::provider_internal

#endif
