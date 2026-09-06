#include "provider_loop.h"

#if !defined(__APPLE__)
#error "provider_loop_test.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_BESPOKE_LOOP_V1)
#error "bespoke decode-loop test requires the production capability guard"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "bespoke decode-loop test requires embedded metallibs"
#endif

#include <ATen/ATen.h>
#include <ATen/Context.h>

#import <Foundation/Foundation.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using deltafin::provider_internal::LoopMetalCanaryReport;
using deltafin::provider_internal::LoopMetalCapabilities;
using deltafin::provider_internal::LoopWeightsTableStats;

void require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/* Worklist step 2 gate: the loop-owned queue/event-pool/mailbox/weights-
 * table plumbing qualifies end to end, independent of the K3_BESPOKE_LOOP
 * env gate (that gate is exercised separately below). */
void canary_and_capabilities() {
  const LoopMetalCapabilities capabilities =
      deltafin::provider_internal::loop_metal_capabilities_v1();
  require(capabilities.abi_version == deltafin::provider_internal::kLoopMetalAbiV1,
          "bespoke-loop ABI version changed");
  require(capabilities.flags ==
              deltafin::provider_internal::kLoopMetalRequiredCapabilitiesV1,
          "bespoke-loop capability flags changed");
  require(capabilities.event_pool_size ==
              deltafin::provider_internal::kLoopMetalSharedEventPoolSizeV1,
          "bespoke-loop shared-event pool size changed");
  require(capabilities.reserved == 0, "bespoke-loop capabilities reserved field is nonzero");

  const LoopMetalCanaryReport report =
      deltafin::provider_internal::loop_metal_canary_v1();
  require(deltafin::provider_internal::loop_metal_canary_passes(report),
          "bespoke-loop canary failed; route=" +
              std::to_string(report.route_ids_matched) + "/" +
              std::to_string(report.top_k) + " pilot=" +
              std::to_string(report.pilot_ids_matched) + "/" +
              std::to_string(report.top_k) + " table_hits=" +
              std::to_string(report.weights_table_hits) +
              " table_invalidations=" +
              std::to_string(report.weights_table_invalidations) +
              " max_abs=" + std::to_string(report.max_absolute_error) +
              " rel_l2=" + std::to_string(report.relative_l2_error));

  // The canary must be re-runnable: it resets the shared weights table before
  // returning, and a second run from a clean process-wide singleton must
  // reproduce byte-identical counters.
  const LoopMetalCanaryReport second =
      deltafin::provider_internal::loop_metal_canary_v1();
  require(deltafin::provider_internal::loop_metal_canary_passes(second),
          "bespoke-loop canary is not repeatable");
  require(second.route_ids_matched == report.route_ids_matched &&
              second.pilot_ids_matched == report.pilot_ids_matched &&
              second.weights_table_hits == report.weights_table_hits &&
              second.weights_table_invalidations ==
                  report.weights_table_invalidations,
          "bespoke-loop canary counters drifted across repeated runs");

  std::cout << "provider_loop.canary rel_l2=" << report.relative_l2_error
            << " max_abs=" << report.max_absolute_error
            << " table_hits=" << report.weights_table_hits
            << " table_invalidations=" << report.weights_table_invalidations
            << '\n'
            << "provider_loop.capability=PASS\n";
}

/* Direct exercise of the device weights-table cache contract: qualifying
 * MPS fp32/int8 tensors are cached and hit under a matching generation; a
 * generation change evicts and reports a miss; non-qualifying tensors
 * (CPU device, fp16 dtype, non-contiguous) are rejected without caching. */
void weights_table_contract() {
  deltafin::provider_internal::loop_weights_table_reset();
  const auto mps_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
  const auto mps_int8 = at::TensorOptions().dtype(at::kChar).device(at::kMPS);
  const auto cpu_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);

  const at::Tensor good = at::zeros({8}, mps_float);
  require(deltafin::provider_internal::loop_weights_table_store(3, 10, 0, good),
          "weights table rejected a qualifying fp32 MPS tensor");
  require(deltafin::provider_internal::loop_weights_table_hit(3, 10, 0),
          "weights table missed a freshly stored entry");
  require(!deltafin::provider_internal::loop_weights_table_hit(3, 11, 0),
          "weights table accepted a stale generation");
  require(!deltafin::provider_internal::loop_weights_table_hit(4, 10, 0),
          "weights table matched the wrong layer index");

  const at::Tensor int8_rows = at::zeros({8}, mps_int8);
  require(deltafin::provider_internal::loop_weights_table_store(5, 1, 2,
                                                                 int8_rows),
          "weights table rejected a qualifying int8 MPS tensor");
  require(deltafin::provider_internal::loop_weights_table_hit(5, 1, 2),
          "weights table missed a freshly stored int8 entry");

  const at::Tensor cpu_tensor = at::zeros({8}, cpu_float);
  require(!deltafin::provider_internal::loop_weights_table_store(3, 12, 1,
                                                                  cpu_tensor),
          "weights table accepted a CPU tensor");
  const at::Tensor half_tensor = good.to(at::kHalf);
  require(!deltafin::provider_internal::loop_weights_table_store(3, 12, 1,
                                                                  half_tensor),
          "weights table accepted an fp16 tensor");
  const at::Tensor noncontiguous = at::zeros({8, 2}, mps_float).transpose(0, 1);
  require(!deltafin::provider_internal::loop_weights_table_store(
              3, 12, 1, noncontiguous),
          "weights table accepted a non-contiguous tensor");

  const LoopWeightsTableStats stats =
      deltafin::provider_internal::loop_weights_table_stats();
  require(stats.generation_invalidations == 1,
          "weights table generation-invalidation counter is wrong");

  deltafin::provider_internal::loop_weights_table_reset();
  const LoopWeightsTableStats cleared =
      deltafin::provider_internal::loop_weights_table_stats();
  require(cleared.entries == 0 && cleared.hits == 0 && cleared.misses == 0 &&
              cleared.generation_invalidations == 0,
          "weights table reset did not clear its counters");
  std::cout << "provider_loop.weights_table=PASS\n";
}

/* K3_BESPOKE_LOOP unset or "0": the sticky gate must stay closed and must
 * not perform any Metal setup, mailbox write, or weights-table mutation --
 * i.e. it is a provable no-op, byte-identical to never having called it. */
void default_gate_is_a_noop() {
  const char* value = std::getenv("K3_BESPOKE_LOOP");
  require(value == nullptr || std::strcmp(value, "1") != 0,
          "test harness must not enable K3_BESPOKE_LOOP for this case");

  const LoopWeightsTableStats before =
      deltafin::provider_internal::loop_weights_table_stats();

  const auto mps_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
  const at::Tensor probe = at::linspace(-3.0, 4.0, 64, mps_float).reshape({8, 8});
  const at::Tensor reference = (probe * 2.5f + 1.0f).to(at::kCPU).clone();

  require(!deltafin::provider_internal::bespoke_loop_decode_ready(),
          "K3_BESPOKE_LOOP must default to closed");
  require(!deltafin::provider_internal::bespoke_loop_decode_ready(),
          "the sticky gate must stay closed across repeated calls");

  const LoopWeightsTableStats after =
      deltafin::provider_internal::loop_weights_table_stats();
  require(before.entries == after.entries && before.hits == after.hits &&
              before.misses == after.misses &&
              before.generation_invalidations == after.generation_invalidations,
          "the closed gate mutated the weights table");

  const at::Tensor repeat = (probe * 2.5f + 1.0f).to(at::kCPU);
  require(at::equal(reference, repeat),
          "the closed gate perturbed unrelated MPS computation -- not a "
          "byte-identical no-op");
  std::cout << "provider_loop.default_noop=PASS\n";
}

/* K3_BESPOKE_LOOP=1 on a qualifying device: the one-shot canary must run and
 * pass, and the sticky result must hold across repeated calls. */
void enabled_gate_qualifies() {
  const char* value = std::getenv("K3_BESPOKE_LOOP");
  require(value != nullptr && std::strcmp(value, "1") == 0,
          "test harness must set K3_BESPOKE_LOOP=1 for this case");
  require(deltafin::provider_internal::bespoke_loop_decode_ready(),
          "K3_BESPOKE_LOOP=1 did not qualify on a capable MPS device");
  require(deltafin::provider_internal::bespoke_loop_decode_ready(),
          "the sticky gate must stay open across repeated calls");
  std::cout << "provider_loop.enabled_gate=PASS\n";
}

}  // namespace

int main() {
  @autoreleasepool {
    try {
      if (!at::hasMPS()) {
        std::cout << "provider_loop.mps=SKIP\n";
        return 0;
      }
      canary_and_capabilities();
      weights_table_contract();
      const char* value = std::getenv("K3_BESPOKE_LOOP");
      if (value != nullptr && std::strcmp(value, "1") == 0) {
        enabled_gate_qualifies();
      } else {
        default_gate_is_a_noop();
      }
      std::cout << "provider_loop.mps=PASS\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "provider_loop=FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
