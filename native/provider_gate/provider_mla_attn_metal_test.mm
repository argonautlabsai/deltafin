#include "provider_mla_attn_metal.h"

#if !defined(__APPLE__)
#error "provider_mla_attn_metal_test.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_MLA_ATTN_METAL_V1)
#error "MLA attention Metal test requires the production capability guard"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "MLA attention Metal test requires embedded metallibs"
#endif

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr std::int64_t kHeads = 96;
constexpr std::int64_t kQkHeadDim = 192;
constexpr std::int64_t kValueHeadDim = 128;
constexpr std::int64_t kOutputWidth = kHeads * kValueHeadDim;

void require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_failure(Function&& function, const char* name) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " unexpectedly succeeded");
}

void require_close(const at::Tensor& actual, const at::Tensor& expected,
                   const std::string& name, const double rtol = 2.0e-5,
                   const double atol = 2.0e-6) {
  const at::Tensor actual_cpu = actual.to(at::kCPU).to(at::kFloat);
  const at::Tensor expected_cpu = expected.to(at::kCPU).to(at::kFloat);
  const double maximum =
      at::max(at::abs(actual_cpu - expected_cpu)).item<double>();
  if (!at::allclose(actual_cpu, expected_cpu, rtol, atol, true)) {
    throw std::runtime_error(name + " parity failed; max_abs=" +
                             std::to_string(maximum));
  }
}

/* The exact production LibTorch attention-core sequence for the expanded
 * T=1 branch (provider_mla.cpp): einsum scores, fp32 softmax, einsum value
 * contraction, transpose+reshape to the pre-gate [1,1,12288] row. */
at::Tensor libtorch_reference(const at::Tensor& query,
                              const at::Tensor& key_states,
                              const at::Tensor& value_states) {
  const double scaling =
      std::pow(static_cast<double>(kQkHeadDim), -0.5);
  at::Tensor scores =
      at::einsum("bhqd,bhkd->bhqk", {query, key_states});
  scores = scores * scaling;
  // Production computes fp32 softmax on fp32 scores; the fp64 cross-check
  // keeps its own dtype so the reference is genuinely double-precision.
  const at::ScalarType softmax_type =
      scores.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  const at::Tensor probabilities =
      at::softmax(scores, -1, softmax_type).to(query.scalar_type());
  at::Tensor attention =
      at::einsum("bhqk,bhkd->bhqd", {probabilities, value_states});
  attention = attention.transpose(1, 2).contiguous();
  return attention.reshape({1, 1, kOutputWidth}).contiguous();
}

struct Fixture {
  at::Tensor query;         // MPS [1,96,1,192] contiguous
  at::Tensor key_states;    // MPS [1,96,S,192] slab view, capacity pitch
  at::Tensor value_states;  // MPS [1,96,S,128] slab view, capacity pitch
  at::Tensor query_cpu;
  at::Tensor key_states_cpu;
  at::Tensor value_states_cpu;
};

Fixture build_fixture(const std::int64_t kv_length,
                      const std::int64_t capacity,
                      const std::uint64_t seed) {
  require(capacity > kv_length,
          "fixture must exercise capacity > length (slab pitch)");
  at::manual_seed(seed);
  const auto cpu_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor query_cpu = at::randn({1, kHeads, 1, kQkHeadDim}, cpu_float);
  at::Tensor key_slab_cpu =
      at::randn({1, kHeads, capacity, kQkHeadDim}, cpu_float);
  at::Tensor value_slab_cpu =
      at::randn({1, kHeads, capacity, kValueHeadDim}, cpu_float);
  // NaN poison beyond the live prefix: indexing by S instead of the slab
  // capacity, or touching uncommitted slots, poisons the comparison.
  key_slab_cpu.narrow(2, kv_length, capacity - kv_length)
      .fill_(std::numeric_limits<float>::quiet_NaN());
  value_slab_cpu.narrow(2, kv_length, capacity - kv_length)
      .fill_(std::numeric_limits<float>::quiet_NaN());

  Fixture fixture;
  fixture.query_cpu = query_cpu;
  fixture.key_states_cpu = key_slab_cpu.narrow(2, 0, kv_length);
  fixture.value_states_cpu = value_slab_cpu.narrow(2, 0, kv_length);
  fixture.query = query_cpu.to(at::kMPS);
  fixture.key_states =
      key_slab_cpu.to(at::kMPS).narrow(2, 0, kv_length);
  fixture.value_states =
      value_slab_cpu.to(at::kMPS).narrow(2, 0, kv_length);
  return fixture;
}

at::Tensor run_kernel(const Fixture& fixture,
                      const std::uint32_t partitions) {
  at::Tensor destination = at::empty(
      {1, 1, kOutputWidth},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  deltafin::provider_internal::mla_attn_metal_decode_f32_with_partitions(
      destination, fixture.query, fixture.key_states, fixture.value_states,
      partitions);
  return destination;
}

void capability_and_canary() {
  const auto capabilities =
      deltafin::provider_internal::mla_attn_metal_capabilities_v1();
  require(capabilities.abi_version ==
              deltafin::provider_internal::kMlaAttnMetalAbiV1,
          "MLA attention Metal ABI changed");
  require(capabilities.flags ==
              deltafin::provider_internal::
                  kMlaAttnMetalRequiredCapabilitiesV1,
          "MLA attention Metal capability flags changed");
  require(capabilities.threads_per_threadgroup == 256 &&
              capabilities.reserved == 0,
          "MLA attention Metal schedule changed");

  const auto report =
      deltafin::provider_internal::mla_attn_metal_canary_v1();
  require(deltafin::provider_internal::mla_attn_metal_canary_passes(report),
          "MLA attention Metal canary failed; rel_l2=" +
              std::to_string(report.relative_l2_error) + " close=" +
              std::to_string(report.close_elements) + "/" +
              std::to_string(report.compared_elements) + " nonfinite=" +
              std::to_string(report.nonfinite));
  std::cout << "provider_mla_attn_metal.canary rel_l2="
            << report.relative_l2_error
            << " max_abs=" << report.max_absolute_error << " close="
            << report.close_elements << '/' << report.compared_elements
            << '\n'
            << "provider_mla_attn_metal.capability=PASS\n";
}

void parity_case(const std::int64_t kv_length, const std::int64_t capacity,
                 const std::uint32_t partitions, const bool fp64_cross_check) {
  const Fixture fixture = build_fixture(
      kv_length, capacity,
      0x51D0'0000ULL + static_cast<std::uint64_t>(kv_length) * 37 +
          partitions);
  const at::Tensor actual = run_kernel(fixture, partitions);
  const at::Tensor expected = libtorch_reference(
      fixture.query, fixture.key_states, fixture.value_states);
  const std::string name = "S=" + std::to_string(kv_length) + " cap=" +
                           std::to_string(capacity) + " P=" +
                           std::to_string(partitions);
  require_close(actual, expected, "kernel-vs-LibTorch " + name);

  if (fp64_cross_check) {
    const at::Tensor expected64 = libtorch_reference(
        fixture.query_cpu.to(at::kDouble),
        fixture.key_states_cpu.to(at::kDouble),
        fixture.value_states_cpu.to(at::kDouble));
    require_close(actual, expected64, "kernel-vs-fp64 " + name);
    require_close(expected, expected64, "LibTorch-vs-fp64 " + name);
  }
}

void parity_sweep() {
  // The plan's sweep: growth-boundary lengths (init capacity 16, x1.5), the
  // canary shape, and every flash-partition extreme, all with capacity > S.
  const std::int64_t lengths[] = {1, 15, 16, 17, 24, 257};
  const std::uint32_t partition_counts[] = {1, 8, 32};
  for (const std::int64_t kv_length : lengths) {
    const std::int64_t capacity = kv_length + 7 + (kv_length % 5);
    for (const std::uint32_t partitions : partition_counts) {
      parity_case(kv_length, capacity, partitions, kv_length == 257);
    }
  }
  std::cout << "provider_mla_attn_metal.parity=PASS\n";
}

void large_context_parity() {
  // Production heuristic partition selection at a deep context.
  parity_case(4095, 4160, 0, false);
  std::cout << "provider_mla_attn_metal.large_context=PASS\n";
}

void contract_rejections() {
  const Fixture fixture = build_fixture(16, 32, 0xC0FFEE);
  const auto mps_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
  at::Tensor destination = at::empty({1, 1, kOutputWidth}, mps_float);

  expect_failure(
      [&] {
        // CPU destination.
        deltafin::provider_internal::mla_attn_metal_decode_f32(
            at::empty({1, 1, kOutputWidth},
                      at::TensorOptions().dtype(at::kFloat)),
            fixture.query, fixture.key_states, fixture.value_states);
      },
      "CPU destination");
  expect_failure(
      [&] {
        // Non-contiguous query view.
        const at::Tensor wide = at::randn({1, kHeads, 1, 2 * kQkHeadDim},
                                          mps_float);
        deltafin::provider_internal::mla_attn_metal_decode_f32(
            destination, wide.narrow(-1, 0, kQkHeadDim), fixture.key_states,
            fixture.value_states);
      },
      "non-contiguous query");
  expect_failure(
      [&] {
        // Value slab with a different capacity pitch than the key slab.
        const at::Tensor other_value_slab =
            at::randn({1, kHeads, 48, kValueHeadDim}, mps_float);
        deltafin::provider_internal::mla_attn_metal_decode_f32(
            destination, fixture.query, fixture.key_states,
            other_value_slab.narrow(2, 0, 16));
      },
      "mismatched value capacity");
  expect_failure(
      [&] {
        // fp16 keys.
        const at::Tensor half_keys =
            fixture.key_states.to(at::kHalf);
        deltafin::provider_internal::mla_attn_metal_decode_f32(
            destination, fixture.query, half_keys, fixture.value_states);
      },
      "fp16 key states");
  expect_failure(
      [&] {
        // Misaligned (non-16-byte) query storage offset.
        const at::Tensor flat =
            at::randn({2 + kHeads * kQkHeadDim}, mps_float);
        const at::Tensor misaligned =
            flat.narrow(0, 2, kHeads * kQkHeadDim)
                .view({1, kHeads, 1, kQkHeadDim});
        deltafin::provider_internal::mla_attn_metal_decode_f32(
            destination, misaligned, fixture.key_states,
            fixture.value_states);
      },
      "misaligned query offset");
  expect_failure(
      [&] {
        // Partition count beyond the kernel ABI.
        deltafin::provider_internal::
            mla_attn_metal_decode_f32_with_partitions(
                destination, fixture.query, fixture.key_states,
                fixture.value_states, 33);
      },
      "partition overflow");
  std::cout << "provider_mla_attn_metal.rejections=PASS\n";
}

}  // namespace

int main() {
  @autoreleasepool {
    try {
      require(at::hasMPS(), "MPS is unavailable");
      capability_and_canary();
      parity_sweep();
      large_context_parity();
      contract_rejections();
      std::cout << "provider_mla_attn_metal.mps=PASS\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "provider_mla_attn_metal=FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
