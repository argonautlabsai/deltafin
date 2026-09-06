#include "provider_abi.h"
#include "provider_spine_debug.h"

#if !defined(__APPLE__)
#error "provider_spine_int8_arena_test.cpp is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_SPINE_INT8_METAL_V1)
#error "int8 spine arena test requires the production Metal dequantizer"
#endif

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_success(const std::int32_t status, const char* error,
                     const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) + ": " + error);
  }
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

void put_u16(std::uint8_t* destination, const std::size_t index,
             const std::uint16_t value) {
  std::memcpy(destination + index * sizeof(value), &value, sizeof(value));
}

struct BoundLayer {
  alignas(256) std::array<std::uint8_t, 256> quantized{};
  alignas(256) std::array<std::uint8_t, 256> scales{};
  DeltafinProviderSpineTensorDescriptorV1 descriptor{};
};

BoundLayer make_layer(const std::int8_t base,
                      const std::array<std::uint16_t, 4>& scales) {
  BoundLayer layer;
  for (std::size_t index = 0; index < 32; ++index) {
    layer.quantized[index] = static_cast<std::uint8_t>(
        static_cast<std::int8_t>(base + static_cast<std::int8_t>(index % 7)));
  }
  for (std::size_t row = 0; row < scales.size(); ++row) {
    put_u16(layer.scales.data(), row, scales[row]);
  }
  layer.descriptor.slot = 13;
  layer.descriptor.encoding =
      DELTAFIN_PROVIDER_SPINE_ROW_I8_F16_SCALE_V1;
  layer.descriptor.rank = 2;
  layer.descriptor.shape[0] = 4;
  layer.descriptor.shape[1] = 8;
  layer.descriptor.data_buffer =
      DELTAFIN_PROVIDER_SPINE_BUFFER_QUANTIZED_V1;
  layer.descriptor.data_length = 32;
  layer.descriptor.auxiliary_buffer =
      DELTAFIN_PROVIDER_SPINE_BUFFER_SCALES_V1;
  layer.descriptor.auxiliary_length = 8;
  return layer;
}

void bind_layer(const DeltafinProviderSessionHandleV1 session,
                const std::uint32_t layer_index,
                const std::uint64_t generation,
                const BoundLayer& layer) {
  DeltafinProviderBindSpineLayerRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  request.layer_index = layer_index;
  request.generation = generation;
  request.descriptors = &layer.descriptor;
  request.descriptor_count = 1;
  request.quantized = layer.quantized.data();
  request.quantized_length = layer.quantized.size();
  request.scales = layer.scales.data();
  request.scales_length = layer.scales.size();
  DeltafinProviderBindSpineLayerReportV1 report{};
  report.struct_size = sizeof(report);
  std::array<char, 1024> error{};
  require_success(deltafin_provider_bind_spine_layer_v1(
                      &request, &report, error.data(), error.size()),
                  error.data(), "bind int8 arena test layer");
  require(report.layer_index == layer_index &&
              report.generation == generation && report.tensor_count == 1 &&
              report.quantized_tensor_count == 1 &&
              report.resident_storage_bytes == 48,
          "int8 arena test bind report changed");
}

void require_materialized(
    const deltafin::provider_internal::SpineFp32ExecutionDebugReport& report,
    const BoundLayer& layer, const std::array<float, 4>& scales,
    const std::uint64_t owner, const std::uint32_t layer_index,
    const std::uint64_t generation) {
  require(report.owner == owner && report.layer_index == layer_index &&
              report.spine_generation == generation &&
              report.required_elements == 64 &&
              report.capacity_elements >= 64 &&
              report.storage_identity != 0 && report.values.size() == 32,
          "FP32 arena debug metadata changed");
  for (std::size_t index = 0; index < report.values.size(); ++index) {
    const float expected = static_cast<float>(
        static_cast<std::int8_t>(layer.quantized[index])) * scales[index / 8];
    require(std::bit_cast<std::uint32_t>(report.values[index]) ==
                std::bit_cast<std::uint32_t>(expected),
            "FP32 arena materialization was not bit exact");
  }
}

void arena_sequence() {
  DeltafinProviderSessionRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.requested_device = DELTAFIN_PROVIDER_DEVICE_MPS_V1;
  request.max_route_positions = 1;
  DeltafinProviderSessionReportV1 session{};
  session.struct_size = sizeof(session);
  std::array<char, 1024> error{};
  require_success(deltafin_provider_session_create_v1(
                      &request, &session, error.data(), error.size()),
                  error.data(), "create int8 arena test session");

  const BoundLayer first = make_layer(
      -11, {UINT16_C(0x3800), UINT16_C(0x3c00),
            UINT16_C(0xbc00), UINT16_C(0x4000)});
  bind_layer(session.session, 0, 1, first);
  const auto first_report =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session.session, 0, 1, 41, 13);
  require_materialized(first_report, first, {0.5F, 1.0F, -1.0F, 2.0F},
                       41, 0, 1);

  // Step 6 contract: a same-owner same-layer same-generation re-request is
  // an idempotent hit publishing the SAME views (pre-commit orchestration
  // materializes one layer early; the ordinary prepare then asks again).
  // The window must not advance and the storage must not be re-encoded.
  const auto repeat_report =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session.session, 0, 1, 41, 13);
  require_materialized(repeat_report, first, {0.5F, 1.0F, -1.0F, 2.0F},
                       41, 0, 1);
  require(repeat_report.storage_identity == first_report.storage_identity,
          "same-layer idempotent hit did not reuse the arena storage");

  const BoundLayer second = make_layer(
      3, {UINT16_C(0x3c00), UINT16_C(0x3c00),
          UINT16_C(0x3c00), UINT16_C(0x3c00)});
  bind_layer(session.session, 1, 2, second);
  const auto second_report =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session.session, 1, 2, 41, 13);
  require_materialized(second_report, second, {1.0F, 1.0F, 1.0F, 1.0F},
                       41, 1, 2);
  require(second_report.storage_identity == first_report.storage_identity,
          "FP32 arena did not reuse its bounded storage");
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_fp32_execution_debug(
                session.session, 1, 2, 42, 13));
      },
      "new arena owner above layer zero");

  bind_layer(session.session, 0, 3, first);
  const auto restarted =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session.session, 0, 3, 42, 13);
  require_materialized(restarted, first, {0.5F, 1.0F, -1.0F, 2.0F},
                       42, 0, 3);
  require(restarted.storage_identity == first_report.storage_identity,
          "new target owner allocated a second FP32 arena");

  DeltafinProviderResourceRequestV1 destroy{};
  destroy.struct_size = sizeof(destroy);
  destroy.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  destroy.session = session.session;
  require_success(deltafin_provider_session_destroy_v1(
                      &destroy, error.data(), error.size()),
                  error.data(), "destroy int8 arena test session");
  std::cout << "provider_spine_int8_arena.sequence=PASS\n";
}

}  // namespace

int main() {
  try {
    arena_sequence();
    std::cout << "provider_spine_int8_arena.mps=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_spine_int8_arena=FAIL: " << error.what() << '\n';
    return 1;
  }
}
