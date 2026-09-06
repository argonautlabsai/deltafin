#include "provider_abi.h"
#include "provider_spine_debug.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t kBf16EncodingCount = 1U << 16;
constexpr std::size_t kSourceAlignment = 256;

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

struct FreeDeleter {
  void operator()(void* pointer) const noexcept { std::free(pointer); }
};

struct BoundMatrix {
  std::unique_ptr<void, FreeDeleter> allocation;
  std::uint16_t* words = nullptr;
  std::size_t elements = 0;
  std::size_t allocation_bytes = 0;
  DeltafinProviderSpineTensorDescriptorV1 descriptor{};
};

BoundMatrix make_matrix(const std::uint32_t slot, const std::uint64_t rows,
                        const std::uint64_t columns) {
  if (rows == 0 || columns == 0 ||
      rows > SIZE_MAX / static_cast<std::size_t>(columns)) {
    throw std::runtime_error("BF16 arena test matrix shape overflows");
  }
  const std::size_t elements =
      static_cast<std::size_t>(rows * columns);
  if (elements > SIZE_MAX / sizeof(std::uint16_t)) {
    throw std::runtime_error("BF16 arena test matrix byte length overflows");
  }
  const std::size_t bytes = elements * sizeof(std::uint16_t);
  const std::size_t allocation_bytes =
      (bytes + kSourceAlignment - 1) / kSourceAlignment * kSourceAlignment;
  void* raw = nullptr;
  if (posix_memalign(&raw, kSourceAlignment, allocation_bytes) != 0 ||
      raw == nullptr) {
    throw std::runtime_error("allocate aligned BF16 arena source failed");
  }
  std::memset(raw, 0, allocation_bytes);

  BoundMatrix matrix;
  matrix.allocation.reset(raw);
  matrix.words = static_cast<std::uint16_t*>(raw);
  matrix.elements = elements;
  matrix.allocation_bytes = allocation_bytes;
  matrix.descriptor.slot = slot;
  matrix.descriptor.encoding = DELTAFIN_PROVIDER_SPINE_RAW_BF16_V1;
  matrix.descriptor.rank = 2;
  matrix.descriptor.shape[0] = rows;
  matrix.descriptor.shape[1] = columns;
  matrix.descriptor.data_buffer = DELTAFIN_PROVIDER_SPINE_BUFFER_OTHER_V1;
  matrix.descriptor.data_length = bytes;
  return matrix;
}

void bind_matrix(const DeltafinProviderSessionHandleV1 session,
                 const std::uint32_t layer_index,
                 const std::uint64_t generation,
                 const BoundMatrix& matrix) {
  DeltafinProviderBindSpineLayerRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  request.layer_index = layer_index;
  request.generation = generation;
  request.descriptors = &matrix.descriptor;
  request.descriptor_count = 1;
  request.other = static_cast<const std::uint8_t*>(matrix.allocation.get());
  request.other_length = matrix.allocation_bytes;
  DeltafinProviderBindSpineLayerReportV1 report{};
  report.struct_size = sizeof(report);
  std::array<char, 1024> error{};
  require_success(deltafin_provider_bind_spine_layer_v1(
                      &request, &report, error.data(), error.size()),
                  error.data(), "bind BF16 arena test matrix");
  require(report.layer_index == layer_index &&
              report.generation == generation && report.tensor_count == 1 &&
              report.raw_tensor_count == 1 &&
              report.resident_storage_bytes == matrix.elements * 2,
          "BF16 arena bind report changed");
}

DeltafinProviderSessionHandleV1 create_session(const std::uint32_t device) {
  DeltafinProviderSessionRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.requested_device = device;
  request.max_route_positions = 64;
  DeltafinProviderSessionReportV1 report{};
  report.struct_size = sizeof(report);
  std::array<char, 1024> error{};
  require_success(deltafin_provider_session_create_v1(
                      &request, &report, error.data(), error.size()),
                  error.data(), "create BF16 arena test session");
  require(report.selected_device == device,
          "BF16 arena test selected an unexpected device");
  return report.session;
}

void destroy_session(const DeltafinProviderSessionHandleV1 session) {
  DeltafinProviderResourceRequestV1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = DELTAFIN_PROVIDER_ABI_VERSION;
  request.session = session;
  std::array<char, 1024> error{};
  require_success(deltafin_provider_session_destroy_v1(
                      &request, error.data(), error.size()),
                  error.data(), "destroy BF16 arena test session");
}

void require_all_bf16_encodings(
    const deltafin::provider_internal::SpineFp32ExecutionDebugReport& report,
    const std::uint64_t owner, const std::uint64_t generation) {
  require(report.owner == owner && report.layer_index == 0 &&
              report.spine_generation == generation &&
              report.required_elements == kBf16EncodingCount &&
              report.capacity_elements >= kBf16EncodingCount &&
              report.storage_identity != 0 &&
              report.values.size() == kBf16EncodingCount,
          "BF16 arena exhaustive report metadata changed");
  for (std::uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    if (std::bit_cast<std::uint32_t>(report.values[bits]) != bits << 16) {
      throw std::runtime_error(
          "ATen BF16-to-FP32 arena conversion lost a raw encoding");
    }
  }
}

void accelerator_arena_sequence(const std::uint32_t device) {
  const auto session = create_session(device);
  BoundMatrix exhaustive = make_matrix(13, 256, 256);
  for (std::uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    exhaustive.words[bits] = static_cast<std::uint16_t>(bits);
  }
  bind_matrix(session, 0, 1, exhaustive);
  std::memset(exhaustive.allocation.get(), 0xa5,
              exhaustive.allocation_bytes);
  const auto first =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session, 0, 1, 41, 13);
  require_all_bf16_encodings(first, 41, 1);

  // Step 6 contract: a same-owner same-layer same-generation re-request is
  // an idempotent hit publishing the SAME views (pre-commit orchestration
  // materializes one layer early; the ordinary prepare then asks again).
  const auto repeat =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session, 0, 1, 41, 13);
  require_all_bf16_encodings(repeat, 41, 1);
  require(repeat.storage_identity == first.storage_identity,
          "same-layer idempotent hit did not reuse the BF16 arena storage");

  BoundMatrix next = make_matrix(13, 4, 8);
  constexpr std::array<std::uint16_t, 8> finite_bits{
      0x0000U, 0x3f00U, 0xbf80U, 0x3f80U,
      0xc000U, 0x4000U, 0x4040U, 0xc080U};
  for (std::size_t index = 0; index < next.elements; ++index) {
    next.words[index] = finite_bits[index % finite_bits.size()];
  }
  bind_matrix(session, 1, 2, next);
  std::memset(next.allocation.get(), 0x5a, next.allocation_bytes);
  const auto second =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session, 1, 2, 41, 13);
  require(second.owner == 41 && second.layer_index == 1 &&
              second.spine_generation == 2 &&
              second.required_elements == 64 && second.values.size() == 32,
          "BF16 arena second-layer metadata changed");
  require(second.storage_identity == first.storage_identity &&
              second.capacity_elements == first.capacity_elements,
          "BF16 arena did not reuse its bounded allocation");
  for (std::size_t index = 0; index < second.values.size(); ++index) {
    require(std::bit_cast<std::uint32_t>(second.values[index]) ==
                static_cast<std::uint32_t>(finite_bits[index % 8]) << 16,
            "BF16 arena finite conversion changed after source poison");
  }
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_fp32_execution_debug(
                session, 1, 2, 42, 13));
      },
      "BF16 arena owner switch above layer zero");

  BoundMatrix restarted = make_matrix(13, 4, 8);
  for (std::size_t index = 0; index < restarted.elements; ++index) {
    restarted.words[index] = finite_bits[(index + 3) % finite_bits.size()];
  }
  bind_matrix(session, 0, 3, restarted);
  const auto third =
      deltafin::provider_internal::spine_fp32_execution_debug(
          session, 0, 3, 42, 13);
  require(third.owner == 42 && third.layer_index == 0 &&
              third.spine_generation == 3 &&
              third.storage_identity == first.storage_identity &&
              third.capacity_elements == first.capacity_elements,
          "BF16 arena owner restart allocated a second arena");
  destroy_session(session);

  std::cout << "provider_spine_bf16_arena.encodings=65536/65536\n"
            << "provider_spine_bf16_arena.source_poison=PASS\n"
            << "provider_spine_bf16_arena.reuse=PASS\n";
}

void cpu_exclusion() {
  const auto session = create_session(DELTAFIN_PROVIDER_DEVICE_CPU_V1);
  BoundMatrix matrix = make_matrix(13, 4, 8);
  for (std::size_t index = 0; index < matrix.elements; ++index) {
    matrix.words[index] = static_cast<std::uint16_t>(0x3f00U + index);
  }
  bind_matrix(session, 0, 1, matrix);
  expect_failure(
      [&] {
        static_cast<void>(
            deltafin::provider_internal::spine_fp32_execution_debug(
                session, 0, 1, 1, 13));
      },
      "CPU BF16 execution arena");
  destroy_session(session);
  std::cout << "provider_spine_bf16_arena.cpu=UNCHANGED\n";
}

}  // namespace

int main() {
  try {
    DeltafinProviderInventoryV1 inventory{};
    inventory.struct_size = sizeof(inventory);
    std::array<char, 1024> error{};
    require_success(deltafin_provider_inventory_v1(
                        &inventory, error.data(), error.size()),
                    error.data(), "read provider inventory");
    cpu_exclusion();
    require(inventory.mps_available != 0,
            "MPS BF16 arena gate requires a physical MPS device");
    accelerator_arena_sequence(DELTAFIN_PROVIDER_DEVICE_MPS_V1);
    std::cout << "provider_spine_bf16_arena.device=mps\n";
    std::cout << "provider_spine_bf16_arena=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "provider_spine_bf16_arena=FAIL: " << error.what() << '\n';
    return 1;
  }
}
