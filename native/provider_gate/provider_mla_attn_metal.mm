#if !defined(__APPLE__)
#error "provider_mla_attn_metal.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_MLA_ATTN_METAL_V1)
#error "MLA attention Metal bridge requires its explicit production capability"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "MLA attention Metal bridge requires an embedded precompiled metallib"
#endif

#include "provider_mla_attn_metal.h"
#include "provider_loop.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "MLA attention Metal bridge requires Objective-C ARC"
#endif

#include "deltafin_embedded_mla_attn_metal_metallib.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::int64_t kHeads = 96;
constexpr std::int64_t kQkHeadDim = 192;
constexpr std::int64_t kValueHeadDim = 128;
constexpr std::int64_t kOutputWidth = kHeads * kValueHeadDim;  // 12288
constexpr std::uint32_t kPartialStride = 132;
constexpr std::uint32_t kPartThreadsPerThreadgroup = 256;
constexpr std::uint32_t kCombineThreadsPerThreadgroup = 32;
constexpr std::uint32_t kMaxPartitions = 32;
constexpr std::uint32_t kPartitionChunkTarget = 256;
// Generous admission bound; the live cache budget admits ~4.4k positions.
constexpr std::int64_t kMaxKvLength = 1048576;
constexpr std::int64_t kMaxCapacity = 1 << 24;

struct MlaAttnDimsV1 {
  std::uint32_t kv_length;
  std::uint32_t capacity;
  std::uint32_t partitions;
  std::uint32_t chunk;
  float scale;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
  std::uint32_t reserved2;
};

static_assert(sizeof(MlaAttnDimsV1) == 32);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
  if (!condition) fail(message);
}

dispatch_data_t copy_embedded_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedMlaAttnMetalMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedMlaAttnMetalMetallib,
              kDeltafinEmbeddedMlaAttnMetalMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedMlaAttnMetalMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}

struct MetalPipeline {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> part = nil;
  id<MTLComputePipelineState> combine = nil;
};

struct PipelineCache {
  std::mutex mutex;
  MetalPipeline pipeline;
};

PipelineCache& pipeline_cache() {
  static PipelineCache cache;
  return cache;
}

id<MTLComputePipelineState> build_pipeline(id<MTLDevice> device,
                                           id<MTLLibrary> library,
                                           NSString* name,
                                           const std::uint32_t threads) {
  NSError* error = nil;
  id<MTLFunction> function = [library newFunctionWithName:name];
  require(function != nil,
          std::string("embedded MLA attention metallib is missing ") +
              name.UTF8String);
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  require(pipeline != nil,
          std::string("create MLA attention Metal pipeline failed: ") +
              (error == nil ? "unknown"
                            : error.localizedDescription.UTF8String));
  require(pipeline.maxTotalThreadsPerThreadgroup >= threads,
          "MLA attention Metal pipeline cannot admit its threadgroup");
  require(pipeline.threadExecutionWidth == 32,
          "MLA attention Metal pipeline requires a 32-lane simdgroup");
  return pipeline;
}

MetalPipeline pipeline_for(id<MTLDevice> device) {
  require(device != nil, "MLA attention Metal device is unavailable");
  PipelineCache& cache = pipeline_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.pipeline.device != device || cache.pipeline.part == nil ||
      cache.pipeline.combine == nil) {
    dispatch_data_t data = copy_embedded_metallib();
    require(data != nullptr, "wrap embedded MLA attention metallib failed");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    require(library != nil,
            std::string("load embedded MLA attention metallib failed: ") +
                (error == nil ? "unknown"
                              : error.localizedDescription.UTF8String));
    id<MTLComputePipelineState> part = build_pipeline(
        device, library, @"deltafin_mla_attn_part_f32_v1",
        kPartThreadsPerThreadgroup);
    id<MTLComputePipelineState> combine = build_pipeline(
        device, library, @"deltafin_mla_attn_combine_f32_v1",
        kCombineThreadsPerThreadgroup);
    cache.pipeline = MetalPipeline{device, library, part, combine};
  }
  return cache.pipeline;
}

id<MTLBuffer> tensor_buffer(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

NSUInteger tensor_byte_offset(const at::Tensor& tensor,
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

void require_aligned_offset(const at::Tensor& tensor, const char* name) {
  require(tensor.storage_offset() % 4 == 0,
          std::string(name) +
              " storage offset is not 16-byte aligned (4 fp32 elements)");
}

float attention_scale() {
  return static_cast<float>(
      std::pow(static_cast<double>(kQkHeadDim), -0.5));
}

std::uint32_t choose_partitions(const std::int64_t kv_length) {
  const std::int64_t raw =
      (kv_length + kPartitionChunkTarget - 1) / kPartitionChunkTarget;
  if (raw <= 1) return 1;
  if (raw >= kMaxPartitions) return kMaxPartitions;
  return static_cast<std::uint32_t>(raw);
}

}  // namespace

bool mla_attn_metal_canary_passes(const MlaAttnMetalCanaryReport& report) {
  return report.heads == kHeads && report.kv_length != 0 &&
         report.capacity > report.kv_length && report.partitions != 0 &&
         report.compared_elements ==
             static_cast<std::uint32_t>(kOutputWidth) &&
         report.close_elements == report.compared_elements &&
         report.nonfinite == 0 && report.query_offset_elements != 0 &&
         report.key_offset_elements != 0 &&
         report.value_offset_elements != 0 &&
         report.destination_offset_elements != 0 && report.reserved == 0 &&
         std::isfinite(report.max_absolute_error) &&
         std::isfinite(report.relative_l2_error) &&
         report.relative_l2_error <= kMlaAttnMetalCanaryMaxRelL2;
}

MlaAttnMetalCapabilities mla_attn_metal_capabilities_v1() {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  const MetalPipeline pipeline = pipeline_for(stream->device());
  require(pipeline.part != nil && pipeline.combine != nil,
          "MLA attention Metal pipeline is incomplete");
  return MlaAttnMetalCapabilities{
      .abi_version = kMlaAttnMetalAbiV1,
      .flags = kMlaAttnMetalRequiredCapabilitiesV1,
      .threads_per_threadgroup = kPartThreadsPerThreadgroup,
      .reserved = 0,
  };
}

void mla_attn_metal_decode_f32_with_partitions(
    const at::Tensor& destination, const at::Tensor& query,
    const at::Tensor& key_states, const at::Tensor& value_states,
    const std::uint32_t partitions) {
  loop_residency_register_activation(destination); loop_residency_register_activation(query); loop_residency_register_activation(key_states); loop_residency_register_activation(value_states);
  require(key_states.defined() && key_states.dim() == 4,
          "MLA attention key states must have rank four");
  const at::Device device = key_states.device();
  require(device.is_mps(), "MLA attention Metal requires MPS tensors");
  require(query.defined() && query.device() == device &&
              query.scalar_type() == at::kFloat && query.is_contiguous() &&
              query.sizes() ==
                  at::IntArrayRef({1, kHeads, 1, kQkHeadDim}),
          "MLA attention query must be a contiguous fp32 [1,96,1,192] view");
  require(destination.defined() && destination.device() == device &&
              destination.scalar_type() == at::kFloat &&
              destination.is_contiguous() &&
              destination.sizes() == at::IntArrayRef({1, 1, kOutputWidth}),
          "MLA attention destination must be contiguous fp32 [1,1,12288]");

  const std::int64_t kv_length = key_states.size(2);
  require(kv_length >= 1 && kv_length <= kMaxKvLength,
          "MLA attention kv length is outside the admitted range");
  require(key_states.scalar_type() == at::kFloat &&
              key_states.sizes() ==
                  at::IntArrayRef({1, kHeads, kv_length, kQkHeadDim}) &&
              key_states.stride(3) == 1 &&
              key_states.stride(2) == kQkHeadDim,
          "MLA attention key states must be an fp32 [1,96,S,192] slab view");
  // The provider narrows [1,96,capacity,192] to S positions, keeping the
  // parent's head pitch: re-derive the slab capacity from stride(1).  The
  // slab is re-checked every call because cache growth swaps storages.
  require(key_states.stride(1) % kQkHeadDim == 0,
          "MLA attention key head pitch is not a whole number of positions");
  const std::int64_t capacity = key_states.stride(1) / kQkHeadDim;
  require(capacity >= kv_length && capacity <= kMaxCapacity,
          "MLA attention key slab capacity is smaller than its length");
  require(value_states.defined() && value_states.device() == device &&
              value_states.scalar_type() == at::kFloat &&
              value_states.sizes() ==
                  at::IntArrayRef({1, kHeads, kv_length, kValueHeadDim}) &&
              value_states.stride(3) == 1 &&
              value_states.stride(2) == kValueHeadDim &&
              value_states.stride(1) == capacity * kValueHeadDim,
          "MLA attention value states disagree with the key slab capacity");
  require(partitions <= kMaxPartitions,
          "MLA attention partition count exceeds the kernel ABI");
  require_aligned_offset(query, "MLA attention query");
  require_aligned_offset(key_states, "MLA attention key states");
  require_aligned_offset(value_states, "MLA attention value states");
  require_aligned_offset(destination, "MLA attention destination");

  const std::uint32_t chosen =
      partitions == 0 ? choose_partitions(kv_length) : partitions;
  const std::uint32_t chunk = static_cast<std::uint32_t>(
      (kv_length + chosen - 1) / chosen);

  const std::size_t query_bytes =
      static_cast<std::size_t>(kHeads) * kQkHeadDim * sizeof(float);
  const std::size_t key_bytes =
      (static_cast<std::size_t>(kHeads - 1) * capacity + (kv_length - 1)) *
          kQkHeadDim * sizeof(float) +
      static_cast<std::size_t>(kQkHeadDim) * sizeof(float);
  const std::size_t value_bytes =
      (static_cast<std::size_t>(kHeads - 1) * capacity + (kv_length - 1)) *
          kValueHeadDim * sizeof(float) +
      static_cast<std::size_t>(kValueHeadDim) * sizeof(float);
  const std::size_t destination_bytes =
      static_cast<std::size_t>(kOutputWidth) * sizeof(float);
  const NSUInteger query_offset =
      tensor_byte_offset(query, query_bytes, "MLA attention query");
  const NSUInteger key_offset =
      tensor_byte_offset(key_states, key_bytes, "MLA attention key states");
  const NSUInteger value_offset = tensor_byte_offset(
      value_states, value_bytes, "MLA attention value states");
  const NSUInteger destination_offset = tensor_byte_offset(
      destination, destination_bytes, "MLA attention destination");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr &&
              stream->device() == tensor_buffer(key_states).device,
          "MLA attention tensors belong to another MPS device");
  const MetalPipeline pipeline = pipeline_for(stream->device());

  const std::int64_t partial_elements =
      kHeads * static_cast<std::int64_t>(chosen) * kPartialStride;
  const at::Tensor partials = at::empty(
      {partial_elements},
      at::TensorOptions().dtype(at::kFloat).device(device));
  const std::size_t partial_bytes =
      static_cast<std::size_t>(partial_elements) * sizeof(float);
  const NSUInteger partials_offset =
      tensor_byte_offset(partials, partial_bytes, "MLA attention partials");

  const MlaAttnDimsV1 dims{
      static_cast<std::uint32_t>(kv_length),
      static_cast<std::uint32_t>(capacity),
      chosen,
      chunk,
      attention_scale(),
      0,
      0,
      0,
  };
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil,
              "current MPS stream has no MLA attention encoder");
      [encoder setComputePipelineState:pipeline.part];
      [encoder setBuffer:tensor_buffer(query)
                  offset:query_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(key_states)
                  offset:key_offset
                 atIndex:1];
      [encoder setBuffer:tensor_buffer(value_states)
                  offset:value_offset
                 atIndex:2];
      [encoder setBuffer:tensor_buffer(partials)
                  offset:partials_offset
                 atIndex:3];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:4];
      [encoder dispatchThreadgroups:MTLSizeMake(
                                        static_cast<NSUInteger>(kHeads) *
                                            dims.partitions,
                                        1, 1)
              threadsPerThreadgroup:MTLSizeMake(kPartThreadsPerThreadgroup,
                                                1, 1)];
      // PyTorch's stream encoder is serial today; the explicit barrier keeps
      // the partial->combine dependency correct under any dispatch type.
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [encoder setComputePipelineState:pipeline.combine];
      [encoder setBuffer:tensor_buffer(partials)
                  offset:partials_offset
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(destination)
                  offset:destination_offset
                 atIndex:1];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(
                                        static_cast<NSUInteger>(kHeads), 1,
                                        1)
              threadsPerThreadgroup:MTLSizeMake(
                                        kCombineThreadsPerThreadgroup, 1,
                                        1)];
    }
  });
}

void mla_attn_metal_decode_f32(const at::Tensor& destination,
                               const at::Tensor& query,
                               const at::Tensor& key_states,
                               const at::Tensor& value_states) {
  mla_attn_metal_decode_f32_with_partitions(destination, query, key_states,
                                            value_states, 0);
}

MlaAttnMetalCanaryReport mla_attn_metal_canary_v1() {
  constexpr std::int64_t kS = 257;
  constexpr std::int64_t kCapacity = 300;
  constexpr std::uint32_t kPartitions = 32;  // chunk 9: partitions 29..31 empty
  constexpr std::int64_t kQueryOffset = 4;
  constexpr std::int64_t kKeyOffset = 8;
  constexpr std::int64_t kValueOffset = 12;
  constexpr std::int64_t kDestinationOffset = 16;
  constexpr std::int64_t kQueryElements = kHeads * kQkHeadDim;
  constexpr std::int64_t kKeyElements = kHeads * kCapacity * kQkHeadDim;
  constexpr std::int64_t kValueElements = kHeads * kCapacity * kValueHeadDim;

  std::uint32_t state = 0x2B7E1516U;
  const auto next_value = [&state]() {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>((state >> 8) & 0xFFFF) / 32768.0F - 1.0F;
  };
  std::vector<float> query_values(kQueryElements);
  std::vector<float> key_values(kKeyElements);
  std::vector<float> value_values(kValueElements);
  for (float& value : query_values) value = next_value();
  const float poison = std::numeric_limits<float>::quiet_NaN();
  for (std::int64_t head = 0; head < kHeads; ++head) {
    for (std::int64_t position = 0; position < kCapacity; ++position) {
      // NaN beyond S: any capacity-pitch indexing bug poisons the output.
      const bool live = position < kS;
      float* key_row =
          key_values.data() + (head * kCapacity + position) * kQkHeadDim;
      for (std::int64_t dim = 0; dim < kQkHeadDim; ++dim) {
        key_row[dim] = live ? next_value() : poison;
      }
      float* value_row =
          value_values.data() + (head * kCapacity + position) * kValueHeadDim;
      for (std::int64_t dim = 0; dim < kValueHeadDim; ++dim) {
        value_row[dim] = live ? next_value() : poison;
      }
    }
  }

  const auto mps_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS);
  const auto cpu_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor query_storage =
      at::zeros({kQueryOffset + kQueryElements + 12}, mps_float);
  at::Tensor key_storage =
      at::zeros({kKeyOffset + kKeyElements + 12}, mps_float);
  at::Tensor value_storage =
      at::zeros({kValueOffset + kValueElements + 12}, mps_float);
  at::Tensor destination_storage = at::full(
      {kDestinationOffset + kOutputWidth + 12}, -123.0F, mps_float);
  at::Tensor query = query_storage.narrow(0, kQueryOffset, kQueryElements)
                         .view({1, kHeads, 1, kQkHeadDim});
  at::Tensor key_slab = key_storage.narrow(0, kKeyOffset, kKeyElements)
                            .view({1, kHeads, kCapacity, kQkHeadDim});
  at::Tensor value_slab =
      value_storage.narrow(0, kValueOffset, kValueElements)
          .view({1, kHeads, kCapacity, kValueHeadDim});
  at::Tensor destination =
      destination_storage.narrow(0, kDestinationOffset, kOutputWidth)
          .view({1, 1, kOutputWidth});
  query.copy_(at::from_blob(query_values.data(),
                            {1, kHeads, 1, kQkHeadDim}, cpu_float),
              false);
  key_slab.copy_(at::from_blob(key_values.data(),
                               {1, kHeads, kCapacity, kQkHeadDim}, cpu_float),
                 false);
  value_slab.copy_(
      at::from_blob(value_values.data(),
                    {1, kHeads, kCapacity, kValueHeadDim}, cpu_float),
      false);
  const at::Tensor key_states = key_slab.narrow(2, 0, kS);
  const at::Tensor value_states = value_slab.narrow(2, 0, kS);

  mla_attn_metal_decode_f32_with_partitions(destination, query, key_states,
                                            value_states, kPartitions);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr,
          "current MPS stream disappeared during MLA attention canary");
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  const at::Tensor output = destination.to(at::kCPU).contiguous();
  const float* actual = output.const_data_ptr<float>();

  // fp64 host reference over the same fp32 inputs (kernel scale widened).
  const double scale = static_cast<double>(attention_scale());
  std::vector<double> expected(kOutputWidth, 0.0);
  std::vector<double> scores(kS);
  for (std::int64_t head = 0; head < kHeads; ++head) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::int64_t position = 0; position < kS; ++position) {
      const float* key_row =
          key_values.data() + (head * kCapacity + position) * kQkHeadDim;
      const float* query_row = query_values.data() + head * kQkHeadDim;
      double score = 0.0;
      for (std::int64_t dim = 0; dim < kQkHeadDim; ++dim) {
        score += static_cast<double>(query_row[dim]) *
                 static_cast<double>(key_row[dim]);
      }
      scores[position] = score * scale;
      maximum = std::max(maximum, scores[position]);
    }
    double total = 0.0;
    for (std::int64_t position = 0; position < kS; ++position) {
      scores[position] = std::exp(scores[position] - maximum);
      total += scores[position];
    }
    for (std::int64_t dim = 0; dim < kValueHeadDim; ++dim) {
      double accumulator = 0.0;
      for (std::int64_t position = 0; position < kS; ++position) {
        accumulator +=
            scores[position] *
            static_cast<double>(
                value_values[(head * kCapacity + position) * kValueHeadDim +
                             dim]);
      }
      expected[head * kValueHeadDim + dim] = accumulator / total;
    }
  }

  MlaAttnMetalCanaryReport report{
      .heads = static_cast<std::uint32_t>(kHeads),
      .kv_length = static_cast<std::uint32_t>(kS),
      .capacity = static_cast<std::uint32_t>(kCapacity),
      .partitions = kPartitions,
      .compared_elements = static_cast<std::uint32_t>(kOutputWidth),
      .close_elements = 0,
      .nonfinite = 0,
      .query_offset_elements = static_cast<std::uint32_t>(kQueryOffset),
      .key_offset_elements = static_cast<std::uint32_t>(kKeyOffset),
      .value_offset_elements = static_cast<std::uint32_t>(kValueOffset),
      .destination_offset_elements =
          static_cast<std::uint32_t>(kDestinationOffset),
      .reserved = 0,
  };
  double difference_squares = 0.0;
  double expected_squares = 0.0;
  for (std::int64_t index = 0; index < kOutputWidth; ++index) {
    const double got = static_cast<double>(actual[index]);
    const double want = expected[index];
    if (!std::isfinite(got) || !std::isfinite(want)) {
      ++report.nonfinite;
      continue;
    }
    const double difference = std::abs(got - want);
    report.max_absolute_error =
        std::max(report.max_absolute_error, difference);
    difference_squares += difference * difference;
    expected_squares += want * want;
    if (difference <=
        kMlaAttnMetalCanaryAtol + kMlaAttnMetalCanaryRtol * std::abs(want)) {
      ++report.close_elements;
    }
  }
  report.relative_l2_error =
      expected_squares > 0.0
          ? std::sqrt(difference_squares / expected_squares)
          : std::sqrt(difference_squares);
  return report;
}

bool mla_attn_metal_decode_ready() {
  static const bool enabled = [] {
    const char* value = std::getenv("K3_MLA_METAL_ATTENTION");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  if (!enabled) {
    return false;
  }
  static const bool qualified = [] {
    try {
      const MlaAttnMetalCapabilities capabilities =
          mla_attn_metal_capabilities_v1();
      if (capabilities.abi_version != kMlaAttnMetalAbiV1 ||
          capabilities.flags != kMlaAttnMetalRequiredCapabilitiesV1 ||
          capabilities.threads_per_threadgroup !=
              kPartThreadsPerThreadgroup ||
          capabilities.reserved != 0) {
        std::fprintf(stderr,
                     "[mla-attn-metal] disabled: capability mismatch\n");
        return false;
      }
      const MlaAttnMetalCanaryReport report = mla_attn_metal_canary_v1();
      const bool passed = mla_attn_metal_canary_passes(report);
      std::fprintf(stderr,
                   "[mla-attn-metal] qualification %s: close=%u/%u "
                   "max_abs=%.3e rel_l2=%.3e\n",
                   passed ? "PASS" : "FAIL", report.close_elements,
                   report.compared_elements, report.max_absolute_error,
                   report.relative_l2_error);
      return passed;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[mla-attn-metal] disabled: %s\n", error.what());
      return false;
    } catch (...) {
      std::fprintf(stderr, "[mla-attn-metal] disabled: unknown failure\n");
      return false;
    }
  }();
  return qualified;
}

}  // namespace deltafin::provider_internal
