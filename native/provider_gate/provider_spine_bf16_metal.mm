#if !defined(__APPLE__)
#error "provider_spine_bf16_metal.mm is Apple-only"
#endif
#if !defined(DELTAFIN_HAVE_SPINE_BF16_METAL_V1)
#error "BF16 spine Metal bridge requires its explicit production capability"
#endif
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#error "BF16 spine Metal bridge requires an embedded precompiled metallib"
#endif

#include "provider_spine_bf16_metal.h"
#include "provider_loop.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/ops/matmul.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "BF16 spine Metal bridge requires Objective-C ARC"
#endif

#include "deltafin_embedded_spine_bf16_metal_metallib.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace deltafin::provider_internal {
namespace {

constexpr std::uint32_t kRowsPerSimdgroup = 4;
constexpr std::uint32_t kSimdgroupsPerThreadgroup = 4;
constexpr std::uint32_t kThreadsPerThreadgroup = 128;
constexpr std::uint32_t kColumnsAlignment = 4;
constexpr std::uint32_t kMaximumPositions = 64;
constexpr std::size_t kProviderWeightAlignment = 256;
constexpr std::uint32_t kRowsPerThreadgroup =
    kRowsPerSimdgroup * kSimdgroupsPerThreadgroup;

struct GemvDimsV1 {
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t reserved0;
  std::uint32_t reserved1;
};

static_assert(sizeof(GemvDimsV1) == 16);

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void require(const bool condition, const std::string& message) {
  if (!condition) fail(message);
}

dispatch_data_t copy_embedded_metallib() {
  void* owned = std::malloc(kDeltafinEmbeddedSpineBf16MetalMetallibBytes);
  if (owned == nullptr) return nullptr;
  std::memcpy(owned, kDeltafinEmbeddedSpineBf16MetalMetallib,
              kDeltafinEmbeddedSpineBf16MetalMetallibBytes);
  dispatch_data_t data = dispatch_data_create(
      owned, kDeltafinEmbeddedSpineBf16MetalMetallibBytes, nullptr,
      DISPATCH_DATA_DESTRUCTOR_FREE);
  if (data == nullptr) std::free(owned);
  return data;
}

struct MetalPipelines {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> decode = nil;
  id<MTLComputePipelineState> rows4 = nil;
};

struct PipelineCache {
  std::mutex mutex;
  MetalPipelines pipelines;
};

PipelineCache& pipeline_cache() {
  static PipelineCache cache;
  return cache;
}

id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> library,
                                          NSString* function_name) {
  id<MTLFunction> function =
      [library newFunctionWithName:function_name];
  require(function != nil,
          std::string("embedded BF16 spine metallib is missing ") +
              function_name.UTF8String);
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline =
      [library.device newComputePipelineStateWithFunction:function
                                                     error:&error];
  require(pipeline != nil,
          std::string("create BF16 spine Metal pipeline failed: ") +
              (error == nil ? "unknown"
                            : error.localizedDescription.UTF8String));
  return pipeline;
}

MetalPipelines pipelines_for(id<MTLDevice> device) {
  require(device != nil, "BF16 spine Metal device is unavailable");
  PipelineCache& cache = pipeline_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.pipelines.device != device || cache.pipelines.rows4 == nil ||
      cache.pipelines.decode == nil) {
    dispatch_data_t data = copy_embedded_metallib();
    require(data != nullptr,
            "wrap embedded BF16 spine metallib failed");
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    require(library != nil,
            std::string("load embedded BF16 spine metallib failed: ") +
                (error == nil ? "unknown"
                              : error.localizedDescription.UTF8String));
    MetalPipelines replacement;
    replacement.device = device;
    replacement.library = library;
    replacement.decode = make_pipeline(
        library, @"deltafin_spine_decode_bf16_bits_v1");
    replacement.rows4 = make_pipeline(
        library, @"deltafin_spine_bf16_gemv_rows4_t1_v1");
    require(replacement.rows4.threadExecutionWidth == 32,
            "BF16 spine rows4 pipeline requires a 32-lane SIMD group");
    require(replacement.rows4.maxTotalThreadsPerThreadgroup >=
                kThreadsPerThreadgroup,
            "BF16 spine rows4 pipeline cannot admit 128 threads");
    cache.pipelines = replacement;
  }
  return cache.pipelines;
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

std::size_t checked_weight_bytes(const std::uint32_t rows,
                                 const std::uint32_t columns) {
  require(rows != 0 && columns != 0,
          "BF16 spine GEMV dimensions must be nonzero");
  require(columns % kColumnsAlignment == 0,
          "BF16 spine GEMV columns must be a multiple of four");
  const std::uint64_t elements =
      static_cast<std::uint64_t>(rows) * columns;
  require(elements <= std::numeric_limits<std::size_t>::max() /
                          sizeof(std::uint16_t),
          "BF16 spine GEMV weight byte count exceeds size_t");
  return static_cast<std::size_t>(elements) * sizeof(std::uint16_t);
}

at::Tensor encode_decode(id<MTLBuffer> source, id<MTLDevice> device,
                         std::size_t logical_bytes,
                         std::size_t source_byte_offset,
                         std::uint32_t elements);

struct FreeAligned {
  void operator()(std::uint8_t* pointer) const noexcept {
    std::free(pointer);
  }
};

std::uint16_t finite_bf16_bits(const std::size_t index) {
  const std::int32_t centered =
      static_cast<std::int32_t>((index * 73U + 19U) % 257U) - 128;
  const float value = static_cast<float>(centered) / 64.0F;
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16);
}

std::int64_t argmax(const float* values, const std::size_t length) {
  require(length != 0, "cannot take argmax of an empty canary");
  std::size_t best = 0;
  for (std::size_t index = 1; index < length; ++index) {
    if (values[index] > values[best]) best = index;
  }
  return static_cast<std::int64_t>(best);
}

}  // namespace

struct SpineBf16MetalBuffer::Impl {
  const void* host_pointer = nullptr;
  std::size_t logical_bytes = 0;
  std::size_t allocation_bytes = 0;
  id<MTLDevice> device = nil;
  id<MTLBuffer> buffer = nil;
  std::size_t buffer_byte_offset = 0;
  SpineBf16MetalStorageKind storage_kind =
      SpineBf16MetalStorageKind::BorrowedNoCopy;
  at::Tensor retained_tensor;
};

SpineBf16MetalBuffer::SpineBf16MetalBuffer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

SpineBf16MetalBuffer::~SpineBf16MetalBuffer() = default;
SpineBf16MetalBuffer::SpineBf16MetalBuffer(
    SpineBf16MetalBuffer&&) noexcept = default;
SpineBf16MetalBuffer& SpineBf16MetalBuffer::operator=(
    SpineBf16MetalBuffer&&) noexcept = default;

std::size_t SpineBf16MetalBuffer::logical_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->logical_bytes;
}

std::size_t SpineBf16MetalBuffer::allocation_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->allocation_bytes;
}

SpineBf16MetalStorageKind SpineBf16MetalBuffer::storage_kind() const noexcept {
  return impl_ == nullptr ? SpineBf16MetalStorageKind::BorrowedNoCopy
                          : impl_->storage_kind;
}

std::size_t SpineBf16MetalBuffer::bytes_per_element() const noexcept {
  return sizeof(std::uint16_t);
}

SpineBf16MetalCapabilities spine_bf16_metal_capabilities_v1() {
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  const MetalPipelines pipelines = pipelines_for(stream->device());
  require(pipelines.rows4 != nil && pipelines.decode != nil,
          "BF16 spine Metal pipelines are incomplete");
  return SpineBf16MetalCapabilities{
      .abi_version = kSpineBf16MetalAbiV1,
      .flags = kSpineBf16MetalRequiredCapabilitiesV1,
      .positions = kMaximumPositions,
      .rows_per_simdgroup = kRowsPerSimdgroup,
      .threads_per_threadgroup = kThreadsPerThreadgroup,
      .column_alignment = kColumnsAlignment,
  };
}

SpineBf16MetalBuffer wrap_spine_bf16_metal_buffer(
    const void* host_pointer, const std::size_t logical_bytes,
    const std::size_t allocation_bytes) {
  require(host_pointer != nullptr,
          "BF16 spine host pointer is null");
  require(logical_bytes != 0 && logical_bytes <= allocation_bytes &&
              logical_bytes % sizeof(std::uint16_t) == 0,
          "BF16 spine logical/allocation lengths are invalid");
  const long raw_page_size = sysconf(_SC_PAGESIZE);
  require(raw_page_size > 0,
          "query BF16 spine host page size failed");
  const std::size_t page_size = static_cast<std::size_t>(raw_page_size);
  require(reinterpret_cast<std::uintptr_t>(host_pointer) % page_size == 0,
          "BF16 spine host pointer is not page aligned");
  require(allocation_bytes % page_size == 0,
          "BF16 spine allocation length is not page aligned");
  require(allocation_bytes <= std::numeric_limits<NSUInteger>::max(),
          "BF16 spine allocation length exceeds NSUInteger");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  static_cast<void>(pipelines_for(device));
  id<MTLBuffer> buffer = [device
      newBufferWithBytesNoCopy:const_cast<void*>(host_pointer)
                        length:allocation_bytes
                       options:MTLResourceStorageModeShared |
                               MTLResourceCPUCacheModeDefaultCache
                   deallocator:nil];
  require(buffer != nil,
          "Metal rejected the page-aligned BF16 spine arena");
  require(buffer.length >= allocation_bytes && buffer.contents == host_pointer,
          "Metal changed the BF16 spine no-copy storage contract");
  auto impl = std::make_unique<SpineBf16MetalBuffer::Impl>();
  impl->host_pointer = host_pointer;
  impl->logical_bytes = logical_bytes;
  impl->allocation_bytes = allocation_bytes;
  impl->device = device;
  impl->buffer = buffer;
  impl->storage_kind = SpineBf16MetalStorageKind::BorrowedNoCopy;
  return SpineBf16MetalBuffer(std::move(impl));
}

SpineBf16MetalBuffer copy_spine_bf16_metal_buffer(
    const void* source, const std::size_t logical_bytes) {
  require(source != nullptr, "BF16 spine copy source is null");
  require(logical_bytes != 0 &&
              logical_bytes % sizeof(std::uint16_t) == 0 &&
              logical_bytes <= std::numeric_limits<NSUInteger>::max(),
          "BF16 spine copy length is invalid");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  static_cast<void>(pipelines_for(device));
  id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>(logical_bytes)
                  options:MTLResourceStorageModeShared |
                          MTLResourceCPUCacheModeDefaultCache];
  require(buffer != nil && buffer.contents != nullptr &&
              buffer.length >= logical_bytes,
          "allocate provider-owned BF16 Metal storage failed");
  std::memcpy(buffer.contents, source, logical_bytes);
  auto impl = std::make_unique<SpineBf16MetalBuffer::Impl>();
  impl->logical_bytes = logical_bytes;
  impl->allocation_bytes = static_cast<std::size_t>(buffer.length);
  impl->device = device;
  impl->buffer = buffer;
  impl->storage_kind = SpineBf16MetalStorageKind::OwnedSharedCopy;
  return SpineBf16MetalBuffer(std::move(impl));
}

SpineBf16MetalBuffer retain_spine_bf16_metal_tensor(
    const at::Tensor& tensor) {
  const bool exact_carrier =
      tensor.defined() &&
      (tensor.scalar_type() == at::kBFloat16 ||
       tensor.scalar_type() == at::kUInt16 ||
       tensor.scalar_type() == at::kShort);
  require(tensor.defined() && tensor.device().is_mps() &&
              exact_carrier &&
              tensor.is_contiguous() && tensor.storage_offset() == 0 &&
              tensor.element_size() == sizeof(std::uint16_t) &&
              tensor.numel() > 0 && !tensor.requires_grad(),
          "retained BF16 spine tensor must be contiguous MPS "
          "BFloat16/UInt16/Short raw storage with zero offset");
  const std::uint64_t elements =
      static_cast<std::uint64_t>(tensor.numel());
  require(elements <= std::numeric_limits<std::size_t>::max() /
                          sizeof(std::uint16_t),
          "retained BF16 spine tensor byte count exceeds size_t");
  const std::size_t logical_bytes =
      static_cast<std::size_t>(elements) * sizeof(std::uint16_t);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  id<MTLDevice> device = stream->device();
  require(device != nil, "current MPS device is unavailable");
  static_cast<void>(pipelines_for(device));
  id<MTLBuffer> buffer = tensor_buffer(tensor);
  require(buffer != nil && logical_bytes <= buffer.length,
          "retained BF16 spine tensor exceeds its MTLBuffer");
  loop_residency_register_tensor(tensor);
  const NSUInteger byte_offset = tensor_byte_offset(
      tensor, logical_bytes, "retained BF16 spine tensor");
  require(byte_offset == 0,
          "retained BF16 spine tensor must begin at its MTLBuffer base");
  auto impl = std::make_unique<SpineBf16MetalBuffer::Impl>();
  impl->logical_bytes = logical_bytes;
  impl->allocation_bytes = static_cast<std::size_t>(buffer.length);
  impl->device = device;
  impl->buffer = buffer;
  impl->buffer_byte_offset = static_cast<std::size_t>(byte_offset);
  impl->storage_kind = SpineBf16MetalStorageKind::RetainedMpsBf16;
  impl->retained_tensor = tensor;
  return SpineBf16MetalBuffer(std::move(impl));
}

at::Tensor spine_bf16_metal_gemv(
    const SpineBf16MetalBuffer& weight,
    const std::size_t weight_byte_offset, const std::uint32_t rows,
    const std::uint32_t columns, const at::Tensor& input) {
  require(weight.impl_ != nullptr && weight.impl_->buffer != nil,
          "BF16 spine GEMV received an empty no-copy wrapper");
  require(weight_byte_offset % kProviderWeightAlignment == 0,
          "BF16 spine GEMV weight offset is not 256-byte aligned");
  const std::size_t weight_bytes = checked_weight_bytes(rows, columns);
  require(weight_byte_offset <= weight.impl_->logical_bytes &&
              weight_bytes <=
                  weight.impl_->logical_bytes - weight_byte_offset,
          "BF16 spine GEMV weight range exceeds its logical slab");
  require(weight.impl_->buffer_byte_offset <=
                  std::numeric_limits<std::size_t>::max() -
                      weight_byte_offset &&
              weight.impl_->buffer_byte_offset + weight_byte_offset <=
                  weight.impl_->buffer.length &&
              weight_bytes <=
                  weight.impl_->buffer.length -
                      (weight.impl_->buffer_byte_offset +
                       weight_byte_offset),
          "BF16 spine GEMV weight range exceeds its physical MTLBuffer");
  require(input.defined() && input.device().is_mps() &&
              input.scalar_type() == at::kFloat && input.is_contiguous() &&
              input.dim() == 2 && input.size(0) >= 1 &&
              input.size(0) <= kMaximumPositions &&
              input.size(1) == static_cast<std::int64_t>(columns) &&
              !input.requires_grad(),
          "BF16 spine GEMV input must be contiguous MPS fp32 "
          "[1..64,columns]");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream is unavailable");
  id<MTLDevice> device = stream->device();
  require(device != nil && device == weight.impl_->device,
          "BF16 spine wrapper belongs to another MPS device");
  const MetalPipelines pipelines = pipelines_for(device);
  const std::uint32_t positions =
      static_cast<std::uint32_t>(input.size(0));
  at::Tensor output = at::empty(
      {static_cast<std::int64_t>(positions),
       static_cast<std::int64_t>(rows)},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  const std::size_t input_row_bytes =
      static_cast<std::size_t>(columns) * sizeof(float);
  const std::size_t output_row_bytes =
      static_cast<std::size_t>(rows) * sizeof(float);
  require(input_row_bytes <= std::numeric_limits<std::size_t>::max() /
                                positions &&
              output_row_bytes <= std::numeric_limits<std::size_t>::max() /
                                     positions,
          "BF16 spine multi-position byte span overflows size_t");
  const std::size_t input_bytes = input_row_bytes * positions;
  const std::size_t output_bytes = output_row_bytes * positions;
  const NSUInteger input_offset =
      tensor_byte_offset(input, input_bytes, "BF16 spine input");
  const NSUInteger output_offset =
      tensor_byte_offset(output, output_bytes, "BF16 spine output");
  const GemvDimsV1 dims{rows, columns, 0, 0};

  loop_residency_register_activation(input);
  loop_residency_register_activation(output);
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil,
              "current MPS stream has no BF16 spine encoder");
      [encoder setComputePipelineState:pipelines.rows4];
      [encoder setBuffer:weight.impl_->buffer
                  offset:static_cast<NSUInteger>(
                             weight.impl_->buffer_byte_offset +
                             weight_byte_offset)
                 atIndex:0];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:3];
      for (std::uint32_t position = 0; position < positions; ++position) {
        [encoder setBuffer:tensor_buffer(input)
                    offset:input_offset +
                           static_cast<NSUInteger>(position) * input_row_bytes
                   atIndex:1];
        [encoder setBuffer:tensor_buffer(output)
                    offset:output_offset +
                           static_cast<NSUInteger>(position) * output_row_bytes
                   atIndex:2];
        [encoder dispatchThreadgroups:
                     MTLSizeMake((rows + kRowsPerThreadgroup - 1) /
                                     kRowsPerThreadgroup,
                                 1, 1)
            threadsPerThreadgroup:
                MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
      }
    }
  });
  return output;
}

at::Tensor spine_bf16_metal_gemv_t1(
    const SpineBf16MetalBuffer& weight,
    const std::size_t weight_byte_offset, const std::uint32_t rows,
    const std::uint32_t columns, const at::Tensor& input) {
  require(input.defined() && input.dim() == 2 && input.size(0) == 1,
          "BF16 spine T=1 GEMV requires exactly one input position");
  return spine_bf16_metal_gemv(weight, weight_byte_offset, rows, columns,
                               input);
}

namespace {

at::Tensor encode_decode(id<MTLBuffer> source, id<MTLDevice> device,
                         const std::size_t logical_bytes,
                         const std::size_t source_byte_offset,
                         const std::uint32_t elements) {
  require(source != nil && device != nil && elements != 0,
          "BF16 spine decode received an empty source");
  require(source_byte_offset % sizeof(std::uint16_t) == 0,
          "BF16 spine decode offset is not ushort aligned");
  const std::size_t bytes =
      static_cast<std::size_t>(elements) * sizeof(std::uint16_t);
  require(source_byte_offset <= logical_bytes &&
              bytes <= logical_bytes - source_byte_offset,
          "BF16 spine decode range exceeds its logical slab");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr && stream->device() == device,
          "BF16 spine decode wrapper belongs to another MPS device");
  const MetalPipelines pipelines = pipelines_for(stream->device());
  at::Tensor output = at::empty(
      {static_cast<std::int64_t>(elements)},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  const NSUInteger output_offset = tensor_byte_offset(
      output, static_cast<std::size_t>(elements) * sizeof(float),
      "BF16 spine decode output");
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      require(encoder != nil,
              "current MPS stream has no BF16 decode encoder");
      [encoder setComputePipelineState:pipelines.decode];
      [encoder setBuffer:source
                  offset:static_cast<NSUInteger>(source_byte_offset)
                 atIndex:0];
      [encoder setBuffer:tensor_buffer(output)
                  offset:output_offset
                 atIndex:1];
      [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(elements, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
  });
  return output;
}

}  // namespace

SpineBf16MetalCanaryReport spine_bf16_metal_canary_v1() {
  constexpr std::uint32_t kDecodeElements = 1U << 16;
  constexpr std::uint32_t kRows = 37;
  constexpr std::uint32_t kColumns = 256;
  constexpr std::size_t kDecodeBytes =
      kDecodeElements * sizeof(std::uint16_t);
  constexpr std::size_t kWeightOffset = kDecodeBytes;
  constexpr std::size_t kWeightBytes =
      static_cast<std::size_t>(kRows) * kColumns * sizeof(std::uint16_t);
  constexpr std::size_t kLogicalBytes = kWeightOffset + kWeightBytes;
  constexpr std::uint32_t kOneHotColumn = 73;
  static_assert(kWeightOffset % kProviderWeightAlignment == 0);

  const long raw_page_size = sysconf(_SC_PAGESIZE);
  require(raw_page_size > 0,
          "query BF16 canary page size failed");
  const std::size_t page_size = static_cast<std::size_t>(raw_page_size);
  const std::size_t allocation_bytes =
      (kLogicalBytes + page_size - 1) / page_size * page_size;
  void* raw = nullptr;
  const int allocation_status =
      posix_memalign(&raw, page_size, allocation_bytes);
  require(allocation_status == 0 && raw != nullptr,
          "allocate BF16 spine canary arena failed");
  std::unique_ptr<std::uint8_t, FreeAligned> arena(
      static_cast<std::uint8_t*>(raw));
  std::memset(arena.get(), 0, allocation_bytes);
  auto* words = reinterpret_cast<std::uint16_t*>(arena.get());
  for (std::uint32_t bits = 0; bits < kDecodeElements; ++bits) {
    words[bits] = static_cast<std::uint16_t>(bits);
  }
  auto* weight_words = reinterpret_cast<std::uint16_t*>(
      arena.get() + kWeightOffset);
  for (std::size_t index = 0;
       index < static_cast<std::size_t>(kRows) * kColumns; ++index) {
    weight_words[index] = finite_bf16_bits(index);
  }

  SpineBf16MetalBuffer shared = wrap_spine_bf16_metal_buffer(
      arena.get(), kLogicalBytes, allocation_bytes);
  std::vector<float> one_hot_values(kColumns, 0.0F);
  one_hot_values[kOneHotColumn] = 1.0F;
  std::vector<float> dense_values(kColumns, 0.0F);
  for (std::uint32_t column = 0; column < kColumns; ++column) {
    const std::int32_t centered =
        static_cast<std::int32_t>((column * 41U + 11U) % 193U) - 96;
    dense_values[column] = static_cast<float>(centered) / 1024.0F;
  }
  const at::Tensor one_hot_cpu = at::from_blob(
      one_hot_values.data(), {1, kColumns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const at::Tensor dense_cpu = at::from_blob(
      dense_values.data(), {1, kColumns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const at::Tensor one_hot = one_hot_cpu.to(at::kMPS).contiguous();
  const at::Tensor dense = dense_cpu.to(at::kMPS).contiguous();
  at::Tensor decoded = encode_decode(
      shared.impl_->buffer, shared.impl_->device,
      shared.impl_->logical_bytes, 0, kDecodeElements);
  at::Tensor one_hot_output = spine_bf16_metal_gemv_t1(
      shared, kWeightOffset, kRows, kColumns, one_hot);
  at::Tensor dense_output = spine_bf16_metal_gemv_t1(
      shared, kWeightOffset, kRows, kColumns, dense);

  const at::Tensor weight_cpu = at::from_blob(
      weight_words, {kRows, kColumns},
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
  at::Tensor weight_fp32 = at::empty(
      {kRows, kColumns},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  weight_fp32.copy_(weight_cpu, false);
  at::Tensor dense_reference =
      at::matmul(dense, weight_fp32.transpose(0, 1)).contiguous();

  // Production GEMV calls above perform no commit or wait. The qualification
  // batches every custom dispatch and its independent reference before this
  // one deliberate canary-only host boundary.
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  require(stream != nullptr, "current MPS stream disappeared during canary");
  stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

  const at::Tensor decoded_cpu = decoded.to(at::kCPU).contiguous();
  const at::Tensor one_hot_result =
      one_hot_output.to(at::kCPU).contiguous();
  const at::Tensor dense_result = dense_output.to(at::kCPU).contiguous();
  const at::Tensor dense_reference_cpu =
      dense_reference.to(at::kCPU).contiguous();
  const float* decoded_values = decoded_cpu.const_data_ptr<float>();
  const float* one_hot_result_values =
      one_hot_result.const_data_ptr<float>();
  const float* dense_result_values = dense_result.const_data_ptr<float>();
  const float* dense_reference_values =
      dense_reference_cpu.const_data_ptr<float>();

  SpineBf16MetalCanaryReport report;
  report.decoded_elements = kDecodeElements;
  report.rows = kRows;
  for (std::uint32_t index = 0; index < kDecodeElements; ++index) {
    const std::uint32_t expected =
        index << 16;
    if (std::bit_cast<std::uint32_t>(decoded_values[index]) == expected) {
      ++report.decoded_equal_bits;
    }
  }
  for (std::uint32_t row = 0; row < kRows; ++row) {
    const std::uint32_t expected =
        static_cast<std::uint32_t>(
            weight_words[static_cast<std::size_t>(row) * kColumns +
                         kOneHotColumn])
        << 16;
    if (std::bit_cast<std::uint32_t>(one_hot_result_values[row]) ==
        expected) {
      ++report.one_hot_equal_bits;
    }
    const float reference = dense_reference_values[row];
    const float candidate = dense_result_values[row];
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
      ++report.nonfinite;
      continue;
    }
    if (std::bit_cast<std::uint32_t>(reference) ==
        std::bit_cast<std::uint32_t>(candidate)) {
      ++report.dense_equal_bits;
    }
    report.dense_maximum_absolute = std::max(
        report.dense_maximum_absolute, std::abs(reference - candidate));
  }
  report.dense_reference_argmax =
      argmax(dense_reference_values, kRows);
  report.dense_candidate_argmax = argmax(dense_result_values, kRows);
  return report;
}

}  // namespace deltafin::provider_internal
