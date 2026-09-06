// Kimi-K3 Metal MoE bridge — host side for tools/metal/moe_mxfp4.metal.
//
// Runs a whole layer's selected experts (MXFP4 dequant + w1/w3 GEMV + SiTU + w2
// GEMV + weighted sum) on the GPU from one command buffer.  Persistent device,
// queue, library and pipelines; expert bytes are wrapped zero-copy where the
// pointer allows it, so a cached expert costs no host-side traffic at all.
//
// Cargo compiles the reviewed MSL to a metallib and embeds those bytes in the
// production executable. Release inference therefore never invokes Metal's
// source compiler and has no loose shader-file dependency. Selection remains
// explicit and versioned:
//   1. NULL, empty, or K3_METAL_MOE_EMBEDDED_SOURCE_V1 -> embedded metallib
//   2. any other non-empty string -> debug/development source path, when built
//      with DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1
//
// ---------------------------------------------------------------------------
// C ABI (all functions are thread-safe; a global mutex serialises GPU work)
//
//   int  k3_metal_available(void)
//        1 if a Metal device + library + pipelines came up, else 0.  Cheap after
//        the first call.  Never throws; inspect k3_metal_last_error() on 0.
//
//   int  k3_metal_init(const char *source_selector)
//        Initialize with the versioned embedded selector (NULL/empty select it
//        too), or an explicit development source path. Returns 0 on success,
//        negative on failure. Idempotent and compiled only once per process.
//
//   int  k3_metal_moe_layer(const uint8_t *const *expert_blobs, int n_experts,
//                           const float *weights, const float *x, float *out)
//        expert_blobs[e] -> the expert's verbatim 17,547,264-byte shard span
//            (w1_p 5505024 | w1_s 344064 | w2_p 5505024 | w2_s 344064 |
//             w3_p 5505024 | w3_s 344064).  Must stay mapped and unmodified for
//            the duration of the call; if it is page-aligned it is ALSO retained
//            in the zero-copy buffer cache afterwards (see k3_metal_drop).
//        weights[n_experts]  fp32 routing weights, applied in the given order.
//        x[3584]             fp32 activations for one token.
//        out[3584]           fp32 result, overwritten.
//        Returns 0 on success, negative on error.  Synchronous.
//
//   int  k3_metal_moe_positions(
//            const uint8_t *const *expert_blobs, int n_edges,
//            const int *position_offsets, int n_positions,
//            const float *weights, const float *x, float *out)
//        Opt-in N5 path. `position_offsets` has n_positions+1 entries, starts at
//        zero, ends at n_edges, and partitions row-major expert_blobs/weights.
//        x/out are [n_positions, 3584]. The ordinary per-position kernels are
//        encoded in the same order into ONE command buffer and waited once.
//        Arithmetic and reduction order within every position are unchanged.
//
//   int  k3_metal_moe_positions_flat(
//            const uint8_t *const *expert_blobs, int n_edges,
//            const int *position_offsets, int n_positions,
//            const float *weights, const float *x, float *out)
//        Same synchronous contract, but a Tier-2 device runs every independent
//        position edge in one GLU dispatch, one W2 dispatch, then one ordered
//        per-position reduction. Returns -6 when bindless execution is not
//        available, allowing the caller to retain the established ABI.
//
//   int  k3_metal_moe_positions_flat_fused(
//            const uint8_t *const *expert_blobs, int n_edges,
//            const int *position_offsets, int n_positions,
//            const float *weights, const float *x, float *out)
//        Experimental exact variant of the flat ABI. The GLU remains one flat
//        dispatch, while W2 and the ordered per-position weighted reduction are
//        fused into one dispatch. Returns -6 when its optional kernel or Tier-2
//        bindless execution is unavailable, so callers can fall back safely.
//
//   void k3_metal_drop(const uint8_t *blob)
//        Release the cached MTLBuffer for one blob pointer.  MUST be called
//        before the caller unmaps/frees that memory, otherwise a later call that
//        re-uses the pointer would read a dangling wrap.
//
//   void k3_metal_flush(void)   drop the whole buffer cache.
//
//   int  k3_metal_set_mode(int bindless)   1 = one dispatch for all experts via
//        a Tier-2 argument buffer (default when supported); 0 = one dispatch per
//        expert inside a single concurrent encoder.  Returns the mode in effect.
//   int  k3_metal_mode(void)
//
//   void k3_metal_shapes(int *hidden, int *inter, long long *span)
//   void k3_metal_stats(long long *five)
//        [calls, zero_copy_wraps, copies, cache_entries, bindless]
//   const char *k3_metal_last_error(void)
// ---------------------------------------------------------------------------

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#if !__has_feature(objc_arc)
#error "Deltafin's Metal bridge requires Objective-C ARC"
#endif

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
#include "deltafin_embedded_moe_mxfp4_metallib.h"
#else
#include "deltafin_embedded_moe_mxfp4_msl.h"
#endif
#include "metal_moe_abi.h"
#include "../native/provider_gate/provider_prep_timer.h"
#include <mach/mach_time.h>

#include <algorithm>
#include <array>

/* Expert-kernel split (2026-09-03): time the synchronous MoE command-buffer
 * wait for the open finish phase scope; the part before the CB's
 * GPUStartTime is queueing behind earlier GPU work (drain). */
static uint64_t k3_prep_media_ns(void) {
  static mach_timebase_info_data_t tb = {0, 0};
  if (tb.denom == 0) { mach_timebase_info(&tb); }
  return mach_absolute_time() * tb.numer / tb.denom;
}
namespace deltafin::provider_internal {
void loop_gpu_timeline_note_cb(id<MTLCommandBuffer> command,
                               const char* klass) noexcept;
void loop_residency_attach_queue(void* queue) noexcept;
void loop_residency_register_allocation(void* mtl_buffer) noexcept;
bool loop_residency_expert_buffers_enabled() noexcept;
}
static inline void k3_reg(id<MTLBuffer> b) {
  if (b != nil && deltafin::provider_internal::loop_residency_expert_buffers_enabled())
    deltafin::provider_internal::loop_residency_register_allocation((__bridge void*)b);
}
static void k3_prep_timed_wait(id<MTLCommandBuffer> cb) {
  if (!deltafin::provider_internal::prep_phase_active()) {
    [cb waitUntilCompleted];
    return;
  }
  const uint64_t t0 = k3_prep_media_ns();
  [cb waitUntilCompleted];
  const uint64_t t1 = k3_prep_media_ns();
  const uint64_t waited = t1 - t0;
  uint64_t drain = 0;
  bool stamped = false;
  const double gs = cb.GPUStartTime;
  if (gs > 0.0) {
    stamped = true;
    const uint64_t g = (uint64_t)(gs * 1e9);
    if (g > t0) { drain = (g - t0 < waited) ? (g - t0) : waited; }
  }
  deltafin::provider_internal::prep_note_wait(waited, drain, stamped);
}
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

// ---- fixed model geometry (mirrors tools/fetch_v2.py LAYOUT) ---------------
static const uint32_t K3_H = 3584;                 // hidden / x length
static const uint32_t K3_I = 3072;                 // intermediate / h length
static const uint32_t P_BYTES = 5505024, S_BYTES = 344064;
static const uint32_t OFF_W1P = 0;
static const uint32_t OFF_W1S = P_BYTES;
static const uint32_t OFF_W2P = P_BYTES + S_BYTES;
static const uint32_t OFF_W2S = 2 * P_BYTES + S_BYTES;
static const uint32_t OFF_W3P = 2 * (P_BYTES + S_BYTES);
static const uint32_t OFF_W3S = 3 * P_BYTES + 2 * S_BYTES;
static const size_t   EXPERT_SPAN = 3 * (size_t)(P_BYTES + S_BYTES);   // 17,547,264
static_assert(EXPERT_SPAN == K3_RAW_V1_EXPERT_SPAN,
              "raw-v1 Metal expert span ABI drift");

// C119 scale4-table-v2 in-memory layout. Storage may be a compact sidecar plus
// the canonical raw packed planes; the descriptor presented here is always one
// assembled contiguous span with these offsets. Metal never assumes file
// provenance.
static const uint32_t SCALE4_HEADER_BYTES = 8192;
static const uint32_t SCALE4_BYTES = S_BYTES / 2;
static const uint32_t SCALE4_OFF_W1P = 8192;
static const uint32_t SCALE4_OFF_W1S = 5513216;
static const uint32_t SCALE4_OFF_W2P = 5685248;
static const uint32_t SCALE4_OFF_W2S = 11190272;
static const uint32_t SCALE4_OFF_W3P = 11362304;
static const uint32_t SCALE4_OFF_W3S = 16867328;
static const size_t SCALE4_SPAN = K3_SCALE4_V2_EXPERT_SPAN;
static_assert(SCALE4_OFF_W1P == SCALE4_HEADER_BYTES,
              "scale4 header/payload boundary drift");
static_assert(SCALE4_OFF_W1S == SCALE4_OFF_W1P + P_BYTES,
              "scale4 w1 packed extent drift");
static_assert(SCALE4_OFF_W2P == SCALE4_OFF_W1S + SCALE4_BYTES,
              "scale4 w1 scale extent drift");
static_assert(SCALE4_OFF_W2S == SCALE4_OFF_W2P + P_BYTES,
              "scale4 w2 packed extent drift");
static_assert(SCALE4_OFF_W3P == SCALE4_OFF_W2S + SCALE4_BYTES,
              "scale4 w2 scale extent drift");
static_assert(SCALE4_OFF_W3S == SCALE4_OFF_W3P + P_BYTES,
              "scale4 w3 packed extent drift");
static_assert(SCALE4_OFF_W3S + SCALE4_BYTES == SCALE4_SPAN,
              "scale4 expert span drift");
static_assert(SCALE4_SPAN % (16 * 1024) == 0,
              "scale4 expert slots must retain 16 KiB alignment");

static const uint32_t ROWS_PER_TG = 16;            // 4 simdgroups x 4 rows
static const uint32_t TG_THREADS  = 128;
static const uint32_t FUSED_TG_THREADS = 512;       // 16 experts x one simdgroup

struct MoeDims {                                   // must match moe_mxfp4.metal
  uint32_t n_experts, H, I;
  uint32_t w1p, w1s, w2p, w2s, w3p, w3s;
  uint32_t expert, nt;
};

// ---------------------------------------------------------------------------
namespace {

std::mutex g_mu;
std::string g_err;
bool g_tried = false, g_ok = false;

id<MTLDevice> g_dev = nil;
id<MTLCommandQueue> g_q = nil;
id<MTLComputePipelineState> g_pGluB = nil, g_pW2B = nil;
id<MTLComputePipelineState> g_pGluPosB = nil, g_pRedPos = nil;
id<MTLComputePipelineState> g_pW2RedPos = nil;
id<MTLComputePipelineState> g_pGlu1 = nil, g_pW21 = nil;
id<MTLComputePipelineState> g_pRed = nil;
id<MTLComputePipelineState> g_pGluBScale4 = nil, g_pW2BScale4 = nil;
id<MTLComputePipelineState> g_pGluPosBScale4 = nil;
id<MTLComputePipelineState> g_pW2RedPosScale4 = nil;
id<MTLComputePipelineState> g_pGlu1Scale4 = nil, g_pW21Scale4 = nil;
bool g_scale4_ok = false;

id<MTLBuffer> g_bX = nil, g_bH = nil, g_bY = nil, g_bOut = nil, g_bW = nil, g_bArgs = nil;
// N5 uses separate pools so the default one-position allocation and execution
// path above remain byte-for-byte untouched until explicitly requested.
id<MTLBuffer> g_bBatchX = nil, g_bBatchH = nil, g_bBatchY = nil;
id<MTLBuffer> g_bBatchOut = nil, g_bBatchW = nil, g_bBatchArgs = nil;
id<MTLBuffer> g_bBatchEdgePosition = nil, g_bBatchOffsets = nil;
NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_cache = nil;   // blob ptr -> wrap
NSMutableDictionary<NSNumber *, NSNumber *> *g_cache_meta = nil; // ptr -> layout/length
NSMutableArray<id<MTLBuffer>> *g_scratch = nil;                  // copy-path staging
int g_cap = 0;                                                   // pool capacity in experts
int g_batch_edge_cap = 0, g_batch_position_cap = 0;
int g_bindless = -1;                                             // -1 = auto
long long g_calls = 0, g_wraps = 0, g_copies = 0;

void seterr(const char *what, NSError *e) {
  g_err = what;
  if (e) { g_err += ": "; g_err += [[e localizedDescription] UTF8String]; }
}

#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
dispatch_data_t copy_embedded_metallib(const unsigned char *bytes,
                                       std::size_t length) {
  void *owned = std::malloc(length);
  if (!owned) return nullptr;
  std::memcpy(owned, bytes, length);
  dispatch_data_t data = dispatch_data_create(
      owned, length, nullptr, DISPATCH_DATA_DESTRUCTOR_FREE);
  if (!data) std::free(owned);
  return data;
}
#endif

bool embedded_source_selected(const char *selector) {
  return selector == nullptr || *selector == '\0' ||
         std::strcmp(selector, K3_METAL_MOE_EMBEDDED_SOURCE_V1) == 0;
}

#if defined(DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1)
NSString *load_development_msl_source(const char *selector) {
#if !defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  if (embedded_source_selected(selector)) {
    NSString *source = [[NSString alloc]
        initWithBytes:kDeltafinEmbeddedMoeMxfp4Msl
               length:kDeltafinEmbeddedMoeMxfp4MslBytes
             encoding:NSUTF8StringEncoding];
    if (!source) seterr("decode embedded MSL v1", nil);
    return source;
  }
#endif

  // Loose source is deliberately an explicit-only development/test escape
  // hatch. Production callers send the versioned selector and never probe the
  // current directory, dylib directory, or environment from this bridge.
  const std::string path(selector);
  NSError *error = nil;
  NSString *source = [NSString stringWithContentsOfFile:@(path.c_str())
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
  if (!source) seterr(("read development MSL " + path).c_str(), error);
  return source;
}
#endif  // DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1

id<MTLLibrary> load_metal_library(const char *selector, NSError **error) {
  if (embedded_source_selected(selector)) {
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
    dispatch_data_t data = copy_embedded_metallib(
        kDeltafinEmbeddedMoeMxfp4Metallib,
        kDeltafinEmbeddedMoeMxfp4MetallibBytes);
    if (!data) {
      seterr("wrap embedded MoE metallib", nil);
      return nil;
    }
    id<MTLLibrary> library = [g_dev newLibraryWithData:data error:error];
    if (!library) seterr("load embedded MoE metallib", *error);
    return library;
#elif defined(DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1)
    NSString *source = load_development_msl_source(selector);
    if (!source) return nil;
    id<MTLLibrary> library =
        [g_dev newLibraryWithSource:source options:nil error:error];
    if (!library) seterr("compile embedded development MSL", *error);
    return library;
#else
    seterr("embedded MoE metallib is unavailable", nil);
    return nil;
#endif
  }

#if defined(DELTAFIN_ENABLE_METAL_SOURCE_RUNTIME_V1)
  NSString *source = load_development_msl_source(selector);
  if (!source) return nil;
  id<MTLLibrary> library =
      [g_dev newLibraryWithSource:source options:nil error:error];
  if (!library) seterr("compile development MSL", *error);
  return library;
#else
  seterr("development Metal source is disabled in this build", nil);
  return nil;
#endif
}

id<MTLComputePipelineState> mkpipe(id<MTLLibrary> lib, NSString *name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) { seterr([[@"missing kernel " stringByAppendingString:name] UTF8String], nil); return nil; }
  NSError *err = nil;
  id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) seterr("pipeline", err);
  return p;
}

id<MTLComputePipelineState> mkpipe_optional(id<MTLLibrary> lib,
                                            NSString *name) {
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  if (!fn) return nil;
  NSError *err = nil;
  return [g_dev newComputePipelineStateWithFunction:fn error:&err];
}

struct LayoutSpec {
  uint32_t id;
  size_t span;
  uint32_t w1p, w1s, w2p, w2s, w3p, w3s;
  bool scale4;
};

LayoutSpec raw_layout() {
  return {K3_LAYOUT_RAW_V1, EXPERT_SPAN,
          OFF_W1P, OFF_W1S, OFF_W2P, OFF_W2S, OFF_W3P, OFF_W3S, false};
}

LayoutSpec scale4_layout() {
  return {K3_LAYOUT_SCALE4_V2, SCALE4_SPAN,
          SCALE4_OFF_W1P, SCALE4_OFF_W1S,
          SCALE4_OFF_W2P, SCALE4_OFF_W2S,
          SCALE4_OFF_W3P, SCALE4_OFF_W3S, true};
}

bool layout_for_id(uint32_t id, LayoutSpec *out) {
  if (!out) return false;
  if (id == K3_LAYOUT_RAW_V1) {
    *out = raw_layout();
    return true;
  }
  if (id == K3_LAYOUT_SCALE4_V2) {
    *out = scale4_layout();
    return true;
  }
  return false;
}

uint32_t read_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t read_le64(const uint8_t *p) {
  return (uint64_t)read_le32(p) | ((uint64_t)read_le32(p + 4) << 32);
}

bool bytes_zero(const uint8_t *p, size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i)
    if (p[i] != 0) return false;
  return true;
}

bool validate_scale4_header(const uint8_t *p) {
  static const uint8_t magic[8] = {'K', '3', 'S', 'C', '4', 'V', '2', 0};
  static const std::array<uint32_t, 12> spans = {
      SCALE4_OFF_W1P, P_BYTES, SCALE4_OFF_W1S, SCALE4_BYTES,
      SCALE4_OFF_W2P, P_BYTES, SCALE4_OFF_W2S, SCALE4_BYTES,
      SCALE4_OFF_W3P, P_BYTES, SCALE4_OFF_W3S, SCALE4_BYTES,
  };
  if (!p || std::memcmp(p, magic, sizeof(magic)) != 0 ||
      read_le32(p + 8) != 2 ||
      read_le32(p + 12) != SCALE4_HEADER_BYTES ||
      read_le64(p + 16) != SCALE4_SPAN ||
      read_le64(p + 24) != EXPERT_SPAN) {
    g_err = "invalid scale4-v2 header identity or length";
    return false;
  }
  for (size_t i = 0; i < spans.size(); ++i) {
    if (read_le32(p + 72 + i * 4) != spans[i]) {
      g_err = "invalid scale4-v2 payload table";
      return false;
    }
  }
  static const std::array<uint32_t, 6> sidecar_spans = {
      8192, SCALE4_BYTES, 180224, SCALE4_BYTES,
      352256, SCALE4_BYTES,
  };
  for (size_t i = 0; i < sidecar_spans.size(); ++i) {
    if (read_le32(p + 320 + i * 4) != sidecar_spans[i]) {
      g_err = "invalid scale4-v2 sidecar payload table";
      return false;
    }
  }
  if (!bytes_zero(p, 67, 72) || !bytes_zero(p, 120, 128) ||
      !bytes_zero(p, 344, SCALE4_HEADER_BYTES)) {
    g_err = "scale4-v2 reserved header bytes are nonzero";
    return false;
  }
  static const uint32_t table_offsets[3] = {128, 192, 256};
  for (size_t matrix = 0; matrix < 3; ++matrix) {
    uint32_t base = p[64 + matrix];
    if (base > 240) {
      g_err = "scale4-v2 base exceeds the exact uint8 range";
      return false;
    }
    for (uint32_t delta = 0; delta < 16; ++delta) {
      uint32_t expected = (base + delta) << 23;
      if (read_le32(p + table_offsets[matrix] + delta * 4) != expected) {
        g_err = "scale4-v2 exponent table does not match its base";
        return false;
      }
    }
  }
  return true;
}

bool descriptors_locked(const K3MetalExpertDescriptorV1 *descs, int count,
                        std::vector<const uint8_t *> *blobs,
                        LayoutSpec *layout) {
  if (count < 0 || !blobs || !layout || (count > 0 && !descs)) {
    g_err = "invalid Metal expert descriptor array";
    return false;
  }
  blobs->clear();
  blobs->reserve((size_t)std::max(count, 0));
  if (count == 0) {
    *layout = raw_layout();
    return true;
  }
  uint32_t selected = 0;
  for (int i = 0; i < count; ++i) {
    const K3MetalExpertDescriptorV1 &d = descs[i];
    if (d.abi_version != K3_DESCRIPTOR_ABI_V1 ||
        d.struct_bytes != sizeof(K3MetalExpertDescriptorV1) ||
        d.reserved != 0 || !d.blob) {
      g_err = "invalid Metal expert descriptor fields";
      return false;
    }
    LayoutSpec candidate {};
    if (!layout_for_id(d.layout_id, &candidate) ||
        d.blob_bytes != candidate.span) {
      g_err = "unknown Metal expert layout or byte length";
      return false;
    }
    if (i == 0) {
      selected = d.layout_id;
      *layout = candidate;
      if (candidate.scale4 && !g_scale4_ok) {
        g_err = "scale4-v2 Metal pipelines are unavailable";
        return false;
      }
    } else if (d.layout_id != selected) {
      g_err = "mixed Metal expert layouts are not allowed in one dispatch";
      return false;
    }
    if (candidate.scale4 && !validate_scale4_header(d.blob)) return false;
    blobs->push_back(d.blob);
  }
  return true;
}

MoeDims dims_for(const LayoutSpec &layout, uint32_t count) {
  return {count, K3_H, K3_I,
          layout.w1p, layout.w1s, layout.w2p, layout.w2s,
          layout.w3p, layout.w3s, 0, 0};
}

// Grow the float pools so they hold `n` experts' worth of h and y.
bool ensure_capacity(int n) {
  if (n <= g_cap) return true;
  int cap = n < 16 ? 16 : n;
  id<MTLBuffer> bH = [g_dev newBufferWithLength:(size_t)cap * K3_I * 4
                                        options:MTLResourceStorageModeShared];
  k3_reg(bH);
  id<MTLBuffer> bY = [g_dev newBufferWithLength:(size_t)cap * K3_H * 4
                                        options:MTLResourceStorageModeShared];
  k3_reg(bY);
  id<MTLBuffer> bW = [g_dev newBufferWithLength:(size_t)cap * 4
                                        options:MTLResourceStorageModeShared];
  k3_reg(bW);
  id<MTLBuffer> bA = [g_dev newBufferWithLength:(size_t)cap * 8
                                        options:MTLResourceStorageModeShared];
  k3_reg(bA);
  if (!bH || !bY || !bW || !bA) { seterr("pool alloc", nil); return false; }
  g_bH = bH; g_bY = bY; g_bW = bW; g_bArgs = bA; g_cap = cap;
  return true;
}

// Separate capacity for the opt-in multi-position ABI. Keeping it out of
// ensure_capacity() prevents a speculative T=8 call from permanently inflating
// the established T=1 scratch pools.
bool ensure_batch_capacity(int edges, int positions) {
  if (edges <= g_batch_edge_cap && positions <= g_batch_position_cap) return true;
  const int edge_cap = std::max(std::max(edges, g_batch_edge_cap), 1);
  const int position_cap =
      std::max(std::max(positions, g_batch_position_cap), 1);
  id<MTLBuffer> bX =
      [g_dev newBufferWithLength:(size_t)position_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bX);
  id<MTLBuffer> bH =
      [g_dev newBufferWithLength:(size_t)edge_cap * K3_I * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bH);
  id<MTLBuffer> bY =
      [g_dev newBufferWithLength:(size_t)edge_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bY);
  id<MTLBuffer> bOut =
      [g_dev newBufferWithLength:(size_t)position_cap * K3_H * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bOut);
  id<MTLBuffer> bW =
      [g_dev newBufferWithLength:(size_t)edge_cap * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bW);
  id<MTLBuffer> bArgs =
      [g_dev newBufferWithLength:(size_t)edge_cap * 8
                         options:MTLResourceStorageModeShared];
  k3_reg(bArgs);
  id<MTLBuffer> bEdgePosition =
      [g_dev newBufferWithLength:(size_t)edge_cap * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bEdgePosition);
  id<MTLBuffer> bOffsets =
      [g_dev newBufferWithLength:(size_t)(position_cap + 1) * 4
                         options:MTLResourceStorageModeShared];
  k3_reg(bOffsets);
  if (!bX || !bH || !bY || !bOut || !bW || !bArgs ||
      !bEdgePosition || !bOffsets) {
    seterr("position-batch pool alloc", nil);
    return false;
  }
  g_bBatchX = bX;
  g_bBatchH = bH;
  g_bBatchY = bY;
  g_bBatchOut = bOut;
  g_bBatchW = bW;
  g_bBatchArgs = bArgs;
  g_bBatchEdgePosition = bEdgePosition;
  g_bBatchOffsets = bOffsets;
  g_batch_edge_cap = edge_cap;
  g_batch_position_cap = position_cap;
  return true;
}

bool init_locked(const char *source_selector) {
  if (g_tried) return g_ok;
  g_tried = true;
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { seterr("no Metal device", nil); return false; }
    g_q = [g_dev newCommandQueue];
    deltafin::provider_internal::loop_residency_attach_queue((__bridge void*)g_q);
    if (!g_q) { seterr("no command queue", nil); return false; }

    NSError *err = nil;
    id<MTLLibrary> lib = load_metal_library(source_selector, &err);
    if (!lib) return false;

    g_pGluB = mkpipe(lib, @"moe_glu_batch");
    g_pGluPosB = mkpipe_optional(lib, @"moe_glu_positions_batch");
    g_pW2B  = mkpipe(lib, @"moe_w2_batch");
    g_pGlu1 = mkpipe(lib, @"moe_glu_one");
    g_pW21  = mkpipe(lib, @"moe_w2_one");
    g_pRed  = mkpipe(lib, @"moe_reduce");
    g_pRedPos = mkpipe_optional(lib, @"moe_reduce_positions");
    g_pW2RedPos = mkpipe_optional(lib, @"moe_w2_reduce_positions");
    g_pGluBScale4 = mkpipe_optional(lib, @"moe_glu_batch_scale4");
    g_pGluPosBScale4 =
        mkpipe_optional(lib, @"moe_glu_positions_batch_scale4");
    g_pW2BScale4 = mkpipe_optional(lib, @"moe_w2_batch_scale4");
    g_pW2RedPosScale4 =
        mkpipe_optional(lib, @"moe_w2_reduce_positions_scale4");
    g_pGlu1Scale4 = mkpipe_optional(lib, @"moe_glu_one_scale4");
    g_pW21Scale4 = mkpipe_optional(lib, @"moe_w2_one_scale4");
    if (!g_pGluB || !g_pW2B || !g_pGlu1 || !g_pW21 || !g_pRed) return false;
    if ([g_pGluB threadExecutionWidth] != 32) { seterr("simd width != 32", nil); return false; }
    g_scale4_ok =
        g_pGluBScale4 && g_pGluPosBScale4 && g_pW2BScale4 &&
        g_pW2RedPosScale4 && g_pGlu1Scale4 && g_pW21Scale4 &&
        [g_pGluBScale4 threadExecutionWidth] == 32;

    g_bX   = [g_dev newBufferWithLength:K3_H * 4 options:MTLResourceStorageModeShared];
  k3_reg(g_bX);
    g_bOut = [g_dev newBufferWithLength:K3_H * 4 options:MTLResourceStorageModeShared];
  k3_reg(g_bOut);
    if (!g_bX || !g_bOut) { seterr("pool alloc", nil); return false; }
    if (!ensure_capacity(16)) return false;

    g_cache = [NSMutableDictionary dictionary];
    g_cache_meta = [NSMutableDictionary dictionary];
    g_scratch = [NSMutableArray array];

    if (g_bindless < 0) {
      bool tier2 = ([g_dev argumentBuffersSupport] == MTLArgumentBuffersTier2) &&
                   [g_bX respondsToSelector:@selector(gpuAddress)];
      const char *env = getenv("K3_METAL_BINDLESS");
      g_bindless = env ? (atoi(env) != 0 && tier2) : (tier2 ? 1 : 0);
    }
    g_ok = true;
  }
  return g_ok;
}

// ---------------------------------------------------------------------------
// Registered host arena (K3_EXPERT_RAM_CACHE).
//
// A RAM cache holds thousands of expert spans live at once. Wrapping each one
// individually costs twice over, and the SECOND cost is the expensive one:
//
//   1. one MTLBuffer per distinct pointer (g_cache grew 112 -> 1364 measured), and
//   2. one [enc useResource:] residency call PER EXPERT PER DISPATCH.
//
// The read path never pays either, because it recycles a small arena and Metal
// therefore sees the same ~112 buffers forever. Measured consequence of not
// doing this: expert_kernel 13.75s -> 18.74s (+36%) purely as a function of how
// many cache hits were served, with copies=0 throughout — so it was never a
// memcpy cost.
//
// Registering the cache arena once gives one wrap and one useResource for the
// whole pool, whatever its size, and expert addresses become base + offset.
// Bindless already consumes raw gpuAddress values, so no shader change is needed.
static id<MTLBuffer> g_arena_buf = nil;
static const uint8_t *g_arena_base = nullptr;
static size_t g_arena_len = 0;
static uint64_t g_arena_gpu = 0;

static inline bool arena_contains(const uint8_t *p, size_t span) {
  return g_arena_buf && p && g_arena_base && p >= g_arena_base &&
         (size_t)(p - g_arena_base) + span <= g_arena_len;
}

// Device address for a blob: arena base + offset when the span lives in the
// registered arena, otherwise the wrapped buffer's own address.
static inline uint64_t blob_gpu_addr_of(const uint8_t *p, id<MTLBuffer> b, size_t span) {
  if (arena_contains(p, span)) return g_arena_gpu + (uint64_t)(p - g_arena_base);
  return [b gpuAddress];
}

// Wrap one versioned expert blob. Both raw-v1 and the assembled scale4-v2 span
// are exact 16 KiB multiples. Cache metadata is part of the identity: a recycled
// address may never resurrect a wrap with the previous layout or length.
id<MTLBuffer> wrap_blob(const uint8_t *p, int slot,
                        const LayoutSpec &layout) {
  // Arena spans share one wrap; they never enter the per-pointer cache.
  if (arena_contains(p, (size_t)layout.span)) return g_arena_buf;
  NSNumber *key = @((unsigned long long)(uintptr_t)p);
  id<MTLBuffer> b = g_cache[key];
  uint64_t signature = ((uint64_t)layout.id << 32) | (uint64_t)layout.span;
  NSNumber *expected = @(signature);
  if (b && [g_cache_meta[key] isEqualToNumber:expected]) return b;
  if (b) {
    [g_cache removeObjectForKey:key];
    [g_cache_meta removeObjectForKey:key];
  }
  size_t pg = (size_t)getpagesize();
  if (((uintptr_t)p % pg) == 0 && (layout.span % pg) == 0) {
    b = [g_dev newBufferWithBytesNoCopy:(void *)p length:layout.span
                                options:MTLResourceStorageModeShared deallocator:nil];
  k3_reg(b);
    if (b) {
      g_cache[key] = b;
      g_cache_meta[key] = expected;
      g_wraps++;
      return b;
    }
  }
  while ((int)[g_scratch count] <= slot) {
    id<MTLBuffer> s = [g_dev newBufferWithLength:EXPERT_SPAN
                                         options:MTLResourceStorageModeShared];
  k3_reg(s);
    if (!s) { seterr("scratch alloc", nil); return nil; }
    [g_scratch addObject:s];
  }
  b = g_scratch[slot];              // never cached by pointer: contents are stale-able
  memcpy([b contents], p, layout.span);
  g_copies++;
  return b;
}

}  // namespace

// ---------------------------------------------------------------------------
extern "C" {

int k3_metal_embedded_library_cycle_test_v1(int cycles) {
#if defined(DELTAFIN_HAVE_PRECOMPILED_METAL_LIBRARIES_V1)
  if (cycles <= 0 || cycles > 256) return -1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -2;
    for (int cycle = 0; cycle < cycles; ++cycle) {
      @autoreleasepool {
        dispatch_data_t data = copy_embedded_metallib(
            kDeltafinEmbeddedMoeMxfp4Metallib,
            kDeltafinEmbeddedMoeMxfp4MetallibBytes);
        if (!data) return -3;
        NSError *error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithData:data error:&error];
        if (!library || error != nil) return -4;
      }
    }
  }
  return 0;
#else
  (void)cycles;
  return -5;
#endif
}

int k3_metal_init(const char *msl_path) {
  std::lock_guard<std::mutex> lk(g_mu);
  return init_locked(msl_path) ? 0 : -1;
}

int k3_metal_available(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  return init_locked(nullptr) ? 1 : 0;
}

const char *k3_metal_last_error(void) { return g_err.c_str(); }

uint32_t k3_metal_descriptor_abi_version(void) {
  return K3_DESCRIPTOR_ABI_V1;
}

uint64_t k3_metal_layout_capabilities(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return 0;
  return K3_CAP_RAW_V1 | (g_scale4_ok ? K3_CAP_SCALE4_V2 : 0);
}

void k3_metal_shapes(int *hidden, int *inter, long long *span) {
  if (hidden) *hidden = (int)K3_H;
  if (inter) *inter = (int)K3_I;
  if (span) *span = (long long)EXPERT_SPAN;
}

void k3_metal_stats(long long *five) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!five) return;
  five[0] = g_calls; five[1] = g_wraps; five[2] = g_copies;
  five[3] = (long long)[g_cache count]; five[4] = g_bindless;
}

int k3_metal_mode(void) { std::lock_guard<std::mutex> lk(g_mu); return g_bindless; }

int k3_metal_set_mode(int bindless) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  bool tier2 = ([g_dev argumentBuffersSupport] == MTLArgumentBuffersTier2) &&
               [g_bX respondsToSelector:@selector(gpuAddress)];
  g_bindless = (bindless && tier2) ? 1 : 0;
  return g_bindless;
}

void k3_metal_drop(const uint8_t *blob) {
  std::lock_guard<std::mutex> lk(g_mu);
  NSNumber *key = @((unsigned long long)(uintptr_t)blob);
  if (g_cache) [g_cache removeObjectForKey:key];
  if (g_cache_meta) [g_cache_meta removeObjectForKey:key];
}

// Register a host arena so every span inside it shares one wrap and one
// residency call. len=0 unregisters. Additive: with no arena registered every
// path behaves exactly as before.
int32_t k3_metal_register_arena(const uint8_t *base, unsigned long long len) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  g_arena_buf = nil;
  g_arena_base = nullptr;
  g_arena_len = 0;
  g_arena_gpu = 0;
  if (!base || len == 0) return 0;
  size_t pg = (size_t)getpagesize();
  if (((uintptr_t)base % pg) != 0) return -2;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base length:(NSUInteger)len
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
  k3_reg(b);
  if (!b) return -3;
  if (![b respondsToSelector:@selector(gpuAddress)]) return -4;
  g_arena_buf = b;
  g_arena_base = base;
  g_arena_len = (size_t)len;
  g_arena_gpu = [b gpuAddress];
  return 0;
}

void k3_metal_flush(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_cache) [g_cache removeAllObjects];
  if (g_cache_meta) [g_cache_meta removeAllObjects];
  if (g_scratch) [g_scratch removeAllObjects];
}

static int k3_metal_moe_layer_locked(
                       const uint8_t *const *expert_blobs, int n_experts,
                       const float *weights, const float *x, float *out,
                       const LayoutSpec &layout) {
  if (!out || !x) return -2;
  if (n_experts < 0) return -2;
  if (n_experts == 0) { memset(out, 0, K3_H * 4); return 0; }
  if (!expert_blobs || !weights) return -2;
  if (!ensure_capacity(n_experts)) return -3;

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e) {
      if (!expert_blobs[e]) return -2;
      bufs[e] = wrap_blob(expert_blobs[e], e, layout);
      if (!bufs[e]) return -4;
    }
    memcpy([g_bX contents], x, K3_H * 4);
    memcpy([g_bW contents], weights, (size_t)n_experts * 4);

    MoeDims D = dims_for(layout, (uint32_t)n_experts);
    id<MTLComputePipelineState> pGluB =
        layout.scale4 ? g_pGluBScale4 : g_pGluB;
    id<MTLComputePipelineState> pW2B =
        layout.scale4 ? g_pW2BScale4 : g_pW2B;
    id<MTLComputePipelineState> pGlu1 =
        layout.scale4 ? g_pGlu1Scale4 : g_pGlu1;
    id<MTLComputePipelineState> pW21 =
        layout.scale4 ? g_pW21Scale4 : g_pW21;

    id<MTLCommandBuffer> cb = [g_q commandBuffer];

    if (g_bindless == 1) {
      uint64_t *addr = (uint64_t *)[g_bArgs contents];
      for (int e = 0; e < n_experts; ++e)
        addr[e] = blob_gpu_addr_of(expert_blobs[e], bufs[e], (size_t)layout.span);
      // serial encoder: glu -> w2 -> reduce are dependent, the implicit barriers
      // between dispatches in a serial encoder are exactly the ordering we need.
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      // Deduped: N experts inside one registered arena cost ONE residency
      // call, not N. This is the half of the wrapping tax that scales with
      // dispatches rather than with distinct pointers.
      for (int e = 0; e < n_experts; ++e) {
        bool seen = false;
        for (int j = 0; j < e; ++j)
          if (bufs[j] == bufs[e]) { seen = true; break; }
        if (!seen) [enc useResource:bufs[e] usage:MTLResourceUsageRead];
      }

      D.nt = (uint32_t)n_experts * K3_I;
      [enc setComputePipelineState:pGluB];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:g_bX offset:0 atIndex:1];
      [enc setBuffer:g_bH offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

      D.nt = (uint32_t)n_experts * K3_H;
      [enc setComputePipelineState:pW2B];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:g_bH offset:0 atIndex:1];
      [enc setBuffer:g_bY offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

      [enc setComputePipelineState:g_pRed];
      [enc setBuffer:g_bY offset:0 atIndex:0];
      [enc setBuffer:g_bW offset:0 atIndex:1];
      [enc setBuffer:g_bOut offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
    } else {
      // stage 1: every expert independent -> concurrent dispatches, no barriers.
      id<MTLComputeCommandEncoder> e1 =
          [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [e1 setComputePipelineState:pGlu1];
      [e1 setBuffer:g_bX offset:0 atIndex:1];
      [e1 setBuffer:g_bH offset:0 atIndex:2];
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e; D.nt = K3_I;
        [e1 setBuffer:bufs[e] offset:0 atIndex:0];
        [e1 setBytes:&D length:sizeof D atIndex:3];
        [e1 dispatchThreadgroups:MTLSizeMake((K3_I + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
      [e1 endEncoding];                       // encoder boundary == full barrier

      id<MTLComputeCommandEncoder> e2 =
          [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      [e2 setComputePipelineState:pW21];
      [e2 setBuffer:g_bH offset:0 atIndex:1];
      [e2 setBuffer:g_bY offset:0 atIndex:2];
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e; D.nt = K3_H;
        [e2 setBuffer:bufs[e] offset:0 atIndex:0];
        [e2 setBytes:&D length:sizeof D atIndex:3];
        [e2 dispatchThreadgroups:MTLSizeMake((K3_H + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
      [e2 endEncoding];

      id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
      [e3 setComputePipelineState:g_pRed];
      [e3 setBuffer:g_bY offset:0 atIndex:0];
      [e3 setBuffer:g_bW offset:0 atIndex:1];
      [e3 setBuffer:g_bOut offset:0 atIndex:2];
      [e3 setBytes:&D length:sizeof D atIndex:3];
      [e3 dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [e3 endEncoding];
    }

    deltafin::provider_internal::loop_gpu_timeline_note_cb(cb, "moe-t1");
    [cb commit];
    k3_prep_timed_wait(cb);
    if ([cb status] == MTLCommandBufferStatusError) {
      seterr("command buffer", [cb error]);
      return -5;
    }
    memcpy(out, [g_bOut contents], K3_H * 4);
    g_calls++;
  }
  return 0;
}

int k3_metal_moe_layer(const uint8_t *const *expert_blobs, int n_experts,
                       const float *weights, const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  LayoutSpec layout = raw_layout();
  return k3_metal_moe_layer_locked(
      expert_blobs, n_experts, weights, x, out, layout);
}

// Encode the T=1 raw-v1 expert stack (glu -> w2 -> reduce) onto a CALLER
// command buffer — the loop-queue CB_tail path (K3-SIDEQUEUE-PLAN.md Step
// 5).  Reuses this file's pipelines, wrap cache, and h/y scratch; the
// caller's x/out buffers are bound directly (no host staging), route
// weights land in g_bW before commit.  The caller's queue must belong to
// the same device (returns -6 otherwise, letting it fall back to stock).
// Sync discipline: the caller commits and host-waits its CB before
// releasing the expert byte lease, so the wrap cache never outlives the
// bytes it wraps.
int k3_metal_moe_encode_t1_raw_v1(id<MTLCommandBuffer> cb,
                                  id<MTLDevice> device,
                                  const uint8_t *const *expert_blobs,
                                  int n_experts, const float *weights,
                                  id<MTLBuffer> x_buf, NSUInteger x_off,
                                  id<MTLBuffer> out_buf,
                                  NSUInteger out_off) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  if (cb == nil || device == nil || !expert_blobs || !weights ||
      x_buf == nil || out_buf == nil) {
    return -2;
  }
  if (device != g_dev) {
    seterr("caller device differs from the MoE Metal device", nil);
    return -6;
  }
  if (n_experts <= 0 || n_experts > 64) return -2;
  if (!ensure_capacity(n_experts)) return -3;
  const LayoutSpec layout = raw_layout();

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_experts);
    for (int e = 0; e < n_experts; ++e) {
      if (!expert_blobs[e]) return -2;
      bufs[e] = wrap_blob(expert_blobs[e], e, layout);
      if (!bufs[e]) return -4;
    }
    memcpy([g_bW contents], weights, (size_t)n_experts * 4);

    MoeDims D = dims_for(layout, (uint32_t)n_experts);
    // Serial encoder on the caller CB: glu -> w2 -> reduce are dependent;
    // implicit barriers between dispatches give exactly that ordering.
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (enc == nil) return -5;
    if (g_bindless == 1) {
      uint64_t *addr = (uint64_t *)[g_bArgs contents];
      for (int e = 0; e < n_experts; ++e)
        addr[e] = blob_gpu_addr_of(expert_blobs[e], bufs[e], (size_t)layout.span);
      // Deduped: N experts inside one registered arena cost ONE residency
      // call, not N. This is the half of the wrapping tax that scales with
      // dispatches rather than with distinct pointers.
      for (int e = 0; e < n_experts; ++e) {
        bool seen = false;
        for (int j = 0; j < e; ++j)
          if (bufs[j] == bufs[e]) { seen = true; break; }
        if (!seen) [enc useResource:bufs[e] usage:MTLResourceUsageRead];
      }
      D.nt = (uint32_t)n_experts * K3_I;
      [enc setComputePipelineState:g_pGluB];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:x_buf offset:x_off atIndex:1];
      [enc setBuffer:g_bH offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(
                                    (D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG,
                                    1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      D.nt = (uint32_t)n_experts * K3_H;
      [enc setComputePipelineState:g_pW2B];
      [enc setBuffer:g_bArgs offset:0 atIndex:0];
      [enc setBuffer:g_bH offset:0 atIndex:1];
      [enc setBuffer:g_bY offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake(
                                    (D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG,
                                    1, 1)
          threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
    } else {
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e;
        D.nt = K3_I;
        [enc setComputePipelineState:g_pGlu1];
        [enc setBuffer:bufs[e] offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:x_off atIndex:1];
        [enc setBuffer:g_bH offset:(size_t)e * K3_I * 4 atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(
                                      (K3_I + ROWS_PER_TG - 1) / ROWS_PER_TG,
                                      1, 1)
            threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
      for (int e = 0; e < n_experts; ++e) {
        D.expert = (uint32_t)e;
        D.nt = K3_H;
        [enc setComputePipelineState:g_pW21];
        [enc setBuffer:bufs[e] offset:0 atIndex:0];
        [enc setBuffer:g_bH offset:(size_t)e * K3_I * 4 atIndex:1];
        [enc setBuffer:g_bY offset:(size_t)e * K3_H * 4 atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(
                                      (K3_H + ROWS_PER_TG - 1) / ROWS_PER_TG,
                                      1, 1)
            threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
    }
    MoeDims R = dims_for(layout, (uint32_t)n_experts);
    [enc setComputePipelineState:g_pRed];
    [enc setBuffer:g_bY offset:0 atIndex:0];
    [enc setBuffer:g_bW offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:out_off atIndex:2];
    [enc setBytes:&R length:sizeof R atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    ++g_calls;
  }
  return 0;
}

int k3_metal_moe_layer_desc_v1(
                       const K3MetalExpertDescriptorV1 *experts,
                       int n_experts, const float *weights,
                       const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  std::vector<const uint8_t *> blobs;
  LayoutSpec layout {};
  if (!descriptors_locked(experts, n_experts, &blobs, &layout)) return -7;
  return k3_metal_moe_layer_locked(
      blobs.empty() ? nullptr : blobs.data(), n_experts,
      weights, x, out, layout);
}

static int k3_metal_moe_positions_locked(
                           const uint8_t *const *expert_blobs, int n_edges,
                           const int *position_offsets, int n_positions,
                           const float *weights, const float *x, float *out,
                           const LayoutSpec &layout) {
  if (n_edges < 0 || n_positions < 0) return -2;
  if (n_positions == 0) return n_edges == 0 ? 0 : -2;
  if (!position_offsets || !x || !out) return -2;
  if (position_offsets[0] != 0) return -2;
  for (int p = 0; p < n_positions; ++p) {
    if (position_offsets[p] < 0 ||
        position_offsets[p] > position_offsets[p + 1] ||
        position_offsets[p + 1] > n_edges) {
      return -2;
    }
  }
  if (position_offsets[n_positions] != n_edges) return -2;
  if (n_edges > 0 && (!expert_blobs || !weights)) return -2;
  if (!ensure_batch_capacity(n_edges, n_positions)) return -3;

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_edges);
    for (int edge = 0; edge < n_edges; ++edge) {
      if (!expert_blobs[edge]) return -2;
      bufs[edge] = wrap_blob(expert_blobs[edge], edge, layout);
      if (!bufs[edge]) return -4;
    }
    memcpy([g_bBatchX contents], x, (size_t)n_positions * K3_H * 4);
    if (n_edges > 0)
      memcpy([g_bBatchW contents], weights, (size_t)n_edges * 4);
    // Empty route rows are legal at the ABI boundary. Their reduce dispatch is
    // skipped, so pre-zero every output row before encoding the nonempty ones.
    memset([g_bBatchOut contents], 0, (size_t)n_positions * K3_H * 4);

    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    if (!cb) {
      seterr("position-batch command buffer", nil);
      return -5;
    }

    if (g_bindless == 1) {
      id<MTLComputePipelineState> pGluB =
          layout.scale4 ? g_pGluBScale4 : g_pGluB;
      id<MTLComputePipelineState> pW2B =
          layout.scale4 ? g_pW2BScale4 : g_pW2B;
      uint64_t *addr = (uint64_t *)[g_bBatchArgs contents];
      for (int edge = 0; edge < n_edges; ++edge)
        addr[edge] = blob_gpu_addr_of(expert_blobs[edge], bufs[edge],
                                      (size_t)layout.span);

      // One serial encoder preserves every dependency in the proven N5 probe:
      // position 0 GLU -> W2 -> reduce, then position 1, and so on.
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int edge = 0; edge < n_edges; ++edge)
        [enc useResource:bufs[edge] usage:MTLResourceUsageRead];

      for (int p = 0; p < n_positions; ++p) {
        const int edge0 = position_offsets[p];
        const int count = position_offsets[p + 1] - edge0;
        if (count == 0) continue;
        const size_t xoff = (size_t)p * K3_H * 4;
        const size_t hoff = (size_t)edge0 * K3_I * 4;
        const size_t yoff = (size_t)edge0 * K3_H * 4;
        const size_t woff = (size_t)edge0 * 4;
        const size_t aoff = (size_t)edge0 * 8;
        const size_t ooff = (size_t)p * K3_H * 4;
        MoeDims D = dims_for(layout, (uint32_t)count);
        D.nt = (uint32_t)count * K3_I;

        [enc setComputePipelineState:pGluB];
        [enc setBuffer:g_bBatchArgs offset:aoff atIndex:0];
        [enc setBuffer:g_bBatchX offset:xoff atIndex:1];
        [enc setBuffer:g_bBatchH offset:hoff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:
                 MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

        D.nt = (uint32_t)count * K3_H;
        [enc setComputePipelineState:pW2B];
        [enc setBuffer:g_bBatchArgs offset:aoff atIndex:0];
        [enc setBuffer:g_bBatchH offset:hoff atIndex:1];
        [enc setBuffer:g_bBatchY offset:yoff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:
                 MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

        [enc setComputePipelineState:g_pRed];
        [enc setBuffer:g_bBatchY offset:yoff atIndex:0];
        [enc setBuffer:g_bBatchW offset:woff atIndex:1];
        [enc setBuffer:g_bBatchOut offset:ooff atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      }
      [enc endEncoding];
    } else {
      id<MTLComputePipelineState> pGlu1 =
          layout.scale4 ? g_pGlu1Scale4 : g_pGlu1;
      id<MTLComputePipelineState> pW21 =
          layout.scale4 ? g_pW21Scale4 : g_pW21;
      // The direct-bind fallback keeps the established concurrent-per-stage
      // schedule. Encoder boundaries preserve GLU/W2/reduce dependencies while
      // all positions still share one command buffer and one completion wait.
      for (int p = 0; p < n_positions; ++p) {
        const int edge0 = position_offsets[p];
        const int count = position_offsets[p + 1] - edge0;
        if (count == 0) continue;
        const size_t xoff = (size_t)p * K3_H * 4;
        const size_t hoff = (size_t)edge0 * K3_I * 4;
        const size_t yoff = (size_t)edge0 * K3_H * 4;
        const size_t woff = (size_t)edge0 * 4;
        const size_t ooff = (size_t)p * K3_H * 4;
        MoeDims D = dims_for(layout, (uint32_t)count);

        id<MTLComputeCommandEncoder> e1 =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e1 setComputePipelineState:pGlu1];
        [e1 setBuffer:g_bBatchX offset:xoff atIndex:1];
        [e1 setBuffer:g_bBatchH offset:hoff atIndex:2];
        for (int edge = 0; edge < count; ++edge) {
          D.expert = (uint32_t)edge;
          D.nt = K3_I;
          [e1 setBuffer:bufs[(size_t)edge0 + edge] offset:0 atIndex:0];
          [e1 setBytes:&D length:sizeof D atIndex:3];
          [e1 dispatchThreadgroups:
                  MTLSizeMake((K3_I + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
        }
        [e1 endEncoding];

        id<MTLComputeCommandEncoder> e2 =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e2 setComputePipelineState:pW21];
        [e2 setBuffer:g_bBatchH offset:hoff atIndex:1];
        [e2 setBuffer:g_bBatchY offset:yoff atIndex:2];
        for (int edge = 0; edge < count; ++edge) {
          D.expert = (uint32_t)edge;
          D.nt = K3_H;
          [e2 setBuffer:bufs[(size_t)edge0 + edge] offset:0 atIndex:0];
          [e2 setBytes:&D length:sizeof D atIndex:3];
          [e2 dispatchThreadgroups:
                  MTLSizeMake((K3_H + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
        }
        [e2 endEncoding];

        id<MTLComputeCommandEncoder> e3 = [cb computeCommandEncoder];
        [e3 setComputePipelineState:g_pRed];
        [e3 setBuffer:g_bBatchY offset:yoff atIndex:0];
        [e3 setBuffer:g_bBatchW offset:woff atIndex:1];
        [e3 setBuffer:g_bBatchOut offset:ooff atIndex:2];
        [e3 setBytes:&D length:sizeof D atIndex:3];
        [e3 dispatchThreadgroups:MTLSizeMake((K3_H + 255) / 256, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e3 endEncoding];
      }
    }

    deltafin::provider_internal::loop_gpu_timeline_note_cb(cb, "moe-batch");
    [cb commit];
    k3_prep_timed_wait(cb);
    if ([cb status] == MTLCommandBufferStatusError) {
      seterr("position-batch command buffer", [cb error]);
      return -5;
    }
    memcpy(out, [g_bBatchOut contents], (size_t)n_positions * K3_H * 4);
    g_calls += n_positions;
  }
  return 0;
}

int k3_metal_moe_positions(const uint8_t *const *expert_blobs, int n_edges,
                           const int *position_offsets, int n_positions,
                           const float *weights, const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  LayoutSpec layout = raw_layout();
  return k3_metal_moe_positions_locked(
      expert_blobs, n_edges, position_offsets, n_positions,
      weights, x, out, layout);
}

int k3_metal_moe_positions_desc_v1(
                           const K3MetalExpertDescriptorV1 *experts,
                           int n_edges, const int *position_offsets,
                           int n_positions, const float *weights,
                           const float *x, float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  std::vector<const uint8_t *> blobs;
  LayoutSpec layout {};
  if (!descriptors_locked(experts, n_edges, &blobs, &layout)) return -7;
  return k3_metal_moe_positions_locked(
      blobs.empty() ? nullptr : blobs.data(), n_edges,
      position_offsets, n_positions, weights, x, out, layout);
}

static int k3_metal_moe_positions_flat_impl_locked(
                                const uint8_t *const *expert_blobs, int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out, bool fused_reduce,
                                const LayoutSpec &layout) {
  if (n_edges < 0 || n_positions < 0) return -2;
  // The exact verifier currently stops at T=9. Keep a small forward-compatible
  // ceiling for separately qualified T=13/T=16 experiments without permitting
  // an accidental unbounded shared-buffer allocation through the C ABI.
  if (n_positions > 16 || n_edges > n_positions * 16) return -2;
  if (n_positions == 0) return n_edges == 0 ? 0 : -2;
  if (!position_offsets || !x || !out) return -2;
  if (position_offsets[0] != 0) return -2;
  for (int p = 0; p < n_positions; ++p) {
    if (position_offsets[p] < 0 ||
        position_offsets[p] > position_offsets[p + 1] ||
        position_offsets[p + 1] > n_edges) {
      return -2;
    }
    if (fused_reduce && position_offsets[p + 1] - position_offsets[p] > 16)
      return -2;
  }
  if (position_offsets[n_positions] != n_edges) return -2;
  if (n_edges > 0 && (!expert_blobs || !weights)) return -2;
  id<MTLComputePipelineState> pGluPosB =
      layout.scale4 ? g_pGluPosBScale4 : g_pGluPosB;
  id<MTLComputePipelineState> pW2B =
      layout.scale4 ? g_pW2BScale4 : g_pW2B;
  id<MTLComputePipelineState> pW2RedPos =
      layout.scale4 ? g_pW2RedPosScale4 : g_pW2RedPos;
  if (g_bindless != 1 || !pGluPosB ||
      (fused_reduce ? (!pW2RedPos ||
                       [pW2RedPos maxTotalThreadsPerThreadgroup] <
                           FUSED_TG_THREADS)
                    : !g_pRedPos)) {
    return -6;
  }
  if ((uint64_t)n_edges * K3_I > UINT32_MAX ||
      (uint64_t)n_edges * K3_H > UINT32_MAX ||
      (uint64_t)n_positions * K3_H > UINT32_MAX) {
    return -2;
  }
  if (!ensure_batch_capacity(n_edges, n_positions)) return -3;

  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_edges);
    std::vector<uint32_t> edge_position((size_t)n_edges);
    std::vector<uint32_t> offsets((size_t)n_positions + 1);
    for (int p = 0; p <= n_positions; ++p)
      offsets[(size_t)p] = (uint32_t)position_offsets[p];
    for (int p = 0; p < n_positions; ++p) {
      for (int edge = position_offsets[p]; edge < position_offsets[p + 1]; ++edge)
        edge_position[(size_t)edge] = (uint32_t)p;
    }
    for (int edge = 0; edge < n_edges; ++edge) {
      if (!expert_blobs[edge]) return -2;
      bufs[(size_t)edge] = wrap_blob(expert_blobs[edge], edge, layout);
      if (!bufs[(size_t)edge]) return -4;
    }

    memcpy([g_bBatchX contents], x, (size_t)n_positions * K3_H * 4);
    if (n_edges > 0) {
      memcpy([g_bBatchW contents], weights, (size_t)n_edges * 4);
      memcpy([g_bBatchEdgePosition contents], edge_position.data(),
             (size_t)n_edges * 4);
    }
    memcpy([g_bBatchOffsets contents], offsets.data(),
           ((size_t)n_positions + 1) * 4);

    uint64_t *addr = (uint64_t *)[g_bBatchArgs contents];
    for (int edge = 0; edge < n_edges; ++edge)
      addr[edge] = blob_gpu_addr_of(expert_blobs[edge], bufs[(size_t)edge],
                                    (size_t)layout.span);

    MoeDims D = dims_for(layout, (uint32_t)n_edges);
    id<MTLCommandBuffer> cb = [g_q commandBuffer];
    if (!cb) {
      seterr("flat position-batch command buffer", nil);
      return -5;
    }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (int edge = 0; edge < n_edges; ++edge)
      [enc useResource:bufs[(size_t)edge] usage:MTLResourceUsageRead];

    if (n_edges > 0) {
      D.nt = (uint32_t)n_edges * K3_I;
      [enc setComputePipelineState:pGluPosB];
      [enc setBuffer:g_bBatchArgs offset:0 atIndex:0];
      [enc setBuffer:g_bBatchX offset:0 atIndex:1];
      [enc setBuffer:g_bBatchH offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc setBuffer:g_bBatchEdgePosition offset:0 atIndex:4];
      [enc dispatchThreadgroups:
               MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];

      if (!fused_reduce) {
        D.nt = (uint32_t)n_edges * K3_H;
        [enc setComputePipelineState:pW2B];
        [enc setBuffer:g_bBatchArgs offset:0 atIndex:0];
        [enc setBuffer:g_bBatchH offset:0 atIndex:1];
        [enc setBuffer:g_bBatchY offset:0 atIndex:2];
        [enc setBytes:&D length:sizeof D atIndex:3];
        [enc dispatchThreadgroups:
                 MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      }
    }

    D.nt = (uint32_t)n_positions * K3_H;
    if (fused_reduce) {
      [enc setComputePipelineState:pW2RedPos];
      [enc setBuffer:g_bBatchArgs offset:0 atIndex:0];
      [enc setBuffer:g_bBatchH offset:0 atIndex:1];
      [enc setBuffer:g_bBatchW offset:0 atIndex:2];
      [enc setBuffer:g_bBatchOut offset:0 atIndex:3];
      [enc setBytes:&D length:sizeof D atIndex:4];
      [enc setBuffer:g_bBatchOffsets offset:0 atIndex:5];
      [enc dispatchThreadgroups:
               MTLSizeMake(D.nt / 4, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(FUSED_TG_THREADS, 1, 1)];
    } else {
      [enc setComputePipelineState:g_pRedPos];
      [enc setBuffer:g_bBatchY offset:0 atIndex:0];
      [enc setBuffer:g_bBatchW offset:0 atIndex:1];
      [enc setBuffer:g_bBatchOut offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc setBuffer:g_bBatchOffsets offset:0 atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + 255) / 256, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];

    deltafin::provider_internal::loop_gpu_timeline_note_cb(cb, "moe-flat");
    [cb commit];
    k3_prep_timed_wait(cb);
    if ([cb status] == MTLCommandBufferStatusError) {
      seterr("flat position-batch command buffer", [cb error]);
      return -5;
    }
    memcpy(out, [g_bBatchOut contents], (size_t)n_positions * K3_H * 4);
    g_calls += n_positions;
  }
  return 0;
}

int k3_metal_moe_positions_flat(const uint8_t *const *expert_blobs, int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  LayoutSpec layout = raw_layout();
  return k3_metal_moe_positions_flat_impl_locked(
      expert_blobs, n_edges, position_offsets, n_positions,
      weights, x, out, false, layout);
}

int k3_metal_moe_positions_flat_fused(
                                const uint8_t *const *expert_blobs, int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  LayoutSpec layout = raw_layout();
  return k3_metal_moe_positions_flat_impl_locked(
      expert_blobs, n_edges, position_offsets, n_positions,
      weights, x, out, true, layout);
}

static int k3_metal_moe_positions_flat_desc_v1_locked(
                                const K3MetalExpertDescriptorV1 *experts,
                                int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out, bool fused_reduce) {
  std::vector<const uint8_t *> blobs;
  LayoutSpec layout {};
  if (!descriptors_locked(experts, n_edges, &blobs, &layout)) return -7;
  return k3_metal_moe_positions_flat_impl_locked(
      blobs.empty() ? nullptr : blobs.data(), n_edges,
      position_offsets, n_positions, weights, x, out, fused_reduce, layout);
}


// ---------------------------------------------------------------------------
// K3_MOE_ATEN_STREAM=1: encode the flat position batch on a caller-provided
// compute encoder (PyTorch's MPS stream) with x and out in caller buffers.
// No commit and no wait here: the caller orders consumption on its stream
// and synchronizes once per tile. A ring of scratch sets keeps consecutive
// waves from overwriting buffers still in flight.
struct K3StreamScratch {
  id<MTLBuffer> H = nil, Y = nil, W = nil, Args = nil, EdgePos = nil, Offsets = nil;
  int edge_cap = 0, position_cap = 0;
};
static K3StreamScratch g_stream_ring[12];
static unsigned g_stream_ring_next = 0;
static bool ensure_stream_scratch(K3StreamScratch &s, int edges, int positions) {
  if (edges <= s.edge_cap && positions <= s.position_cap) return true;
  const int ec = std::max(std::max(edges, s.edge_cap), 16);
  const int pc = std::max(std::max(positions, s.position_cap), 16);
  id<MTLBuffer> H = [g_dev newBufferWithLength:(size_t)ec * K3_I * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> Y = [g_dev newBufferWithLength:(size_t)ec * K3_H * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> W = [g_dev newBufferWithLength:(size_t)ec * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> A = [g_dev newBufferWithLength:(size_t)ec * 8 options:MTLResourceStorageModeShared];
  id<MTLBuffer> E = [g_dev newBufferWithLength:(size_t)ec * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> O = [g_dev newBufferWithLength:(size_t)(pc + 1) * 4 options:MTLResourceStorageModeShared];
  if (!H || !Y || !W || !A || !E || !O) { seterr("stream scratch alloc", nil); return false; }
  s.H = H; s.Y = Y; s.W = W; s.Args = A; s.EdgePos = E; s.Offsets = O;
  s.edge_cap = ec; s.position_cap = pc;
  return true;
}
extern "C" int k3_metal_moe_encode_positions_flat_stream_v1(
    void *encoder_handle, void *device_handle,
    const uint8_t *const *expert_blobs, int n_edges,
    const int *position_offsets, int n_positions, const float *weights,
    void *x_handle, size_t x_off, void *out_handle, size_t out_off) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder_handle;
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
  id<MTLBuffer> xb = (__bridge id<MTLBuffer>)x_handle;
  id<MTLBuffer> ob = (__bridge id<MTLBuffer>)out_handle;
  if (!enc || !device || !xb || !ob || !position_offsets) return -2;
  if (device != g_dev) { seterr("caller device differs from the MoE Metal device", nil); return -6; }
  if (n_edges < 0 || n_positions <= 0 || n_positions > 16 || n_edges > n_positions * 16) return -2;
  if (position_offsets[0] != 0) return -2;
  for (int p = 0; p < n_positions; ++p) {
    if (position_offsets[p] < 0 || position_offsets[p] > position_offsets[p + 1] ||
        position_offsets[p + 1] > n_edges) return -2;
  }
  if (position_offsets[n_positions] != n_edges) return -2;
  if (n_edges > 0 && (!expert_blobs || !weights)) return -2;
  LayoutSpec layout = raw_layout();
  id<MTLComputePipelineState> pGluPosB = layout.scale4 ? g_pGluPosBScale4 : g_pGluPosB;
  id<MTLComputePipelineState> pW2B = layout.scale4 ? g_pW2BScale4 : g_pW2B;
  if (g_bindless != 1 || !pGluPosB || !pW2B || !g_pRedPos) return -6;
  if ((uint64_t)n_edges * K3_I > UINT32_MAX || (uint64_t)n_edges * K3_H > UINT32_MAX ||
      (uint64_t)n_positions * K3_H > UINT32_MAX) return -2;
  if ((size_t)x_off + (size_t)n_positions * K3_H * 4 > [xb length] ||
      (size_t)out_off + (size_t)n_positions * K3_H * 4 > [ob length]) return -2;
  K3StreamScratch &s = g_stream_ring[g_stream_ring_next++ % 12u];
  if (!ensure_stream_scratch(s, n_edges, n_positions)) return -3;
  @autoreleasepool {
    std::vector<id<MTLBuffer>> bufs((size_t)n_edges);
    std::vector<uint32_t> edge_position((size_t)n_edges);
    std::vector<uint32_t> offsets((size_t)n_positions + 1);
    for (int p = 0; p <= n_positions; ++p) offsets[(size_t)p] = (uint32_t)position_offsets[p];
    for (int p = 0; p < n_positions; ++p)
      for (int edge = position_offsets[p]; edge < position_offsets[p + 1]; ++edge)
        edge_position[(size_t)edge] = (uint32_t)p;
    for (int edge = 0; edge < n_edges; ++edge) {
      if (!expert_blobs[edge]) return -2;
      bufs[(size_t)edge] = wrap_blob(expert_blobs[edge], edge, layout);
      if (!bufs[(size_t)edge]) return -4;
    }
    if (n_edges > 0) {
      memcpy([s.W contents], weights, (size_t)n_edges * 4);
      memcpy([s.EdgePos contents], edge_position.data(), (size_t)n_edges * 4);
    }
    memcpy([s.Offsets contents], offsets.data(), ((size_t)n_positions + 1) * 4);
    uint64_t *addr = (uint64_t *)[s.Args contents];
    for (int edge = 0; edge < n_edges; ++edge)
      addr[edge] = blob_gpu_addr_of(expert_blobs[edge], bufs[(size_t)edge], (size_t)layout.span);
    MoeDims D = dims_for(layout, (uint32_t)n_edges);
    for (int edge = 0; edge < n_edges; ++edge)
      [enc useResource:bufs[(size_t)edge] usage:MTLResourceUsageRead];
    if (n_edges > 0) {
      D.nt = (uint32_t)n_edges * K3_I;
      [enc setComputePipelineState:pGluPosB];
      [enc setBuffer:s.Args offset:0 atIndex:0];
      [enc setBuffer:xb offset:(NSUInteger)x_off atIndex:1];
      [enc setBuffer:s.H offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc setBuffer:s.EdgePos offset:0 atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
      D.nt = (uint32_t)n_edges * K3_H;
      [enc setComputePipelineState:pW2B];
      [enc setBuffer:s.Args offset:0 atIndex:0];
      [enc setBuffer:s.H offset:0 atIndex:1];
      [enc setBuffer:s.Y offset:0 atIndex:2];
      [enc setBytes:&D length:sizeof D atIndex:3];
      [enc dispatchThreadgroups:MTLSizeMake((D.nt + ROWS_PER_TG - 1) / ROWS_PER_TG, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(TG_THREADS, 1, 1)];
    }
    D.nt = (uint32_t)n_positions * K3_H;
    [enc setComputePipelineState:g_pRedPos];
    [enc setBuffer:s.Y offset:0 atIndex:0];
    [enc setBuffer:s.W offset:0 atIndex:1];
    [enc setBuffer:ob offset:(NSUInteger)out_off atIndex:2];
    [enc setBytes:&D length:sizeof D atIndex:3];
    [enc setBuffer:s.Offsets offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((D.nt + 255) / 256, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    g_calls += n_positions;
  }
  return 0;
}

int k3_metal_moe_positions_flat_desc_v1(
                                const K3MetalExpertDescriptorV1 *experts,
                                int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  return k3_metal_moe_positions_flat_desc_v1_locked(
      experts, n_edges, position_offsets, n_positions,
      weights, x, out, false);
}

int k3_metal_moe_positions_flat_fused_desc_v1(
                                const K3MetalExpertDescriptorV1 *experts,
                                int n_edges,
                                const int *position_offsets, int n_positions,
                                const float *weights, const float *x,
                                float *out) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!init_locked(nullptr)) return -1;
  return k3_metal_moe_positions_flat_desc_v1_locked(
      experts, n_edges, position_offsets, n_positions,
      weights, x, out, true);
}

}  // extern "C"
