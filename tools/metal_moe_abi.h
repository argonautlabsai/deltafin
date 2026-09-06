#ifndef K3_METAL_MOE_ABI_H
#define K3_METAL_MOE_ABI_H

#include <stddef.h>
#include <stdint.h>

/*
 * Versioned, additive expert-storage ABI shared by Deltafin's native provider
 * and the established Objective-C++ Metal bridge. Legacy raw-pointer entry
 * points remain unchanged. Compact storage is accepted only through these
 * descriptors, so no caller can accidentally reinterpret scale4 bytes as the
 * larger raw-v1 layout.
 */
enum {
  K3_LAYOUT_RAW_V1 = 1,
  K3_LAYOUT_SCALE4_V2 = 2,
  K3_DESCRIPTOR_ABI_V1 = 1,
};

#define K3_RAW_V1_EXPERT_SPAN UINT64_C(17547264)
#define K3_SCALE4_V2_EXPERT_SPAN UINT64_C(17039360)
#define K3_CAP_RAW_V1 (UINT64_C(1) << K3_LAYOUT_RAW_V1)
#define K3_CAP_SCALE4_V2 (UINT64_C(1) << K3_LAYOUT_SCALE4_V2)

/*
 * Source selector ABI, not a filesystem path. Production callers pass this
 * value to k3_metal_init() so the bridge loads the precompiled metallib
 * embedded into the native binary. Debug builds may admit another non-empty
 * string as an explicit development source path; NULL and the empty string
 * also select this embedded version.
 */
#define K3_METAL_MOE_EMBEDDED_SOURCE_V1 \
  "deltafin:embedded-metal-moe-mxfp4:v1"

typedef struct K3MetalExpertDescriptorV1 {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t layout_id;
  uint32_t reserved;
  uint64_t blob_bytes;
  const uint8_t* blob;
} K3MetalExpertDescriptorV1;

#ifdef __cplusplus
static_assert(sizeof(K3MetalExpertDescriptorV1) == 32,
              "Metal expert descriptor ABI drift");
static_assert(offsetof(K3MetalExpertDescriptorV1, blob_bytes) == 16,
              "Metal expert descriptor byte-count offset drift");
static_assert(offsetof(K3MetalExpertDescriptorV1, blob) == 24,
              "Metal expert descriptor pointer offset drift");
extern "C" {
#endif

uint32_t k3_metal_descriptor_abi_version(void);
uint64_t k3_metal_layout_capabilities(void);

int k3_metal_moe_layer_desc_v1(
    const K3MetalExpertDescriptorV1* experts, int expert_count,
    const float* weights, const float* input, float* output);

int k3_metal_moe_positions_desc_v1(
    const K3MetalExpertDescriptorV1* experts, int edge_count,
    const int* position_offsets, int position_count, const float* weights,
    const float* input, float* output);

#ifdef __cplusplus
}
#endif

#ifdef __OBJC__
#import <Metal/Metal.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Loop-queue CB_tail entry (K3-SIDEQUEUE-PLAN.md Step 5): encodes the T=1
 * raw-v1 expert stack onto a caller command buffer, binding the caller's
 * x/out buffers directly.  Same-device required (-6 otherwise); the caller
 * must host-wait its CB before releasing the expert byte lease. */
int k3_metal_moe_encode_t1_raw_v1(id<MTLCommandBuffer> cb,
                                  id<MTLDevice> device,
                                  const uint8_t* const* expert_blobs,
                                  int n_experts, const float* weights,
                                  id<MTLBuffer> x_buf, NSUInteger x_off,
                                  id<MTLBuffer> out_buf, NSUInteger out_off);
#ifdef __cplusplus
}
#endif
#endif

#endif
