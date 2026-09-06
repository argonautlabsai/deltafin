#include <metal_stdlib>
using namespace metal;

/*
 * Fused MLA decode attention core (T=1) over deltafin's production fp32
 * expanded KV slabs.  Ported from the validated k3-proto flash-decoding
 * prototype (kernels.metal mla_attn_part/mla_attn_combine, relL2 ~1e-7 vs an
 * fp64 reference); the only structural deltas are (a) fp32 KV reads instead
 * of fp16 dedup buffers — the 64 shared "rope" dims are read from each head's
 * own 192-wide key row — and (b) slab indexing by the cache capacity pitch
 * (`capacity`), never by the live length S: the provider narrows a
 * [1,96,capacity,192] slab to [1,96,S,192], so a token's key row lives at
 * ((head*capacity)+t)*192 and its value row at ((head*capacity)+t)*128.
 *
 * scores = (q . k) * 192^-0.5 over the full 192-dim rows, online softmax in
 * fp32, weighted V accumulation in fp32.  S is split into `partitions`
 * flash-decoding partitions; the combine kernel merges partials per head and
 * writes out[96,128] fp32 — exactly the pre-gate [1,1,12288] row (the
 * production transpose is a pure view at T=1).
 *
 * This source is compiled with -fno-fast-math: the fp32 accumulation, the
 * exp() calls, and the -INFINITY sentinels must keep IEEE semantics.
 */

#include "provider_mla_attn_core.metalh"
