# Credits

## Upstream: gavamedia/deltafin (MIT)

Everything below this fork's changes is the work of the deltafin authors at
GAVAMEDIA, published at https://github.com/gavamedia/deltafin under the MIT
licence, whose notice is kept intact in `LICENSE`. In particular this fork
inherits, unchanged in kind:

- the Rust engine core and its native provider gate;
- the Metal MoE expert path (`tools/metal_moe.mm`) and the raw-v1 expert files;
- the KDA and MLA attention implementations;
- the int8 spine conversion (`deltafin convert-spine-int8`) and its explicit
  non-weight-exact status;
- PILOT speculative expert prefetch;
- the DSpark and Qwen drafters with exact-argmax acceptance;
- the F_NOCACHE positional read path for expert files;
- the OpenAI-compatible server and the audited chat template;
- the documentation under `docs/`.

The M5 Max reference figure this package compares against (0.684 tok/s on the
17-token prompt) was posted in upstream issue #15 by a contributor whose
account has since been deleted, building on the profile in issue #13 by
@trueimage. Both are acknowledged here since they cannot be credited by name.

## This fork: ARGODRIVE Deltafin (Argonaut Labs)

Ninety-three commits on top of upstream `441fbfd`, squashed for publication.
Every item below corresponds to commits in that history (`git log
upstream/main..local-patches`); nothing is claimed that the list does not
support. All additions are environment-gated and default-off except where the
benchmark configuration (`k3-public-bench/env.sh`) turns them on.

**Storage layout and read path**
- multi-volume expert overlay: hot-tier placement (`K3_EXPERT_HOT_DIR`), a
  second canonical volume (`K3_EXPERT_DIR_B`) and a third (`K3_EXPERT_DIR_C`),
  resolved by role;
- the mirror layout of record and the usage-weighted staging that builds it;
- the plan-path balancer (`K3_PLAN_BALANCE`): the prefetch/whole-file read path
  previously bypassed every balancer;
- split-homed chunked reads (`K3_SPLIT_READ`) and the in-flight ETA router
  across holders (`K3_SPLIT_ETA`); mirror-striped reads;
- prefetch/read thread-pool separation (`K3_EXPERT_PREFETCH_THREADS`,
  `K3_EXPERT_PREFETCH_GENERATIONS`) and the tier-balance probe order.

**Compute and memory**
- the spine `MTLResidencySet` (`K3_RESIDENCY_SET`);
- arrival-driven expert compute (`K3_ARRIVAL_GROUPS`);
- the Summer pool (read-into-slot, static preload, and the wide-tile arena-base
  corruption fix in `metal_moe.mm`);
- the drafter survival pair (`K3_QWEN_ROUNDTRIP_SOFT`, `K3_QWEN_REARM`) with the
  acceptance-gated re-arm; the DFlash2 block-drafter port (experimental, not in
  the benchmark configuration).

**Instruments and method**
- the per-read provenance trace, the blocking-wait classifier, the read-path
  phase split, the GPU timeline and the attention/kernel split timers;
- the measurement campaign and its harness, published separately as ARGODRIVE:
  https://github.com/argonautlabsai/argodrive.

A number of mechanisms were built, measured and left default-off because they
did not pay on this machine (expert retention, RAM pools above 8 GB, the mirror
scheduler, route oracles, deeper draft trees). They remain in the tree with
their measurements in the commit messages.

Correction to an earlier internal draft of this file: split-homed chunked
reads were first added in this fork (commit 0e59d45, 2026-08-25), not upstream.
