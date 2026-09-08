# Champion environment of record (2026-09-06) for Kimi K3 on the deltafin engine, Apple M5 Max 128 GB.
# Edit the four paths for your machine; everything else is the promoted configuration, unchanged.
export DELTAFIN_ROOT="${DELTAFIN_ROOT:-$HOME/deltafin-root-local}"          # model root: int8 spine (via k3-resident-int8), drafter, primary expert set
export K3_EXPERT_DIR_B="${K3_EXPERT_DIR_B:-/Volumes/Green/k3-experts-full}"   # second base: a full copy of all 82,432 experts (a 2 TB enclosure is a recipe requirement)
export K3_EXPERT_DIR_C="${K3_EXPERT_DIR_C:-/Volumes/Yellow/serve3-green}"     # replica band on the slowest enclosure (26,684 files; placement/manifest-yellow.txt.gz)
export K3_EXPERT_HOT_DIR="${K3_EXPERT_HOT_DIR:-/Volumes/White/k3-experts-b}"  # hot replica (50,265 files; placement/manifest-white.txt.gz); primary = internal SSD with the most-used experts copied in (placement/manifest-internal.txt.gz)
export K3_EXPERT_PREFETCH_GENERATIONS=4   # layers of prefetch lookahead; the engine default is 2 — the measured configuration ran with 4 (the measurement harness), so the package must set it (omission found 2026-09-08 by diffing every K3_ export against the harness)
# expert reads
export K3_EXPERT_READ_THREADS=64 K3_EXPERT_PREFETCH_THREADS=8   # PROMOTED 2026-09-06: 12 → 8 prefetch threads = +2.0% at 200 tok (1.1260 vs 1.1044, uncovered wait 70.9 → 58.6 s), +2.2% at 60; the five-knob stack was +1.9% and this knob carries it
export K3_SPLIT_READ=2 K3_TIER_BALANCE=1
export K3_SPLIT_ETA=1 K3_SPLIT_ETA_GBPS=7.1,5.5,6.5,13.5   # in-flight router for chunked reads (rates: hot, dir_c, dir_b, primary GB/s)
export K3_PLAN_BALANCE=1        # the prefetch/whole-file path also picks the least-loaded holder (2026-09-06, +11%)
# drafter (Qwen3-0.6B, exact-argmax acceptance: output is byte-identical to non-speculative decoding)
export K3_UAG_DRAFT=on K3_DSPARK=off K3_NGRAM_DRAFT=off K3_SPEC_DEPTH=8 K3_QWEN_PREFIX_COMMIT=1
export K3_QWEN_ROUNDTRIP_SOFT=1 K3_QWEN_REARM=4 K3_QWEN_REARM_MIN_ACCEPT=500
# memory
export K3_SPINE_RESIDENT_GB=16 K3_PROVIDER_RESIDENT_LAYERS=93 K3_HOST_RESERVE_GB=8 K3_SPINE_LOAD_RESERVE_GB=8
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 K3_MEMORY_PATIENCE_SECONDS=300
# compute path
export K3_ROUTE_ASYNC=1 K3_KDA_LOOP=on K3_ROUTE_SIDEQUEUE=on K3_KDA_SHARED_BOUNDARY=1 K3_KDA_PRECOMMIT=1 K3_KDA_INT8_DIRECT=1
export K3_TRUE_ROUTE_TOPUP=1 K3_PILOT_EARLY=1 K3_CB_FUSION=1 K3_LOOP_CAT=1 K3_SPINE_FP32_ARENA=0
export K3_ARRIVAL_GROUPS=8      # arrival-driven expert compute (tile dispatched in waves as experts land)
export K3_RESIDENCY_SET=1       # pin the spine heaps in one MTLResidencySet on every command queue
