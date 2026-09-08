#!/usr/bin/env python3
"""Widen the White (hot) and Yellow (dir_c) replica bands on the space freed on
2026-09-06: copy the next-most-used experts each band lacks from Green's full set,
up to a byte budget per drive, so the ETA router has more holders per file.
Real copies (cross-volume). Writes a manifest + rollback script per band.
Usage: python3 k3-stage-wider-bands.py <white_GB> <yellow_GB> [--dry]
"""
import json, os, shutil, sys, time
R = "$K3_DIR"
SRC = "/Volumes/Green/k3-experts-full"
BANDS = {"White": (f"/Volumes/White/k3-experts-b", float(sys.argv[1]) * 1e9),
         "Yellow": (f"/Volumes/Yellow/serve3-green", float(sys.argv[2]) * 1e9)}
dry = "--dry" in sys.argv
use = json.load(open(f"{R}/k3-soak-logs/expert-usage-2026-09-03.json"))
def fname(k):
    if k.endswith(".bin"): return k
    if "-" in k and not k.startswith("L"):
        l, e = k.split("-", 1); return f"L{l}-E{e}.bin"
    return k
ranked = sorted(((int(v), fname(k)) for k, v in use.items()), reverse=True)
traffic_all = sum(n for n, _ in ranked)
stamp = time.strftime("%Y-%m-%d-%H%M")
for name, (band, budget) in BANDS.items():
    have = set(os.listdir(band))
    free = shutil.disk_usage(band).free
    if budget > free - 20e9:
        print(f"{name}: budget {budget/1e9:.0f} GB exceeds free-20 GB ({free/1e9:.0f} GB); clamping"); budget = free - 20e9
    chosen, total = [], 0
    for n, f in ranked:
        if f in have: continue
        src = f"{SRC}/{f}"
        if not os.path.exists(src): continue
        size = os.path.getsize(src)
        if total + size > budget: break
        chosen.append((f, size, n)); total += size
    traffic_have = sum(n for n, f in ranked if f in have)
    traffic_chosen = sum(n for _, _, n in chosen)
    print(f"{name}: band {len(have)} files carries {100*traffic_have/traffic_all:.1f}% of recorded reads; "
          f"+{len(chosen)} files / {total/1e9:.0f} GB adds {100*traffic_chosen/traffic_all:.1f}% -> {100*(traffic_have+traffic_chosen)/traffic_all:.1f}%")
    if dry: continue
    man = f"{R}/k3-layout-snapshots/{stamp}-wider-{name}.json"; os.makedirs(os.path.dirname(man), exist_ok=True)
    json.dump({"band": band, "files": [f for f, _, _ in chosen]}, open(man, "w"))
    with open(f"{R}/k3-layout-snapshots/{stamp}-wider-{name}-rollback.sh", "w") as rb:
        rb.write("#!/bin/sh\n" + "".join(f"rm -f '{band}/{f}'\n" for f, _, _ in chosen))
    t0 = time.time(); done = 0
    for f, size, _ in chosen:
        tmp = f"{band}/.{f}.part"
        shutil.copyfile(f"{SRC}/{f}", tmp); os.rename(tmp, f"{band}/{f}"); done += size
    dt = time.time() - t0
    print(f"{name}: copied {len(chosen)} files, {done/1e9:.0f} GB in {dt:.0f} s ({done/1e9/max(dt,1):.2f} GB/s); manifest {man}")
