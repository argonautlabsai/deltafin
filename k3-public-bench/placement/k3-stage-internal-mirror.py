#!/usr/bin/env python3
"""Usage-weighted partial mirror on the internal SSD (2026-09-06).

Copies the most-used experts that the internal SSD does NOT hold yet from Green's
base copy into the primary set, up to a byte budget, so the fastest device can
serve them and the ETA router has a fast second home for the heaviest traffic.
Real copies (cross-volume), not clones. Writes a manifest and a rollback script.

Usage: python3 k3-stage-internal-mirror.py <budget_GB> [--dry]
"""
import json, os, shutil, sys, time

R = "$K3_DIR"
PRIMARY = f"{R}/deltafin-root-local/k3-experts"
SRC = "/Volumes/Green/deltafin-root-b/k3-experts"      # B half (the experts the internal lacks live here)
SRC_A = "/Volumes/Green/deltafin-root-a/k3-experts"
budget = float(sys.argv[1]) * 1e9; dry = "--dry" in sys.argv
have = set(os.listdir(PRIMARY))
use = json.load(open(f"{R}/k3-soak-logs/expert-usage-2026-09-03.json"))
# usage json: {"L<layer>-E<expert>.bin": reads} or {"<layer>-<expert>": reads}; normalise to file names
def fname(k):
    if k.endswith(".bin"): return k
    if "-" in k and not k.startswith("L"):
        l, e = k.split("-", 1); return f"L{l}-E{e}.bin"
    return k
ranked = sorted(((int(v), fname(k)) for k, v in use.items()), reverse=True)
missing = [(n, f) for n, f in ranked if f not in have]
chosen, total = [], 0
for n, f in missing:
    src = f"{SRC}/{f}" if os.path.exists(f"{SRC}/{f}") else (f"{SRC_A}/{f}" if os.path.exists(f"{SRC_A}/{f}") else None)
    if not src: continue
    size = os.path.getsize(src)
    if total + size > budget: break
    chosen.append((f, src, n)); total += size
traffic_all = sum(n for n, _ in ranked); traffic_missing = sum(n for n, _ in missing); traffic_chosen = sum(n for _, _, n in chosen)
print(f"internal holds {len(have)} files; {len(missing)} used experts are absent from it ({100*traffic_missing/traffic_all:.1f}% of trace reads); "
      f"copying {len(chosen)} files = {total/1e9:.1f} GB covering {100*traffic_chosen/traffic_all:.1f}% of trace reads ({100*traffic_chosen/max(1,traffic_missing):.1f}% of the absent traffic)")
snap = f"{R}/k3-layout-snapshots/2026-09-06-internal-mirror"; os.makedirs(snap, exist_ok=True)
open(f"{snap}/manifest.txt", "w").write("".join(f"{f}\t{n}\n" for f, _, n in chosen))
open(f"{snap}/ROLLBACK.sh", "w").write("#!/bin/sh\n# removes the copies this staging added to the primary set (they exist on Green/White too)\n" +
                                       "".join(f"rm -f {PRIMARY}/{f}\n" for f, _, _ in chosen))
if dry: sys.exit(0)
t0 = time.time(); done = 0
for f, src, _ in chosen:
    dst = f"{PRIMARY}/{f}"
    if os.path.exists(dst): continue
    shutil.copyfile(src, dst + ".part"); os.rename(dst + ".part", dst); done += 1
    if done % 1000 == 0: print(f"  {done}/{len(chosen)} copied, {time.time()-t0:.0f} s", flush=True)
print(f"copied {done} files in {time.time()-t0:.0f} s; primary now {len(os.listdir(PRIMARY))} files; rollback: sh {snap}/ROLLBACK.sh")
