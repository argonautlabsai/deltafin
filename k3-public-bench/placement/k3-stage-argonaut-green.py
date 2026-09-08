#!/usr/bin/env python3
"""GREEN rebalance (2026-09-05): usage-weighted Argonaut placement on 4 drives with Green (SN8100, direct) as K3B and Yellow (SN7100, behind the hub) as K3A. Targets are TRAFFIC shares.
Writes serve3-green (K3A) / serve-b-green (K3B) and re-applies the argu4 stall clones; profile 4h-green.
Chain: dir_c = K3A/serve3-argu (first) -> hot = K3B/serve-b-argu -> dir_b = K3C b-set (+ copies) -> primary = internal.
Usage: k3-stage-argonaut-usage.py <internal%> <K3A%> <K3C%> <K3B%> [--dry] [--k3c-copies N]"""
import json, os, subprocess, sys, time
R = "$K3_DIR"
tI, tA, tC, tB = [float(x) for x in sys.argv[1:5]]; dry = "--dry" in sys.argv
copies_max = int(sys.argv[sys.argv.index("--k3c-copies") + 1]) if "--k3c-copies" in sys.argv else 5000
L = lambda d: set(f for f in os.listdir(d) if f.endswith(".bin"))
prim = L(f"{R}/deltafin-root-local/k3-experts"); k3a = L("/Volumes/Yellow/deltafin-root/k3-experts"); k3b = L("/Volumes/Green/deltafin-root-b/k3-experts"); k3c = L("/Volumes/White/k3-experts-b")
use = json.load(open(f"{R}/k3-soak-logs/expert-usage-2026-09-03.json"))
ALL = k3a | k3b; tot = sum(use.values())
target = {"internal": tI, "K3A": tA, "K3C": tC, "K3B": tB}; got = {k: 0.0 for k in target}
assign = {}; k3c_copies = []
for f, c in sorted(use.items(), key=lambda kv: -kv[1]):
    if f not in ALL: continue
    w = 100 * c / tot
    cands = []
    if f in k3a: cands.append("K3A")
    if f in k3b: cands.append("K3B")
    if f in k3c: cands.append("K3C")
    elif len(k3c_copies) < copies_max: cands.append("K3C-copy")
    if f in prim and f not in k3c: cands.append("internal")     # K3C (dir_b) is probed before primary: a K3C-b file cannot be internal-served
    def deficit(d): return target[d.replace("-copy", "")] - got[d.replace("-copy", "")]
    best = max(cands, key=deficit)
    if best == "K3C-copy": k3c_copies.append(f); best = "K3C"
    assign[f] = best; got[best] += w
# files never seen in traces: leave where they are (internal if present, else K3C-b serves them)
print("traffic shares: " + ", ".join(f"{k} {got[k]:.1f}% (target {target[k]:.0f})" for k in ("internal", "K3A", "K3C", "K3B")))
subA = [f for f, d in assign.items() if d == "K3A"]; subB = [f for f, d in assign.items() if d == "K3B"]
print(f"subsets: K3A {len(subA)} files, K3B {len(subB)} files, K3C extra copies {len(k3c_copies)} ({len(k3c_copies)*17.5/1000:.0f} GB), K3C-b native {sum(1 for f,d in assign.items() if d=='K3C' and f in k3c)}")
if dry: sys.exit(0)
snap = f"{R}/k3-layout-snapshots/2026-09-05-green"; os.makedirs(snap, exist_ok=True)
for name, base, dest, chosen in (("A", "/Volumes/Yellow/deltafin-root/k3-experts", "/Volumes/Yellow/serve3-green", sorted(subA)), ("B", "/Volumes/Green/deltafin-root-b/k3-experts", "/Volumes/Green/serve-b-green", sorted(subB))):
    os.makedirs(dest, exist_ok=True); have = L(dest); todo = [f for f in chosen if f not in have]; t0 = time.time()
    for k in range(0, len(todo), 400): subprocess.run(["cp", "-c"] + [f"{base}/{f}" for f in todo[k:k+400]] + [dest], check=True)
    for f in have - set(chosen): os.remove(f"{dest}/{f}")
    assert len(L(dest)) == len(chosen); print(f"{name}: {dest} {len(chosen)} files ({time.time()-t0:.0f}s)")
    open(f"{snap}/{name}.list", "w").write("\n".join(chosen) + "\n")
# real copies onto K3C-b for files it lacks (from whichever base half has them)
t0 = time.time(); n = 0
for f in k3c_copies:
    src = f"/Volumes/Yellow/deltafin-root/k3-experts/{f}" if f in k3a else f"/Volumes/Green/deltafin-root-b/k3-experts/{f}"
    dst = f"/Volumes/White/k3-experts-b/{f}"
    if not os.path.exists(dst): subprocess.run(["cp", src, dst], check=True); n += 1
open(f"{snap}/K3C-copies.list", "w").write("\n".join(k3c_copies) + "\n")
open(f"{snap}/ROLLBACK.sh", "w").write("rm -rf /Volumes/Yellow/serve3-green /Volumes/Green/serve-b-green\n")
print(f"K3C: {n} files copied ({time.time()-t0:.0f}s); K3C-b now {len(L('/Volumes/White/k3-experts-b'))} files; staged")

# re-apply the argu4 stall clones (1,000 top-stall experts) onto the drive that holds each file's base half
stall = [l.split()[0] for l in open(f"{R}/k3-layout-snapshots/2026-09-04-argu4-stall/added.list") if l.strip()]
nA = nB = 0
for f in stall:
    if f in k3a and not os.path.exists(f"/Volumes/Yellow/serve3-green/{f}"): subprocess.run(["cp", "-c", f"/Volumes/Yellow/deltafin-root/k3-experts/{f}", "/Volumes/Yellow/serve3-green/"], check=True); nA += 1
    elif f in k3b and not os.path.exists(f"/Volumes/Green/serve-b-green/{f}"): subprocess.run(["cp", "-c", f"/Volumes/Green/deltafin-root-b/k3-experts/{f}", "/Volumes/Green/serve-b-green/"], check=True); nB += 1
print(f"stall clones re-applied: K3A +{nA}, K3B +{nB}; serve3-green {len(L('/Volumes/Yellow/serve3-green'))} files, serve-b-green {len(L('/Volumes/Green/serve-b-green'))} files")
pf = f"{R}/k3-summer-profiles.txt"; t = open(pf).read()
line = f"4h-green|DELTAFIN_ROOT={R}/deltafin-root-local K3_EXPERT_DIR_B=/Volumes/White/k3-experts-b K3_EXPERT_HOT_DIR=/Volumes/Green/serve-b-green K3_EXPERT_DIR_C=/Volumes/Yellow/serve3-green\n"
if "4h-green|" not in t: open(pf, "a").write(line); print("profile 4h-green added")
