#!/bin/sh
# Reproducible speed benchmark for Kimi K3 on deltafin (Apple silicon, SSD-streamed experts).
# Runs the public 17-token "France" prompt 3x and the 200-token prompt of record 2x, cold (no resident engine),
# checks text identity against the texts of record, and writes results/RESULTS.md.
# Usage: sh run-bench.sh [path/to/deltafin]   (defaults to ../engines/deltafin/target/release/deltafin)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); DF=${1:-$HERE/../engines/deltafin/target/release/deltafin}
. "$HERE/env.sh"
[ -d "$DELTAFIN_ROOT/k3-experts" ] || { echo "model root not found: DELTAFIN_ROOT=$DELTAFIN_ROOT (export DELTAFIN_ROOT or edit env.sh)"; exit 1; }
OUT=$HERE/results; mkdir -p "$OUT"; TS=$(date '+%Y-%m-%d %H:%M')
md5_of() { python3 - "$1" <<'PY'
import hashlib, re, sys
parts=[]; speed=""; gen=""
for line in open(sys.argv[1], errors="ignore"):
    if line.startswith("[stats]"):
        f=dict(kv.split("=",1) for kv in line.split()[1:] if "=" in kv); speed=f.get("speed",speed); gen=f.get("generated",gen)
    elif line.startswith("WALL_SECONDS="): continue
    elif not line.startswith("[") and not line.startswith("deltafin:") and line.strip(): parts.append(line)
t="".join(parts).replace("\n",""); t=re.sub(r"^ARM [^]]*\]","",t)
print(hashlib.md5(t.encode()).hexdigest()[:12], gen or "-", speed or "-")
PY
}
run() { tag=$1; tokens=$2; prompt=$3
  pgrep -x deltafin >/dev/null && { echo "a deltafin process is already running; benchmarks must run cold"; exit 1; }
  t0=$(python3 -c "import time; print(time.time())")
  caffeinate -is "$DF" run --model-root "$DELTAFIN_ROOT" --stats --max-new "$tokens" --prompt "$prompt" > "$OUT/$tag.log" 2>&1
  rc=$?
  t1=$(python3 -c "import time; print(time.time())"); python3 -c "print('WALL_SECONDS=%.3f' % ($t1 - $t0))" >> "$OUT/$tag.log"
  # Fail closed: an engine error (missing expert volumes, census incomplete) must
  # stop the benchmark, not be recorded as an empty row.
  if [ "$rc" -ne 0 ] || ! grep -q '^\[stats\] generated=' "$OUT/$tag.log"; then
    echo "$tag: engine run failed (exit $rc); last lines of $OUT/$tag.log:"; grep -vE '^\s*$' "$OUT/$tag.log" | tail -5
    echo "no result recorded; check DELTAFIN_ROOT and the expert volumes named in env.sh"; exit 1
  fi
  set -- $(md5_of "$OUT/$tag.log"); echo "$tag md5=$1 generated=$2 speed=$3 wall=$(python3 -c "print('%.1fs' % ($t1 - $t0))")"
}
echo "deltafin speed benchmark  $TS  $(sysctl -n machdep.cpu.brand_string)  $(sw_vers -productVersion)"
for i in 1 2 3; do run FRANCE_$i 17 "The capital of France is"; done
for i in 1 2;   do run RECORD_$i 200 "The three main financial statements are"; done
python3 - "$OUT" "$TS" <<'PY'
import sys, os, statistics, hashlib, re
out, ts = sys.argv[1], sys.argv[2]
def parse(tag):
    parts=[]; speed=""; gen=""
    for line in open(os.path.join(out, tag + ".log"), errors="ignore"):
        if line.startswith("[stats]"):
            f=dict(kv.split("=",1) for kv in line.split()[1:] if "=" in kv); speed=f.get("speed",speed); gen=f.get("generated",gen)
        elif line.startswith("WALL_SECONDS="): continue
        elif not line.startswith("[") and not line.startswith("deltafin:") and line.strip(): parts.append(line)
    t="".join(parts).replace("\n",""); t=re.sub(r"^ARM [^]]*\]","",t)
    wall = 0.0
    for line in open(os.path.join(out, tag + ".log"), errors="ignore"):
        if line.startswith("WALL_SECONDS="): wall = float(line.split("=")[1])
    return float(speed or 0), gen, hashlib.md5(t.encode()).hexdigest()[:12], wall
ref = {"FRANCE": "777f5ef1d9f3", "RECORD": "6d8c4f50a22c"}
rows = []; fr = []; rec = []
for tag in ["FRANCE_1","FRANCE_2","FRANCE_3","RECORD_1","RECORD_2"]:
    sp, gen, md5, wall = parse(tag); ok = "identical" if md5 == ref[tag.split("_")[0]] else "DIFFERS"
    wall_tps = (int(gen) / wall) if (wall and gen) else 0.0
    rows.append(f"| {tag} | {gen} | {sp:.4f} | {wall_tps:.3f} ({wall:.0f} s) | {md5} | {ok} |"); (fr if tag.startswith("FRANCE") else rec).append(sp)
md = f"""# Results — {ts}

| arm | tokens | steady tok/s ([stats] speed) | fresh-process wall tok/s (wall) | text md5 | vs text of record |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

- France (17 tokens, 3 runs): median **{statistics.median(fr):.4f}** tok/s, best {max(fr):.4f}.
- Prompt of record (200 tokens, 2 runs): {rec[0]:.4f} / {rec[1]:.4f} tok/s.
- Texts of record: results/text-of-record-17.txt (md5 777f5ef1d9f3), results/text-of-record-200.txt (md5 6d8c4f50a22c).
"""
open(os.path.join(out, "RESULTS.md"), "w").write(md); print(md)
PY
