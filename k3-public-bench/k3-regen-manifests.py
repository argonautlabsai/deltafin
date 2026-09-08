#!/usr/bin/env python3
"""Regenerate the public placement manifests from the live directories.

The package's manifests must describe the layout the headline was measured on,
and the only authority for that is the disk. This lists every `L<layer>-E<expert>.bin`
in each role directory, writes it sorted (one name per line, gzipped) to
k3-public-bench/placement/manifest-<role>.txt.gz, then re-reads each manifest
and checks it against the directory it came from. Exit 1 on any mismatch.

Roles (from env.sh):
  internal  DELTAFIN_ROOT/k3-experts          primary
  green     /Volumes/Green/k3-experts-full    K3_EXPERT_DIR_B  (census: primary ∪ dir_b must be complete)
  white     /Volumes/White/k3-experts-b       K3_EXPERT_HOT_DIR
  yellow    /Volumes/Yellow/serve3-green      K3_EXPERT_DIR_C
"""
import gzip
import os
import sys

ROOT = "$K3_DIR"
OUT = f"{ROOT}/k3-public-bench/placement"
ROLES = {
    "internal": f"{ROOT}/deltafin-root-local/k3-experts",
    "green": "/Volumes/Green/k3-experts-full",
    "white": "/Volumes/White/k3-experts-b",
    "yellow": "/Volumes/Yellow/serve3-green",
}
FULL = 82432


def listing(path):
    return sorted(f for f in os.listdir(path) if f.endswith(".bin"))


def main():
    sets = {}
    for role, path in ROLES.items():
        if not os.path.isdir(path):
            print(f"{role}: {path} is not a directory — refusing to write a manifest", file=sys.stderr)
            return 1
        names = listing(path)
        target = f"{OUT}/manifest-{role}.txt.gz"
        with gzip.open(target, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        with gzip.open(target, "rt", encoding="utf-8") as fh:
            back = [l.strip() for l in fh if l.strip()]
        if back != names:
            print(f"{role}: manifest re-read does not match the directory", file=sys.stderr)
            return 1
        sets[role] = set(names)
        print(f"{role:<9} {len(names):>6} files  -> manifest-{role}.txt.gz  (verified against {path})")

    union = sets["internal"] | sets["green"]
    print(f"census: primary ∪ dir_b = {len(union)} (full set {FULL}) — {'COMPLETE' if len(union) == FULL else '*** INCOMPLETE ***'}")
    for role in ("white", "yellow"):
        extra = sets[role] - sets["green"]
        print(f"{role}: files not in the full set: {len(extra)} {'(ok)' if not extra else '*** unexpected ***'}")
    return 0 if len(union) == FULL else 1


if __name__ == "__main__":
    sys.exit(main())
