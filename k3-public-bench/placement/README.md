# Placement inputs and manifests (layout of record, 2026-09-08)

The manifests here are the layout the headline was measured on. They were
regenerated from the live directories on 2026-09-08 (`k3-regen-manifests.py`,
in the package root) and verified against those directories after writing; the
previous manifests in this directory described an older staging generation and
were wrong (see "History").

## Roles and manifests

| role | env var | volume | manifest | files |
|---|---|---|---|---:|
| primary | `DELTAFIN_ROOT/k3-experts` | internal SSD | `manifest-internal.txt.gz` | 69,776 |
| second base (census) | `K3_EXPERT_DIR_B` | Green, WD_BLACK SN8100 2 TB, direct port | `manifest-green.txt.gz` | 82,432 — every expert |
| hot replica | `K3_EXPERT_HOT_DIR` | White, WD_BLACK SN8100 1 TB, direct port | `manifest-white.txt.gz` | 50,265 |
| replica on the slowest drive | `K3_EXPERT_DIR_C` | Yellow, WD_BLACK SN7100 1 TB, behind the TB5 hub | `manifest-yellow.txt.gz` | 26,684 |

Manifests are sorted file names, one per line (`L<layer>-E<expert>.bin`).
Primary ∪ second base is the complete set of 82,432 experts; the engine's
census requires that and will otherwise download during a run. The hot replica
and the Yellow band are strict subsets of the full set.

Resolve chain, first hit wins: Yellow band → hot replica → second base →
primary. Reads are then split across holders by the in-flight ETA router
(`K3_SPLIT_READ=2 K3_SPLIT_ETA=1`, rates in `env.sh`), so a file with several
homes is served by whichever is expected to finish first.

## What a reader needs

- Green must hold every expert (a 2 TB enclosure is a recipe requirement).
- The internal SSD holds the primary set listed in `manifest-internal.txt.gz`.
- White and Yellow hold the listed subsets. Copy the listed files from Green;
  on this machine they are real cross-volume copies, not clones.
- Per-drive share of reads on the prompt of record at 200 tokens, measured
  2026-09-08: internal 43.7%, White 21.3%, Green 20.4%, Yellow 14.5%.

## Provenance

The bands were chosen by traffic share from the usage trace
`expert-usage-2026-09-03.json` (per-(layer, expert) read counts from 19 router
traces of 200-token completions), not by file count.

| step | date | script | effect |
|---|---|---|---|
| Yellow band and White hot set, usage-weighted | 2026-09-05 | `k3-stage-argonaut-green.py` | first bands on the colour-named drives |
| most-used experts copied into the primary set | 2026-09-06 | `k3-stage-internal-mirror.py` | internal 56,669 → 69,776 |
| both bands widened into freed space | 2026-09-06 | `k3-stage-wider-bands.py` (+`wider-White-2026-09-06.json`, `wider-Yellow-2026-09-06.json`) | White → 50,265, Yellow → 26,684; +5.9% at 200 tokens |

The 2026-09-06 additions are listed in the two `wider-*.json` files so the
membership can be audited step by step: the previous manifests' 36,018 White
and 15,287 Yellow files plus the 14,247 and 11,397 additions listed there give
exactly the 50,265 and 26,684 in the current manifests. The manifests are the
authority; the scripts are history.

## History

Until 2026-09-08 this directory shipped manifests from the 2026-09-05
layout (internal 56,669 / White 36,018 / Green 18,017 / Yellow 15,287, with
different role assignments). The headline was never measured on that layout.
On 2026-09-08 Yellow was temporarily widened to 53,468 files (coverage of
recorded reads 66.5% → 96.1%) and its share of traffic moved by 0.2 points;
no benefit was demonstrated and the band was restored before the manifests were
regenerated. What the restoration verifies: the 26,784 files removed are exactly
the set the widening script recorded adding, every one of them is present in
Green's full set, and the resulting directory is set-identical to (widened set −
additions). What it does not verify: no independent record of the pre-widening
membership survives (the previous manifests were overwritten by the
regeneration and no snapshot holds them), so the claim that today's 26,684 files
are the same 26,684 the headline was measured on rests on the widening script
having only added files, which its code and manifest show, plus the file counts
reconciling with the 2026-09-06 additions. Restoring membership restores the
logical layout; it does not restore identical physical SSD placement or cache
state.
