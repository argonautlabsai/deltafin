# Prefill — where the six minutes go (2026-09-08)

The 512-token prompt takes about 6.3 minutes to its first token on the layout
of record. This page classifies that time from the engine's own phase timers
and the per-device read counters, both recorded on the standard-length arms
(`RESULTS-2026-09-08-standard-lengths.md`).

## Where the time goes

Arm `PP512_OFF_A`, four drives, first token at 373.1 s:

| phase (engine timers, first chunk) | seconds | share |
|---|---:|---:|
| `[arrival] uncovered_wait` — waiting for expert reads to land | 282.8 | 76% |
| expert kernels (`[arrival] dispatch` 60.8 / `[kernel-sub] moe` 58.4) | ~60 | 16% |
| attention (`attention_resident`) | 20.2 | 5% |
| bind, plan, other | ~9 | 2% |

The drafter setting is irrelevant: `PP512_ON_A` differs by one second.

## How much is read

| quantity | value |
|---|---:|
| device bytes read during the prefill window, all four drives (arm csv) | **8,977 GB at 24.1 GB/s aggregate** |
| engine `[opens] requested_bytes` for the chunk | 9,033.8 GB (agrees) |
| the whole expert set, read once | 1,443 GB |
| **amplification** | **6.2×** |

The control: the 6-token prompt requests 169 GB — one pass over what it needs,
no amplification.

## Why: row passes re-read the layer

`layer_passes=744` on the 512-token prompt = 8 × 93: the 512 rows are processed
as **eight row-passes of 64**, and each pass re-reads the layer's expert union
in tiles of at most 64 experts. An expert routed to rows in k of the eight
passes is read k times; the observed average is 6.2. The drives were at their
combined ceiling for the whole prefill (24 GB/s), so the time is the reads.

## Cross-check from the drive ladder

First-token time on the same prompt scales with storage bandwidth exactly as
nine terabytes of reads must: 799 s on one drive (13.6 GB/s; `uncovered_wait`
691 s, 86%), 545 s on two (20.7 GB/s), 375 s on four. See `SCALING.md`.

## Earlier classification

An earlier measurement on this engine (2026-08-26, ~119-row prompt) found
prefill attention-compute-bound: attention then cost about 140 s and reads
completed behind it. That was true for that engine. Attention now costs 20 s at
512 rows, the read path delivers 24 GB/s, and the balance has flipped. The
earlier verdict is superseded, not contradicted.

## The lever, and its gate

Process the prompt expert-major per layer: gather every row's routes, group
rows by expert, read each expert once, run its kernel over its rows. Reads fall
from 9.0 TB toward 1.44 TB (the per-layer union for 512 rows is smaller still);
at 24 GB/s that is about a minute of reads instead of 283 s of waiting, for a
first token of roughly two to two and a half minutes before any overlap of reads
with kernels.

The gate for that change has two parts, and both must move: `[opens]
requested_bytes` toward one pass of the expert set, **and** `layer_passes`
falling from 744 to 93. Bytes falling while passes stay at 744 would mean the
experts were cached, not the schedule restructured. Output must remain
token-identical on the 6-token and 512-token prompts, and decode must stay
within the settled repeat spread of the standard-length set.

Status: classified and planned; not built.
