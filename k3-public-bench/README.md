# Kimi K3 on a MacBook Pro M5 Max — SSD-streamed MoE decode at 1.00 tok/s sustained

Reproducible speed benchmark for the deltafin engine running Kimi K3 (93 layers,
92 MoE layers × 896 experts, 16 routed per position) with the experts streamed from
SSD and a small exact drafter. Every arm's text is checked against a text of
record before a number counts. Token-identical between drafter on and off on a
given storage layout, checked on every promotion. Not claimed across storage
layouts: one near-tie flip was observed at token ~305 on a two-holder layout;
both continuations were plausible and 40 of 40 sampled expert files were
byte-identical across holders (see `results/SCALING.md`). On general prompts
the text can differ between execution paths because they use different
floating-point arithmetic (see "Limits").

## Headline (2026-09-08, standard lengths)

**1.00 tok/s steady decode over a 512-token completion; 1.13 tok/s over 128
tokens; 0.96 median on the 17-token benchmark from upstream issue #15 against
the 0.684 posted there. Time to first token on a 512-token prompt is ~6.3
minutes.**

Measured in raw completion mode (not the chat template), greedy, on the single
prompt of record, on the hardware and quantization described below. Twelve cold
arms, order partially balanced (forward then reversed) to reduce order effects,
no monitoring process running; median of two runs at every setting. Morning
repeats differed by less than 2% within each setting; a separate later set of
200-token tests after large staging writes showed a 3.6% spread between drafted
runs, cause not isolated — small differences on this rig need replication.

| test | drafter off | drafter on | drafter gain |
|---|---:|---:|---:|
| steady decode, 512 generated tokens | 0.9232 | **1.0015** | +8.5% |
| steady decode, 128 generated tokens | 0.9261 | **1.1252** | +21.5% |
| inclusive throughput, 512 generated tokens | 0.9063 | **0.9849** | +8.7% |
| inclusive throughput, 128 generated tokens | 0.8630 | **1.0377** | +20.2% |
| prompt processing, 512-token prompt | ≈1.4 tok/s | ≈1.4 tok/s | none, as expected |
| first token, 512-token prompt | ≈376 s | ≈375 s | negligible |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/drives-live-dark.svg">
  <img src="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/drives-live.svg" alt="Animated replay of the four drives' read throughput during the 200-token record arm">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/drive-draw-dark.svg">
  <img src="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/drive-draw.svg" alt="Per-drive draw under the engine vs standalone ceiling">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/read-timeline-dark.svg">
  <img src="https://raw.githubusercontent.com/argonautlabsai/deltafin/main/k3-public-bench/results/charts/read-timeline.svg" alt="Per-drive read throughput during one 200-token run, all four drives">
</picture>

**Steady** excludes the first-token phase: `(generated - 1) / (elapsed_final -
elapsed_first_token)`. **Inclusive** is `generated / elapsed` from process start
and therefore contains the cold model load. Both are reported because they
answer different questions and because engines differ in which they quote.
Prompt processing is derived by subtracting first-token latencies and is **not
yet classified** as compute-, storage- or serialization-bound.

Method, per-arm detail and definitions:
[`results/RESULTS-2026-09-08-standard-lengths.md`](results/RESULTS-2026-09-08-standard-lengths.md).
Reproduce with `k3-stdbench-0908.sh`; table via `k3-stdbench-table.py`.

## Historical results (200-token rung, 2026-09-06)

The project ledger was built at 200 generated tokens. Those numbers are kept for
continuity and are no longer the headline: 200 tokens overstates sustained
throughput relative to a 512-token answer, because the speculative advantage
decays as the answer lengthens.

| test | result | reference |
|---|---|---|
| public prompt "The capital of France is", 17 tokens, 3 runs (`run-bench.sh`, 2026-09-06) | steady median **0.9631** tok/s (0.8909 / 0.9631 / 0.9633); fresh-process wall 0.80–0.87 tok/s (20–21 s per run) | +41% vs the public 0.6840 |
| prompt of record "The three main financial statements are", 200 tokens, 2 runs | steady **1.1000 / 1.1211** tok/s; wall 1.087 / 1.105 tok/s (181–184 s per run) | text identical to the text of record |

Champion of record with the internal harness (interleaved same-session pair, 2026-09-06): **1.0586 / 1.0725** tok/s against 0.9645 / 0.9562 for the previous configuration in the same hour (+11%). The package numbers above are what `run-bench.sh` reproduces in a later, slower machine state; 200-token results move by several percent with machine state across a day, and 17-token results scatter by about 3% run to run, so compare medians and same-hour pairs.

DFlash2 (optional block drafter, `K3_DSPARK=on K3_DSPARK_MODEL=dflash2 K3_DSPARK_MAX_DRAFTS=4`, not part of the headline): on ten general prompts at 200 tokens it is +20–30% where the Qwen drafter dies and its own acceptance stays above ~60%, and a loss where Qwen is alive or its acceptance is under ~50% (median +12.6%). Its generated text can differ from the headline path (see "Limits"); the headline numbers above are Qwen-drafter numbers with token-identical text.

Update 2026-09-06: the package headline above was re-run cold on the final configuration (wider bands + prefetch threads 8, no monitoring process running): France median 0.9631, record 1.1000 / 1.1211, texts identical.

Update 2026-09-06: widening the two replica bands on the freed enclosure space (see "placement by role") gave **1.0515 / 1.0558** tok/s on the prompt of record against 0.9949 / 0.9976 twenty minutes earlier on the same layout without the copies (+5.9%, text identical). The package headline below is being re-run on this layout.

What the number is: the engine's `[stats] speed`, the steady decode rate over
the generated tokens (generated / decode elapsed, prefill excluded), which is
the same definition as the public figure. `run-bench.sh` also reports the
fresh-process wall throughput (tokens / wall seconds from process start,
including model load and prefill), which is the user-visible number and is
much lower on a 17-token completion; quote the two side by side.

Public reference: https://github.com/gavamedia/deltafin/issues/15

Instruments used for every measurement here (read monitor, barrier trace,
harness, drive tools): https://github.com/argonautlabsai/argodrive — deltafin-specific today.

## Hardware

- MacBook Pro, Apple M5 Max, 128 GB unified memory, macOS 26.6.2.
- Internal SSD plus three NVMe enclosures (OWC Express 1M2, Thunderbolt 5,
  80 Gb/s links): two WD_BLACK SN8100 (1 TB and 2 TB, each on its own port)
  and one WD_BLACK SN7100 1 TB behind an OWC Thunderbolt 5 hub (the machine
  has three ports; one carries the hub). Cold single-file read latency at
  queue depth 1 (17.5 MB expert file): SN8100 2.6 ms, SN7100 3.3 ms; the
  SN7100 measured the same latency behind the hub as on a direct port. The
  engine-level cost of the hub has not been measured (a four-drive versus
  three-drive test showed −10.5% for removing the SN7100 entirely; that is
  the value of a fourth spindle, not of the hub).
- Measured read draw on the 200-token record arm: internal 9.2 GB/s mean
  (13.5 peak), SN8100 1 TB 5.1 (7.1), SN8100 2 TB 3.8 (6.3), SN7100 3.2
  (5.5); 21.6 GB/s in total for 24.0 GB of expert bytes per generated token.
- The layout of record needs one enclosure holding the complete 82,432-file
  expert set (about 1.44 TB, so a 2 TB drive); the other drives hold subsets.
- Expert reads use positional reads (`pread`/`preadv`) with F_NOCACHE; nothing
  is served from the host page cache.

## Software

- deltafin engine: fork of https://github.com/gavamedia/deltafin at upstream base
  441fbfd; the published tree is one commit on top of it (branch
  `publish-2026-09-05`) carrying arrival-driven expert compute, split-homed
  expert reads with tier balancing, the verify-width admission fix, the spine
  residency set, the Metal MoE bridge changes and the chat/drafter knobs;
  libtorch 2.13 (MPS).
- Kimi K3 experts as MXFP4 files (17.5 MB each, one file per expert, 82,432
  files, raw-v1 layout); int8 row-quantized spine resident in unified memory
  (50.7 GiB mmapped); Qwen3-0.6B drafter with exact-argmax acceptance, depth 8,
  prefix commit. Expert weights untouched at their released MXFP4 precision;
  the resident trunk is int8, which upstream labels non-weight-exact. No
  bit-exactness against BF16 is claimed.
- Placement, by role (drive names change; roles do not). The internal SSD
  holds the primary set (69,803 files: its original 56,669 plus the 13,107
  most-used experts copied from the base). One enclosure holds a complete copy
  of all 82,432 experts as the second base (`K3_EXPERT_DIR_B`; a 2 TB drive is
  therefore a recipe requirement). The other two enclosures hold replica bands
  (`K3_EXPERT_HOT_DIR`, 50,265 files — 88.9% of recorded reads; `K3_EXPERT_DIR_C`, 26,684 files — 66.5%; widened 2026-09-06 from 36,018 / 15,287 by copying usage-ranked experts from the full set), so
  every expert has at least two fast homes. Reads are routed to the holder with
  the lowest expected completion time from shared in-flight counters, on both
  read paths (`K3_SPLIT_ETA=1` for the chunked path, `K3_PLAN_BALANCE=1` for the
  prefetch and whole-file path). Before `K3_PLAN_BALANCE` existed the second
  path used a fixed probe order and every full-mirror layout lost 24–33%; with
  it the same layout gained 11% (2026-09-06). The staging script, the usage
  trace, the parameters and the resulting manifests are in `placement/`.
- Storage ceiling, measured 2026-09-08 with whole-file reads and no page cache,
  one drive at a time: internal 13.6 GB/s, SN8100 1 TB 7.1, SN8100 2 TB 7.1 (the
  same enclosure wall; capacity does not add speed), SN7100 5.7 falling to 5.1
  at high queue depth. Under the engine every drive peaks at 90–100% of its own
  ceiling; the four together drew about 24 GB/s averaged over a 512-token
  prefill (device counters) with one-second peaks near 27–30 GB/s. Before the
  plan-path balancer the engine drew 21.6 GB/s on the 200-token record arm.

## How to run

1. Build deltafin at the commit above; prepare the model root (`deltafin setup`),
   the int8 spine (`deltafin convert-spine-int8`) and the expert directories.
2. Edit the four paths at the top of `env.sh`.
3. `sh run-bench.sh` (or `sh run-bench.sh /path/to/deltafin`). It refuses to run
   while another deltafin process is resident: every arm is cold.
4. Read `results/RESULTS.md`. A text md5 that differs from the text of record means
   the run is not comparable (wrong quantization, missing expert files, or a
   numerics change); speed from such a run must not be quoted.

## Measurement rules we hold ourselves to

- One arm at a time, nothing else on the machine; Spotlight indexing off on every
  volume that holds expert files (`mdutil -s` to check).
- Compare candidates only inside one interleaved same-hour bracket: machine state
  moves 200-token results by several percent within a day.
- The engine's expert census must report `lazy-missing=0`; otherwise it downloads
  during the run and the number is void.
- Promotion needs a paired 200-token bracket, not a single arm; 17-token runs
  scatter by about 3% and are reported as a median of three.

## Limits

- The decode rate depends on how well the 0.6B drafter predicts the target.
  On draft-friendly completions (the finance prompt of record, the France
  prompt) acceptance is 68–100% and the champion sustains its headline rate.
  On general instruction prompts answered in chat mode (thinking off),
  acceptance falls to a quarter or less, the engine switches the drafter off
  by design (forcing it back on measured −18%), and decode runs single-row
  at about 0.64–0.74 tok/s on this configuration (0.50 on a translation
  prompt, 0.91 on a code prompt where the drafter stays alive). The
  ten-prompt table in `results/general-0906/` gives the measured range; the
  same prompts on the previous configuration are in `results/general/`.
- Identity: draft tokens are verified against K3's greedy predictions in the selected execution path. On the prompts of record the text is token-identical with the drafter on and off; on general prompts the generated text can differ from single-row decoding because execution paths use different floating-point arithmetic (each path is repeatable, and repeatability within a configuration is not equivalence between configurations). A tie-breaking rule would not remove this, since small numerical differences can reverse two unequal logits or change expert routing upstream.
  drafter on and off, and identical across every arm (27 of 27 two-hundred-
  token arms on 2026-09-06). On general prompts the text can differ at a
  near-tied token: between the drafter-on and drafter-off paths (8 of 10
  prompts, e.g. "kind of like" versus "just like"), and between two
  drafter-on runs on different storage configurations (1 of 10: "Then,
  subtract" versus "Then subtract"), because the wide tile accumulates the
  routed expert outputs in arrival order. None of these is an accepted wrong
  draft; they are rounding differences at ties, and a run is deterministic
  only up to that arrival order.
- Checkpoints: the drafters are the Qwen3 Base models. The post-trained
  Qwen3-0.6B and 1.7B were measured as drafters and were 20–24% slower on the
  prompt of record and no better on chat prompts.

## Texts of record

17 tokens: `Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also`

200 tokens (md5 6d8c4f50a22c): see `results/text-of-record-200.txt`.

## What is in this directory

- `env.sh` — the promoted configuration (every knob is a measured promotion).
- `run-bench.sh` — the runner: France ×3, prompt of record ×2, identity check, results table.
- `results/` — texts of record and the latest `RESULTS.md` with its raw logs.
- `placement/` — the staging script that produced the two replica directories,
  the expert-usage trace it read, the parameters used, and gzipped manifests
  (file lists) of all four expert directories as they were on 2026-09-05.
