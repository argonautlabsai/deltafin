# Standard-length benchmark set — 2026-09-08

Twelve cold arms on the champion configuration. Eleven ran as one uninterrupted sequence in one sitting; the twelfth (`TG512_OFF_B`) was added afterwards to
give every setting a repeat. No monitoring process was running during any of
them and no other work ran on the machine. Every length was run with the
speculative drafter enabled and disabled, and the order was partially balanced
to reduce order effects: the first pass forward, the second reversed. The design
reduces the risk that drift favours one variant; it cannot guarantee
cancellation, and the twelfth arm ran after the main sequence.

Prompt token counts are exact, measured with the model's own tiktoken
vocabulary. The short prompt ("The three main financial statements are") is
6 tokens; the long prompt is a fixed self-authored passage of exactly 512
tokens, included in this package as `prompt-512-tokens.txt`.

## Headline

**K3 achieves 1.00 token/s steady decode over a 512-token completion, and 1.13
tokens/s over 128 tokens, with speculative drafting. A 512-token prompt takes
approximately 6.3 minutes to produce its first token.**

Raw completion mode (not the chat template), greedy, single prompt of record,
on the hardware and quantization described in the package README.

## Results

Median of two runs at every setting.

| test | drafter off | drafter on | drafter gain |
|---|---:|---:|---:|
| steady decode, 512 generated tokens | 0.9232 | **1.0015** | +8.5% |
| steady decode, 128 generated tokens | 0.9261 | **1.1252** | +21.5% |
| inclusive throughput, 512 generated tokens | 0.9063 | **0.9849** | +8.7% |
| inclusive throughput, 128 generated tokens | 0.8630 | **1.0377** | +20.2% |
| prompt processing, 512-token prompt | ≈1.4 tok/s | ≈1.4 tok/s | none, as expected |
| first token, 512-token prompt | ≈376 s | ≈375 s | negligible |

First-token latency on the 6-token prompt, which contains the cold model load,
was 9.3 to 11.7 seconds across all arms.

## Per-arm detail

| arm | generated | inclusive | steady | first token, s | drafts accepted | memory rejections |
|---|---:|---:|---:|---:|---:|---:|
| TG128_ON_A | 128 | 1.0341 | 1.1329 | 11.68 | 94/146 | 0 |
| TG128_ON_B | 128 | 1.0413 | 1.1176 | 9.29 | 94/146 | 0 |
| TG128_OFF_A | 128 | 0.8694 | 0.9340 | 11.26 | 0/0 | — |
| TG128_OFF_B | 128 | 0.8567 | 0.9181 | 11.09 | 0/0 | — |
| TG512_ON_A | 512 | 0.9901 | 1.0065 | 9.42 | 247/417 | 0 |
| TG512_ON_B | 512 | 0.9796 | 0.9966 | 9.90 | 247/417 | 0 |
| TG512_OFF_A | 512 | 0.9108 | 0.9282 | 11.58 | 0/0 | — |
| TG512_OFF_B | 512 | 0.9018 | 0.9183 | 11.27 | 0/0 | — |
| PP512_ON_A | 8 | — | — | 371.72 | 5/5 | 0 |
| PP512_ON_B | 8 | — | — | 378.33 | 5/5 | 0 |
| PP512_OFF_A | 8 | — | — | 373.07 | 0/0 | — |
| PP512_OFF_B | 8 | — | — | 378.97 | 0/0 | — |

The generation rate of the prefill arms is not reported: eight tokens is too few
to measure a decode rate, and those arms exist only to supply the first-token
latency of a long prompt.

## Variability

Repeats within the main set differed by less than 2% within each setting. A separate
later set of 200-token tests on the same day showed a 3.6% spread between drafted
runs following substantial staging writes to the drives. The cause was not
isolated; small performance differences on this rig require further
replication, and neither figure should be read as a universal noise floor.

Every pair in the main set agrees within two percent. The two drafted arms at each length
produced identical chunk and verify-transaction counts across separate cold runs
(128 tokens: 34 and 27; 512 tokens: 265 and 77). That establishes **repeatable
aggregate drafting behaviour**; it does not by itself establish identical
outputs or identical proposal sequences, which were not compared.

## What these numbers say, and what they do not

**Prompt processing is the weak point, by a wide margin.** A 512-token prompt
costs about six minutes before the first token, and both drafter settings agree
to within one second, which is expected because drafting does not participate
in prefill. The 1.4 tok/s figure is derived by subtracting first-token
latencies, which removes the cold model load only approximately. It has since
been classified from the engine's own phase timers and the device counters (see
`SCALING.md`): 76% of the prefill is spent waiting
for expert reads, and the prompt is read as roughly nine terabytes for a 1.4 TB
expert set because the 512 rows are processed in eight passes of 64 that each
re-read a layer's experts. That is an engine scheduling cost, not a storage
limit, and it is the next thing to fix.

**Plain decode does not degrade with answer length.** Drafter off, the steady
rate moves from 0.9261 at 128 tokens to 0.9232 at 512, a change of about a third
of a percent, which is inside the agreement of the repeats. The read path
sustains its rate over a long answer.

**The speculative advantage degrades, and that is where to look first.** Drafter
on, steady falls from 1.1252 to 1.0015, and the gain over plain decode falls
from +21.5% to +8.5% — a reduction of roughly sixty percent in the advantage,
not a halving. Acceptance declines more modestly, from 64.4% (94 of 146) to
59.2% (247 of 417), so acceptance alone does not account for the whole change.
Speculative verification also alters expert unions, read traffic and scheduling,
so storage effects are not excluded by the flat drafter-off line. The honest
statement is that **the loss of speculative benefit motivates investigating
drafting and verification first.** The measurement that would settle it is
accepted tokens per verification, and verification time, across successive
output windows.

## Definitions

**Inclusive tok/s** — `generated / elapsed` from the engine's final `[stats]`
line. The clock starts when the run begins, so cold model load, prompt
processing and the first forward pass are all inside it. This is the figure the
project ledger has always used and it is the pessimistic one.

**Steady tok/s** — `(generated - 1) / (elapsed_final - elapsed_first_token)`.
The first-token phase is excluded, so this is the decode rate once generation is
under way. It is the figure most comparable with a generation benchmark that
loads the model separately.

**First-token latency** — the `elapsed` field of the first `[stats]` line. On
this engine it contains the cold model load as well as prompt processing, so it
is not a prefill measurement on its own.

**Prompt processing tok/s** — derived, not printed by the engine. Two arms
differing only in prompt length give `(512 - 6) / (latency_512 - latency_6)`.
Subtracting cancels the model load, which is common to both, though only
approximately: see the caveat above.

## Conditions held constant

| condition | value |
|---|---|
| generation | greedy, no sampling |
| mode | raw completion, not the chat template |
| process | cold start per arm, no warm server resident |
| page cache | not cleared between arms; the working set far exceeds memory |
| background load | no monitoring process running, no builds, one arm at a time |
| memory guard | an arm aborts if swap exceeds 100 MB unless overridden |

Matching token counts with another engine does not by itself make two systems
comparable. The model, the quantization, the sampling settings, the storage
layout and the definition of elapsed time all differ, and each of ours is
recorded above so a reader can judge what a comparison is worth.

Reproduce: each arm is one cold `deltafin run` with the `env.sh` configuration
(as `run-bench.sh` does for the headline), `--max-new` set to the length and
`--prompt` set to the prompt of record or `prompt-512-tokens.txt`; drafter-off
arms add `K3_UAG_DRAFT=off K3_QWEN_ROUNDTRIP_SOFT=0 K3_QWEN_REARM=0`. The
figures above are read from the engine's `[stats]` lines as defined here.
