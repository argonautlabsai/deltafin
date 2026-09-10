# Share an ARGODRIVE installation or benchmark result

Copy the template below into a new [Discussion](https://github.com/argonautlabsai/deltafin/discussions).
Suggested title: `[Result] <chip> / <RAM> / <drive count> / <installation or comparison>`.
Leave unknown measurements as `not measured`. An installation report does not
need a benchmark; setup failures and slower results are welcome too.

Use a public test prompt. Before attaching logs, remove private prompts,
credentials, personal paths and drive serial numbers. Drive models, capacities
and connection types are sufficient.

## Template

### Outcome

- Report type: installation / benchmark comparison / setup failure
- Outcome and anything that required manual changes:
- Repository and commit (`git rev-parse HEAD` from the checkout used to build):
- Binary: built from that commit / other (describe)
- Local modifications:

### Hardware and software

- Mac model / chip / GPU core count:
- Unified memory:
- macOS version:
- Available memory and swap in use before the run:
- Power mode and AC/battery:
- Other active workloads, dashboard and samplers:

| Drive role | SSD model and capacity | Enclosure / connection | Direct port or hub | Model data held |
|---|---|---|---|---|
| Internal | | | | |
| External 1 | | | | |
| External 2 (if used) | | | | |
| External 3 (if used) | | | | |

### Installation or reproduction

- Model, weight format and resident-spine precision:
- Setup mode: full / stream / existing model data
- Benchmark guide or script used, if any:
- Exact command and effective environment overrides (redact personal paths):
- Exact public prompt, or a link to its file:
- Prompt token count / requested and actual generated token counts:
- Raw completion or chat template:
- Drafter model, enabled/disabled and settings:
- Start state: fresh process / persistent server; known cache state:

A fresh process alone does not establish that storage caches are cold. Describe
what was done instead of inferring the cache state.

### Measurements (optional for installation reports)

Use one row per run, in execution order. For a comparison, identify the exact
control and change under test. Keep the model, prompt, generation length and
measurement definition matched; disclose other differences. Report all repeats.

| Order | Arm / change | Generated tokens | Steady decode tok/s | Inclusive tok/s | First-token time (s) | Output identity / errors |
|---|---|---:|---:|---:|---:|---|
| 1 | Control | | | | | |
| 2 | Candidate | | | | | |
| 3 | Control repeat | | | | | |
| 4 | Candidate repeat | | | | | |

- Timing source and definitions (including when each clock starts):
- Output comparison method (token IDs / text hash / not checked):
- Storage measurements, if collected: sampling interval, phase, average vs peak:
- Deviations from the published recipe:
- Redacted logs, configuration and results files:

For long-answer claims, include a 512-generated-token result where practical.
Report prefill and decode separately. An output mismatch should be reported,
not omitted or treated as a successful identity check.

See the [benchmark guide](../k3-public-bench/README.md) for the published
definitions and limitations, and [ARGODRIVE](https://github.com/argonautlabsai/argodrive)
for the dashboard and measurement instruments.
