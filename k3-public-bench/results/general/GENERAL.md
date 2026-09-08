# Ten-prompt generalisation bracket — champion env, 200 tokens, cold, k3-pressure 60; mode: chat

| # | tokens | chunks | drafts accepted/proposed | steady tok/s | fused tok/s (200 / wall s incl. load+prefill) | GB requested / token | re-arm lines | md5 ON / OFF | identity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 196 | 4/16 (25%) | (0.5886) | 0.578 | 38.77 | 0 | 71064368ebdd / 45111349c3cd | DIFFERS (whitespace-insensitive) at word ~185 of 190/189 |
| 2 | 200 | 38 | 162/244 (66%) | (0.7793) | 0.758 | 27.21 | 0 | 18eb9b1c4329 / 04a10a92ae19 | DIFFERS (whitespace-insensitive) at word ~77 of 120/171 |
| 3 | 68 | 24 | 44/64 (69%) | (0.5968) | 0.562 | 34.62 | 0 | 0f067c1ba768 / e044421289d6 | identical text (whitespace-insensitive) |
| 4 | 200 | 200 | 0/4 (0%) | (0.5943) | 0.583 | 37.37 | 0 | b4625ad9f888 / 4fc6e6c7ecc7 | DIFFERS (whitespace-insensitive) at word ~80 of 200/200 |
| 5 | 200 | 199 | 2/14 (14%) | (0.5370) | 0.526 | 42.17 | 0 | 3899dbf4bf21 / 3899dbf4bf21 | identical text (whitespace-insensitive) |
| 6 | 200 | 196 | 4/10 (40%) | (0.5934) | 0.581 | 37.34 | 0 | c8cdefa86d96 / 292800bb4174 | DIFFERS (whitespace-insensitive) at word ~56 of 198/199 |
| 7 | 200 | 198 | 2/12 (17%) | (0.5853) | 0.575 | 38.76 | 0 | 816ebcba56f0 / ecef22d1f2e3 | DIFFERS (whitespace-insensitive) at word ~12 of 199/198 |
| 8 | 200 | 200 | 0/4 (0%) | (0.5814) | 0.570 | 37.74 | 0 | ef1eb19217ba / f34edde813db | DIFFERS (whitespace-insensitive) at word ~56 of 179/180 |
| 9 | 75 | 33 | 42/108 (39%) | (0.4347) | 0.417 | 48.05 | 0 | 32f02774cf04 / 6672c5cfcaca | DIFFERS (whitespace-insensitive) at word ~2 of 50/162 |
| 10 | 200 | 190 | 10/26 (38%) | (0.5646) | 0.554 | 38.74 | 0 | a62618b88bba / 869f5c33eb5e | DIFFERS (whitespace-insensitive) at word ~120 of 192/197 |

Prompts (fixed before any run):
1. Explain how a central bank sets interest rates and what happens when it raises them.
2. Write a Python function that parses a CSV of transactions and returns the top five payees by total amount, with a short explanation.
3. List twelve capital cities in Europe with their countries, one per line.
4. Describe the water cycle for a ten-year-old.
5. A company has revenue of 4,200,000, cost of sales of 2,650,000, operating expenses of 910,000 and tax at 20%. Calculate gross profit, operating profit and net profit, showing each step.
6. Summarise the plot of a heist story you invent, in three paragraphs.
7. What are the main differences between VAT and sales tax, and which countries use each?
8. Give a step-by-step recipe for a simple tomato pasta sauce.
9. Translate the following into formal French and then back into English: "The quarterly report is due on Friday and the board expects a summary of cash flow."
10. Explain what a Mixture-of-Experts model is and why it is hard to run on a laptop.

Mode: chat (the prompts are instructions). The only deviation from k3-public-bench/env.sh is K3_QWEN_ALLOW_CHAT=1: upstream scopes the Qwen drafter to raw completions and leaves chat undrafted; without it every chat arm decodes single-row. Raw-completion mode was tried first on prompts 1 and 2 (G1_ON_raw, G2_ON_raw): the model continued them as dataset-like fragments, the drafter died within ~10 tokens (11/31 and 7/20 accepted, 189 and 193 chunks) and both ran at 0.587 tok/s — kept as an appendix, not part of the ten.

Notes: steady = the engine's [stats] speed (generated / decode elapsed). fused = generated / wall seconds of the whole arm (process start, model load, prefill, decode, harness helpers). GB requested / token = the engine's expert bytes requested ([opens] requested_bytes) per generated token. Identity = md5 of the generated text with the drafter on vs off for the same prompt; a DIFFERS row keeps its speed in parentheses and it is not quoted.
