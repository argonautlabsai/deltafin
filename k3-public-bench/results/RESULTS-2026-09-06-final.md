# Results — 2026-09-06

| arm | tokens | steady tok/s ([stats] speed) | fresh-process wall tok/s (wall) | text md5 | vs text of record |
|---|---|---|---|---|---|
| FRANCE_1 | 17 | 0.8909 | 0.803 (21 s) | 777f5ef1d9f3 | identical |
| FRANCE_2 | 17 | 0.9633 | 0.866 (20 s) | 777f5ef1d9f3 | identical |
| FRANCE_3 | 17 | 0.9631 | 0.864 (20 s) | 777f5ef1d9f3 | identical |
| RECORD_1 | 200 | 1.1000 | 1.087 (184 s) | 6d8c4f50a22c | identical |
| RECORD_2 | 200 | 1.1211 | 1.105 (181 s) | 6d8c4f50a22c | identical |

- France (17 tokens, 3 runs): median **0.9631** tok/s, best 0.9633.
- Prompt of record (200 tokens, 2 runs): 1.1000 / 1.1211 tok/s.
- Texts of record: results/text-of-record-17.txt (md5 777f5ef1d9f3), results/text-of-record-200.txt (md5 6d8c4f50a22c).
