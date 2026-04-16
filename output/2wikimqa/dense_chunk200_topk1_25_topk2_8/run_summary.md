# 2WikiMQA Baseline Run Summary

Generated on `2026-04-16` (UTC).

## Run Snapshot

- Dataset: `2wikimqa`
- Evaluation subset: `data/longbench/2wikimqa_100_seed42.jsonl`
- Retrieval method: `dense`
- Dense index root: `data/embeddings/2wikimqa/200_2_2/`
- Runtime entrypoint: `python main/load_data.py`
- Active retrieval config: `chunk_size=200`, `min_sentence=2`, `overlap=2`, `topk1=25`, `topk2=8`
- Output directory: `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/`
- Main result file: `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/1.txt`

## Main Metrics

- Total evaluated questions: `100`
- Exact Match (EM): `69.00%`
- F1: `79.26%`
- Strictly correct by EM: `69`
- Strictly incorrect by EM: `31`

## Runtime Notes

- Average per-question runtime: `320.73s`
- Median per-question runtime: `121.74s`
- Maximum observed runtime: `15122.62s`
- Average retrieval calls per question: `3.21`
- Average generation calls per question: `6.42`
- Direct fallback triggered: `8` questions
- Timeout flagged: `1` question

Timeout/outlier note:
`qid=81c8cd41355e5f0489dad4010b5fd414b817f7a9134affc2`

Question:
`Who was born first, Vytautas Straižys or Mirjam Polkunen?`

This example recorded the longest runtime (`15122.62s`) and is worth checking separately before using latency numbers in the report.

## Current Project Status

- MuSiQue baseline fixed-subset run remains the earlier reference baseline.
- The existing MuSiQue error analysis reports `38` strict correct and `62` strict errors on the final deduplicated `100`-question set.
- MuSiQue baseline references:
  - `output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_report.md`
  - `output/musique/dense_chunk200_topk1_25_topk2_8/bad_case_analysis.md`

## Suggested Citation In Notes / Commit Message

Use the following short description if needed:

`Baseline on 2WikiMQA-100 completed with dense retrieval (chunk200 / topk1=25 / topk2=8), EM 69.00 and F1 79.26 on 100 fixed-seed examples.`

## Follow-up

- `2wikimqa` error analysis is not yet written.
- Current baseline output path follows the existing runtime naming convention:
  `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/`
  rather than the planned alias `output/2wikimqa/baseline/`.
