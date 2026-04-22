# adaptive_typed_v1_final

日期：2026-04-19

## 目的

本说明用于固定当前最终方法版本，并记录报告前各轮正式评估的身份。

当前阶段的目标不是继续探索新机制，而是冻结版本、补齐评估，并拿到一份可用于下周报告的完整结果。

---

## 输入与输出

- 方法标签：`adaptive_typed_v1_final`
- MuSiQue 输出目录：`output/musique/adaptive_typed_v1_final/`
- 2WikiMQA 输出目录：`output/2wikimqa/adaptive_typed_v1_final/`

本阶段的运行对象按顺序分成三类：

1. MuSiQue bad 62：`output/musique/bad_cases_qids.txt`
2. MuSiQue 剩余 38（即 fixed-100 中 baseline 原本正确的部分）
3. 2WikiMQA 对应评测集

MuSiQue bad 62 的运行命令：

```bash
python main/load_data_adaptive.py --qid-file output/musique/bad_cases_qids.txt
```

---

## 当前版本包含的改动

- 保留 fallback gate 改进
- 保留 multi-query fusion
- 保留 category type guidance
- 保留当前 single-attempt runtime 配置（`MAX_VARIANTS = 0`）

从现在开始，不再继续加入新的机制性改动，除非发现明确 bug。

---

## 评估定位

`adaptive_typed_v1_final` 的评估定位分三层：

1. MuSiQue bad 62：看 hardest bad cases 中修复了多少题，以及哪些错误类型出现了改善
2. MuSiQue 剩余 38：看 adaptive 是否会破坏 baseline 原本正确的题
3. 另一个数据集：看当前最终版本在跨数据集时是否还能保持合理表现

不预设一定出现显著 EM 提升。

---

## 结果解读原则

- 方法版本和评测集合分开管理：`adaptive_typed_v1_final` 是方法名，`62 / 38 / 2WikiMQA` 是评测对象
- 如果 EM 有提升：记录具体改善题数与对应失败类型
- 如果 EM 提升有限：重点总结机制正信号与失败归因
- 不将这次结果与旧的 pilot、12 题子集、gate 单独验证 run 混在一起解释

---

## 执行状态（更新于 2026-04-21）

1. ✅ 已完成：MuSiQue bad 62 final run → 4/62 EM
2. ✅ 已完成：MuSiQue fixed-38 → 32/38 EM（bugfix 后稳定运行）
3. ✅ 已完成：2WikiMQA-100 → 68/100 EM
4. ✅ 已完成：三数据集结果汇总，见 `plan/results_adaptive_typed_v1_final_2026-04-21.md`

后续所有汇报与总结，默认以 `adaptive_typed_v1_final` 作为最终方法版本名。

---

## 补充实验：baseline_single_attempt（MAX_VARIANTS=0）

**目的**：消除计算量差异，在相同单次 attempt 设定下，单独量化 adaptive 机制的贡献。

**做法**：`config.py` 中 MAX_VARIANTS 通过环境变量读取，直接覆盖即可，无需新建 config 文件。

**运行命令**：

MuSiQue：
```bash
RT_RAG_MAX_VARIANTS=0 python main/load_data.py
```

2WikiMQA：
```bash
RT_RAG_MAX_VARIANTS=0 \
RT_RAG_DATASET=2wikimqa \
RT_RAG_DATA_PATH=data/longbench/2wikimqa_100_seed42.jsonl \
python main/load_data.py
```

**输出目录**：
- `output/musique/baseline_single_attempt/`
- `output/2wikimqa/baseline_single_attempt/`

**执行状态**：
- ✅ MuSiQue-100 baseline (MAX_VARIANTS=0) → 40/100，结果在 `output/musique/dense_chunk200_topk1_25_topk2_8/5.txt`
- ✅ 2WikiMQA-100 baseline (MAX_VARIANTS=0) → 71/100，结果在 `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/2.txt`
- ✅ 与 adaptive 结果对比，汇总表已更新（见 `plan/results_adaptive_typed_v1_final_2026-04-21.md`）
