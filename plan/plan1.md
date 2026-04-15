# Next-Step Experiment Plan for Baseline Expansion and Revised Method

## Summary

接下来的主线分三步走，按成本优先来安排：

1. 先把现有 baseline 扩展到另外两个数据集，但不直接跑全量。
   为 `2WikiMQA` 和 `HotpotQA` 各构造一个固定 `100` 题子集，口径对齐当前 `MuSiQue-100`。
2. 基于 baseline 的失败模式，修订 proposed method。
   新方法做成一个组合方法，而不是两个完全独立项目：`typed nodes + self-adaptive control` 合在一起实现。
3. 新方法必须走独立脚本和独立输出目录，不覆盖 baseline。
   baseline 保持可复现、可对照；新方法作为平行实验线。

默认策略：
- 三个数据集都先跑固定 `100` 题子集。
- 只有当某个数据集上结果明显有价值，才考虑扩到全量。
- 新方法只在 baseline 子集跑完并完成失败模式汇总后再实现。

## Implementation Changes

### 1. 数据与 baseline 扩展

为另外两个数据集建立与 MuSiQue 相同风格的固定子集：
- `data/longbench/2wikimqa_100_seed42.jsonl`
- `data/longbench/2wikimqa_100_seed42.meta.json`
- `data/longbench/hotpotqa_100_seed42.jsonl`
- `data/longbench/hotpotqa_100_seed42.meta.json`

子集规则固定如下：
- 每个数据集抽 `100` 题
- 固定随机种子 `42`
- metadata 记录原始文件、样本数、seed、抽到的原始索引
- 后续 baseline 和新方法都统一跑这三个 fixed subsets，不换样本

baseline 运行策略：
- 保留当前 `main/config.py` 的轻量级思路
- 不再把 `main/config.py` 绑定死到 MuSiQue
- 为 baseline 增加“按数据集切换”的独立配置入口，建议新建 dataset-specific baseline configs，而不是继续频繁手改一个文件
- baseline 输出目录继续沿用现有风格，但按数据集分别落到独立目录，避免相互覆盖

### 2. 修订后的 proposed method

把最初 proposal 改写成一个更聚焦、由 baseline 经验驱动的方法。核心从“通用 typed tree”收缩到“对当前错误最有针对性的 typed control”。

新方法主线保留两部分，但优先级重新排序：

- `Typed Nodes`
  - 节点至少显式带上 `answer_type`
  - 同时保留少量高价值字段：`op_type`、`entity_anchor`、`depends_on`
  - 不把 `relation` 等字段做得太重，避免把项目拖成结构化知识工程

- `Self-Adaptive Control`
  - `answer-type validation`
  - `temporal consistency check`
  - `geographic granularity check`
  - `fallback evidence gate`
  - `canonical short answer compression`
  - `direct-solve pruning`
  - `single retry rewrite` 仅保留一次，不做高成本多轮恢复

修订后的方法重点要明确写进文档：
- baseline 的主要问题不是单纯 retrieval miss，而是 `slot/type mismatch`、`temporal anchor drift`、`granularity mismatch`、`fallback guessing`
- 因此新方法的目标不是“生成更多树”或“加更强模型”，而是“约束中间状态，避免错误向上传播”

### 3. 独立脚本与代码组织

不要覆盖 baseline 逻辑。新方法使用独立入口和独立配置。

建议代码组织：
- baseline 继续保留现有入口：`main/load_data.py`
- 新方法新增独立入口：`main/load_data_adaptive.py`
- baseline 保留原树执行逻辑
- 新方法新增独立执行逻辑文件，建议命名：
  - `main/tree_decompose_adaptive.py`
  - `main/config_adaptive.py`

脚本职责固定：
- `load_data.py` / baseline configs:
  - 只负责原始 RT-RAG baseline
- `load_data_adaptive.py` / adaptive configs:
  - 只负责修订后的新方法
- 两条线共享底层 retrieval 工具可以接受，但不能让 baseline 被新逻辑隐式改写

输出隔离规则：
- baseline 输出继续放在 `output/<dataset>/...`
- 新方法输出单独放在新的方法目录命名下，建议包含方法标识，如 `output/<dataset>/adaptive_typed_...`
- debug JSON、统计日志、error analysis 全部分开

### 4. 文档更新

新增一份修订版方法说明，单独保存，不覆盖最初 proposal。
建议新增：
- `note/revised_proposed_method.md`

内容结构固定：
- baseline 观察到的关键失败模式
- 为什么最初 proposal 需要收缩和重排优先级
- revised method 的核心机制
- 该方法与 baseline 的最小实现差异
- 预期改善的错误类型
- 不解决什么问题

保留 `note/initial_proposed_method.md` 作为最初版本，不改写历史记录。

## Experiment Plan

### Phase 1: Baseline completion on three fixed subsets

先完成三套 baseline 子集实验：
- MuSiQue-100
- 2WikiMQA-100
- HotpotQA-100

对每个数据集统一收集：
- EM
- F1
- 去重后的 unique-qid 结果
- median / p95 runtime
- `used_direct_fallback` 比例
- `retrieval_call_count`
- `generation_call_count`

完成后输出一张 baseline 总表：
- 三个数据集的子集成绩
- 成本统计
- 每个数据集最突出的失败模式摘要

### Phase 2: Failure-driven method revision

只在 baseline 三个子集结果出来后，做 cross-dataset failure review。
重点判断下面哪些问题是跨数据集稳定出现的：
- answer-type confusion
- temporal / historical anchor errors
- geographic granularity mismatch
- free-form long answer not matching gold
- fallback producing unsupported guesses

只有跨数据集重复出现的问题，才进入 revised method 的核心机制。
不要把只在单个题型偶发的问题塞进主方法。

### Phase 3: Adaptive method evaluation

新方法先跑三套 `100` 题子集，不跑全量。

主比较实验：
- `Baseline`
- `Adaptive Typed-Control Method`

最少需要的 ablation：
- `Node typing only`
- `Control layer only`
- `Full adaptive method`

其中：
- full method 在三个 `100` 题子集都跑
- 两个 ablation 默认只在三套 `100` 题子集上跑，不扩全量
- 如果成本压力仍大，则 ablation 至少保证在 MuSiQue-100 全跑，在另外两个数据集保底跑 `30-50` 题 fixed slice

### Phase 4: Optional expansion gates

只有满足以下条件，才考虑扩到全量数据集：
- 新方法在某个数据集的 `100` 题子集上，相对 baseline 有稳定提升
- 提升不是主要来自格式伪错误
- 成本增幅可接受
- 错误分析能解释“为什么提升了”

默认门槛：
- `EM` 提升至少 `+2.0`
- 或 `F1` 提升至少 `+1.5`
- 且 median runtime 不超过 baseline 的 `2x`

如果不满足，项目结论就停留在 fixed-subset 级别，不扩全量。

## Test Cases and Scenarios

每个数据集都要至少覆盖这几类分析场景：
- 正常题的整体 EM/F1 对比
- bad cases 的定向复盘
- long-answer / over-generation 样本
- fallback 触发样本
- type mismatch 样本
- temporal mismatch 样本

针对 revised method，需要单独验证：
- answer type mismatch 是否减少
- temporal anchor 错误是否减少
- county/city、province/country 这类粒度错误是否减少
- final answer 是否更短、更贴近 canonical gold
- fallback 是否更少“无证据硬猜”

## Assumptions and Defaults

- 默认继续使用“成本优先”策略，不主动追求全量三数据集
- 三个数据集的对比统一在 fixed `100` 题子集上进行
- baseline 不再重构，只做最小必要的配置拆分与数据集扩展
- revised method 采用“一个组合方法 + 少量关键 ablation”，不拆成两个独立项目并行推进
- baseline 与新方法必须双轨并存，任何新逻辑都不能覆盖 baseline 入口
