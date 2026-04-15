# Revised Experiment Plan v2

## Summary

采纳这轮反馈后的核心调整如下：

- `2WikiMQA` 和 `HotpotQA` 改为 `dev30 / eval70`，避免开发集过小。
- 不再预设三维分层抽样，先做轻量分布探查，再决定是否只按 `问题类型 + 答案类型` 两维分层；如果分布太稀疏，就退回“约束随机抽样”。
- ablation 统一只在同一数据集口径下比较，主消融固定在 `MuSiQue-100` 完整集合上。
- `fallback evidence gate` 返回 `[none]` 被视为“保守失败”而非“绝对退步”；评估时必须单独报告“unsupported guess 减少”和“[none] 增加”这组权衡。
- 明确定义 `entity_anchor`，并封死 Phase 3 期间的“选择性重跑”风险。

默认结论口径：
- `MuSiQue-100` 是开发与主消融集合。
- `2Wiki dev30 / Hotpot dev30` 用于补充开发阶段的跨数据集失败模式验证。
- `2Wiki eval70 / Hotpot eval70` 是 held-out 泛化结论的主要依据。

## Data Protocol

### 1. Fixed subsets and splits

数据集划分固定为：

- `MuSiQue-100`
  - 全部用于开发、方法设计、主消融
- `2WikiMQA-100`
  - `dev30`
  - `eval70`
- `HotpotQA-100`
  - `dev30`
  - `eval70`

所有 subset 都必须产出：
- `.jsonl`
- `.meta.json`

metadata 必须记录：
- source file
- seed
- selected indices
- split assignment
- sampling policy
- 轻量分布统计摘要

### 2. Sampling policy

不直接实现三维复杂分层抽样。

先做一次分布探查：
- 问题类型粗分：`temporal / comparison / bridge / location / numeric / other`
- 答案类型粗分：`person / org / location / date / number / other`
- 问题长度只做统计，不作为默认分层轴

然后按以下规则选样：
- 若 `问题类型 × 答案类型` 的二维分桶没有明显稀疏，再做二维分层抽样
- 若二维分桶仍明显稀疏，则改用“约束随机抽样”：
  - 保证每个高频问题类型至少有最小覆盖
  - 保证每个高频答案类型至少有最小覆盖
  - 其余样本随机补齐

不允许为了追求“形式上的分层”引入过高实现复杂度。

## Method Scope

### 3. Typed node schema

v1 typed node 只保留四个字段：

- `answer_type`
- `op_type`
- `entity_anchor`
- `depends_on`

字段定义固定如下：

- `answer_type`
  - 当前 node 期望输出的答案类型
  - 取值集合预定义，不允许自由文本漂移

- `op_type`
  - 当前 node 的推理操作类别
  - 取值固定为：`direct / bridge / comparison / composition / temporal / numeric`

- `entity_anchor`
  - 当前 node 中必须保留、不可在 rewrite 中丢失的已知实体或核心名词短语
  - 不是“所有实体”，而是当前 node 的 retrieval target 中最关键的 anchor
  - 若 node 是由父节点答案生成的，则 `entity_anchor` 可包含父节点填充值和原题显式实体

- `depends_on`
  - 该 node 依赖哪些上游 node 输出
  - 只记录依赖关系，不承载额外语义

### 4. Adaptive control v1

保留这 5 个机制：

- `type-aware validation`
- `temporal consistency check`
- `granularity check`
- `single retry rewrite`
- `fallback evidence gate`

关键约束：

`single retry rewrite`
- 只在 node 级触发
- 触发条件必须同时满足：
  - 当前 node retrieval 弱或空
  - 或候选答案被 validator 判 invalid
  - 该 node 尚未 rewrite
  - 仍有足够剩余 timeout
- rewrite 后必须保留 `entity_anchor` 和 `answer_type`

`fallback evidence gate`
- 若无足够证据支撑，返回 `[none]`
- 不允许“高频常识补洞”
- `[none]` 被视为一种明确、可诊断的失败类型

### 5. [none] handling policy

计划里必须显式承认：
- `[none]` 会让 EM/F1 直接记为 0
- 因此它可能在短期内拉低总体分数
- 但如果它替代的是“unsupported guess”，则方法可能在行为上更可靠

所以 adaptive 结果必须同时报告：
- 总体 EM / F1
- `[none]` rate
- unsupported-guess case 数量
- guess-to-none 转移数
- none-to-correct 数量
- none-to-wrong 与 wrong-to-none 的净变化

解释规则：
- 若 EM 小幅下降，但 unsupported guess 明显下降，且 held-out 上 validator 误杀有限，则该结果不能直接判定为退步
- 若 `[none]` 大幅增加而 unsupported guess 没显著减少，则视为方法失败

## Code Isolation

### 6. Frozen baseline rule

baseline 和 adaptive 必须物理隔离：

- `main/baseline/`
- `main/adaptive/`

baseline 冻结后：
- 记录 commit hash
- 后续不允许再修改 baseline 核心执行逻辑
- adaptive 不得回改 baseline

共享规则：
- 只允许共享无状态纯工具函数
- retrieval execution、tree execution、answer post-processing、validation logic 一律分线维护

## Experiment Design

### 7. Phase 1: Baseline runs

先完成 baseline：

- `MuSiQue-100`
- `2Wiki dev30`
- `2Wiki eval70`
- `Hotpot dev30`
- `Hotpot eval70`

Phase 1 可并行执行。

每个 run 必须产出：
- EM / F1
- unique-qid 去重结果
- wall-clock runtime
- retrieval / generation calls
- direct fallback rate
- timeout summary
- 粗略 API 成本估计

### 8. Phase 2: Failure analysis and spec freeze

失败分析采用“自动预标 + 有上限的人工复核”。

自动预标标签：
- type mismatch
- temporal drift
- granularity mismatch
- unsupported guess
- long-answer / normalization
- retrieval miss / unclear

人工复核目标：
- MuSiQue：30 个错误样本
- 2Wiki dev30：最多 20 个错误样本
- Hotpot dev30：最多 20 个错误样本

Phase 2 完成标准：
- 至少完成 `70` 个错误样本人工复核
- top 3 非格式类 failure families 覆盖至少 `60%` 的错误
- adaptive v1 spec 冻结
- 冻结后不再根据 held-out 结果增删机制

### 9. Phase 3: Adaptive evaluation

主比较：
- `Frozen Baseline`
- `Adaptive v1`

主消融只在同一口径下做：
- `MuSiQue-100`
  - `Adaptive v1`
  - `Adaptive v1 - validators`
  - `Adaptive v1 - recovery`

开发补充消融：
- `2Wiki dev30`
  - baseline vs adaptive v1
- `Hotpot dev30`
  - baseline vs adaptive v1

held-out 只跑主方法，不跑消融：
- `2Wiki eval70`
  - baseline vs adaptive v1
- `Hotpot eval70`
  - baseline vs adaptive v1

这样保证：
- 消融结论来自同一数据集同一分布
- held-out 只承担泛化验证，不承担方法拆解

### 10. Leakage prevention rule

Phase 3 期间严格禁止“选择性重跑 eval”。

冻结后允许修改的只有：
- 运行故障修复
- 非语义性 bug fix
- 基础设施问题修复

一旦任何会影响方法行为的参数、validator、rewrite 逻辑发生改变：
- 必须提升版本号
- 必须同时重跑：
  - `MuSiQue-100` 主消融全套
  - `2Wiki dev30`
  - `Hotpot dev30`
  - `2Wiki eval70`
  - `Hotpot eval70`

不允许：
- 只重跑 `eval70`
- 看完 held-out 后再偷偷调参数
- 只对表现差的数据集选择性补跑

## Test Plan

必须统一报告：

- EM
- F1
- paired per-question delta
- bootstrap 或重复重采样稳定性结果
- wall-clock runtime
- retrieval / generation call counts
- fallback rate
- `[none]` rate
- unsupported guess 数
- validator rejection counts

必须展示的案例类型：
- temporal errors
- answer-type confusion
- granularity mismatch
- unsupported guess 被改成 `[none]`
- `[none]` 最终通过 rewrite 或 validator 变成正确答案的案例
- 长答案被确定性 cleanup 改善的案例

## Assumptions

- 成本优先仍然成立，但 held-out 结论不能只靠 MuSiQue
- `dev30 / eval70` 是当前最平衡的开发/验证划分
- sampling 以“低复杂度、可解释”优先，不追求形式上最复杂的分层
- `[none]` 是方法行为的一部分，必须被显式分析，而不是当作普通 wrong 静默吞掉
- Phase 3 一旦冻结，任何影响行为的改动都必须整套重跑，防止间接泄露
