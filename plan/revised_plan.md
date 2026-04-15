# Revised Experiment Plan After Critique Review

## Summary

采用 Claude 的大部分意见，并把原计划改成一个更稳的“pilot + held-out + frozen baseline”方案。

关键调整：
- `100` 题子集不再作为硬阈值决策的终点，只作为低成本 pilot。
- 不再用同一批 MuSiQue 失败模式来设计、再在同一批 MuSiQue 上下结论。
- 不再做 `node typing only` 这种逻辑上站不住的 ablation。
- baseline 与 adaptive 方法彻底分线，避免共享代码被后续修改污染复现。
- 先补成本校准和样本代表性，再跑大批量实验。

## Protocol Changes

### 1. 数据集与划分

三个数据集都保留 fixed subset 思路，但改成“开发集 + 保留评测集”：

- `MuSiQue-100`：主开发集，用于失败模式分析、方法设计、方法调参
- `2WikiMQA-100`：拆成 `dev20 + eval80`
- `HotpotQA-100`：拆成 `dev20 + eval80`

固定规则：
- 仍使用固定 seed 和 metadata
- 但不再做纯随机抽样
- 改为轻量分层抽样：按问题表面模式和答案类型启发式分桶后按比例抽样
- metadata 额外记录 bucket 和 split

分层维度统一用可自动推断的启发式：
- 问题类型：temporal / comparison / bridge-entity / location / numeric / other
- 预期答案类型：person / organization / location / date / number / work-title / other
- 问题长度分桶：short / medium / long

默认结论口径：
- MuSiQue 只用于方法开发，不作为最终泛化结论的主证据
- 最终跨数据集结论主要看 `2Wiki eval80 + Hotpot eval80`

### 2. baseline 冻结与代码隔离

不再允许 baseline 和 adaptive 共用可变核心模块。

代码组织固定为两套：
- `main/baseline/`
- `main/adaptive/`

baseline 冻结内容：
- `load_data.py`
- `tree_decompose.py`
- `retrieve.py`
- 对应 config
- 运行所需的最小辅助模块

adaptive 方法：
- 新增独立入口与独立实现
- 不修改 baseline 目录下任何逻辑
- 如需复用工具，只能复用纯工具函数；涉及 retrieval / tree execution / answer post-processing 的逻辑一律复制到 adaptive 线中单独维护

复现要求：
- Phase 1 结束后记录 baseline commit hash
- 后续所有 adaptive 结果都必须引用该 frozen baseline 作为对照

### 3. 成本与运行流程

在任何 100 题运行前，先做 `10` 题校准 run：

- `MuSiQue-10`
- `2Wiki dev10`
- `Hotpot dev10`

校准 run 必须产出：
- 总 wall-clock 时间
- 平均每题 API 请求数
- 平均每题 retrieval / generation 调用数
- timeout / direct fallback 比例
- 粗略成本区间估计

注意：
- 当前累计式 timing 不能直接拿来做成本预算
- 预算依据以校准 run 的真实 wall-clock 和 API request 统计为准

并行化策略：
- Phase 1 三个 baseline run 可并行
- subset 构建、评测脚本、error analysis tooling 可并行准备
- 不要求整条链严格串行，只要求“方法冻结前不看 held-out 误差细节”

## Revised Method Scope

### 4. 方法主线

保留“typed nodes + self-adaptive control”的组合方法，但收缩到一个可定义、可测量的 v1。

v1 typed node 只保留这几个字段：
- `answer_type`
- `op_type`
- `entity_anchor`
- `depends_on`

不把 `relation` 等更重字段放进 v1。

v1 adaptive control 只做这 5 个机制：
- `type-aware validation`
- `temporal consistency check`
- `granularity check`
- `single retry rewrite`
- `fallback evidence gate`

两个原计划机制从 v1 删除：
- `direct-solve pruning`
  - 原因：触发条件太模糊，容易把多跳题误剪成单跳
- 强依赖生成式再压缩的 `canonical short answer compression`
  - 改成低风险的确定性 root-level 后处理，不作为方法核心贡献点

### 5. 关键机制定义

`type-aware validation`
- 每个 node 必须先预测 `answer_type`
- 若候选答案与预期类型不匹配，则该 node 判为 invalid，不进入上层组合

`temporal consistency check`
- 只在问题或 node 含明显时间锚点时触发
- 触发词包括：`first / last / current / before / after / abolished / founded / when / year`
- 若候选答案对应证据不含兼容时间线索，则判 invalid

`granularity check`
- 只对 location hierarchy 触发
- 重点限制：`county / city / province / state / country`
- 若问题要求更细粒度，但答案落在更粗层级，且证据中明确标出层级，则判 invalid

`single retry rewrite`
- 触发条件：
  - retrieval 返回空或弱证据
  - 或候选答案被任一 validator 判 invalid
  - 且该 node 尚未 rewrite 过
  - 且剩余 timeout 预算仍高于问题预算的 `25%`
- rewrite 次数最多一次
- rewrite 必须显式保留 `entity_anchor + answer_type`

`fallback evidence gate`
- 只有在至少一条证据里出现与 `answer_type` 相容的候选 span，且与问题 anchor 对齐时，才允许 fallback 输出答案
- 否则返回 `[none]`，不允许“无证据硬猜”

`deterministic root answer cleanup`
- 不是核心方法，只作为统一评测前后处理
- 仅做确定性规则：
  - 去掉 lead-in 话术
  - 去掉 “Note:” 之后的解释
  - 去掉额外并列解释子句
  - 对 date / number / entity questions 只保留首个证据对齐 span
- 不增加额外 LLM 调用

## Experiment Design

### 6. Phase 1: Frozen baseline runs

先完成这些 baseline 运行：
- MuSiQue-100
- 2Wiki dev20 + eval80
- Hotpot dev20 + eval80

输出：
- EM / F1
- unique-qid 去重结果
- wall-clock 成本表
- timeout / fallback / call-count summary

命名统一：
- 外部实验名用 `musique / 2wikimqa / hotpotqa`
- 若底层 raw/index 构建需要 `2wikimultihopqa`，则通过显式 alias map 转换
- 不允许在代码里混用未声明别名

### 7. Phase 2: Failure analysis and spec freeze

失败模式分析改为“自动预标 + 小规模人工审核”，不做人海手标。

自动部分：
- 为 dev pool 生成 error CSV
- 用规则预标以下标签：
  - type mismatch
  - temporal drift
  - granularity mismatch
  - unsupported fallback guess
  - long-answer / normalization issue
  - retrieval miss / unclear

人工审核量固定：
- MuSiQue：30 个错误样本
- 2Wiki dev20：最多 15 个错误样本
- Hotpot dev20：最多 15 个错误样本

Phase 2 完成标准：
- 已完成至少 `60` 个错误样本人工复核
- top 3 failure families 覆盖至少 `60%` 的非格式类错误
- adaptive 方法 spec 冻结
- 冻结后不再根据 held-out 结果改方法

### 8. Phase 3: Adaptive evaluation

adaptive 方法评测只在方法冻结后进行。

主比较：
- `Frozen Baseline`
- `Adaptive v1`

ablation 改成机制可解释的版本：
- `Adaptive v1`
- `Adaptive v1 - validators`
  - 去掉 type / temporal / granularity checks
- `Adaptive v1 - recovery`
  - 去掉 single retry rewrite + fallback evidence gate

不再做 `node typing only`。
如果 `answer_type` 只是日志字段、不参与决策，那它不构成独立 ablation 条件。

运行范围：
- full adaptive 与 frozen baseline 都跑：
  - MuSiQue-100
  - 2Wiki eval80
  - Hotpot eval80
- 两个 ablation 默认只跑：
  - MuSiQue-100
  - 2Wiki dev20
  - Hotpot dev20

### 9. 决策与扩展标准

删除原先 `EM +2.0 / F1 +1.5` 的硬门槛。

改为三类证据联合判断是否值得扩到全量：
- `paired delta` 为正
- `bootstrap CI` 或重复子样本重采样显示改进方向稳定
- 成本增幅在预算内且没有明显增加 timeout / unsupported fallback

扩全量的触发条件：
- 在 `2Wiki eval80` 和 `Hotpot eval80` 中，至少一个指标方向稳定为正，且没有数据集出现明显退化
- 改进主要来自真实错误减少，而不是纯 normalization 修补
- 预算经校准 run 外推后仍可接受

如果只在 MuSiQue 改善而 held-out 不稳定：
- 结论停留在 “MuSiQue-dev 有效，跨数据集泛化未证实”
- 不扩全量

## Test Plan

必须报告的指标：
- EM
- F1
- paired per-question delta
- bootstrap CI 或重复重采样区间
- wall-clock runtime
- retrieval / generation calls
- direct fallback rate
- validator rejection counts

必须单独展示的案例：
- temporal errors
- answer-type confusion
- county/city 或 province/country 粒度错误
- unsupported fallback guesses
- long-answer to short-answer cleanup cases

## Assumptions

- 仍然坚持成本优先，不追求立即全量三数据集
- `100` 题子集是 pilot 资源单元，不是最终显著性结论单元
- MuSiQue 是开发主场，不承担主要泛化结论
- 2Wiki 和 Hotpot 的 held-out 部分在 spec freeze 前不参与方法修改
- baseline 与 adaptive 的核心执行代码必须物理隔离，而不是逻辑上“尽量不改”
