# Adaptive Bad Case 阶段性总结

日期：2026-04-18

## 目的

本文档用于总结当前 `adaptive_typed_v1` 在 MuSiQue bad case 上的实验状态，明确现阶段结果支持什么、不支持什么。

这样做的目的，是在尝试下一种方法之前，先把当前阶段的中间结论固定下来，避免后续继续改方法时把现在的观察冲掉。

## 当前 baseline 背景

- MuSiQue 固定 100 题 baseline：`38%` EM
- MuSiQue baseline bad case：`62` 题
- 2WikiMQA 固定 100 题 baseline：`69.00%` EM / `79.26%` F1

相关 baseline 参考：
- [output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_report.md](/workspace/projects/rt-rag-repro/output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_report.md:1)
- [output/musique/dense_chunk200_topk1_25_topk2_8/bad_case_analysis.md](/workspace/projects/rt-rag-repro/output/musique/dense_chunk200_topk1_25_topk2_8/bad_case_analysis.md:1)
- [output/2wikimqa/dense_chunk200_topk1_25_topk2_8/run_summary.md](/workspace/projects/rt-rag-repro/output/2wikimqa/dense_chunk200_topk1_25_topk2_8/run_summary.md:1)

## Adaptive v1 的设计初衷

`adaptive_typed_v1` 的出发点，是针对 MuSiQue 错误分析里相对明确的三类错误：

- 答案类型混淆
- 时间锚点错误
- direct fallback 无依据乱猜

当前 v1 已实现的机制包括：

- typed query formulation
- suspicious-answer rewrite
- low-confidence 传播
- parent-level direct retrieval fallback
- fallback evidence gate

## 已完成的实验

### 1. 初始 bad-case pilot

初始 adaptive pilot 输出：
- [output/musique/adaptive_typed_v1/1.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/1.txt:1)

这轮 pilot 暴露了一个 query builder bug：
- `build_typed_retrieval_query()` 在部分 `answer_type + entity_anchor` 场景下会生成 double-question prompt

代表性样例：
- `a621901602313fe0bb3278547bfd2dc7a7eeb93905d04b78`

### 2. 修复后的单题重跑

修复后的单题重跑结果：
- [output/musique/adaptive_typed_v1/2.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/2.txt:1)

观察结论：
- double-question bug 的确被修掉了
- 但该 sanity-check 样例仍然失败，只是失败原因从 prompt 写坏，转成了 retrieval / fallback 层面的问题

### 3. 修复后的 12 题小规模重跑

修复后的小规模 bad-case 重跑结果：
- [output/musique/adaptive_typed_v1/3.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/3.txt:1)

子集规模：
- `12` 道代表性 bad case

观察结果：
- 相比 baseline bad-case 行为，`0/12` EM 提升

## 主要发现

### 1. double-question bug 的确存在，但不是当前主要瓶颈

这个 bug 值得修，因为它会污染一部分 typed-query 场景。

但修完之后的小规模重跑说明：

- 这个 bug 不是当前 hardest bad cases 失败的主导原因
- 即使移除它，很多题仍然主要失败在 retrieval miss、timeout 和 fallback 行为上

### 2. 机制层面有局部正信号，但没有转化成 EM 提升

目前能观察到一些有限的正信号：

- 至少有一道题出现了 `adaptive_rewrite_effective = 1`
- 至少有一道题从“完全错误实体”进步到了“部分正确实体字符串”

代表性例子：

- `6e64c9c62d8e1170c6341cde12aa2fe75a35762012ab8d30`
  rewrite 第一次真正起效了，但最终答案仍然不是 gold county
- `f4f04befbe7bb6e1ca4055ffd495356951166e116a15b3de`
  adaptive 输出了 `Philip`，而 baseline 是 `Camilla`。方向上更接近正确答案，但对 `Philip Mountbatten` 的 EM 仍然失败

这说明：

- adaptive control 不是完全空转
- 但当前收益过弱，或者停留在“部分命中”层面，还不足以稳定转成 exact match 提升

### 3. 当前主导失败模式已经比较清楚

在这轮重跑子集里，最重要的失败链路是：

`复杂多跳题 -> 搜索慢 / 超时或树求解退化 -> direct fallback -> evidence gate 拦截 candidate -> [none]`

这个模式在多道题里重复出现，已经可以看作当前 v1 的结构性限制。

### 4. evidence gate 在语义上合理，但对 EM 不友好

fallback gate 当前确实完成了它的设计目的：

- 拦截无依据答案
- 减少 unsupported guessing

但在 EM 评测下：

- `[none]` 和错误答案得分相同
- 所以 gate 往往提升了回答约束性，却没有带来指标收益

也就是说，当前 gate 更偏 precision-oriented，而课程评测更偏 strict exact match。

### 5. hardest bad cases 的主要问题，并不只是 query wording

这轮重跑说明，很多剩余失败更像是：

- retrieval miss
- 多跳中间实体选错
- 时间锚点漂移
- timeout 太早发生，adaptive 控制链路来不及发挥作用

这削弱了继续仅靠 prompt phrasing 微调当前 v1 的必要性。

## 12 题小样本的日志级归类

为了进一步判断 bad case 更像“evidence 缺失”还是“evidence 使用失败”，这里对修复后重跑的 12 题代表样本做了日志级别的人工归类。分类依据主要来自：

- debug JSON 中的 retrieval query
- top passage preview
- 是否触发 direct fallback
- 最终输出与 gold 的距离

这里采用四类工作性标签：

- A 类：evidence 基本够用，或已经非常接近 gold，但系统没有正确抽取/规范化
- B 类：evidence 只覆盖到“答案附近的信息”，但没有直接覆盖题目要求的精确槽位
- C 类：evidence 只覆盖到链路局部，或支持了一条错误/不完整链
- D 类：当前日志看起来没有召回到足够支持 gold 的关键 evidence

### 12 题归类表

| qid | 问题简述 | 预测答案 | gold | 归类 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `f4f04bef...` | current queen 的 spouse | `Philip` | `Philip Mountbatten` | A | retrieval 多次命中 `Elizabeth II`，答案已经从完全错误实体变成部分正确实体，更像答案规范化/完整性不足。 |
| `6e64c9c6...` | Zhu Qinan 出生于哪个 county | `Wenzhou` | `Yongjia County` | B | top passage 只明确给出 `Wenzhou, Zhejiang`，说明出生地信息被召回了，但粒度停在 city，没有直接支持 county。 |
| `6d57f1e8...` | Snappy Tomato Pizza 成立于哪个 county | `Campbell County` | `Kenton County` | B | retrieval 命中 `Fort Mitchell, Kentucky`，说明 founding location 已召回，但还需要额外地理映射才能落到正确 county。 |
| `096c9e28...` | Sean Hampton 出生地在 Florida 的哪里 | `Ocala` | `Northern Florida` | B | 输出是更具体的 city，而 gold 是区域级答案，说明 evidence 很可能覆盖到相关位置，但没有对齐题目要求的粒度。 |
| `a6219016...` | 谁继承了 Menucha Rochel Slonim 的父亲 | `[none]` | `Menachem Mendel Schneersohn` | C | retrieval 已命中 `Menucha Rochel Slonim` 和其父亲 `Rabbi Dovber Schneuri`，但没有形成支持 successor 关系的完整证据链。 |
| `2df2586b...` | Satyo Husodo 相关新国家的总统 | `[none]` | `Francisco Guterres` | C | 日志中已出现 `Indonesia` 与 `Commission of Truth and Friendship` 两个关键环节，但最终没有把链条稳定拼到目标总统。 |
| `d51c30bc...` | BBC Footballer of the Year 属于哪支队伍 | `[none]` | `Egypt national football team` | C | 结合原始错误分析，该题更像中间实体或关系链没有稳定走通，最终落入 fallback 而不是直接 evidence 空缺。 |
| `102eaee0...` | 与 Closer performer 一样从 adult contemporary radio 起步的艺人 | `[none]` | `Michael Bublé` | D | retrieval 结果主要是泛化的 adult contemporary 背景文本，没有看到支持 gold 的关键人物级证据。 |
| `efa29c58...` | 与 All That Echoes performer 一样从 adult contemporary radio 起步的艺人 | `[none]` | `Michael Bublé` | D | 与上一题同属一类，表现为 query 落在宽泛音乐背景文本，而不是目标人物级证据。 |
| `865a174e...` | FDA food safety system 的名称 | `[none]` | `Food Safety Modernization Act (FSMA)` | D | retrieval 命中 OTC / prescription 与 FDA regulation 相关文本，但没有看到直接把概念收敛到 `FSMA` 的强证据。 |
| `a3854b7b...` | Roncalli 为什么离开某地 | `[none]` | `for the conclave in Rome` | D | retrieval 同时打到 `Gozzi Altarpiece` 和 Roncalli 的背景信息，但没有在日志中看到支撑 gold reason 的关键关系句。 |
| `0da908a9...` | 某 explorer 到达总部地点的时间 | `[none]` | `August 3, 1769` | D | 该题 timeout 很重，且最终走到 fallback + gate blocked，更像关键时序证据没有稳定召回。 |

### 这 12 题的阶段性结论

从这 12 题的日志看，不能把 bad case 简单概括成“都没有检索到 gold evidence”。

更准确的情况是：

- 少数题已经非常接近成功，主要死在答案抽取或规范化上
- 有一批题已经召回了“答案附近”的信息，但没有精确对齐题目要求的槽位或粒度
- 还有一批题只召回了链路中的局部环节，没能形成完整多跳链
- 只有部分题更像“当前日志里确实没有看到足够支持 gold 的关键 evidence”

因此，这批 hardest bad cases 更像是：

`evidence 缺失`、`evidence 只覆盖到相邻答案`、`链路中间子问题出错`、`答案抽取与规范化失败`

这几类问题共同存在，而不是单一的 retrieval miss。

## 当前阶段结论

截至目前，`adaptive_typed_v1` 的结论可以概括为：

- 机制层面存在一些有用信号
- 但在修复 bug 后的代表性 bad-case 子集上，尚未观察到可证明的 EM 提升

因此目前最公平的表述是：

`adaptive_typed_v1` 作为一种控制机制思路是合理的，但在当前 retrieval、timeout 和 evidence-gating 设定下，它还没有在最难的 MuSiQue bad case 上转化成可测量的 exact-match 增益。

## 轻量版配置补充说明

在后续运行中，又引入了一个更轻量的 adaptive 配置，用于解决 hardest bad cases 上运行时间过长的问题。

该配置可以视为：

- `adaptive_typed_v1_single_attempt`

核心变化是：

- 保留 `TREES_PER_QUESTION = 2`
- 将 `MAX_VARIANTS = 0`

这意味着：

- 每道题只允许单轮顶层 attempt
- 最多只生成两棵主树
- 不再进入 variant 扩展后的第二轮建树与求解

这样做的主要动机不是提升方法能力，而是：

- 压缩运行时间
- 避免顶层 repeated attempt/variant 把预算大量耗尽
- 让完整 bad-case rerun 在现实时间内可完成

因此，后续若使用该配置得到新结果，应视为一个单独的实验设置，而不是与早期 `adaptive_typed_v1` 结果直接混合。

建议后续分析时明确区分：

- 原始/较完整的 `adaptive_typed_v1`
- 运行时收缩后的 `adaptive_typed_v1_single_attempt`

## `adaptive_typed_v1_single_attempt` 完整 62 题 bad-case 结果

在完成轻量版配置定义后，又对全部 `62` 道 MuSiQue bad cases 跑了一轮完整实验：

- 输出文件：
  [output/musique/adaptive_typed_v1_single_attempt/1.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1_single_attempt/1.txt:1)

该版本配置要点：

- `TREES_PER_QUESTION = 2`
- `MAX_VARIANTS = 0`
- 也就是每题只允许单轮 attempt，最多生成两棵主树

### 结果概览

相对于 baseline bad cases，这轮结果是：

- `improved = 0`
- `regressed = 0`
- `same_wrong = 62`

也就是说：

**62/62 全部仍然是错误，没有任何一题从 baseline bad case 变成正确。**

### 行为模式

从结果文件统计看，这一版方法在 bad cases 上的主导行为已经很明确：

- `57/62` 题触发了 `used_direct_fallback = True`
- `56/62` 题最终输出为 `[none]`
- `56/62` 题触发了 `adaptive_fallback_gate_blocked_count > 0`

这说明当前系统在 hardest bad cases 上，主要不是“把错误答案修正成正确答案”，而是在走：

`求解失败 -> direct fallback -> evidence gate 拦截 -> [none]`

也就是说，它更像是在提升拒答倾向，而不是提升 exact match。

### 机制信号

本轮统计还显示：

- `adaptive_rewrite_triggered = 15`
- `adaptive_rewrite_effective = 0`
- `adaptive_parent_direct_fallback = 8`

其中最关键的是：

- `rewrite_effective = 0`

这说明 rewrite 虽然有触发，但**没有一题通过 rewrite 真正被修正成功**。

因此，在这批 hardest bad cases 上，当前 adaptive typed-control 的核心机制并没有转化成可见收益。

### 按错误类型观察

将本轮结果与 baseline 错误类型对照后，可以看到：

- `multi_hop_intermediate_entity_error`：`16/16` 仍然输出 `[none]`
- `temporal_anchor_error`：`13` 题中 `12` 题输出 `[none]`
- `answer_type_confusion`：`8` 题中 `7` 题输出 `[none]`
- `geographic_hierarchy_relation_error`：`8` 题中 `5` 题输出 `[none]`
- `overgenerated_noncanonical_answer`：`5/5` 输出 `[none]`

这意味着，连原本理论上最该受益的：

- 时间锚点错误
- 答案类型混淆

也没有展现出稳定修复效果。

### 运行时间观察

虽然这版已经通过 `MAX_VARIANTS = 0` 压缩了顶层 repeated attempt 的开销，但完整 62 题运行时仍然偏重：

- 平均 `timing_total_seconds ≈ 6731s`
- 中位数 `≈ 4398s`
- 最大值 `≈ 18257s`
- `44/62` 题总耗时仍然超过 `600s`

另外：

- `35/62` 题被标记为 `timeout_triggered = True`
- 只有 `7` 道题出现了 `adaptive_timeout_cutoff_nodes > 0`
- 总共记录了 `11` 个 timeout-cutoff 节点

这说明：

- 运行时问题虽然有所缓解，但并没有真正解决
- node-level timeout cutoff 机制不是当前主要止损来源
- 主要耗时瓶颈仍然不只是在 `solve_node` 内部

### 阶段性结论

这一轮 `adaptive_typed_v1_single_attempt` 的完整 62 题结果，应被视为一个明确的负结果：

- 没有 EM 改善
- rewrite 没有真正起效
- 大量题目退化为 `[none]`
- fallback gate 成了主导行为
- 运行时间仍然偏高

因此，对 MuSiQue hardest bad cases 来说，可以得出更明确的结论：

> 当前这条 adaptive typed-control 路线，在最难的 MuSiQue bad cases 上没有展现出可用的修复能力；它更多表现为提高拒答倾向，而没有把错误答案稳定转化成正确答案。

这也意味着，继续在这个 v1 / single-attempt v1 上做小修小补的边际收益已经非常有限。

## 报告写法建议

对课程报告来说，这一阶段实验仍然有价值，但更适合被表述为：

- 一次有针对性的机制验证
- 一个 mixed / negative result
- 一次关于“为什么直觉上合理的修复没有转成 EM 提升”的分析

建议报告重点放在：

- 方法原本想对准哪些错误类型
- 哪些内部信号说明它并非完全无效
- 为什么这些局部信号最终没有转成 end metric 提升
- 这件事说明了真正瓶颈更可能在 retrieval coverage、多跳稳健性和 timeout 行为

## 下一步值得尝试的方法

下一阶段不建议继续对当前 gate 行为做零碎补丁。

更值得尝试的是以下几个方向：

### 方案一：gate 失败后的 retrieval-first rescue path

思路：
- 当 direct fallback 被 evidence gate 拦截后，不是立刻返回 `[none]`
- 而是触发一次受控的额外 retrieval，使用明显不同的 query formulation
- 最终仍然要求证据支持后才能返回答案

为什么值得试：

- 保持了“证据优先”的方法语义
- 直接针对当前最明显的 `gate -> [none]` 死路

主要风险：

- 运行时间可能继续上升
- 如果 index / top-k 本身拿不到 gold evidence，这条路也可能无效

### 方案二：root answer cleanup / canonicalization

思路：
- 只在 root answer 层做确定性后处理
- 压缩冗长实体答案
- 对少量常见短实体输出做安全规范化

为什么值得试：

- 风险低
- 成本小
- 有机会把一些 near miss 转成 EM 命中

主要风险：

- ceiling 很有限
- 无法解决 retrieval miss

### 方案三：把重点从 control 转向 retrieval 干预

思路：
- 承认当前主要瓶颈更像 retrieval，而不是 answer validation
- 可尝试方向包括：
  - 调整 top-k 策略
  - 对困难 node 做 multi-query retrieval
  - 将 rewrite 更明确地设计成 evidence recall 导向，而不是仅仅类型引导

为什么值得试：

- 更贴近当前 bad case 的主导失败模式

主要风险：

- 工程范围更大
- 运行成本和对比公平性会更难控制

## 实际建议

建议的下一步顺序如下：

1. 先把本文档作为当前 adaptive v1 阶段的冻结总结
2. 不再对 `adaptive_typed_v1` 做零碎修补
3. 下一阶段只选一个新方向尝试
4. 优先考虑：
   - `gate 失败后的 retrieval-first rescue path`
   - 或 `root answer cleanup / canonicalization`
5. 新方法仍然先在一个小规模代表性 bad-case 子集上验证，再决定是否扩大实验

## 相关文件

- [output/musique/adaptive_typed_v1/1.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/1.txt:1)
- [output/musique/adaptive_typed_v1/2.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/2.txt:1)
- [output/musique/adaptive_typed_v1/3.txt](/workspace/projects/rt-rag-repro/output/musique/adaptive_typed_v1/3.txt:1)
- [output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_question_summary.csv](/workspace/projects/rt-rag-repro/output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_question_summary.csv:1)
