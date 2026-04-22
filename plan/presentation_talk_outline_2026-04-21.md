# Poster Presentation 讲稿提纲

日期：2026-04-22  
对应最终结果：`plan/results_adaptive_typed_v1_final_2026-04-21.md`

## 0. 这份文档的定位

这份提纲的目标是把项目的**最终可讲故事线**固定下来，方便后面直接做 poster / slide 和口头 presentation。它不是完整实验报告，而是把以下三件事讲清楚：

1. 我们为什么会提出这条 adaptive 方法线
2. 方法是如何通过实验一步步收敛的
3. 最终结果到底支持什么、不支持什么

当前汇报默认使用的最终方法版本是 `adaptive_typed_v1_final`。后续如无特别说明，所有数字都以 2026-04-21 汇总结果为准。

---

## 1. 一句话先讲清楚这个项目

我们在 **不改 RT-RAG 主体框架** 的前提下，尝试加入一套轻量的 adaptive typed-control 机制，看看它能不能修复一部分 hardest multi-hop failures，同时尽量不破坏原本已经答对的题。

更短的版本可以直接讲成：

> We study whether lightweight adaptive control can help RT-RAG recover some hard multi-hop failures without redesigning the whole pipeline.

---

## 2. 任务背景与 baseline

本项目主要看两个固定的 100 题子集：

- MuSiQue-100：baseline `38/100` EM
- 2WikiMQA-100：baseline `69/100` EM，`79.26%` F1

MuSiQue 是主分析对象，因为它更难，也更能暴露 RT-RAG 在复杂多跳问答上的典型失败模式。在 MuSiQue fixed-100 中：

- baseline 答对 `38` 题
- baseline 答错 `62` 题

所以后续故事线的主线很自然：

- 先分析这 `62` 个 hardest bad cases 到底为什么错
- 再看轻量 adaptive 方法能不能修掉其中一部分
- 最后检查这种修复会不会伤到 baseline 原本已经答对的 easy cases

---

## 3. 为什么会想到做 adaptive typed-control

方法不是先拍脑袋设计出来的，而是从 baseline 错误分析里推出来的。

MuSiQue baseline 的主要错误类型包括：

- 多跳中间实体选错
- 时间锚点错误
- 答案类型混淆
- 地理层级 / 关系层级错误
- direct fallback 猜错，或者正确答案被 gate 拦截

这里有一个关键判断：

- 有些错误需要更强的 retrieval / reasoning 才能解决
- 但也有一部分错误更像是 query、answer type 和 fallback control 没有对齐

因此，这条方法线一开始瞄准的是三类**相对可操作**的问题：

- 答案类型混淆
- 时间锚点偏移
- direct fallback 无依据乱猜

所以这项工作的定位从一开始就不是“重做一个新系统”，而是：

> 在现有 RT-RAG 流程上加入少量、可解释的控制逻辑，验证能否带来可信的局部收益。

---

## 4. 方法是怎么收敛出来的

这一部分建议按时间顺序讲，因为它最能体现项目是如何从“有想法”走到“有结论”的。

### 阶段 1：初始方案 `adaptive_typed_v1`

最初的设计包括：

- typed query formulation
- suspicious-answer rewrite
- low-confidence propagation
- parent-level direct retrieval fallback
- fallback evidence gate

最初假设是：如果 retrieval query 和答案类型更对齐，再加上对 fallback 的轻量控制，应该能修掉一部分类型错误、时间错误和 fallback 乱猜。

### 阶段 2：pilot 和 12 题小样本没有出现可观测 EM 提升

早期实验先暴露出一个 query builder bug：某些 `answer_type + entity_anchor` 组合会生成 double-question prompt。这个 bug 修掉之后，单题 sanity check 的确变正常了，但小规模重跑仍然没有带来明显 EM 改善。

最重要的结果是：

- 修复 bug 后的 12 题代表性 bad-case 子集：`0/12` EM 提升

这一步非常重要，因为它说明：

- adaptive control 不是完全空转，局部日志里有正信号
- 但这些信号还不足以稳定转成最终 exact match 提升

也就是说，这一轮实验让我们意识到：问题并不只是 prompt phrasing 不够好。

### 阶段 3：bad-case 诊断暴露出真正瓶颈

后续诊断把失败模式梳理得更清楚了，最重要的三个点是：

- **leaf node retrieval fail**
- **timeout / runtime chain instability**
- **fallback gate 和 `direct_answer` 之间的错配**

这里 timeout 需要讲清楚，但不要讲得太长。可以概括为：

> baseline 的 per-question timeout 机制并不能真正中断已发出的长调用，所以它常常只是把求解过程截断，最后留下 `[none]`，并没有真正控制住 wall-clock 开销。

这个阶段是整个故事线的转折点。因为从这里开始，项目的重点不再是继续微调 wording，而是判断：

- 哪些 failure mode 值得继续修
- 哪些修复真的能转成 EM 收益

### 阶段 4：方法冻结为 `adaptive_typed_v1_final`

在完成 gate 改进、多查询检索尝试和 category guidance 补强之后，方法最终冻结为 `adaptive_typed_v1_final`。

这里 presentation 里一定要强调两点：

- 它是当前阶段的**最终汇报版本**
- 它不是“最优方法”，而是“在时间和证据约束下最值得汇报的稳定版本”

---

## 5. 最终方法到底包含什么

如果做 poster 或 slide，不需要把所有机制都铺开讲。只需要讲最重要、最能和结果对上的四点。

### 5.1 Typed guidance

系统在树分解求解过程中利用答案类型和实体锚点，构造更聚焦的 query。它的作用不是硬规则判错，而是给 retrieval 和 answer generation 提供方向。

现场最好直接讲这句：

> typed guidance is a soft steering signal, not a hard validator.

### 5.2 Candidate-aware gate 改进

后期实验确认，旧 gate 会误拦截一部分本来正确的 `direct_answer`。因此 final 版本保留了两点改动：

- gate 检索时把 candidate answer 也纳入 query hint
- gate 判断改成“有明确矛盾才拦截”，而不是“证据不充分就拦截”

从现有结果看，**最明确、最可信的 bad-case 收益主要来自这一部分**。

### 5.3 Multi-query fusion 与 category guidance

这两部分的动机分别是：

- multi-query fusion：减少 leaf node 因单个 query 偏掉而直接 `[none]`
- category guidance：避免把“类别问题”答成实例列表

就目前数据来说，这两部分更适合讲成：

- 它们有合理动机
- 日志层面有正信号
- 但还不能声称它们已经带来了大规模、稳定的 EM 提升

### 5.4 Timeout 关闭带来的稳定性改善

final 版本使用 `QUESTION_TIMEOUT_SECONDS = 99999`，目的是避免 baseline 那种“树还没跑完就被截断”的情况。

这里表达要非常小心：

- 这项改动的主要收益是**稳定性**
- 不是算法更快，也不是能力更强

如果老师追问“为什么 adaptive 更快”，标准回答应是：

> raw runtime 变快主要是因为 final adaptive 使用 `MAX_VARIANTS = 0`，计算量更少；不是 timeout 关闭本身带来了加速。

---

## 6. 最终结果该怎么讲

这一部分一定要把“局部收益”和“整体结果”同时讲出来，不能只讲其中一半。

### 6.1 MuSiQue hardest bad cases：有小幅但真实的修复

- 评测对象：MuSiQue bad-62
- baseline：`0/62`
- `adaptive_typed_v1_final`：`4/62`
- 净增：`+4`

最合适的解读方式是：

> final 方法在 hardest bad cases 上修复了少量但真实的失败案例，说明这条 adaptive 方向不是无效的。

但一定不要把它讲成“整体能力显著提升”。

### 6.2 MuSiQue easy cases：出现了可解释的回退

- Fixed-38：baseline `38/38` → adaptive `32/38`
- 净变化：`-6`

这些回退主要不是随机噪声，而是有比较一致的模式：

- 答案过宽
- 输出格式不符合 gold 的短答案形式
- 实体混淆
- type rewrite / multi-query 改变了原本正确题的检索方向

这部分一定要诚实讲，因为它正是最终结论成立的基础。

### 6.3 MuSiQue 整体：局部修复没有抵消 easy-case 回退

把 bad-62 和 fixed-38 合起来看：

- Fixed-100 合计：baseline `38/100` → adaptive `36/100`
- 净变化：`-2`

完整重跑结果也一致支持这个判断：

- MuSiQue-100：baseline `41/100` → adaptive `38/100`
- 净变化：`-3`

如果看**公平对比**（两边都 `MAX_VARIANTS=0`），MuSiQue-100 的结果是：

- EM：baseline `40/100` → adaptive `38/100`，净 `-2`
- F1：baseline **49.21%** → adaptive **48.77%**，净 `-0.44%`

也就是说，在统一计算预算之后，EM 和 F1 都是**小幅低于 baseline**，但差距不大。这和 bad-case `+4` 的结果放在一起，最合理的解读是：

- adaptive 在 hard cases 上有局部真实收益
- 但整体上仍然没有超过 baseline
- easy cases 上的负面干预抵消了这部分收益

所以目前最准确的表述不是”提升了整体 EM”，而是：

> 我们看到了 hard-case 的局部正收益，但整体上还没有超过 baseline。

### 6.4 2WikiMQA：跨数据集基本持平

| 指标 | Baseline | Adaptive | 变化 |
|---|---|---|---|
| EM | 69/100 (69%) | 68/100 (68%) | -1 |
| F1 | 79.26% | 76.52% | -2.74% |

它的作用更像 cross-dataset sanity check，而不是主故事线的中心结果。比较稳妥的讲法是：

> 在 2WikiMQA 上，EM 基本持平，但 F1 有小幅下降，因此这组结果更适合被表述为“跨数据集表现尚可”，而不是明显提升。

---

## 7. 公平对比和 runtime 应该怎么表述

这是 presentation 里最容易被追问的地方，必须统一口径。

### 7.1 什么是 raw 对比

raw 对比是：

- baseline 默认 `MAX_VARIANTS = 1`
- adaptive final 使用 `MAX_VARIANTS = 0`

在这个口径下，adaptive 的 raw runtime 更短，但这**主要来自计算量减少**，不能被表述为“算法本身更高效”。

### 7.2 什么是公平对比

公平对比是让两边都使用 `MAX_VARIANTS = 0`：

| 数据集 | Baseline EM | Adaptive EM | EM 变化 | Baseline F1 | Adaptive F1 | F1 变化 |
|---|---|---|---|---|---|---|
| MuSiQue-100 | 40% | 38% | **-2%** | 49.21% | 48.77% | -0.44% |
| 2WikiMQA-100 | 71% | 68% | **-3%** | 80.02% | 76.52% | -3.50% |

这一组结果的意义非常关键：

> 当计算预算被控制到一致时，adaptive 仍然没有超过 baseline。  
> 这说明目前的问题不是“baseline 算得更多”，而是 adaptive 机制本身对 easy cases 存在负面干预。

### 7.3 runtime 最稳妥的说法

最稳妥的 runtime 表述建议固定成下面这三句：

1. adaptive final 做到了 `0 timeout`、`0 retry`，运行更稳定
2. raw runtime 更短主要因为 `MAX_VARIANTS = 0`，不是算法加速
3. 在同预算下，baseline 和 adaptive 的平均耗时接近，adaptive 没有明显的速度优势，但稳定性更好

---

## 8. 最终 takeaway 应该怎么落

如果只保留最核心的四条结论，我建议固定成下面这版。

### Takeaway 1

这条 adaptive 方法线是**局部有效且可解释的**，不是一个已经验证成功的通用提分方案。

### Takeaway 2

目前最可信的收益来自 **candidate-aware gate 改进**，它确实修复了一小部分 hardest bad cases。

### Takeaway 3

当前主要剩余瓶颈仍然是 **retrieval** 和 **runtime chain stability**，而不是继续堆更多 control logic。

### Takeaway 4

这套方法目前缺少“何时该触发 adaptive、何时不要干预”的判断，因此会对 easy cases 产生过度干预。

如果需要一句最核心的 summary，我建议使用：

> We extend RT-RAG with a lightweight adaptive typed-control strategy. The method recovers a few hardest MuSiQue failures, but the overall results show that the dominant remaining bottlenecks are still retrieval quality and runtime stability.

---

## 9. Poster / Slide 最推荐的讲法

### 9.1 1 分钟版本

1. 我们先做了 RT-RAG baseline，MuSiQue-100 只有 `38%` EM，所以我重点分析了其中 `62` 个 hardest bad cases。
2. 错误分析发现，除了 retrieval 问题外，还有一些错误来自答案类型、fallback gate 和 query direction 不对齐。
3. 所以我设计了一个轻量的 adaptive typed-control 方法，不改主体框架，只在 query guidance、gate 和 leaf retrieval 上做补强。
4. 最终方法在 MuSiQue bad-62 上从 `0/62` 提升到 `4/62`，说明 hard cases 上有真实但有限的收益。
5. 但它也让 fixed-38 从 `38/38` 降到 `32/38`，所以整体并没有超过 baseline。结论是：这条方向有局部价值，但主要瓶颈仍然是 retrieval 和 runtime stability。

### 9.2 2 分钟版本

1. baseline 阶段我先跑了 MuSiQue 和 2WikiMQA，其中 MuSiQue 更难，所以被选为主分析对象。
2. 基于错误分析，我提出 `adaptive_typed_v1`，目标是修答案类型错误、时间锚点错误和无依据 fallback。
3. 但 pilot 和 12 题小样本没有带来 EM 提升，所以我转去做 bad-case diagnosis。
4. 诊断结果显示，真正的主要问题是 leaf retrieval fail、runtime timeout / instability，以及 fallback gate 和 `direct_answer` 的错配。
5. 基于这些发现，方法收敛为 `adaptive_typed_v1_final`，保留 typed guidance、candidate-aware gate 改进、multi-query fusion 和 category guidance。
6. 最终结果是：MuSiQue bad-62 `0/62 -> 4/62`，Fixed-38 `38/38 -> 32/38`，MuSiQue 整体 `38 -> 36`，2WikiMQA `69 -> 68`。
7. 所以最准确的结论不是“整体提分”，而是“hard cases 有小幅真实改善，但 adaptive 仍会过度干预 easy cases，主瓶颈依然在 retrieval 和 runtime stability”。

---

## 10. 最适合直接做成 slide 的结构

如果后面要做 5 到 7 页 slide，我建议直接按下面结构来拆：

1. Problem & Motivation  
   RT-RAG baseline 在 MuSiQue 上只有 `38%` EM，hardest bad cases 值得单独分析

2. Error Analysis  
   展示 baseline 的主要错误类型，说明为什么会想到做 lightweight adaptive control

3. Method  
   只讲 3 到 4 个关键模块：typed guidance、candidate-aware gate、multi-query fusion、category guidance

4. Experiment Timeline  
   baseline -> early pilot -> diagnosis -> final method

5. Main Results  
   一张核心表：Bad-62 `+4`，Fixed-38 `-6`，Fixed-100 `-2`，2Wiki `-1`

6. Failure Analysis & Takeaways  
   为什么会回退；为什么结论指向 retrieval / runtime，而不是继续堆 control logic

7. Future Work  
   如果要继续做，重点应该是 trigger policy、better retrieval、减少对 easy cases 的干预

---

## 11. 汇报时不要踩的坑

这部分可以当作 presentation 的口径检查表。

- 不要把 bad-62 的 `+4` 讲成整体系统提升
- 不要把 raw runtime 更短讲成算法更快；主要原因是 `MAX_VARIANTS=0`
- 不要把 multi-query / category guidance 讲成“已被充分证明有效”；目前证据更偏行为正信号
- 一定要主动承认 easy-case 回退，因为这是整个故事最关键的诚实部分
- 如果被问“那这个方法到底值不值得做”，最好的回答是：**值，因为它帮助我们确认了哪类 adaptive control 有局部价值，也更明确地暴露出真正主瓶颈仍在 retrieval 和 runtime**

---

## 12. 当前文档的用途

这份提纲现在最适合用于三件事：

1. 直接准备 poster / slide 的讲述顺序
2. 统一 presentation 时的数字口径
3. 作为最终项目总结的讲稿底稿

如果后面继续细化，我建议优先补两样材料：

1. 一张最核心结果表  
   Bad-62 `+4`、Fixed-38 `-6`、Fixed-100 `-2`、2Wiki `-1`

2. 一张 failure mode / regression type 图  
   说明为什么它修了 hard cases，但也伤到了 easy cases

---

## 13. 根据课程要求，poster 应该怎么落版

现在已经看到了课程给的 poster 要求，里面最关键的约束其实很简单：

- poster 主要回答三件事：`problem` / `method` / `results`
- 推荐使用三栏布局
- 左栏讲 1， 中栏讲 2，右栏讲 3
- 中间栏可以更宽

这和我们现在的故事线是匹配的，所以不需要再发明新的结构，直接把现有内容压缩映射过去就可以。

### 13.1 最推荐的三栏结构

#### 左栏：Problem

这一栏只回答两个问题：

1. 我们在解决什么问题？
2. 为什么这个问题值得做？

建议放的内容：

- 一句任务定义  
  `We study whether lightweight adaptive control can help RT-RAG recover hard multi-hop QA failures.`
- baseline 背景  
  MuSiQue-100 baseline 只有 `38%` EM，因此我们重点分析 hardest bad cases
- 一张简单的 error analysis 图或表  
  展示 baseline 的主要错误类型
- 一句 motivation  
  有些错误来自 retrieval / reasoning 本身，但也有一部分来自 type、query 和 fallback control 没有对齐

这一栏的核心目标不是讲方法细节，而是让观众在 10 秒内明白：

> baseline 在 hardest multi-hop cases 上不够稳，所以我们尝试做 lightweight adaptive control。

#### 中栏：Method

这一栏是海报的主体，应该最宽。

建议只保留 3 到 4 个模块，不要把所有实现细节都塞上去：

- Typed guidance
- Candidate-aware gate
- Multi-query fusion
- Category guidance

最推荐的呈现方式不是长段落，而是：

- 一张 RT-RAG 原流程图
- 用高亮框标出你们改动的几个位置
- 每个模块配一句非常短的话

例如：

- `Typed guidance`: steer retrieval toward the expected answer type
- `Candidate-aware gate`: avoid blocking correct fallback answers
- `Multi-query fusion`: reduce leaf-node retrieval miss
- `Category guidance`: answer type/category questions more directly

如果版面不够，优先保留：

1. typed guidance
2. candidate-aware gate
3. 一个总流程图

### 13.2 右栏：Results

这一栏只讲结论，不讲实现细节。

建议分成三个小块：

1. Main result table  
   Bad-62 `+4`，Fixed-38 `-6`，Fixed-100 `-2`，2Wiki `-1`

2. Failure / regression analysis  
   简要说明为什么会出现回退：  
   type rewrite / multi-query 有时会改变原本正确题的检索方向

3. Final takeaway  
   hard cases 有小幅真实收益，但整体瓶颈仍然是 retrieval 和 runtime stability

这一栏最重要的是诚实和清楚。不要只放 `+4`，一定要同时放 `-6` 和整体 `-2`，因为这是整个项目结论成立的关键。

---

## 14. 按课程要求，现场讲解应该怎么配合 poster

课程要求的目标之一是练习解释你们做过的工作，所以讲法要尽量贴着海报三栏走，不要变成口头念论文。

最自然的讲法顺序就是：

1. 左栏先讲问题  
   baseline 在 MuSiQue hardest cases 上不够好，所以我们先做错误分析

2. 中栏讲方法  
   我们没有重做 RT-RAG，而是在几个关键节点上加入 lightweight adaptive control

3. 右栏讲结果  
   hard cases 有少量修复，但 easy cases 出现回退，所以整体还没有超过 baseline

这样讲的好处是：

- 完全符合老师给的 `problem -> method -> results` 结构
- 观众看海报和听你讲是同步的
- 也最容易在短时间内讲清楚这项工作真正支持什么结论

---

## 15. 现在最建议直接放上海报的内容

如果要从现有材料里直接挑最值得上 poster 的内容，我建议固定成下面这些。

### 必放

- Problem statement
- MuSiQue baseline `38/100` EM
- 一张错误类型分布图
- 方法流程图
- 核心结果表：Bad-62 `+4`，Fixed-38 `-6`，Fixed-100 `-2`，2Wiki `-1`
- 一句 final takeaway

### 能放更好

- 4 个修复成功的 bad cases 里挑 1 到 2 个做 case study
- 6 个回退题里归纳 2 到 3 类 regression pattern
- 一句 future work：need better trigger policy / retrieval improvement

### 不建议占太多版面

- 过多早期 run 编号
- 太细的 debug 日志
- 过细的 runtime 数字表
- 过多 implementation-level code detail

海报不是实验日志墙。最重要的是让别人一眼看懂：

1. 问题是什么
2. 你改了哪里
3. 结果支持什么结论
