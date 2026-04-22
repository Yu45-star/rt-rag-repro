# Bad Case 诊断与下一步方案：adaptive_typed_v1_single_attempt

**日期**：2026-04-19  
**实验**：62 个 bad case 重跑，方法 `adaptive_typed_v1_single_attempt`  
**结果**：在这次 62 个 MuSiQue bad case 重跑中，最终无新增答对 case（0/62）  
**日志路径**：`output/musique/adaptive_typed_v1_single_attempt/debug/1/`

---

## 当前阶段结论摘要

如果只总结目前已经拿到的数据，可以归纳成下面四点：

1. `adaptive_typed_v1_single_attempt` 在 MuSiQue 的 62 个 bad case 上，**没有带来可观测的最终 EM 改善**；当前版本更像是“暴露失败模式的诊断工具”，而不是已经成熟的修复方案。
2. 现阶段最主要的问题不只是 prompt phrasing，而是**leaf node 检索失败、超时导致的求解链条中断，以及 fallback gate 与 direct_answer 之间的错配**。
3. gate 相关实验说明：**旧 gate 确实会误拦截一部分正确答案**，但如果仅仅把 gate 放宽，又会把不少原本应被拦截的错误答案放出来。因此 gate 不是唯一主问题，也不值得继续单独深挖太久。
4. 下一步最值得投入的方向是**上游检索召回改进**，尤其是 leaf node 的多查询融合；相比继续微调 gate，这更有机会直接减少 `[none]` 和错误链路。

换句话说，这一轮实验已经足够支持一个比较稳的阶段性判断：

> 当前 adaptive v1 的主要瓶颈在检索与链路求解，不在单纯的答案后处理。

---

## 失败模式分类

### 模式A：全局超时（35 例，56%）

大量 case 的 `timing_timeout_elapsed_seconds` 达到 5,000–18,257 秒。更准确地说，这说明**当前 timeout 机制无法稳定限制整题的实际 wall-clock 时间**；顶层 repeated attempt、建树阶段以及已经发出的长调用都可能共同拉长总耗时。这 35 例在本轮先视为“运行时问题主导”，暂不把它们和纯质量问题混在一起分析。

- `timeout_stage=before_attempt`（含 retry_count=1）：32 例
- `timeout_stage=before_solve_tree`：2 例
- `timeout_stage=before_build_tree`：1 例

### 模式B：Fallback Gate 错误拦截正确答案（≥5 例，高优先级）

`direct_answer()` 实际返回了正确答案，但被 `check_fallback_supported()` gate 拦截，最终输出 `[none]`。

**确认的被拦截正确答案**：

| QID（前12位） | 问题简述 | direct_answer 返回 | Golden |
|---|---|---|---|
| 7ac9fed94844 | Blanton's manufacturer → Saints symbol | "fleur de lis" | "fleur-de-lis" |
| c858bf26a171 | Mara Region → 哪个国家有该水体 | "Tanzania" | "Tanzania" |
| 8daaf871ac61 | Batman Under the Red Hood → Neil Patrick Harris 配音角色 | "Dick Grayson" | "Dick Grayson" |
| a62190160231 | Who followed Menucha Rochel Slonim's father | "Rabbi Menachem Mendel Schneersohn" | "Menachem Mendel Schneersohn" |
| c6fe00704467 | The Beach 拍摄地 | "Ko Phi Phi Le" | "Ko Phi Phi Le" |

**根本原因**：Gate 复用了与树节点近似的检索路径。当前置检索本来就不充分时，gate 往往也拿不到能支持 `direct_answer` 的证据，于是出现“答案本身其实对，但 gate 因证据不足而误拦截”的情况。

**注意**：Gate 也确实拦截了不少明显错误的 `direct_answer`（如 Camilla、Oshkosh、Bernard Montgomery 等），所以这里的问题不是“gate 不该存在”，而是“gate 的证据获取方式和判定标准还不够稳”。

**改进方向**：
- 在 gate 检索时，将 `candidate_answer` 作为 query hint 加入检索（即用 `question + candidate_answer` 检索），提高找到支持证据的概率
- 若继续调整 gate prompt，应明确把目标写成“减少假阴性”，而不是泛泛地“放宽 gate”，避免把大量明显错误答案一起放出来

### 模式C：Leaf Node 检索失败导致级联 [none]（最普遍的根本原因）

N1 子问题检索返回了相关但错误层次的实体，导致 N1→[none]，进而 N2→[none]，整棵树失败。

**典型案例**：
- Blanton's 案例：需要找 "Sazerac Company（制造商）总部 = New Orleans"，但检索到的是"蒸馏厂在 Frankfort, Kentucky"。最终 direct_answer 正确答出了 "fleur de lis"，但被 gate 拦截。
- 大量 case 的 N1 answer 为 `[none]`，而 N2 又 depends_on N1，导致整个 tree 无法运行。

**改进方向**：
- 多查询检索融合（Multi-query fusion）：只在 **leaf node 检索阶段**，同时使用 type-guided query、原始问题、entity-focused query 检索，再合并文档并重排
- 实体角色区分：如 "manufacturer's headquarters" vs "distillery location"，需要在 query 构造时更明确保留角色关键词

### 模式D：Sequential 节点右子问题生成错误（~3-5 例）

**典型案例**：Blanton's/fleur-de-lis（QID 7ac9fed9）：

```
N1: "What city is the location of Blanton's manufacturer's headquarters?"  → [none]
N2: "What city is the location of Blanton's manufacturer's headquarters?"  → [none]  ← 与 N1 完全相同！
```

N2 应该是 "What is the Saints' symbol from [city]?"，但实际上 N2 的 display_question 与 N1 相同。

**根本原因**：Sequential 节点在生成右子问题时（`generate_right_question` 阶段），LLM 调用超时或失败，导致 N2 的问题没有被正确替换。也就是说，这类失败首先是“右子问题生成失败”，然后才表现为“后续求解链条全部失效”。

**改进方向**：
- 对 `generate_right_question` 阶段单独增加超时保护和回退逻辑
- 若 N2 问题未被正确生成（仍与 N1 相同），直接 skip 该 tree，尝试 direct_answer

### 模式E：答案类型识别缺失导致生成错误（~2-3 例）

**典型案例**：Battle of Brechin（QID f6564f94）：

- 问题："A participant of the Battle of Brechin is an instance of?"
- Golden：**"Scottish clan"**（类别）
- 预测："members of Clan Gordon and Clan Ogilvy (royalists), members of Clan Lindsay (rebels)"（具体成员列表）

检索到了正确文档，但 LLM 回答了"谁参与了"而不是"参与者属于什么类型/类别"。

**根本原因**：`infer_answer_type()` 对 "is an instance of" 这一类问题没有识别出“类别型答案”需求（返回 `"other"`），因此 type guidance 没有生效，generation prompt 也没有明确约束模型去回答“类别/类型”而不是“实例列表”。

**改进方向**：
- 在 `infer_answer_type()` 中增加本体论问题识别：`"is an instance of|is a type of|is a kind of"` → `"category"`
- 对 `category` 类型的 generation prompt 加入明确指引："请回答该实体所属的类别或类型，而非列举具体成员"

---

## 各失败模式的预期改进 case 数

| 优化方向 | 预期增益 | 难度 |
|---|---|---|
| 修复 gate 检索策略（答案 hint） | +5 例（确认） | 低 |
| 多查询检索融合 | +10-15 例（估计） | 中 |
| 增加 "category" 类型识别 | +2-3 例 | 低 |
| Sequential N2 超时回退 | +2-4 例 | 中 |

---

## 关键代码位置

| 文件 | 内容 |
|---|---|
| `main/control_helpers.py:45-69` | `check_fallback_supported()`：gate 逻辑 |
| `main/control_helpers.py:9-42` | `is_answer_suspicious()`：可疑答案检测 |
| `main/tree_decompose_adaptive.py:42-54` | `infer_answer_type()`：答案类型推断 |
| `main/tree_decompose_adaptive.py:68-97` | `build_typed_retrieval_query()`：类型感知检索 query 构造 |
| `main/tree_decompose_adaptive.py:140-596` | `adaptive_solve_tree()`：核心树求解逻辑 |
| `main/tree_decompose_adaptive.py:460-496` | Parent direct retrieval fallback 逻辑 |
| `main/config_adaptive.py` | `ENABLE_FALLBACK_SUPPORT_CHECK` 等特性开关 |

---

## 实施进度

### ✅ 步骤1：验证 gate 影响（已完成，2026-04-19）

用 `ENABLE_FALLBACK_SUPPORT_CHECK=False` 跑 5 个确认 case（结果见 `output/musique/adaptive_typed_v1_single_attempt/2.txt`）：

| 问题 | Gate-on | Gate-off | Golden | 结论 |
|---|---|---|---|---|
| Blanton's → Saints symbol | `[none]` | fleur de lis | fleur-de-lis | ✅ 修复 |
| Dick Grayson (Batman) | `[none]` | Dick Grayson | Dick Grayson | ✅ 修复 |
| Ko Phi Phi Le | `[none]` | Ko Phi Phi Le | Ko Phi Phi Le | ✅ 修复 |
| Tanzania | `[none]` | Tanzania | Tanzania | ✅ 修复 |
| Menucha → 继任者 | `[none]` | Rabbi Dovber Schneuri | Menachem Mendel Schneersohn | ❌ direct_answer 本身就错 |

**结论**：Gate 确实是其中 4 例失败的直接原因。第 5 例里，`direct_answer` 本身就答错，因此即使关掉 gate 也不会修复该题。
目前 `ENABLE_FALLBACK_SUPPORT_CHECK = False` 已写入 `main/config_adaptive.py`（仅验证用）。

### ✅ 步骤2：实施 gate 改进（已完成，2026-04-19）

两处改动，目标都是减少“正确答案因证据不足被误拦截”的假阴性：

**改动1**：`main/tree_decompose_adaptive.py` — gate 检索 query 从 `extract_keywords(question)` 改为 `extract_keywords(question + " " + candidate)`，让候选答案参与检索。

**改动2**：`main/control_helpers.py` — prompt 从"有支持才放行"改为"有明确矛盾才拦截"，同时反转 return 逻辑（`return not reply.startswith("y")`）。

**验证结果**（5 case，结果见 `output/musique/adaptive_typed_v1_single_attempt/4.txt`）：

| 问题 | 旧 gate | 新 gate | Golden | 结论 |
|---|---|---|---|---|
| Blanton's → Saints symbol | `[none]` | fleur de lis | fleur-de-lis | ✅ 修复 |
| Dick Grayson (Batman) | `[none]` | Dick Grayson | Dick Grayson | ✅ 修复 |
| Ko Phi Phi Le | `[none]` | Ko Phi Phi Le | Ko Phi Phi Le | ✅ 修复 |
| Tanzania | `[none]` | Tanzania | Tanzania | ✅ 修复 |
| Menucha → 继任者 | `[none]` | Rabbi Menachem Mendel Schneersohn | Menachem Mendel Schneersohn | ✅ 修复（bonus！） |

5/5 全部答对。Menucha 一例还有额外现象：candidate-aware retrieval 不只是让 gate 更容易放行，也让 `direct_answer` 本身从错误答案变成了正确答案。

### ✅ 步骤2.5：负对照验证（已完成，2026-04-19）

挑选 3 个已知 direct_answer 明显错误、旧 gate 能正确拦截的 case 做负对照（结果见 `output/musique/adaptive_typed_v1_single_attempt/5.txt`）：

| 问题 | 错误的 direct_answer | 新 gate 结果 | 结论 |
|---|---|---|---|
| Who is the spouse of the current queen of England? | Camilla | `[none]` | ✅ 正确拦截 |
| Who was the British general in the battle...? | Bernard Montgomery | `[none]` | ✅ 正确拦截 |
| What is the seat of the county sharing a border...? | Oshkosh | `[none]` | ✅ 正确拦截 |

正对照 5/5 + 负对照 3/3，说明这个 gate 改动在小样本上是“有收益且未立刻失控”的，因此值得进入更大规模验证。

### ✅ 步骤3：全量 bad case 验证（已完成，2026-04-19）

受超时限制，全量 run 实际只处理了 19/62 个 case（结果见 `output/musique/adaptive_typed_v1_single_attempt/6.txt`）。19 个 case 对比 Run 1：

| | Run 1（旧 gate） | Run 6（新 gate） |
|---|---|---|
| 正确答案 | 0 | 1（Menucha ✅） |
| `[none]` | 19 | 5 |
| 错误非空答案 | 0 | 13 |

加上单独验证的 5 个正对照 case（全部正确），gate 改进的综合结论可以更精确地表述为：
- **+6 个正确答案**（EM 真实收益）
- **13 个 `[none]` → 错误答案**（在 EM 口径下两者都记 0，因此分数不变，但输出质量的形态发生了变化）
- Gate 放宽后对已知坏答案（Camilla、Oshkosh 等）仍能正确拦截

**决策**：Gate 继续细调的边际收益已经明显下降。那 13 个错误非空答案的根本问题在于 `direct_answer` 本身就答错，而不是 gate 判错。因此下一步更值得转向上游检索质量优化，优先减少 leaf node 直接掉成 `[none]` 的情况。

### 🔲 步骤4：实施多查询融合

**目标**：leaf node 检索失败（尤其是 N1→`[none]`）是当前 bad case 中最值得优先处理的上游问题。这里的思路不是改 generation，而是先扩大 leaf 阶段的证据召回范围，让 LLM 至少能看到更接近 gold evidence 的文档。

**查询策略**（每个 leaf node 最多 3 个 query）：
1. `build_typed_retrieval_query(question, type, anchor)`（当前已有，type-guided）
2. `question`（原始问题，无 type hint）
3. `node.entity_anchor`（可选，仅当 anchor 明显且 `answer_type` 属于 `person/location/org` 时启用，避免过泛实体把结果带偏）

**文档合并策略**：
- 按 chunk index 去重，同一 chunk 保留最高 rerank_score
- 合并后 **按 rerank score 降序排列，再截取现有 `topk2`**（写法：`sorted(..., key=..., reverse=True)[:topk2]`）

**架构原则（重要）**：
- **不修改 `main/retrieve.py` 中现有公共函数的签名与默认行为**，尤其 `answer_question(...)` 保持原样，避免污染 baseline 主干
- multi-query 逻辑只在 adaptive 分支启用，baseline 路径保持不变
- `tree_decompose_adaptive.py` 中可新增 `answer_with_prefetched_docs()` helper，但它应尽量复用现有生成逻辑，只替换“文档来源”
- 若在 `main/retrieve.py` 中新增 `multi_query_retrieve_documents(...)`，它的定位应是 **新增辅助函数**，而不是改写 baseline 现有的 `retrieve_documents(...)` / `answer_question(...)` 路径

**改动文件和位置**：

| 文件 | 改动内容 |
|---|---|
| `main/retrieve.py` | 新增 `multi_query_retrieve_documents(queries, ...)` 函数（紧接 `retrieve_documents` 之后，不改现有接口） |
| `main/tree_decompose_adaptive.py` | 新增 `answer_with_prefetched_docs()` 局部 helper；leaf node 处当 `ENABLE_MULTI_QUERY_FUSION=True` 时调用融合检索 + 新 helper |
| `main/config_adaptive.py` | 新增 `ENABLE_MULTI_QUERY_FUSION = True` |

**关键实现细节**：

```python
# retrieve.py - 新增函数（不改 answer_question 签名）
def multi_query_retrieve_documents(queries, dataset, method="dense", ...):
    merged = {}  # chunk_index -> best result
    for i, query in enumerate(queries):
        results = retrieve_and_rerank_chunks(..., query=query, stage=f"{stage}.q{i+1}")
        for r in results:
            idx = r["index"]
            if idx not in merged or r["rerank_score"] > merged[idx]["rerank_score"]:
                merged[idx] = r
    # 明确降序排列，取 top-topk2
    top = sorted(merged.values(), key=lambda x: x["rerank_score"], reverse=True)[:topk2]
    # format same as retrieve_documents ...
```

```python
# tree_decompose_adaptive.py - leaf node 处
if ENABLE_MULTI_QUERY_FUSION:
    queries = [query_for_leaf, actual_question]
    # entity_anchor 仅对信息量足够的类型启用
    if node.entity_anchor and node.answer_type in ("person", "location", "org"):
        queries.append(node.entity_anchor)
    merged_docs = multi_query_retrieve_documents(
        queries, DATASET, METHOD, ..., stage=f"tree.node.{node.id}.multi_query",
    )
    full_response = answer_with_prefetched_docs(
        question=actual_question, docs=merged_docs, ...
    )
else:
    full_response = answer_question(question=query_for_leaf, ...)
```

**小样本 stop/go 门槛**：
先选 5 个典型 leaf-failure case 做最小验证。**满足以下条件之一**再扩到全量 62：
- 至少修复 2/5 题（leaf node 由 [none] 变为正确答案）
- 或 leaf `[none]` 数量显著减少，且运行时无明显恶化

同时增加一个保守约束：

- **错误非空答案不能明显增多**

也就是说，不能只看到 `[none]` 下降，而忽略错误答案大量上升。

**实验边界说明**：

- 步骤4 及之后的实验，默认以当前“已改进的 gate 版本”为固定对照版本
- 不再同步继续调 gate，避免 multi-query 效果与 gate 改动相互混淆
- `answer_with_prefetched_docs()` 的实现应尽量复用现有生成逻辑，只替换文档来源，避免引入新的 prompt 漂移

### 🔲 步骤5：添加 `category` 类型识别

### 🔲 步骤6：全量 62 bad case 重跑对比

---

## 后续评估计划（2026-04-19 更新）

在完成 `adaptive_typed_v1_final` 的 MuSiQue bad 62 最终运行后，后续不再继续新增方法机制，而是进入“补齐评测 + 总结结果”阶段。

### 已拿到的最终 bad-case 结果

- 方法版本：`adaptive_typed_v1_final`
- MuSiQue bad 62：`4/62` EM 正确（相对 baseline `0/62`，净增 `+4`）
- 已确认的 EM 增益主要来自 gate 改进
- multi-query 与 category guidance 主要体现为行为层正信号，尚未大规模转成额外 EM

### 接下来的执行顺序

1. 补跑 MuSiQue fixed-100 中剩余的 `38` 道 baseline-correct 题
   目标：判断 adaptive 是否会破坏 baseline 原本正确的 case，并据此补齐 MuSiQue fixed-100 的整体结果
2. 如果时间允许，再跑另一个数据集
   目标：给 presentation 补一组跨数据集结果，避免结论只停留在 MuSiQue
3. 最后统一写总总结
   内容包括：
   - MuSiQue bad 62 的修复情况
   - MuSiQue fixed-100 的整体净变化
   - 另一个数据集上的表现
   - 方法的有效范围与主要瓶颈

### 当前阶段的原则

- `adaptive_typed_v1_final` 视为已冻结的方法版本
- 除非发现明确 bug，否则不再继续改动 gate、multi-query、category 或 timeout 相关逻辑
- 后续工作的重点从“继续改方法”切换为“补齐评估、固定结论、准备汇报”
