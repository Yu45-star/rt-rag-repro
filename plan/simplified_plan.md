# 实现计划：Adaptive Typed-Control Method

## 背景与现状

课程项目，复现并扩展 RT-RAG。Baseline 已完成：
- **MuSiQue-100** (`data/longbench/musique_100_seed42.jsonl`)：38% EM，62道题答错
  - 配置：`main/config.py`（轻量：2棵树，topk1=25，topk2=8）
  - 错误分析已完成，见 `output/musique/dense_chunk200_topk1_25_topk2_8/`
- **2WikiMQA-100** (`data/longbench/2wikimqa_100_seed42.jsonl`)：**69% EM，79.26% F1**（2026-04-16 完成）
  - 结果见 `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/run_summary.md`
  - 注：2WikiMQA 的 100 题子集、dense index (`data/embeddings/2wikimqa/`)、baseline 输出均已就绪

**错误分布**（来自 error_analysis_report.md）：
| 错误类型 | 数量 | 占比 |
|---|---|---|
| 多跳中间实体选错 | 16 | 25.8% |
| 时间锚点错误 | 13 | 21.0% |
| 答案类型混淆 | 8 | 12.9% |
| 地理层级错误 | 8 | 12.9% |
| 别名/格式（伪错误） | 7 | 11.3% |
| 62个错误中有23个来自 direct fallback 乱猜 | 23 | 37% of errors |

**Proposed method 的出发点**：类型混淆 + 时间错误 + 无依据 fallback 猜测合计占约 47% 的错误，且都可以用轻量验证解决。多跳中间实体选错（26%）需要更好的检索能力，暂不作为目标。

---

## 方法核心思路

论文消融实验中，去掉 query rewriting 的掉分最为显著。本方法的核心因此对准同一机制：用 typed node 的元信息（`answer_type`、`entity_anchor`）来生成**更有约束力的检索查询**，而不是事后验证答案然后硬截断。

**设计原则**：
- `answer_type` 和 `entity_anchor` 是 prompt 引导材料，不是外部裁判
- 子节点答案可疑时，触发一次 type-aware query rewrite，不是直接拒绝
- 重写后无论结果如何，都以"置信度标记"向上传播，不断链
- fallback 门控保留，但用 LLM 判断而非 span 匹配规则

## 针对的错误类型

| 错误类型 | 解决方案 |
|---|---|
| 答案类型混淆（13%） | 通过 typed query formulation 和 type-aware rewrite，引导模型检索并生成更符合目标类型的答案 |
| 时间锚点错误（21%） | 在检索 query 里嵌入时间上下文，引导模型关注对应时期 |
| Fallback 乱猜（37% of errors） | Fallback 时让 LLM 自判"文档是否支持该答案"，不支持则返回 `[none]` |
| 多跳中间节点误差传播（26%） | **目标限定为 [none]-型阻塞**：子节点无法返回任何答案（`answer=="[none]"` 且 `confidence=="low"`）时，父节点触发 direct retrieval 绕过死掉的子树。子节点返回错误实体（非 [none]）的情况不在本机制覆盖范围——那是检索层根因，parent fallback 对此基本无效。 |

---

## 实验范围说明

**数据集选择**：第二数据集选择 2WikiMQA，HotpotQA 不在本次课程项目实验范围内。两个数据集均使用 dense 检索，保持口径一致。

## 新建文件列表

**不修改任何现有 baseline 文件。**

```
main/
  control_helpers.py             ← 轻量控制辅助工具，非硬性规则验证器 ✓ 已实现
  tree_decompose_adaptive.py     ← typed node 标注 + 自适应求解 ✓ 已实现
  config_adaptive.py             ← 新方法配置，含 METHOD_TAG ✓ 已实现
  load_data_adaptive.py          ← 新入口，额外记录 adaptive stats ✓ 已实现

output/musique/
  bad_cases_qids.txt             ← 62 个 bad case 的 qid 列表 ✓ 已生成
  adaptive_typed_v1/             ← MuSiQue 上的新方法结果（待跑）

output/2wikimqa/
  adaptive_typed_v1/             ← 2WikiMQA 上的新方法结果（待跑）
```

已就绪（无需再建）：
- `data/longbench/2wikimqa_100_seed42.jsonl` ✓
- `data/embeddings/2wikimqa/` ✓
- `output/2wikimqa/dense_chunk200_topk1_25_topk2_8/` ✓（baseline 结果）

---

## 实现细节

### 第一步：`main/control_helpers.py`

不命名为 `validators.py`——这里的函数不是硬性规则验证器，而是轻量控制辅助工具（lightweight control helpers）。只保留两个函数：

```python
def is_answer_suspicious(answer: str, expected_type: str) -> bool:
    """
    轻量启发式：判断答案是否值得触发一次 query rewrite。
    不作为硬性拦截，只是 rewrite 的触发信号。
    例如：
    - expected_type="date" 但答案里没有任何数字 → suspicious
    - expected_type="person" 但答案是纯数字 → suspicious
    - answer == "[none]" → 始终 suspicious
    返回 True 表示"建议 rewrite"，False 表示"可直接上传"。
    """

def check_fallback_supported(question: str, candidate: str,
                              retrieved_docs: str) -> bool:
    """
    调用 LLM 判断 retrieved_docs 是否支持 candidate 作为 question 的答案。
    Prompt 示例：
      "Documents: {docs}
       Question: {question}
       Candidate answer: {candidate}
       Do the documents provide evidence for this answer? Reply yes or no."
    返回 True（有证据支持）或 False（无证据，输出 [none]）。
    """
```

---

### 第二步：`main/tree_decompose_adaptive.py`

#### 2a. 节点标注：`annotate_tree_nodes()`

**不新建 TypedQuestionNode 子类**——`build_question_tree()` 返回的是 `QuestionNode` 实例且节点之间通过 `.left` / `.right` 互相引用，替换节点类型需要更新所有引用，容易引入 bug。改用 **duck typing**：遍历树后通过 `setattr` 给每个 `QuestionNode` 实例动态附加以下属性：

```python
def annotate_tree_nodes(root):
    """遍历树，给每个节点附加 adaptive 扩展属性。"""
    def walk(node):
        if node is None:
            return
        node.answer_type = infer_answer_type(node.display_question)
        node.entity_anchor = extract_entity_anchor(node.display_question)
        node.confidence = "high"
        node.retry_done = False
        node.used_direct_retrieval_fallback = False
        walk(node.left)
        walk(node.right)
    walk(root)
```

附加属性含义：
- `answer_type`：预测类型（person/org/location/date/number/other）
- `entity_anchor`：重写时保留的关键实体短语，抽不到则为 `None`
- `confidence`：软置信度标记（"high"/"low"），不硬截断
- `retry_done`：是否已触发过 query rewrite
- `used_direct_retrieval_fallback`：是否触发过 parent-level direct retrieval fallback

#### 2b. 辅助函数

**`infer_answer_type(question)`**：规则推断（不增加 LLM 调用）：
- "who" → person；"where" / "in which" → location
- "when" / "what year" → date；"how many" / "how much" → number
- 其余 → other

**`extract_entity_anchor(question)`**：正则提取首个大写多词短语（如 "Battle of Brechin"）作为锚点实体；若抽不到则返回 `None`，不强行生成。

**`build_typed_retrieval_query(question, answer_type, entity_anchor)`**：将类型和实体信息**融入问题表述**，生成自然语言重写（不使用方括号元语言）。例如：`"Who is the person that [original question]"`，而不是 `"[Looking for a person] Who..."`。

> 原因：`answer_question()` 会把传入的 question 字符串同时用于 `extract_keywords(question)`（检索）和 `call_api_for_answer(question, docs)`（生成 prompt），没有独立的 retrieval_query 参数。如果类型提示用方括号元语言写成，既会在关键词提取里产生噪声，也会污染 LLM 的生成 prompt。改用自然语言重写，在检索和生成两侧都合法，同时保留方向性引导。

**`build_rewritten_query(question, entity_anchor, answer_type)`**：可疑时的重写版本，加 `Focus on '...' The answer should be a ...` 前缀，锚定关键实体 + 类型。

#### 2c. `adaptive_solve_tree(root, ..., stats)`

核心机制是 **type-aware query rewriting + 软置信度传播**，不做硬截断。

实现方式：复制 `solve_tree()`（line 1450）的完整代码，在两处插入新逻辑，其余保持一致。新增参数 `stats: dict`，由调用方（`adaptive_decompose_and_answer_with_variants`）创建并传入，用于累计各项指标。

**叶节点求解（三层策略）**（插入位置：原 `answer_question` 调用后、`node.answer = answer` 赋值前）：

**叶节点求解（三层策略）**：

```python
# 第一层：type-aware query formulation
#   answer_type 和 entity_anchor 注入到检索 query 的构造阶段，
#   不是拼接到原始问题文本末尾（避免污染检索）。
#   具体做法：build_typed_retrieval_query() 生成一个更聚焦的检索式，
#   例如 "Who is the [person] that ..." 或 "What [location] is associated with ..."
# typed_query 是"用于检索和问答的重构问题"，不是系统提示词字符串。
# build_typed_retrieval_query() 将 answer_type 和 entity_anchor 融入问题表述，
# 使检索目标更聚焦，而不是把标签拼接成 prompt 再丢进 LLM。
typed_query = build_typed_retrieval_query(
    question=node.display_question,
    answer_type=node.answer_type,
    entity_anchor=node.entity_anchor,
)
leaf_answer = answer_question(typed_query, ...)

# 第二层：可疑时做一次 type-aware query rewrite
#   rewrite 进一步收窄检索方向，用 entity_anchor 锚定关键实体
#   不管 rewrite 后结果如何，都继续向上传，只打 confidence 标记，不硬截断
if is_answer_suspicious(leaf_answer, node.answer_type) and not node.retry_done:
    rewritten_q = build_rewritten_query(
        question=node.display_question,
        entity_anchor=node.entity_anchor,
        answer_type=node.answer_type,
    )
    retry_answer = answer_question(rewritten_q, ...)
    # 选择规则：只有当 retry_answer 明显不再 suspicious 时才替换原答案，
    # 避免"只要不是 [none] 就无脑抢占"。若两者都可疑，保留 answer_1。
    if retry_answer != "[none]" and not is_answer_suspicious(retry_answer, node.answer_type):
        leaf_answer = retry_answer
    node.retry_done = True
    node.confidence = "low"   # 标记可疑，不硬截断
else:
    node.confidence = "high"

node.answer = leaf_answer
```

**父节点兜底（第三层）—— Parent-Level Direct Retrieval Fallback**：

父节点在组合子节点答案前检查依赖的子节点是否为低置信度。若某子节点 `confidence == "low"` 且答案为 `[none]`，触发 **parent-level direct retrieval fallback**：父节点直接对自身问题做检索，绕过分解逻辑，而不是被阻塞。

这个机制需要在报告中**单独命名和单独统计**，因为它是性能提升的一个可分离来源，需要区分"typed guidance 带来的提升"和"父节点绕过分解带来的提升"：

```python
# 父节点组合前
if any(child.confidence == "low" and child.answer == "[none]" for child in deps):
    # Parent-Level Direct Retrieval Fallback：绕过子节点分解
    parent.answer = answer_question(parent.display_question, ...)
    parent.confidence = "low"
    parent.used_direct_retrieval_fallback = True   # 单独计数
else:
    # 正常生成式组合
    parent.answer = get_final_answer(parent, child_answers, ...)
    parent.used_direct_retrieval_fallback = False
```

#### 2d. Fallback 门控（内嵌于 `adaptive_decompose_and_answer_with_variants`）

不单独写 `adaptive_direct_answer` 函数——门控逻辑直接内嵌在 `adaptive_decompose_and_answer_with_variants` 中所有调用 `direct_answer()` 的位置（`finalize_timeout_fallback` 和 exhausted-variants fallback 两处）：

```python
# 原来：return direct_answer(question, ...)
# 改为：
candidate = direct_answer(question, ...)
# 需要检索文档用于 support check：用 extract_keywords + retrieve_documents 取文档
# （direct_answer 内部已检索但不返回，这里额外检索一次，仅在 fallback 路径触发，可接受）
docs = retrieve_documents(extract_keywords(question), ...)
if ENABLE_FALLBACK_SUPPORT_CHECK:
    stats["fallback_gate_checks"] += 1
    if not check_fallback_supported(question, candidate, docs):
        stats["fallback_gate_blocked_count"] += 1
        return "[none]"
return candidate
```

> 注：`retrieve_documents` 和 `extract_keywords` 从 `retrieve.py` 导入。docs 截断至 3000 字符再传入 `check_fallback_supported`。

#### 2e. `adaptive_decompose_and_answer_with_variants()` 返回值

函数签名：
```python
def adaptive_decompose_and_answer_with_variants(question, ..., stats=None) -> tuple[str, dict]:
```
- `stats` 默认为 `None`，调用方传入一个 `dict` 用于累计（允许跨多次 attempt 累计）
- 返回 `(answer_str, stats_dict)` 而非纯字符串
- `load_data_adaptive.py` 中：
  ```python
  result = await loop.run_in_executor(None, lambda: adaptive_decompose_and_answer_with_variants(...))
  predicted_answer, adaptive_stats = result
  ```

---

### 第三步：`main/config_adaptive.py`

继承 `main/config.py` 所有基础配置，仅覆盖以下参数：

```python
from config import *

# 输出路径：保持现有动态命名规则，新增 METHOD_TAG 插入路径
METHOD_TAG = "adaptive_typed_v1"
# loader 拼路径时将 METHOD_TAG 插入，如 output/musique/adaptive_typed_v1/...

ANSWER_TYPE_METHOD = "rules"          # 使用规则推断类型，不加 LLM 调用
# 注意：answer_type 为粗粒度分类，主要用于引导检索方向，不保证完全准确
ENABLE_TYPE_GUIDANCE = True           # 是否在检索 query 构造时注入 answer_type
ENABLE_TYPE_AWARE_REWRITE = True      # 是否在答案可疑时触发 type-aware query rewrite
ENABLE_FALLBACK_SUPPORT_CHECK = True  # 是否对 direct fallback 做 LLM 支持度检验
RETRY_TIMEOUT_BUDGET_FRACTION = 0.25
```

---

### 第四步：`main/load_data_adaptive.py`

基于 `main/load_data.py` 修改，差异仅三处：
1. 从 `config_adaptive` 导入配置（`METHOD_TAG = "adaptive_typed_v1"`），输出目录拼为 `output/{dataset}/adaptive_typed_v1/`
2. executor lambda 调用 `adaptive_decompose_and_answer_with_variants`，拆包 `(predicted_answer, adaptive_stats)` tuple
3. `write_result_to_file` 额外写入 6 个 adaptive stats 字段：
   - `adaptive_rewrite_triggered`：int
   - `adaptive_rewrite_effective`：int
   - `adaptive_low_confidence_nodes`：int
   - `adaptive_parent_direct_fallback`：int
   - `adaptive_fallback_gate_checks`：int（`check_fallback_supported` 被调用次数）
   - `adaptive_fallback_gate_blocked_count`：int（门控拦截次数，即返回 `[none]` 次数）

---

### 第五步：第二数据集（2WikiMQA-100）

**已就绪**：2WikiMQA-100 子集、dense index、baseline 结果均已完成（见背景与现状）。

只需要在 2WikiMQA 上跑 adaptive 方法。由于 `load_data_adaptive.py` 在模块导入时静态读取配置，"新建一份 config 文件"并不会让同一条命令自动切换数据集。正确做法是**通过环境变量覆盖**：

```bash
RT_RAG_DATASET=2wikimqa \
RT_RAG_DATA_PATH=data/longbench/2wikimqa_100_seed42.jsonl \
python main/load_data_adaptive.py
```

这利用了 `config.py` 中已有的 `os.getenv()` 机制，无需新建 config 文件或修改 loader 代码。

---

## 实验执行顺序

1. ✓ 实现 `control_helpers.py` + `tree_decompose_adaptive.py`（含顶层入口函数）
2. ✓ 实现 `config_adaptive.py` + `load_data_adaptive.py`
3. ✓ 生成 `output/musique/bad_cases_qids.txt`（从 `output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_strict_errors.csv` 提取，共 62 行）
4. **Bad-case 快速验证**：`python main/load_data_adaptive.py --qid-file output/musique/bad_cases_qids.txt`
5. 对比 adaptive 在 bad cases 上的改善（若无明显改进，调整后重来）
6. **全量 MuSiQue**（验证满意后）：`python main/load_data_adaptive.py` → `output/musique/adaptive_typed_v1/`
7. **2WikiMQA adaptive**：`RT_RAG_DATASET=2wikimqa RT_RAG_DATA_PATH=data/longbench/2wikimqa_100_seed42.jsonl python main/load_data_adaptive.py` → `output/2wikimqa/adaptive_typed_v1/`

---

## 需要报告的指标

每个 数据集 × 方法 组合：
- EM、F1
- Query rewrite 触发率（`is_answer_suspicious` 返回 True 的节点比例）
- Rewrite 后答案发生变化的比例（rewrite 实际有效的频率）
- 低置信度节点传播数量（`confidence == "low"` 的节点）
- **Parent-Level Direct Retrieval Fallback** 触发次数及这些题的正确率（单独统计，用于区分"typed guidance 的贡献"与"父节点绕过分解的贡献"，预期被问到）
- Fallback 触发率（`adaptive_fallback_gate_checks` / 总题数）、Gate 拒绝率（`adaptive_fallback_gate_blocked_count` / `adaptive_fallback_gate_checks`）
- 按错误类型分类的改善/退步情况（与 baseline 错误分析口径一致）

**课程报告叙述结构**：
1. Baseline 结果 + 错误分析 → 为什么提出这个方法
2. 每个机制对应一类错误，motivation 清晰
3. 结果：哪些错误类型减少了，哪些没有，原因分析

---

## 关键文件位置

| 文件 | 内容 |
|---|---|
| `main/tree_decompose.py:1169` | `QuestionNode` 定义 |
| `main/tree_decompose.py:1208` | `build_question_tree()` |
| `main/tree_decompose.py:1450` | `solve_tree()` |
| `main/tree_decompose.py:1923` | `decompose_and_answer_with_variants()` — 顶层入口 |
| `main/retrieve.py:995` | `answer_question()` |
| `main/retrieve.py:483` | `retrieve_documents()` |
| `main/config.py` | Baseline 配置（不改动） |
| `output/musique/dense_chunk200_topk1_25_topk2_8/error_analysis_report.md` | 错误分析结果 |

---

## 改善预期

基于 baseline 62 个错误的逐类分析：

| 错误类型 | 案例数 | 预期改善 | 原因 |
|---|---|---|---|
| Fallback 乱猜 | 23 | 有把握 | `check_fallback_supported()` 直接针对此类，LLM 判断文档不支持则拦截 |
| 答案类型混淆 | 8 | 有合理把握 | typed query formulation 注入类型约束，mechanism 对准错误根因 |
| 时间锚点错误 | 13 | 部分有效 | entity_anchor 可能携带时间信息，但原 temporal check 已删除，覆盖不完整 |
| 多跳中间实体选错 | 16 | 基本无效 | 根因在检索层，相关文档排不进 top-k，query 改写帮助有限；parent fallback 仅覆盖子节点返回 [none] 的情况，子节点返回错误实体不触发 |
| 地理层级错误 | 8 | 基本无效 | answer_type="location" 粒度太粗，无法区分 county/city/province |
| 别名/格式伪错误 | 7 | 完全无效 | 评测规范问题，方法层面无法干预 |
| 过度生成 | 5 | 轻微可能 | typed query 可能使 generation 稍聚焦，但无专门机制 |

**预期可覆盖的 case 约 30 个**（fallback 乱猜 + 答案类型混淆为主），其余 30+ 来自检索层根本问题和评测规范，超出本方法设计范围。

报告叙述策略：明确说明方法针对哪类错误、在这些错误上的改善、以及哪些错误超出设计范围——比过度承诺后指标不达预期更可信。

---

## 风险与边界

- **本方法不直接解决 retrieval miss**：若正确答案所在文档根本未被检索到，typed guidance 和 rewrite 都无法弥补。
- **多跳中间实体选错的改善可能有限**：该类错误（25.8%）根本原因在于检索层，不是 query 表述问题，本方法能覆盖的主要是表述偏差导致的检索失焦。
- **LLM-based support check 本身存在不稳定性**：若 `check_fallback_supported` 的判断出现误判，fallback gate 可能把正确答案也拦掉，带来额外波动；需在实验中观察其误杀率。
