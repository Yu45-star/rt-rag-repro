# 实现计划：Adaptive Typed-Control Method

## 背景与现状

课程项目，复现并扩展 RT-RAG。Baseline 已完成：
- **数据集**: MuSiQue-100 (`data/longbench/musique_100_seed42.jsonl`)
- **配置**: `main/config.py`（轻量：2棵树，topk1=25，topk2=8）
- **结果**: 38% EM，62道题答错
- **错误分析**: 已完成，见 `output/musique/dense_chunk200_topk1_25_topk2_8/`

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
| 多跳中间节点误差传播（26%） | 子节点低置信度时父节点触发 direct retrieval，绕过坏掉的子节点 |

---

## 实验范围说明

**数据集选择**：第二数据集选择 2WikiMQA，HotpotQA 不在本次课程项目实验范围内。两个数据集均使用 dense 检索，保持口径一致。

## 新建文件列表

**不修改任何现有 baseline 文件。**

```
main/
  control_helpers.py             ← 轻量控制辅助工具，非硬性规则验证器（新建）
  tree_decompose_adaptive.py     ← typed node + 自适应求解（新建）
  config_adaptive.py             ← 新方法配置（新建）
  load_data_adaptive.py          ← 新入口（新建）

scripts/
  sample_subset.py               ← 数据集抽样脚本（新建）

data/longbench/
  2wikimqa_100_seed42.jsonl      ← 第二数据集子集（新建）
  2wikimqa_100_seed42.meta.json  ← 抽样元数据（新建）

data/embeddings/2wikimqa/        ← 2WikiMQA dense index（新建，需构建）

output/
  musique/adaptive_typed_v1/     ← MuSiQue 上的新方法结果
  2wikimqa/baseline/             ← 2WikiMQA 上的 baseline 结果
  2wikimqa/adaptive_typed_v1/    ← 2WikiMQA 上的新方法结果
```

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

#### 2a. TypedQuestionNode

继承 `main/tree_decompose.py` 中的 `QuestionNode`（定义在 line 1169）：

```python
from main.tree_decompose import QuestionNode

class TypedQuestionNode(QuestionNode):
    def __init__(self, ...):
        super().__init__(...)
        self.answer_type = "other"    # 预测类型: person/org/location/date/number/other
        self.entity_anchor = None     # 重写时必须保留的关键实体
        self.confidence = "high"      # "high" 或 "low"，软置信度标记，不硬截断
        self.retry_done = False       # 是否已经做过 query rewrite
```

#### 2b. `build_typed_question_tree()`

复用 `build_question_tree()`（line 1208）建树，然后对每个节点用**启发式规则**推断 `answer_type`（优先不增加 LLM 调用）：
- "who" → person
- "where" / "in which" → location
- "when" / "what year" → date
- "how many" / "how much" → number
- 其余 → other

同时从问题文本中提取 `entity_anchor`，规则如下：
- 取当前 node 问句中**不是 WH-word（who/what/where/when/how）、也不是占位符**的主实体短语
- 如果当前 node 依赖父节点答案（`depends_on` 非空），优先保留父节点填入后的实体
- 如果抽不出来，允许为 `None`，不强行生成
- 用于重试时约束查询，防止 rewrite 丢失关键检索锚点

#### 2c. `adaptive_solve_tree()`

核心机制是 **type-aware query rewriting + 软置信度传播**，不做硬截断。

实现方式：在 `tree_decompose_adaptive.py` 中重新实现 `adaptive_solve_tree()`，复制 `solve_tree()` 的结构，在叶节点求解阶段插入以下逻辑：

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

#### 2d. `adaptive_direct_answer()`（fallback 门控）

替换 `tree_decompose_and_answer()`（line 1890）中对 `direct_answer()` 的调用：

```python
def adaptive_direct_answer(question, root_answer_type, ...):
    candidate, docs = direct_answer_with_docs(question, ...)
    # 用 LLM 判断文档是否支持该答案，而不是 span 匹配规则
    if check_fallback_supported(question, candidate, docs):
        return candidate
    else:
        return "[none]"  # 无证据支撑时明确失败，不猜测
```

---

### 第三步：`main/config_adaptive.py`

继承 `main/config.py` 所有基础配置，仅覆盖以下参数：

```python
from main.config import *

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

基于 `main/load_data.py` 修改，差异仅两处：
- 从 `config_adaptive` 导入配置（含 `METHOD_TAG`，用于拼输出路径）
- 调用 `tree_decompose_adaptive.py` 中的**顶层入口函数**（对应原 `tree_decompose_and_answer()`），而不是在 loader 里直接替换底层 `solve_tree` / `direct_answer`
- 每题额外记录 rewrite 触发次数、parent-level direct retrieval fallback 触发次数、fallback support check 结果

---

### 第五步：第二数据集（2WikiMQA-100）

**5a. 构建子集**

写脚本 `scripts/sample_subset.py`，从 `data/longbench/2wikimqa.jsonl` 用 seed=42 随机抽 100 题，输出 `.jsonl` + `.meta.json`。

**5b. 构建 dense index**

与 MuSiQue 保持相同检索口径，使用 `main/build_dense_index/dense_build_index.py` 在 **2WikiMQA 全量 corpus**（而非 100 题子集）上构建 FAISS 索引，输出到 `data/embeddings/2wikimqa/`。100 题子集仅作为评测 query 集合，不参与 index 构建。参考 MuSiQue 的 `data/embeddings/musique/200_2_2/config.json` 保持相同的 chunk 参数（chunk_size=200, overlap=2）。

**5c. 在 2WikiMQA 上跑 baseline 和 adaptive**

各新建一份 config 文件，指定 `DATASET = "2wikimqa"`、`METHOD = "dense"`、对应 dense index 路径和输出目录。

---

## 实验执行顺序

1. 实现 `control_helpers.py` + `tree_decompose_adaptive.py`（含顶层入口函数）
2. 实现 `config_adaptive.py` + `load_data_adaptive.py`
3. `python main/load_data_adaptive.py` → `output/musique/adaptive_typed_v1/`
4. 对比 MuSiQue baseline 与 adaptive 的 EM/F1
5. `python scripts/sample_subset.py` → 构建 2WikiMQA-100 子集
6. `python main/build_dense_index/dense_build_index.py` → 构建 2WikiMQA dense index
7. `python main/load_data.py`（2wiki baseline config）→ `output/2wikimqa/baseline/`
8. `python main/load_data_adaptive.py`（2wiki adaptive config）→ `output/2wikimqa/adaptive_typed_v1/`

---

## 需要报告的指标

每个 数据集 × 方法 组合：
- EM、F1
- Query rewrite 触发率（`is_answer_suspicious` 返回 True 的节点比例）
- Rewrite 后答案发生变化的比例（rewrite 实际有效的频率）
- 低置信度节点传播数量（`confidence == "low"` 的节点）
- **Parent-Level Direct Retrieval Fallback** 触发次数及这些题的正确率（单独统计，用于区分"typed guidance 的贡献"与"父节点绕过分解的贡献"，预期被问到）
- Fallback 触发率、Fallback gate 拒绝率（返回 `[none]` 的比例）
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
| `main/tree_decompose.py:1890` | `tree_decompose_and_answer()` — 最终答案选择 |
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
| 多跳中间实体选错 | 16 | 基本无效 | 根因在检索层，相关文档排不进 top-k，query 改写帮助有限 |
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
