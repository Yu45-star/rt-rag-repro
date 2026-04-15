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

## 针对的错误类型

| 错误类型 | 解决方案 |
|---|---|
| 答案类型混淆（13%） | 对每个节点预测 `answer_type`，验证答案是否匹配 |
| 时间锚点错误（21%） | 对含时间关键词的问题触发一致性检查 |
| Fallback 乱猜（37% of errors） | Fallback 证据门控：只有文档中出现类型相容的答案片段时才接受 fallback |

---

## 新建文件列表

**不修改任何现有 baseline 文件。**

```
main/
  validators.py                  ← 验证逻辑（新建）
  tree_decompose_adaptive.py     ← typed node + 自适应求解（新建）
  config_adaptive.py             ← 新方法配置（新建）
  load_data_adaptive.py          ← 新入口（新建）

data/longbench/
  2wikimqa_100_seed42.jsonl      ← 第二数据集子集（新建）
  2wikimqa_100_seed42.meta.json  ← 抽样元数据（新建）

output/
  musique/adaptive_typed_v1/     ← MuSiQue 上的新方法结果
  2wikimqa/baseline/             ← 2WikiMQA 上的 baseline 结果
  2wikimqa/adaptive_typed_v1/    ← 2WikiMQA 上的新方法结果
```

---

## 实现细节

### 第一步：`main/validators.py`

三个纯函数，不调用 LLM：

```python
ANSWER_TYPE_PATTERNS = {
    "date":   [r'\b\d{4}\b', r'\b(january|february|...)\b'],
    "number": [r'^\d+[\.,]?\d*$'],
    "person": [],   # 启发式：首字母大写，不含数字
    "location": [], # 启发式：首字母大写，可含 of/in/the
    "organization": [],
    "other":  [],
}

def validate_answer_type(answer: str, expected_type: str) -> bool:
    """基于规则检查答案是否属于预期类型。"""

TEMPORAL_TRIGGERS = ["first", "last", "current", "before", "after",
                     "abolished", "founded", "when", "year", "president"]

def check_temporal_consistency(answer: str, question: str, context_docs: str) -> bool:
    """
    仅在问题含 TEMPORAL_TRIGGER 关键词时触发。
    检查：答案对应的时间线索是否出现在检索文档中。
    返回 True（一致）或 False（疑似错误）。
    """

def check_fallback_evidence(candidate_answer: str, retrieved_docs: str,
                             expected_type: str) -> bool:
    """
    仅当检索文档中存在与 candidate_answer 共享至少1个实词、
    且类型与 expected_type 相容的片段时，返回 True。
    否则返回 False → fallback 应输出 [none]。
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
        self.answer_type = "other"       # 预测类型: person/org/location/date/number/other
        self.entity_anchor = None        # 重写时必须保留的关键实体
        self.validator_rejected = False  # 是否被验证器拒绝
        self.retry_done = False          # 是否已经重试过
```

#### 2b. `build_typed_question_tree()`

复用 `build_question_tree()`（line 1208）建树，然后对每个节点用**启发式规则**推断 `answer_type`（优先不增加 LLM 调用）：
- "who" → person
- "where" / "in which" → location
- "when" / "what year" → date
- "how many" / "how much" → number
- 其余 → other

同时从问题文本中提取 `entity_anchor`（最重要的已知实体名词短语），用于重试时保持查询锚定。

#### 2c. `adaptive_solve_tree()`

先调用原始 `solve_tree()`（line 1450）完成全树求解，**再**对已求解的叶节点做验证 pass：

```python
def adaptive_solve_tree(root, original_question, timeout_budget, ...):
    # 1. 用原始逻辑求解整棵树
    result = solve_tree(root, original_question, ...)

    # 2. 验证每个叶节点的答案
    for node in get_leaf_nodes(root):
        if node.answer and node.answer != "[none]":
            type_ok = validate_answer_type(node.answer, node.answer_type)
            temporal_ok = (not has_temporal_trigger(node.display_question) or
                           check_temporal_consistency(node.answer, ...))
            if not (type_ok and temporal_ok):
                node.validator_rejected = True
                # 单次重试：改写查询但保留 entity_anchor
                if not node.retry_done and has_budget(timeout_budget, 0.25):
                    new_q = rewrite_with_anchor(node.display_question,
                                                node.entity_anchor,
                                                node.answer_type)
                    node.answer = re_answer(new_q, ...)
                    node.retry_done = True

    # 3. 重新组合受影响的父节点答案（简单情况：只更新直接父节点）
    propagate_updated_answers(root)
    return root.answer
```

这种方式避免了重写400行的 `solve_tree()` 底层遍历逻辑。

#### 2d. `adaptive_direct_answer()`（fallback 门控）

替换 `tree_decompose_and_answer()`（line 1890）中对 `direct_answer()` 的调用：

```python
def adaptive_direct_answer(question, root_answer_type, ...):
    candidate, docs = direct_answer_with_docs(question, ...)
    if check_fallback_evidence(candidate, docs, root_answer_type):
        return candidate
    else:
        return "[none]"  # 无证据时明确失败，不猜测
```

---

### 第三步：`main/config_adaptive.py`

继承 `main/config.py` 所有基础配置，仅覆盖以下参数：

```python
from main.config import *

OUTPUT_DIR = "output/musique/adaptive_typed_v1"
ANSWER_TYPE_METHOD = "rules"       # 使用规则推断类型，不加 LLM 调用
ENABLE_TYPE_VALIDATION = True
ENABLE_TEMPORAL_CHECK = True
ENABLE_FALLBACK_GATE = True
RETRY_TIMEOUT_BUDGET_FRACTION = 0.25
```

---

### 第四步：`main/load_data_adaptive.py`

基于 `main/load_data.py` 修改，差异仅三处：
- 从 `config_adaptive` 导入配置
- 调用 `adaptive_solve_tree()` 替代 `solve_tree()`
- 调用 `adaptive_direct_answer()` 替代 `direct_answer()`
- 每题额外记录 validator rejection count 和 retry count

---

### 第五步：第二数据集（2WikiMQA-100）

**5a. 构建子集**

写脚本 `scripts/sample_subset.py`，从 `data/longbench/2wikimqa.jsonl` 用 seed=42 随机抽 100 题，输出 `.jsonl` + `.meta.json`。

**5b. 关于检索索引**

- 目前只有 MuSiQue 的 dense embeddings（`data/embeddings/musique/200_2_2/`）
- 2WikiMQA **没有**预建索引
- 两种选项：
  - **选项 A（推荐）**：用 BM25（`METHOD = "bm25"`），不需要预建索引，直接用原始文本
  - **选项 B**：用 `main/build_dense_index/dense_build_index.py` 建新 FAISS 索引，成本更高
- 建议先用 BM25，在报告中注明这一差异，作为方法跨数据集泛化的正当近似

**5c. 在 2WikiMQA 上跑 baseline 和 adaptive**

各新建一份 config 文件，指定 `DATASET = "2wikimqa"`、`METHOD = "bm25"`、对应输出目录。

---

## 实验执行顺序

1. 实现 `validators.py` + `tree_decompose_adaptive.py`
2. `python main/load_data_adaptive.py` → `output/musique/adaptive_typed_v1/`
3. 对比 MuSiQue baseline 与 adaptive 的 EM/F1
4. `python scripts/sample_subset.py` → 构建 2WikiMQA-100 子集
5. `python main/load_data.py`（2wiki baseline config）→ `output/2wikimqa/baseline/`
6. `python main/load_data_adaptive.py`（2wiki adaptive config）→ `output/2wikimqa/adaptive_typed_v1/`

---

## 需要报告的指标

每个 数据集 × 方法 组合：
- EM、F1
- Validator 拒绝节点数（type 和 temporal 各自多少）
- 重试次数、重试后答案变化数
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
