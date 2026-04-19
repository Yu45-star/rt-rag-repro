# Plan: 补全 solve_node 内部的 timeout 检查点

## Context

上一轮对 `adaptive_solve_tree` 做了两处修改：

1. `build_typed_retrieval_query` 的 double-question bug fix（完成，保留）
2. `solve_node` 入口加了 deadline 检查（已完成，但不够）

Codex 指出的问题是：

- `solve_node` 入口的检查只阻止“开始一个新节点”
- 但无法中断已经进入节点后的各个耗时调用

当前 `solve_node` 内部的昂贵调用不止一种，包括：

- `answer_question(...)`
- `build_question_tree(...)`
- `build_enhanced_right_subtree(...)`
- `generate_right_question_with_llm(...)`
- `get_final_answer(...)`

其中任何一处都可能独占运行几百秒。

另一个问题是：timeout 截断产生的 `[none]` 和方法机制产生的 `[none]` 在 debug 日志里无法区分，污染后续的 bad-case 归因分析。

**目标**：

在不修改任何 baseline 文件（`retrieve.py`、`tree_decompose.py` 等）的前提下，把 deadline 检查点补到 `solve_node` 内每个昂贵调用之前，并增加统一的 debug 标记与单独统计，用来区分 timeout 截断。

---

## 修改内容（仅 `main/tree_decompose_adaptive.py`）

### 修改 0：增加统一 helper，避免重复写 cutoff 逻辑

在 `adaptive_solve_tree()` 内新增一个局部 helper，例如：

```python
def cutoff_node_due_to_timeout(node, current_depth, stage, return_answer="[none]"):
    node.answer = return_answer
    node.confidence = "low"
    placeholder_answers[node.id] = return_answer
    stats["timeout_cutoff_nodes"] = stats.get("timeout_cutoff_nodes", 0) + 1
    log_node_event(
        "node_timeout_cutoff",
        node,
        answer=return_answer,
        metadata={
            "current_depth": current_depth,
            "timeout_cutoff": True,
            "stage": stage,
        },
    )
    return {node.id: return_answer}
```

作用：

- 统一写入 `node.answer`
- 统一写入 `confidence = "low"`
- 统一写入 `placeholder_answers`
- 统一记录 `node_timeout_cutoff` 事件
- 单独累计 `timeout_cutoff_nodes`

这样后面所有 pre-call timeout 检查都直接复用这个 helper，避免漏字段或写法漂移。

### 修改 1：`solve_node` 入口的 timeout 截断改为复用 helper

当前代码（约 line 200）：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    node.answer = "[none]"
    node.confidence = "low"
    placeholder_answers[node.id] = "[none]"
    return {node.id: "[none]"}
```

改为：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "solve_node_entry")
```

### 修改 2：叶节点主调用之前加检查

在 `full_response = answer_question(question=query_for_leaf, ...)` 之前：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "leaf_answer")
```

### 修改 3：rewrite 重试调用之前加检查，并记录“跳过 rewrite”

在 `retry_response = answer_question(question=rewritten_q, ...)` 之前：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    node.retry_done = True
    node.confidence = "low"
    log_node_event(
        "node_timeout_skip_rewrite",
        node,
        answer=leaf_answer,
        metadata={
            "current_depth": current_depth,
            "timeout_cutoff": True,
            "stage": "leaf_rewrite",
        },
    )
    # 保留 leaf_answer 原值继续传播，不重新置 [none]
else:
    retry_response = answer_question(...)
    ...（原有 rewrite 逻辑）
```

注意：

- 这里不要直接 `return`
- 因为叶节点已经拿到了 `leaf_answer`
- 正确做法是跳过 rewrite 调用，并把原始 `leaf_answer` 继续向上传播

### 修改 4：`generate_right_question_with_llm(...)` 之前加检查

当前计划只覆盖了 `build_question_tree(...)`，但右子树重建前还有 `generate_right_question_with_llm(...)`，它本身也是 LLM 调用，同样可能很慢。

因此在以下几个位置都要补检查：

- `enhanced_right_subtree` 分支里的 `generate_right_question_with_llm(...)`
- 普通 `needs_reconstruction` 分支里的 `generate_right_question_with_llm(...)`
- `refresh_right_question_in_place` 分支里的 `generate_right_question_with_llm(...)`

统一模式：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "generate_right_question")
```

### 修改 5：`build_enhanced_right_subtree(...)` 和 `build_question_tree(...)` 之前加检查

在以下调用之前分别加检查：

- `build_enhanced_right_subtree(...)`
- `build_question_tree(...)`

统一模式：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "right_subtree_rebuild")
```

### 修改 6：parent-level direct retrieval fallback 之前加检查

在 `full_response = answer_question(node.display_question, ...)`（parent fallback）之前：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "parent_direct_fallback")
```

### 修改 7：`get_final_answer(...)` 之前加检查

在 `final_answer = get_final_answer(node.display_question, child_questions, api_url)` 之前：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "get_final_answer")
```

### 修改 8：aggregate-none fallback 和 internal-direct fallback 之前加检查

同样模式，stage 分别标注：

- `"aggregate_none_fallback"`
- `"internal_direct_answer"`

例如：

```python
if question_deadline is not None and time.perf_counter() >= question_deadline:
    return cutoff_node_due_to_timeout(node, current_depth, "aggregate_none_fallback")
```

### 修改 9：在 stats 初始化中增加 `timeout_cutoff_nodes`

当前 `stats` 里只有：

- `rewrite_triggered`
- `rewrite_effective`
- `low_confidence_nodes`
- `parent_direct_fallback_triggered`
- `fallback_gate_checks`
- `fallback_gate_blocked_count`

需要新增：

```python
"timeout_cutoff_nodes": 0,
```

目的：

- 让超时导致的 low-confidence 节点可单独统计
- 避免把 timeout 产物误解释成方法机制本身的低置信度信号

### 修改 10：`load_data_adaptive.py` 同步写出 `timeout_cutoff_nodes`

当前 `load_data_adaptive.py` 在 `write_result_to_file(...)` 中已经写出了以下 adaptive stats：

- `adaptive_rewrite_triggered`
- `adaptive_rewrite_effective`
- `adaptive_low_confidence_nodes`
- `adaptive_parent_direct_fallback`
- `adaptive_fallback_gate_checks`
- `adaptive_fallback_gate_blocked_count`

需要新增一行：

```python
await fout.write(
    f"adaptive_timeout_cutoff_nodes: {int(adaptive.get('timeout_cutoff_nodes', 0))}\n"
)
```

目的：

- 让 timeout 截断节点数真正落到结果文件里
- 方便后续对整轮 run 做统计
- 让 `low_confidence_nodes` 和 `timeout_cutoff_nodes` 可以结合解读
- 避免 `timeout_cutoff_nodes` 只存在于内存里的 `stats dict`，却无法在 `.txt` 输出中使用

---

## 关键约束

- **不修改** `retrieve.py`、`tree_decompose.py`（baseline 文件）
- **不修改** `answer_question()` 签名（无法加 API 级别 request timeout）
- 所有修改集中在 `main/tree_decompose_adaptive.py` 的 `solve_node` 函数内
- pre-call 检查只能阻止“开始新的 LLM 调用”，不能中断已经在进行中的 API 请求，这是已知限制

补充说明：

- 本轮修复重点是 `solve_node` 内部的昂贵调用
- 顶层的 `direct_answer(...)`、`retrieve_documents(...)`、`generate_question_variants(...)`、attempt 级 `build_question_tree(...)` 仍可能受单次调用耗时影响
- 因此本计划的目标不是实现真正的 request-level hard timeout，而是尽量减少进入新重调用的机会，并让 timeout 截断在日志里可见、可区分

---

## 验证

跑几道 pilot run 里超时严重的题，例如：

- `0da908a928c3...`
- `e62b87ef3116...`

确认：

- `timing_total_seconds` 尽量接近 600s，而不是 1300–1900s
- debug JSON 里出现 `node_timeout_cutoff` 事件，`stage` 字段明确标注截断位置
- 若 rewrite 因 timeout 被跳过，debug JSON 里出现 `node_timeout_skip_rewrite`
- `low_confidence_nodes` 中，超时截断的部分可从 `node_timeout_cutoff` 事件区分出来
- `timeout_cutoff_nodes` 有单独计数，不再和普通 low-confidence 完全混在一起

---

## 关键文件

| 文件 | 修改 |
|---|---|
| `main/tree_decompose_adaptive.py` | `solve_node` 内多个 pre-call deadline 检查 + 统一 cutoff helper + timeout 统计/日志 |
| `main/retrieve.py` | **不修改** |
| `main/tree_decompose.py` | **不修改** |
