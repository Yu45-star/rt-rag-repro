# 修复计划：无限 retry + 重建子树缺少 annotation

日期：2026-04-21

## 背景

运行 `fixed38_remaining_qids.txt` 时长时间无输出。两个 bug 叠加导致：

1. **无限 retry**：`load_data_adaptive.py` 的 `while True` 没有最大重试次数。某道题只要一直报 `AttributeError`，就永远不写结果文件。

2. **重建子树缺少 annotation**：`adaptive_solve_tree()` 中动态重建右子树（`build_enhanced_right_subtree` / `build_question_tree`）后没有调用 `annotate_tree_nodes()`，导致新节点缺少 `answer_type`、`entity_anchor` 等属性，访问时抛 `AttributeError: 'QuestionNode' object has no attribute 'answer_type'`。

---

## 改动1：`main/load_data_adaptive.py` — 添加最大重试次数

在 `while True` 循环中，`current_retry += 1` 之后加退出条件（约第 207 行）：

```python
current_retry += 1
collector.set_retry_count(current_retry)
if current_retry >= 3:
    print(f"Max retries reached for qid: {qid}, skipping.")
    predicted_answer = "[none]"
    break
```

---

## 改动2：`main/tree_decompose_adaptive.py` — 重建子树后补 annotation

两处紧接在 `node.right = new_right_node` 之后、`solve_node(...)` 之前各加一行：

**位置1**（`build_enhanced_right_subtree` 路径，约第 452 行）：
```python
node.right = new_right_node
annotate_tree_nodes(new_right_node)   # 补充 annotation，避免 AttributeError
right_answers = solve_node(node.right, True, current_depth + 1)
```

**位置2**（`build_question_tree` 路径，约第 478 行）：
```python
node.right = new_right_node
annotate_tree_nodes(new_right_node)   # 补充 annotation，避免 AttributeError
right_answers = solve_node(node.right, True, current_depth + 1)
```

`annotate_tree_nodes` 已在文件顶部 import，无需新增 import。

---

## 验证

修改后 kill 当前卡死进程，重跑：

```bash
kill <pid>
python main/load_data_adaptive.py --qid-file output/musique/fixed38_remaining_qids.txt
```

验收标准：
- 不再长时间无输出
- 每道题最多 3 次 retry 后写入结果
- 不再出现 `AttributeError: 'QuestionNode' object has no attribute 'answer_type'`

---

## 关键文件

- `main/load_data_adaptive.py`：retry 循环（约第 200-210 行）
- `main/tree_decompose_adaptive.py`：`adaptive_solve_tree()` 中两处右子树重建（约第 452、478 行）
