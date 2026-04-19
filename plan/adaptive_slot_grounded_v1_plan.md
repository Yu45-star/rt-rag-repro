# 计划：Adaptive Slot-Grounded v1

日期：2026-04-18

## 一、背景

前一阶段的 `adaptive_typed_v1` 已完成 MuSiQue bad-case pilot、bug 修复后单题重跑、以及 12 题代表性 bad-case 小规模重跑。当前阶段结论见：

- [plan/adaptive_badcase_status_2026-04-18.md](/workspace/projects/rt-rag-repro/plan/adaptive_badcase_status_2026-04-18.md:1)

从该阶段总结可以得到两个关键判断：

1. 当前 hardest bad cases 中，问题不只是“有没有检索到相关文档”，而是经常出现：
   - evidence 已经覆盖到答案附近
   - 但没有对齐题目要求的槽位、粒度或关系
2. `adaptive_typed_v1` 的 typed query / rewrite / fallback gate 有一定机制信号，但没有稳定转化成 EM 提升

因此，下一阶段不再优先继续调 query wording，也不直接转向高成本 retrieval 重构，而是尝试一个更贴近当前错误形态的新方向：

**slot-aware grounded answering**

也就是：

> 在生成候选答案之后，不立即接受该答案，而是结合检索到的 evidence，显式检查候选答案是否符合题目要求的答案槽位、粒度与关系；若不符合，则基于现有 evidence 再做一次受约束的 grounded extraction，尝试提取一个更符合题目要求的短答案。

---

## 二、方法动机

### 2.1 为什么不是继续优化 typed query

当前小样本分析表明，很多 bad case 并不是 query 完全跑偏，而是：

- 题目要 `county`，模型答了 `city`
- 题目要 `reason`，模型答了 `location`
- 题目要 `symbol`，模型答了 `manufacturer`
- 题目要标准短答案，模型给了接近正确但不 canonical 的输出

这意味着，仅靠 query rewrite 已经不是最直接的发力点。

### 2.2 为什么优先打 B/C 类错误

根据 12 题日志级归类：

- A 类：evidence 基本够用，但抽取/规范化失败
- B 类：evidence 覆盖到答案附近，但没有对齐精确槽位
- C 类：evidence 只覆盖到链路局部，或支持了错误/不完整链
- D 类：当前日志看起来没有召回到足够支持 gold 的关键 evidence

其中：

- A 类可通过后处理缓解，但天花板有限
- D 类更像 retrieval 问题，改动成本大、风险高
- **B 类和部分 C 类** 最适合通过“基于现有 evidence 的受约束抽取”来尝试修复

因此，本计划优先针对：

- 答案槽位不对
- 地理/层级粒度不对
- 候选答案与 evidence 有关，但不满足问题要求
- 中间链条部分成立，但最终目标关系没有被正确落到答案上

---

## 三、方法定义

工作名称：

- `adaptive_slot_grounded_v1`

核心思想：

1. 先沿用当前系统生成一个候选答案 `candidate`
2. 预测题目需要的答案槽位 `answer_slot`
3. 用规则或轻量判断检查 `candidate` 是否与 `answer_slot` 匹配
4. 若不匹配，则在现有 evidence 上做一次 grounded re-extraction
5. 返回重新抽取的短答案，或 `[none]`

这个方法不是新做一次大规模 retrieval，而是：

- 尽量利用已经检索到的 evidence
- 强化答案槽位、粒度与关系约束
- 把“相关但不对”的答案拉回到更符合题目要求的输出

---

## 四、目标错误类型

### 4.1 主要目标

#### 目标一：地理层级 / 粒度错误

代表问题：

- `county` vs `city`
- `province` vs `country`
- `region` vs `city`

目标是避免：

- evidence 已经给出地点信息，但答案粒度不对

#### 目标二：答案槽位错误

代表问题：

- `reason` vs `location`
- `symbol` vs `manufacturer`
- `role` vs `actor`
- `person` vs `organization`

目标是避免：

- evidence 与题目相关，但回答的不是题目要求的那一类对象

#### 目标三：接近正确但不 canonical 的短答案

代表问题：

- `Philip` vs `Philip Mountbatten`
- 包含 gold 但缺少必要限定词
- 输出过长但主体已经接近正确

目标是：

- 在不增加大规模生成开销的前提下，尽量把接近正确的候选答案压成更标准的短答案

### 4.2 次要目标

对部分 C 类问题尝试缓解：

- 当 evidence 覆盖到链路局部，但最终关系没有落到答案时
- 通过受约束抽取，让模型从已有 evidence 中找出更符合题目问法的答案 span

注意：

本版本不承诺系统性解决真正的 retrieval miss，也不把 D 类问题作为主要攻克对象。

---

## 五、最小实现范围

### 5.1 新增文件

建议新建以下文件，不改 baseline：

```text
main/
  slot_grounding_helpers.py
  tree_decompose_slot_grounded.py
  config_slot_grounded.py
  load_data_slot_grounded.py
```

若希望更轻量，也可以在 adaptive 分支上最小复用，但仍建议和当前 `adaptive_typed_v1` 区分开，避免混淆结果来源。

### 5.2 不做的事

本版本明确不做：

- 不大改 retrieval top-k 策略
- 不增加多轮 retrieval fusion
- 不放宽 evidence gate 成为 fail-open
- 不引入复杂 relation parser
- 不尝试一次性解决所有 multi-hop 链路错误

目标是做一个最小、可解释、易验证的 v1。

---

## 六、核心模块设计

### 6.1 `infer_answer_slot(question)`

根据问题表面形式推断更细的答案槽位，而不是只做粗粒度 `answer_type`。

建议初版使用规则，不增加额外 LLM 调用。

示例映射：

- `which county` -> `county`
- `which city` -> `city`
- `which province` -> `province`
- `which country` -> `country`
- `what is the seat of` -> `county_seat`
- `why` -> `reason`
- `who plays` / `what character` -> `role_or_character`
- `who is the spouse` -> `person`
- `what is the symbol` -> `symbol_name`
- `when` / `what year` -> `date`
- 其他 -> `generic`

### 6.2 `is_slot_mismatch(question, candidate, answer_slot, evidence)`

判断当前候选答案是否与问题要求的槽位相冲突。

初版可以采用规则 + 轻量启发式：

- `answer_slot = county`，candidate 像 city / state / country -> mismatch
- `answer_slot = reason`，candidate 像地点名 -> mismatch
- `answer_slot = symbol_name`，candidate 像 company/org -> mismatch
- `answer_slot = role_or_character`，candidate 明显是演员本人 -> mismatch
- `candidate` 过长、解释性太强，也可视为弱 mismatch

这里的重点不是“严格证明错误”，而是识别“很可能答错槽位”的情况。

### 6.3 `grounded_reextract_answer(question, candidate, answer_slot, evidence)`

在 mismatch 时，从已有 evidence 中做一次受约束的重新抽取。

输入：

- 原问题
- 当前 candidate
- answer_slot
- 已检索 evidence

建议 prompt：

```text
Question: ...
Required answer slot: county / city / reason / symbol_name / ...
Current candidate answer: ...
Evidence:
...

Return the shortest answer span from the evidence that best matches the required answer slot.
If the evidence does not support such an answer, return [none].
Output only the answer.
```

输出要求：

- 必须是短答案
- 不带解释
- 找不到就返回 `[none]`

### 6.4 `root_answer_cleanup(answer, answer_slot)`

在最终输出前做轻量确定性清洗。

可考虑：

- 去掉前缀解释话术
- 去掉多余括注或多余并列说明
- 只保留首个短实体 span
- date / person / location 类答案做保守清洗

注意：

这个模块只作为辅助，不作为方法主贡献点。

---

## 七、系统接入位置

### 7.1 候选答案产生后

无论候选答案来自：

- 正常 tree solve
- parent-level direct retrieval fallback
- direct fallback

都统一经过：

1. `infer_answer_slot`
2. `is_slot_mismatch`
3. 若 mismatch，则 `grounded_reextract_answer`
4. 再做 `root_answer_cleanup`

### 7.2 与 evidence gate 的关系

本版本不取消现有 evidence gate。

更合理的接入顺序建议为：

1. 产生 candidate
2. 若已有 evidence，可先做 slot-aware mismatch 检查
3. mismatch 时先尝试 grounded re-extraction
4. 最终答案再进入 evidence gate 或与其逻辑协调

具体实现时可以简化为：

- 对正常 solve 路径直接做 slot-grounded extraction
- 对 fallback 路径也做一次 slot-grounded extraction
- gate 逻辑保留，但不做 fail-open

---

## 八、实验设计

### 8.1 第一阶段：代表性 bad-case 小样本

先不跑全量，优先在当前已经分析过的 12 题上验证。

参考文件：

- [output/musique/bad_cases_rerun_pilot_qids.txt](/workspace/projects/rt-rag-repro/output/musique/bad_cases_rerun_pilot_qids.txt:1)

重点关注的题型：

- B 类：`6e64...`, `6d57...`, `096c9...`
- A 类：`f4f04...`
- C 类：`a621...`, `2df258...`, `d51c...`

预期：

- B 类应最有机会改善
- A 类可能有少量提升
- C 类可能只有部分改善
- D 类不应期待显著收益

### 8.2 第二阶段：若小样本有信号，再扩到 bad-case 子集

若小样本中至少出现以下之一，可考虑扩大：

- 至少 `2-3` 题 strict EM 改善
- 若 EM 未改善，但明显减少“槽位错误/粒度错误/近 miss”
- `Philip` 这类 near miss 被压到更完整 canonical answer
- county/city 这类题出现正确层级输出

若小样本几乎无变化，则停止扩大，不继续在此方向上投入。

---

## 九、评估重点

除了 EM 以外，本版本应额外记录：

- `slot_mismatch_triggered_count`
- `grounded_reextract_attempted_count`
- `grounded_reextract_success_count`
- `cleanup_changed_answer_count`

并重点观察：

- 原本 B 类错误是否减少
- 原本 A 类 near miss 是否转成正确短答案
- 是否产生新的过度保守 `[none]`
- 是否显著增加 runtime

---

## 十、成功标准

这个版本的成功，不要求“大幅提升所有 bad case”。

更现实的标准是：

1. 在 B 类错误上出现可解释的修复
2. 在 A 类 near miss 上出现少量 EM 改善
3. 不明显加剧 timeout
4. 可以明确说明方法作用于哪类错误、为什么有效

若这些条件都不满足，则说明：

- 当前 bad case 的主瓶颈仍然更接近 retrieval / multi-hop chain failure
- slot-grounded 方法不足以成为主线

---

## 十一、与前一阶段的关系

本计划不是推翻 `adaptive_typed_v1`，而是沿着其失败分析继续收缩问题：

- `adaptive_typed_v1` 证明了 typed-control 有一定机制信号
- 但也暴露出：很多 hardest bad cases 不是 query wording 问题，而是答案槽位与 evidence 对齐失败

因此：

- `adaptive_typed_v1` 可以看作前一阶段
- `adaptive_slot_grounded_v1` 则是基于日志分析提出的下一阶段

---

## 十二、当前建议

建议下一步执行顺序：

1. 先冻结当前 `adaptive_typed_v1` 结论
2. 实现 `adaptive_slot_grounded_v1` 的最小版本
3. 只在 12 题代表性 bad-case 子集上验证
4. 根据结果决定是否扩大实验

一句话总结：

> 下一阶段最值得尝试的，不是继续改 query，也不是直接大改 retrieval，而是利用现有 evidence 做一次面向答案槽位、粒度与关系的 grounded extraction，把“相关但不对”的答案尽量拉回到更符合题目要求的短答案。
