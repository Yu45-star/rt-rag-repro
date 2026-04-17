"""
Adaptive Typed-Control extension of tree_decompose.py.

New files only — no modifications to existing baseline code.
Entry point: adaptive_decompose_and_answer_with_variants()
"""

import re
import time
from collections import Counter

from retrieve import answer_question, direct_answer, retrieve_documents, extract_keywords
from tree_decompose import (
    build_question_tree,
    solve_tree,
    get_tree_statistics,
    get_examples_database,
    print_all_nodes,
    build_tree_shape_summary,
    serialize_question_tree,
    save_tree_stats,
    generate_question_variants,
    global_node_counter,
)
from control_helpers import is_answer_suspicious, check_fallback_supported
import tree_decompose as _td  # to reset global_node_counter

from config_adaptive import (
    DATASET, METHOD, CHUNK_SIZE, MIN_SENTENCE, OVERLAP, TOPK1, TOPK2,
    MAX_ITERATIONS, TREES_PER_QUESTION, MAX_TOKENS, DECOMPOSE_TEMPERATURE,
    TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY, NUM_EXAMPLES, MAX_HEIGHT,
    ENHANCED_RIGHT_SUBTREE, RIGHT_SUBTREE_VARIANTS, RIGHT_SUBTREE_TREES_PER_VARIANT,
    MAX_VARIANTS, STATS_FILE_PATH, QUESTION_TIMEOUT_SECONDS,
    ENABLE_TYPE_GUIDANCE, ENABLE_TYPE_AWARE_REWRITE, ENABLE_FALLBACK_SUPPORT_CHECK,
)


# ---------------------------------------------------------------------------
# Helper: rule-based answer type inference
# ---------------------------------------------------------------------------

def infer_answer_type(question: str) -> str:
    q = question.lower()
    if re.search(r'\bwho\b', q):
        return "person"
    if re.search(r'\bwhere\b|\bin which\b|\bwhich (city|country|place|location|state|region|continent)\b', q):
        return "location"
    if re.search(r'\bwhen\b|\bwhat year\b|\bwhat date\b|\bin what year\b', q):
        return "date"
    if re.search(r'\bhow many\b|\bhow much\b|\bwhat (number|count|total|amount)\b', q):
        return "number"
    if re.search(r'\bwhich (company|organization|org|institution|team|party|group)\b', q):
        return "org"
    return "other"


def extract_entity_anchor(question: str):
    """Return the first multi-word capitalised phrase, or None."""
    # Match runs of Title-Cased words (2+ tokens) not at sentence start
    matches = re.findall(r'(?<!\. )(?<![?!] )\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Helper: query builders  (natural-language rewrites, no meta-brackets)
# ---------------------------------------------------------------------------

def build_typed_retrieval_query(question: str, answer_type: str, entity_anchor) -> str:
    """
    Produce a natural-language rewrite that guides both retrieval and generation
    toward the expected answer type.  No bracket meta-language is used because
    answer_question() passes this string to both extract_keywords() and the LLM.
    """
    if answer_type == "person" and entity_anchor:
        return f"Who is the person associated with {entity_anchor}? {question}"
    if answer_type == "person":
        return f"Who is the person that {question.lstrip('Who').lstrip('who').strip()}"
    if answer_type == "location" and entity_anchor:
        return f"What location is related to {entity_anchor}? {question}"
    if answer_type == "date" and entity_anchor:
        return f"What date or year is associated with {entity_anchor}? {question}"
    if answer_type == "number" and entity_anchor:
        return f"What is the number or quantity related to {entity_anchor}? {question}"
    if answer_type == "org" and entity_anchor:
        return f"Which organisation is associated with {entity_anchor}? {question}"
    # Fallback: return as-is when no useful signal
    return question


def build_rewritten_query(question: str, entity_anchor, answer_type: str) -> str:
    """
    A stronger rewrite used after is_answer_suspicious fires.
    Anchors to entity_anchor and adds an explicit type expectation.
    """
    type_phrases = {
        "person": "The answer should be a person's name.",
        "location": "The answer should be a location (city, country, or region).",
        "date": "The answer should be a date or year.",
        "number": "The answer should be a number or quantity.",
        "org": "The answer should be an organisation or company name.",
    }
    type_hint = type_phrases.get(answer_type, "")
    anchor_hint = f"Focus on '{entity_anchor}'. " if entity_anchor else ""
    return f"{anchor_hint}{type_hint} {question}".strip()


# ---------------------------------------------------------------------------
# Node annotation
# ---------------------------------------------------------------------------

def annotate_tree_nodes(root) -> None:
    """Duck-type adaptive attributes onto every QuestionNode in the tree."""
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


# ---------------------------------------------------------------------------
# adaptive_solve_tree — copy of solve_tree with two insertion points
# ---------------------------------------------------------------------------

def adaptive_solve_tree(root, original_question, api_url=None, max_tokens=4000,
                        temperature=0, top_p=1.0, frequency_penalty=0.0,
                        presence_penalty=0.0, examples_db=None, num_examples=20,
                        enhanced_right_subtree=True, right_subtree_variants=2,
                        right_subtree_trees_per_variant=2, max_height=3,
                        placeholder_answers=None, debug_collector=None,
                        stats=None):
    """
    Adaptive variant of solve_tree.  Two behavioural insertions relative to
    the baseline:

    1. Leaf nodes: typed query formulation + optional rewrite on suspicious answers.
    2. Parent nodes: direct retrieval fallback when a dependency child is
       confidence=="low" and answer=="[none]" (blocks the aggregation).
    """
    if placeholder_answers is None:
        placeholder_answers = {}
    if stats is None:
        stats = {}

    processed_node_ids = set()

    # ---- helpers copied verbatim from solve_tree ----

    def log_node_event(stage, node, question=None, answer=None, metadata=None):
        if debug_collector is None:
            return
        event_metadata = {
            "node_type": node.type if node else None,
            "display_question": node.display_question if node else None,
            "depends_on": node.depends_on if node else None,
        }
        if metadata:
            event_metadata.update(metadata)
        debug_collector.add_node_event(
            stage=stage,
            node_id=node.id if node else None,
            question=question if question is not None else (node.display_question if node else None),
            answer=answer,
            metadata=event_metadata,
        )

    def build_trace_metadata(node, current_depth, resolve_mode, actual_question=None):
        return {
            "node_id": node.id,
            "node_type": node.type,
            "node_display_question": node.display_question,
            "node_original_question": node.question,
            "current_depth": current_depth,
            "resolve_mode": resolve_mode,
            "depends_on": node.depends_on,
            "actual_question": actual_question or node.display_question,
        }

    # ---- inner recursive solver ----

    def solve_node(node, updated_tree=False, current_depth=0):
        if node is None:
            return {}

        if node.id in processed_node_ids:
            return {node.id: placeholder_answers.get(node.id, "[none]")}

        processed_node_ids.add(node.id)
        node_answers = {}

        if node.depends_on and node.depends_on not in placeholder_answers:
            def find_node_by_id(search_node, target_id):
                if search_node is None:
                    return None
                if search_node.id == target_id:
                    return search_node
                left_result = find_node_by_id(search_node.left, target_id)
                if left_result:
                    return left_result
                return find_node_by_id(search_node.right, target_id)

            dependent_node = find_node_by_id(root, node.depends_on)
            if dependent_node:
                dependent_answers = solve_node(dependent_node, updated_tree, current_depth)
                node_answers.update(dependent_answers)

        if node.id in placeholder_answers:
            node.answer = placeholder_answers[node.id]
            node_answers[node.id] = node.answer
            log_node_event("reuse_placeholder", node, answer=node.answer,
                           metadata={"current_depth": current_depth})
            return node_answers

        # ---- LEAF NODE ----
        if (node.left is None and node.right is None) or node.type == "None":
            actual_question = node.question

            if node.depends_on and node.depends_on in placeholder_answers:
                dependent_answer = placeholder_answers[node.depends_on]
                if dependent_answer.lower() == "[none]":
                    node.answer = "[none]"
                    node.confidence = "low"
                    placeholder_answers[node.id] = "[none]"
                    node_answers[node.id] = "[none]"
                    return node_answers

                if "[answer_subquestion1]" in actual_question:
                    actual_question = actual_question.replace("[answer_subquestion1]", dependent_answer)
                elif f"[answer from {node.depends_on}]" in actual_question:
                    actual_question = actual_question.replace(
                        f"[answer from {node.depends_on}]", dependent_answer)

                node.display_question = actual_question
                log_node_event("dependency_resolved", node, question=actual_question,
                               answer=dependent_answer,
                               metadata={"current_depth": current_depth,
                                         "depends_on_node": node.depends_on})

            log_node_event("leaf_question", node, question=actual_question,
                           metadata={"current_depth": current_depth})

            # --- INSERTION 1: type-aware leaf query ---
            if ENABLE_TYPE_GUIDANCE:
                query_for_leaf = build_typed_retrieval_query(
                    actual_question, node.answer_type, node.entity_anchor)
            else:
                query_for_leaf = actual_question

            full_response = answer_question(
                question=query_for_leaf,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                max_iterations=MAX_ITERATIONS,
                debug_collector=debug_collector,
                trace_metadata=build_trace_metadata(node, current_depth, "leaf_answer", query_for_leaf),
                stage_prefix=f"tree.node.{node.id}",
            )

            from tree_decompose import extract_answer
            leaf_answer = extract_answer(full_response)

            # --- INSERTION 2: rewrite on suspicious answer ---
            if (ENABLE_TYPE_AWARE_REWRITE
                    and is_answer_suspicious(leaf_answer, node.answer_type)
                    and not node.retry_done):
                stats["rewrite_triggered"] = stats.get("rewrite_triggered", 0) + 1
                rewritten_q = build_rewritten_query(
                    actual_question, node.entity_anchor, node.answer_type)
                retry_response = answer_question(
                    question=rewritten_q,
                    dataset=DATASET,
                    method=METHOD,
                    chunk_size=CHUNK_SIZE,
                    min_sentence=MIN_SENTENCE,
                    overlap=OVERLAP,
                    topk1=TOPK1,
                    topk2=TOPK2,
                    max_iterations=MAX_ITERATIONS,
                    debug_collector=debug_collector,
                    trace_metadata=build_trace_metadata(node, current_depth, "leaf_rewrite", rewritten_q),
                    stage_prefix=f"tree.node.{node.id}.rewrite",
                )
                retry_answer = extract_answer(retry_response)
                if (retry_answer != "[none]"
                        and not is_answer_suspicious(retry_answer, node.answer_type)):
                    leaf_answer = retry_answer
                    stats["rewrite_effective"] = stats.get("rewrite_effective", 0) + 1

                node.retry_done = True
                node.confidence = "low"
                stats["low_confidence_nodes"] = stats.get("low_confidence_nodes", 0) + 1
            else:
                node.confidence = "high"

            node.answer = leaf_answer
            placeholder_answers[node.id] = leaf_answer
            node_answers[node.id] = leaf_answer
            log_node_event("leaf_answer", node, question=actual_question, answer=leaf_answer,
                           metadata={"current_depth": current_depth,
                                     "typed_query": query_for_leaf if ENABLE_TYPE_GUIDANCE else None,
                                     "confidence": node.confidence})

            if (("[answer_subquestion1]" in node.question or "[answer from" in node.question)
                    and not updated_tree
                    and node.answer and node.answer.lower() != "[none]"):
                if node.parent and not node.is_left_child:
                    print(f"node {node.id}: Contains replacement tag and got answer. May need rebuild")

            return node_answers

        # ---- INTERNAL NODE: solve children first ----
        left_answers = solve_node(node.left, updated_tree, current_depth + 1)
        node_answers.update(left_answers)

        needs_reconstruction = False
        if node.right and node.type == "Sequential":
            if ("[answer_subquestion1]" in node.right.question or
                    (node.right.depends_on and
                     f"[answer from {node.right.depends_on}]" in node.right.question)):
                if node.left and node.left.id in placeholder_answers:
                    left_answer = placeholder_answers[node.left.id]
                    if left_answer.lower() != "[none]":
                        needs_reconstruction = True

        if needs_reconstruction and not updated_tree and enhanced_right_subtree:
            from tree_decompose import (generate_right_question_with_llm,
                                        build_enhanced_right_subtree)
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            log_node_event("reconstruct_right_subtree_question", node,
                           question=new_right_question,
                           answer=placeholder_answers[node.left.id],
                           metadata={"current_depth": current_depth,
                                     "left_node_id": node.left.id if node.left else None,
                                     "original_right_question": node.right.question if node.right else None,
                                     "enhanced_right_subtree": True})
            remaining_height = max(max_height - current_depth - 1, 1)
            new_right_node = build_enhanced_right_subtree(
                original_question=new_right_question,
                left_answer=placeholder_answers[node.left.id],
                api_url=api_url, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                examples_db=examples_db, num_examples=num_examples,
                max_height=remaining_height,
                num_variants=right_subtree_variants,
                trees_per_variant=right_subtree_trees_per_variant,
            )
            node.right = new_right_node
            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)

        elif needs_reconstruction and not updated_tree:
            from tree_decompose import generate_right_question_with_llm
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            remaining_height = max(max_height - current_depth - 1, 1)
            new_right_node = build_question_tree(
                new_right_question, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth=current_depth + 1, parent=node, is_left_child=False,
                max_height=remaining_height, placeholder_answers=placeholder_answers,
            )
            node.right = new_right_node
            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)

        else:
            if (node.right and ("[answer_subquestion1]" in node.right.question or
                    (node.right.depends_on and
                     f"[answer from {node.right.depends_on}]" in node.right.question))):
                if node.left and node.left.id in placeholder_answers:
                    from tree_decompose import generate_right_question_with_llm
                    new_question = generate_right_question_with_llm(
                        parent_question=node.question,
                        left_question=node.left.question,
                        left_answer=placeholder_answers[node.left.id],
                        original_right_question=node.right.question,
                    )
                    node.right.question = new_question
                    node.right.display_question = new_question
                    log_node_event("refresh_right_question_in_place", node.right,
                                   question=new_question,
                                   answer=placeholder_answers[node.left.id],
                                   metadata={"current_depth": current_depth,
                                             "parent_node_id": node.id})
            right_answers = solve_node(node.right, updated_tree, current_depth + 1)
            node_answers.update(right_answers)

        # ---- PARENT AGGREGATION ----

        # --- INSERTION 3: parent-level direct retrieval fallback ---
        # Only for [none]-type blocking: a dependency child is low-confidence
        # and returned no answer at all.
        deps = [c for c in (node.left, node.right) if c is not None]
        blocking_children = [
            c for c in deps
            if getattr(c, "confidence", "high") == "low" and c.answer == "[none]"
        ]
        if blocking_children:
            print(f"node {node.id}: child blocking with [none]+low-confidence, "
                  "triggering parent-level direct retrieval fallback")
            full_response = answer_question(
                node.display_question,
                dataset=DATASET, method=METHOD,
                chunk_size=CHUNK_SIZE, min_sentence=MIN_SENTENCE, overlap=OVERLAP,
                topk1=TOPK1, topk2=TOPK2, max_iterations=MAX_ITERATIONS,
                debug_collector=debug_collector,
                trace_metadata=build_trace_metadata(node, current_depth,
                                                    "parent_direct_retrieval_fallback"),
                stage_prefix=f"tree.node.{node.id}.parent_fallback",
            )
            from tree_decompose import extract_answer
            answer = extract_answer(full_response)
            node.answer = answer
            node.confidence = "low"
            node.used_direct_retrieval_fallback = True
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
            stats["parent_direct_fallback_triggered"] = (
                stats.get("parent_direct_fallback_triggered", 0) + 1)
            log_node_event("parent_direct_retrieval_fallback", node,
                           question=node.display_question, answer=answer,
                           metadata={"current_depth": current_depth,
                                     "blocking_children": [c.id for c in blocking_children]})
            return node_answers

        # ---- normal aggregation (unchanged from solve_tree) ----
        child_questions = []
        valid_child_answers = False

        if node.left and node.left.id in node_answers:
            left_answer = node_answers[node.left.id]
            child_questions.append((node.left.display_question, left_answer))
            if left_answer.lower() != "[none]":
                valid_child_answers = True

        if node.right and node.right.id in node_answers:
            right_answer = node_answers[node.right.id]
            child_questions.append((node.right.display_question, right_answer))
            if right_answer.lower() != "[none]":
                valid_child_answers = True

        if node.type == "Sequential" and node.left and node.left.id in node_answers:
            if node_answers[node.left.id].lower() == "[none]":
                valid_child_answers = False

        if valid_child_answers and child_questions:
            from tree_decompose import get_final_answer
            final_answer = get_final_answer(node.display_question, child_questions, api_url)
            extracted = re.search(r'final answer is:\s*(.*)', final_answer, re.DOTALL)
            if not extracted:
                extracted = re.search(r'answer is:\s*(.*)', final_answer, re.DOTALL)
            node.answer = extracted.group(1).strip() if extracted else final_answer
            placeholder_answers[node.id] = node.answer
            node_answers[node.id] = node.answer
            log_node_event("aggregate_from_children", node,
                           question=node.display_question, answer=node.answer,
                           metadata={"current_depth": current_depth,
                                     "child_questions": [{"question": q, "answer": a}
                                                         for q, a in child_questions]})

            if node.answer.lower() == "[none]":
                full_response = answer_question(
                    node.display_question,
                    dataset=DATASET, method=METHOD,
                    chunk_size=CHUNK_SIZE, min_sentence=MIN_SENTENCE, overlap=OVERLAP,
                    topk1=TOPK1, topk2=TOPK2, max_iterations=MAX_ITERATIONS,
                    debug_collector=debug_collector,
                    trace_metadata=build_trace_metadata(node, current_depth,
                                                        "aggregate_none_fallback"),
                    stage_prefix=f"tree.node.{node.id}.fallback",
                )
                from tree_decompose import extract_answer
                answer = extract_answer(full_response)
                node.answer = answer
                placeholder_answers[node.id] = answer
                node_answers[node.id] = answer
                log_node_event("aggregate_fallback_answer", node,
                               question=node.display_question, answer=answer,
                               metadata={"current_depth": current_depth,
                                         "reason": "aggregate_returned_none"})
        else:
            full_response = answer_question(
                node.display_question,
                dataset=DATASET, method=METHOD,
                chunk_size=CHUNK_SIZE, min_sentence=MIN_SENTENCE, overlap=OVERLAP,
                topk1=TOPK1, topk2=TOPK2, max_iterations=MAX_ITERATIONS,
                debug_collector=debug_collector,
                trace_metadata=build_trace_metadata(node, current_depth,
                                                    "internal_direct_answer"),
                stage_prefix=f"tree.node.{node.id}.fallback",
            )
            from tree_decompose import extract_answer
            answer = extract_answer(full_response)
            node.answer = answer
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
            log_node_event("internal_direct_answer", node,
                           question=node.display_question, answer=answer,
                           metadata={"current_depth": current_depth,
                                     "reason": "no_valid_child_answers"})

        return node_answers

    # ---- top-level call ----
    all_answers = solve_node(root, False, 0)
    final_result = root.answer if root.answer else "[none]"

    if debug_collector is not None:
        debug_collector.add_node_event(
            stage="final_aggregated_answer",
            node_id=root.id if root else None,
            question=original_question,
            answer=final_result,
            metadata={"root_id": root.id if root else None,
                      "resolved_node_count": len(all_answers)},
        )

    return final_result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def adaptive_decompose_and_answer_with_variants(
        question,
        trees_per_question=TREES_PER_QUESTION,
        api_url=None,
        max_tokens=MAX_TOKENS,
        temperature=DECOMPOSE_TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        num_examples=NUM_EXAMPLES,
        max_height=MAX_HEIGHT,
        enhanced_right_subtree=ENHANCED_RIGHT_SUBTREE,
        right_subtree_variants=RIGHT_SUBTREE_VARIANTS,
        right_subtree_trees_per_variant=RIGHT_SUBTREE_TREES_PER_VARIANT,
        max_variants=MAX_VARIANTS,
        stats_file_path=STATS_FILE_PATH,
        question_started_at=None,
        question_deadline=None,
        timeout_budget_seconds=QUESTION_TIMEOUT_SECONDS,
        debug_collector=None,
        stats=None,
) -> tuple:
    """
    Adaptive variant of decompose_and_answer_with_variants.
    Returns (answer_str, stats_dict).
    stats_dict keys:
      rewrite_triggered, rewrite_effective, low_confidence_nodes,
      parent_direct_fallback_triggered,
      fallback_gate_checks, fallback_gate_blocked_count
    """
    if stats is None:
        stats = {
            "rewrite_triggered": 0,
            "rewrite_effective": 0,
            "low_confidence_nodes": 0,
            "parent_direct_fallback_triggered": 0,
            "fallback_gate_checks": 0,
            "fallback_gate_blocked_count": 0,
        }

    _td.global_node_counter = 0
    tree_decompose_started = time.perf_counter()

    try:
        examples_db = get_examples_database()

        attempt_count = 0
        current_question = question
        attempted_questions = [question]
        initial_height = 0
        final_height = 0
        success = False

        def timed_out():
            return question_deadline is not None and time.perf_counter() >= question_deadline

        def timeout_elapsed_seconds():
            if question_started_at is None:
                return None
            return time.perf_counter() - question_started_at

        def _gated_direct_answer(stage, attempt_record=None):
            """Call direct_answer then optionally gate with support check."""
            candidate = direct_answer(
                question, dataset=DATASET, method=METHOD,
                chunk_size=CHUNK_SIZE, min_sentence=MIN_SENTENCE, overlap=OVERLAP,
                topk1=TOPK1, topk2=TOPK2,
                debug_collector=debug_collector,
            )
            if ENABLE_FALLBACK_SUPPORT_CHECK and candidate.lower() != "[none]":
                stats["fallback_gate_checks"] += 1
                docs = retrieve_documents(
                    query=extract_keywords(question),
                    dataset=DATASET, method=METHOD,
                    chunk_size=CHUNK_SIZE, min_sentence=MIN_SENTENCE, overlap=OVERLAP,
                    topk1=TOPK1, topk2=TOPK2,
                    debug_collector=debug_collector,
                    stage=f"adaptive.fallback_gate.{stage}",
                )
                if not check_fallback_supported(question, candidate, docs):
                    stats["fallback_gate_blocked_count"] += 1
                    print(f"Fallback gate blocked candidate '{candidate}' at stage '{stage}'")
                    return "[none]"
            return candidate

        def finalize_timeout_fallback(stage, attempt_record=None):
            elapsed = timeout_elapsed_seconds()
            print(f"\n{'-'*80}")
            print(f"Question exceeded timeout budget at stage '{stage}'. Using direct lookup.")
            print(f"{'-'*80}")

            if debug_collector is not None:
                debug_collector.set_timeout(True, stage=stage, elapsed_seconds=elapsed,
                                            budget_seconds=timeout_budget_seconds)
                debug_collector.set_direct_fallback(True, reason=f"timeout:{stage}")
                debug_collector.add_error(
                    "tree_decompose.timeout",
                    f"Timeout budget exceeded at stage {stage}",
                    {"stage": stage, "timeout_budget_seconds": timeout_budget_seconds,
                     "timeout_elapsed_seconds": round(elapsed or 0.0, 4)},
                )

            direct_fallback_started = time.perf_counter()
            final_answer = _gated_direct_answer(f"timeout.{stage}", attempt_record)
            direct_fallback_elapsed = time.perf_counter() - direct_fallback_started

            if attempt_record is not None:
                attempt_record["final_answer"] = final_answer
                attempt_record["direct_fallback_seconds"] = direct_fallback_elapsed

            if debug_collector is not None:
                debug_collector.add_timing("direct_fallback_seconds", direct_fallback_elapsed)
                if attempt_record is not None:
                    if attempt_record.get("selected_tree"):
                        debug_collector.set_selected_tree(
                            {"attempt_index": attempt_record["attempt_index"],
                             **attempt_record["selected_tree"]})
                    debug_collector.add_tree_attempt(attempt_record)

            save_tree_stats(question, final_answer, initial_height, final_height,
                            stats_file_path, False)
            return final_answer

        while attempt_count <= max_variants:
            if timed_out():
                return (finalize_timeout_fallback("before_attempt"), stats)

            attempt_started = time.perf_counter()
            attempt_record = {
                "attempt_index": attempt_count + 1,
                "question": current_question,
                "trees_generated": 0,
                "tree_shapes": [],
                "selected_tree": None,
                "final_answer": None,
                "solve_tree_seconds": 0.0,
                "variant_generation_seconds": 0.0,
                "attempt_total_seconds": 0.0,
            }

            print(f"\n{'='*80}")
            print(f"ADAPTIVE ATTEMPT {attempt_count + 1} with question: '{current_question}'")
            print(f"{'='*80}")

            all_trees = []
            print(f"\nGenerating {trees_per_question} trees (max height: {max_height})")

            for j in range(trees_per_question):
                if timed_out():
                    attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                    return (finalize_timeout_fallback("before_build_tree", attempt_record), stats)

                tree_temp = 0
                print(f"\nBuilding tree {j + 1} (temperature={tree_temp}, max_height={max_height}):")
                root = build_question_tree(
                    current_question, api_url, max_tokens, tree_temp, top_p,
                    frequency_penalty, presence_penalty, examples_db, num_examples,
                    depth=0, placeholder_answers={}, max_height=max_height,
                )
                height, node_count = get_tree_statistics(root)
                all_trees.append({
                    "tree": root, "tree_num": j + 1,
                    "height": height, "node_count": node_count,
                    "question_text": current_question,
                })
                print(f"Tree {j + 1} - Height: {height}, Node count: {node_count}")

            attempt_record["trees_generated"] = len(all_trees)

            tree_shape_counter = Counter()
            for tree_info in all_trees:
                shape = (tree_info["height"], tree_info["node_count"])
                tree_shape_counter[shape] += 1

            attempt_record["tree_shapes"] = build_tree_shape_summary(tree_shape_counter)

            print("\nTree shape frequencies (height, node count):")
            for shape, count in tree_shape_counter.most_common():
                print(f"Height: {shape[0]}, Node count: {shape[1]} - Frequency: {count}")

            if tree_shape_counter:
                most_common_shape, _ = tree_shape_counter.most_common(1)[0]
                attempt_record["most_common_shape"] = {
                    "height": most_common_shape[0],
                    "node_count": most_common_shape[1],
                    "frequency": tree_shape_counter[most_common_shape],
                }

                most_common_trees = [
                    t for t in all_trees
                    if (t["height"], t["node_count"]) == most_common_shape
                ]

                if most_common_trees:
                    tree_info = most_common_trees[0]
                    tree_root = tree_info["tree"]
                    question_text = tree_info["question_text"]
                    initial_height = tree_info["height"]

                    print(f"\n{'-'*80}")
                    print(f"Solving tree {tree_info['tree_num']} "
                          f"(height={tree_info['height']}, nodes={tree_info['node_count']})")
                    print(f"{'-'*80}")
                    print_all_nodes(tree_root)

                    # Annotate the chosen tree with adaptive attributes
                    annotate_tree_nodes(tree_root)

                    placeholder_answers = {}
                    if timed_out():
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        return (finalize_timeout_fallback("before_solve_tree", attempt_record), stats)

                    solve_tree_started = time.perf_counter()
                    try:
                        answer = adaptive_solve_tree(
                            tree_root, question_text, api_url, max_tokens,
                            temperature, top_p, frequency_penalty, presence_penalty,
                            examples_db, num_examples,
                            enhanced_right_subtree=enhanced_right_subtree,
                            right_subtree_variants=right_subtree_variants,
                            right_subtree_trees_per_variant=right_subtree_trees_per_variant,
                            max_height=max_height,
                            placeholder_answers=placeholder_answers,
                            debug_collector=debug_collector,
                            stats=stats,
                        )
                    except Exception as e:
                        attempt_record["solve_tree_seconds"] = time.perf_counter() - solve_tree_started
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        if debug_collector is not None:
                            debug_collector.add_error(
                                "tree_decompose.solve_tree", str(e),
                                {"exception_type": type(e).__name__,
                                 "attempt_index": attempt_record["attempt_index"],
                                 "tree_num": tree_info["tree_num"],
                                 "solve_tree_seconds": round(attempt_record["solve_tree_seconds"], 4),
                                 "attempt_total_seconds": round(attempt_record["attempt_total_seconds"], 4)},
                            )
                            debug_collector.add_tree_attempt(attempt_record)
                        raise

                    attempt_record["solve_tree_seconds"] = time.perf_counter() - solve_tree_started
                    final_height, _ = get_tree_statistics(tree_root)
                    attempt_record["selected_tree"] = {
                        "tree_num": tree_info["tree_num"],
                        "question": question_text,
                        "initial_height": initial_height,
                        "final_height": final_height,
                        "node_count": tree_info["node_count"],
                        "answer": answer,
                        "tree": serialize_question_tree(tree_root),
                    }
                    attempt_record["final_answer"] = answer

                    print(f"\nSelected tree returned answer: '{answer}'")

                    if answer.lower() != "[none]":
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        if debug_collector is not None:
                            debug_collector.set_selected_tree(
                                {"attempt_index": attempt_record["attempt_index"],
                                 **attempt_record["selected_tree"]})
                            debug_collector.add_tree_attempt(attempt_record)
                        print("Found valid answer, stopping")
                        success = True
                        save_tree_stats(question, answer, initial_height, final_height,
                                        stats_file_path, success)
                        return (answer, stats)

                    if debug_collector is not None:
                        debug_collector.add_error(
                            "tree_decompose.none_answer", "Selected tree returned [none]",
                            {"attempt_index": attempt_record["attempt_index"],
                             "tree_num": tree_info["tree_num"],
                             "solve_tree_seconds": round(attempt_record["solve_tree_seconds"], 4)},
                        )
                    print("Tree returned [none], will try with a new question variant")

            attempt_count += 1
            if attempt_count <= max_variants:
                print(f"\n{'-'*80}")
                print(f"No valid answer. Generating variant {attempt_count}")
                print(f"{'-'*80}")

                if timed_out():
                    attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                    return (finalize_timeout_fallback("before_variant_generation", attempt_record), stats)

                variant_started = time.perf_counter()
                new_variants = generate_question_variants(question, num_variants=1)
                attempt_record["variant_generation_seconds"] = time.perf_counter() - variant_started

                if len(new_variants) > 1:
                    new_question = new_variants[1]
                else:
                    print("Warning: variant generation failed, using original question")
                    if debug_collector is not None:
                        debug_collector.add_error(
                            "tree_decompose.variant_generation",
                            "Failed to produce a new variant",
                            {"attempt_index": attempt_count,
                             "variant_generation_seconds": round(
                                 attempt_record["variant_generation_seconds"], 4)},
                        )
                    new_question = question

                attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                if debug_collector is not None:
                    if attempt_record.get("selected_tree"):
                        debug_collector.set_selected_tree(
                            {"attempt_index": attempt_record["attempt_index"],
                             **attempt_record["selected_tree"]})
                    debug_collector.add_variant(
                        attempt_record["attempt_index"], current_question, new_question,
                        reason="generated_variant",
                        metadata={"duration_seconds": round(
                            attempt_record["variant_generation_seconds"], 4)},
                    )
                    debug_collector.add_tree_attempt(attempt_record)
                current_question = new_question
                attempted_questions.append(current_question)

            else:
                print(f"\n{'-'*80}")
                print("Exhausted all variants. Using direct lookup on original question.")
                print(f"{'-'*80}")

                if debug_collector is not None:
                    debug_collector.set_direct_fallback(True, reason="exhausted_variants")
                    debug_collector.add_error(
                        "tree_decompose.direct_fallback",
                        "Exhausted variants, falling back to direct_answer",
                        {"attempt_count": attempt_count},
                    )

                direct_fallback_started = time.perf_counter()
                final_answer = _gated_direct_answer("exhausted_variants", attempt_record)
                direct_fallback_elapsed = time.perf_counter() - direct_fallback_started

                attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                attempt_record["final_answer"] = final_answer
                attempt_record["direct_fallback_seconds"] = direct_fallback_elapsed
                if debug_collector is not None:
                    debug_collector.add_timing("direct_fallback_seconds", direct_fallback_elapsed)
                    debug_collector.add_tree_attempt(attempt_record)

                save_tree_stats(question, final_answer, initial_height, final_height,
                                stats_file_path, False)
                return (final_answer, stats)

        save_tree_stats(question, "Could not determine an answer", initial_height, final_height,
                        stats_file_path, False)
        return ("Could not determine an answer after trying original question and variants.", stats)

    finally:
        if debug_collector is not None:
            debug_collector.add_timing("tree_decompose_total_seconds",
                                       time.perf_counter() - tree_decompose_started)
