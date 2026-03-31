import re
from copy import deepcopy
from datetime import datetime


MAX_DOCUMENT_PREVIEW_CHARS = 600
MAX_GENERATION_PREVIEW_CHARS = 4000
MAX_DOCUMENTS_PER_EVENT = 5


DEFAULT_TIMING = {
    "question_total_seconds": 0.0,
    "tree_decompose_total_seconds": 0.0,
    "retrieval_total_seconds": 0.0,
    "generation_total_seconds": 0.0,
    "refined_query_total_seconds": 0.0,
    "direct_fallback_seconds": 0.0,
    "timeout_budget_seconds": 0.0,
    "timeout_elapsed_seconds": 0.0,
    "retry_count": 0,
}

DEFAULT_SUMMARY = {
    "attempt_count": 0,
    "retrieval_failed": False,
    "generation_failed": False,
    "used_direct_fallback": False,
    "retry_count": 0,
    "retrieval_call_count": 0,
    "generation_call_count": 0,
    "timeout_triggered": False,
    "timeout_stage": None,
}


def utc_timestamp():
    return datetime.utcnow().isoformat() + "Z"


def truncate_text(value, limit):
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def sanitize_filename(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown"))
    cleaned = cleaned.strip("._")
    return (cleaned or "unknown")[:120]


def classify_failure_reason(predicted_answer):
    normalized = (predicted_answer or "").strip()
    lowered = normalized.lower()

    if lowered == "[none]":
        return "none_answer"
    if normalized.startswith("Error:"):
        return "error_answer"
    if normalized.startswith("Could not determine an answer"):
        return "undetermined_answer"
    return None


def preview_retrieval_items(items):
    previews = []
    for rank, item in enumerate((items or [])[:MAX_DOCUMENTS_PER_EVENT], start=1):
        preview = {
            "rank": item.get("rank", rank),
        }
        for key in ("index", "id", "title", "score", "rerank_score"):
            if key in item:
                preview[key] = item[key]

        content = item.get("content") or item.get("text") or item.get("preview") or ""
        preview["preview"] = truncate_text(content, MAX_DOCUMENT_PREVIEW_CHARS)
        previews.append(preview)
    return previews


class BadCaseDebugCollector:
    def __init__(self, run_metadata, sample_metadata):
        run_payload = dict(run_metadata)
        run_payload.setdefault("timestamp", utc_timestamp())

        self.payload = {
            "run": run_payload,
            "sample": dict(sample_metadata),
            "outcome": {
                "status": "pending",
                "predicted_answer": None,
                "failure_reason": None,
            },
            "timing": dict(DEFAULT_TIMING),
            "tree": {
                "attempts": [],
                "variants": [],
                "selected_tree": None,
                "used_direct_fallback": False,
            },
            "retrieval": [],
            "generation": [],
            "errors": [],
            "summary": dict(DEFAULT_SUMMARY),
        }

    def add_timing(self, key, seconds):
        if seconds is None:
            return
        current = float(self.payload["timing"].get(key, 0.0) or 0.0)
        self.payload["timing"][key] = current + max(0.0, float(seconds))

    def set_timing(self, key, seconds):
        if seconds is None:
            return
        self.payload["timing"][key] = max(0.0, float(seconds))

    def increment_counter(self, key, amount=1):
        current = int(self.payload["summary"].get(key, 0) or 0)
        self.payload["summary"][key] = current + amount

    def set_retry_count(self, count):
        retry_count = max(0, int(count or 0))
        self.payload["timing"]["retry_count"] = retry_count
        self.payload["summary"]["retry_count"] = retry_count

    def add_retrieval_event(self, stage, query, method, topk1, topk2, items, metadata=None):
        event = {
            "stage": stage,
            "query": query,
            "method": method,
            "topk1": topk1,
            "topk2": topk2,
            "result_count": len(items or []),
            "results": preview_retrieval_items(items),
        }
        if metadata:
            event["metadata"] = deepcopy(metadata)
        self.payload["retrieval"].append(event)

    def add_generation_event(
        self,
        stage,
        model,
        temperature,
        max_tokens,
        raw_response,
        parsed_answer=None,
        metadata=None,
    ):
        event = {
            "stage": stage,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "parsed_answer": parsed_answer,
            "raw_response": truncate_text(raw_response, MAX_GENERATION_PREVIEW_CHARS),
        }
        if metadata:
            event["metadata"] = deepcopy(metadata)
        self.payload["generation"].append(event)

    def add_error(self, stage, message, details=None):
        event = {
            "stage": stage,
            "message": str(message),
        }
        if details:
            event["details"] = deepcopy(details)
        self.payload["errors"].append(event)

    def add_tree_attempt(self, attempt_record):
        self.payload["tree"]["attempts"].append(deepcopy(attempt_record))
        self.payload["summary"]["attempt_count"] = len(self.payload["tree"]["attempts"])

    def add_variant(self, attempt_index, from_question, variant_question, reason="generated", metadata=None):
        event = {
            "attempt_index": attempt_index,
            "from_question": from_question,
            "variant_question": variant_question,
            "reason": reason,
        }
        if metadata:
            event["metadata"] = deepcopy(metadata)
        self.payload["tree"]["variants"].append(event)

    def set_selected_tree(self, tree_record):
        self.payload["tree"]["selected_tree"] = deepcopy(tree_record)

    def set_direct_fallback(self, used=True, reason=None):
        self.payload["tree"]["used_direct_fallback"] = used
        self.payload["summary"]["used_direct_fallback"] = used
        if reason:
            self.payload["tree"]["direct_fallback_reason"] = reason

    def set_timeout(self, triggered=True, stage=None, elapsed_seconds=None, budget_seconds=None):
        self.payload["summary"]["timeout_triggered"] = bool(triggered)
        self.payload["summary"]["timeout_stage"] = stage
        if elapsed_seconds is not None:
            self.payload["timing"]["timeout_elapsed_seconds"] = max(0.0, float(elapsed_seconds))
        if budget_seconds is not None:
            self.payload["timing"]["timeout_budget_seconds"] = max(0.0, float(budget_seconds))

    def set_outcome(self, status, predicted_answer=None, failure_reason=None):
        self.payload["outcome"] = {
            "status": status,
            "predicted_answer": predicted_answer,
            "failure_reason": failure_reason,
        }

    def update_summary(self, **kwargs):
        self.payload["summary"].update(kwargs)

    def to_dict(self):
        return deepcopy(self.payload)
