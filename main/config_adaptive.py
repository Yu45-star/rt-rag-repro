from config import *  # noqa: F401, F403

METHOD_TAG = "adaptive_typed_v1"

ANSWER_TYPE_METHOD = "rules"        # rule-based type inference, no extra LLM calls
ENABLE_TYPE_GUIDANCE = True         # inject answer_type / entity_anchor into retrieval query
ENABLE_TYPE_AWARE_REWRITE = True    # trigger rewrite when answer is suspicious
ENABLE_FALLBACK_SUPPORT_CHECK = True  # gate direct_answer() fallback with LLM support check
RETRY_TIMEOUT_BUDGET_FRACTION = 0.25
