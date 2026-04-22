from config import *  # noqa: F401, F403

METHOD_TAG = "adaptive_typed_v1_final"

# Runtime trimming for the current bad-case rerun:
# keep 2 trees in the initial attempt, but disable extra variant attempts.
MAX_VARIANTS = 0

ANSWER_TYPE_METHOD = "rules"        # rule-based type inference, no extra LLM calls
ENABLE_TYPE_GUIDANCE = True         # inject answer_type / entity_anchor into retrieval query
ENABLE_TYPE_AWARE_REWRITE = True    # trigger rewrite when answer is suspicious
ENABLE_FALLBACK_SUPPORT_CHECK = True  # gate direct_answer() fallback with LLM support check
ENABLE_MULTI_QUERY_FUSION = True      # fuse multiple retrieval queries at leaf nodes
RETRY_TIMEOUT_BUDGET_FRACTION = 0.25
QUESTION_TIMEOUT_SECONDS = 99999
