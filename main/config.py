import os

# Tree generation parameters for hierarchical QA
TREES_PER_QUESTION = 2           # Number of trees to generate per question (for consensus-based QA)
MAX_TOKENS = 1600                # Maximum number of tokens allowed per tree
DECOMPOSE_TEMPERATURE = 0.8
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
NUM_EXAMPLES = 15                # Number of few-shot examples
MAX_HEIGHT = 3                   # Maximum depth of the generated tree
ENHANCED_RIGHT_SUBTREE = True
RIGHT_SUBTREE_VARIANTS = 1
RIGHT_SUBTREE_TREES_PER_VARIANT = 1
MAX_VARIANTS = 1

# Path to save run-time statistics and logs
STATS_FILE_PATH = "output/statistics_log.txt"

# OpenAI-compatible language model API settings
BASE_URL = os.getenv("RT_RAG_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("RT_RAG_API_KEY", "YOUR_KEY")
MODEL_NAME = os.getenv("RT_RAG_MODEL", "gpt-4o-mini")

# Path to save generated dense embeddings
EMBEDDING_DATA = "data/embeddings"

# Embedding service settings used for dense retrieval query encoding
RANKER_URL = os.getenv("RT_RAG_RANKER_URL", os.getenv("RT_RAG_EMBED_BASE_URL", "http://localhost:8001/v1"))
RANKER_KEY = os.getenv("RT_RAG_RANKER_KEY", os.getenv("RT_RAG_EMBED_API_KEY", os.getenv("OPENAI_API_KEY", "YOUR_KEY")))

# Retrieval configuration
RETRIEVE_TEMPERATURE = 0.3
DATASET = "musique"             # Dataset name (e.g., "musique", "hotpotqa", etc.)
METHOD = "dense"                # Retrieval method: "dense" or "bm25"
CHUNK_SIZE = 200                # Max number of words per chunk
MIN_SENTENCE = 2                # Minimum number of sentences per chunk
OVERLAP = 2                     # Number of overlapping sentences between chunks
TOPK1 = 25                      # Top-K candidates from initial retrieval
TOPK2 = 8                      # Top-K reranked candidates
SAMPLING_ITERATIONS = 2        # Number of sampling iterations for consensus
MAX_ITERATIONS = 2             # Maximum number of iterations for query rewriting

# Root output directory for saving predictions/results
OUTPUT_DIR_ROOT = "output"

# Concurrency control
MAX_CONCURRENT = 1              # Maximum number of concurrent QA jobs

# Whole-question timeout budget in seconds before falling back to direct retrieval
QUESTION_TIMEOUT_SECONDS = 600

# Path to evaluation dataset (in .jsonl format)
DATA_PATH = "data/longbench/musique_100_seed42.jsonl"
