

# 🧠🌳Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering




## 🔍 What is RT-RAG? 
![RT-RAG Overview](assets/overview.png)
**RT-RAG** systematically decomposes complex multi-hop questions into explicit **binary reasoning trees**. It leverages structured entity analysis and **consensus-based tree selection** to ensure e decomposition, clearly separating core queries, known entities, and unknown targets.

Once the tree is built, a **bottom-up traversal strategy** is used to iteratively rewrite and refine sub-questions. This process efficiently collects high-quality evidence while mitigating error propagation through recursive reasoning.



---

## ⚙️ 1. Environment Setup

### ✅ Quick Start With uv

Cloning the repo on a fresh machine, the shortest setup path is:

```bash
python3 -m pip install --user uv
uv sync --frozen
source .venv/bin/activate
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

This project now uses `pyproject.toml` + `uv.lock` as the source of truth for dependencies, so everyone installs the same resolved environment.

### ✅ Create or Refresh the Environment

Install `uv` first if it is not already available:

```bash
python3 -m pip install --user uv
```

Create the environment and sync dependencies from `pyproject.toml`:

```bash
uv venv
uv sync
source .venv/bin/activate
```

Install the spaCy English model separately so dependency sync stays portable:

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

`requirements.txt` is kept only as a compatibility file. The primary dependency source is `pyproject.toml`.

The config files now read API secrets from environment variables first, so you do not need to hardcode keys in the repository.

For OpenAI-only usage:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export RT_RAG_BASE_URL="https://api.openai.com/v1"
export RT_RAG_API_KEY="$OPENAI_API_KEY"
export RT_RAG_RANKER_URL="https://api.openai.com/v1"
export RT_RAG_RANKER_KEY="$OPENAI_API_KEY"
export RT_RAG_EMBED_BASE_URL="https://api.openai.com/v1"
export RT_RAG_EMBED_API_KEY="$OPENAI_API_KEY"
```

For local Qwen plus OpenAI embeddings:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export RT_RAG_BASE_URL="http://localhost:8000/v1"
export RT_RAG_API_KEY="your-local-llm-key"
export RT_RAG_MODEL="Qwen/Qwen2.5-14B-Instruct"
export RT_RAG_RANKER_URL="https://api.openai.com/v1"
export RT_RAG_RANKER_KEY="$OPENAI_API_KEY"
export RT_RAG_EMBED_BASE_URL="https://api.openai.com/v1"
export RT_RAG_EMBED_API_KEY="$OPENAI_API_KEY"
```

If `uv` cannot write to its default cache directory in a restricted environment, use:

```bash
uv sync --cache-dir /path/to/uv-cache
```

### ✅ (Optional) Store Hugging Face cache in the workspace volume

If your persistent volume is mounted at `/workspace`, set these variables before running any `huggingface-cli download`, `vllm serve`, or first-time model load so Hugging Face caches are stored there instead of the default `~/.cache/huggingface` location:

```bash
mkdir -p /workspace/hf-cache
export HF_HOME=/workspace/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf-cache/hub
export TRANSFORMERS_CACHE=/workspace/hf-cache/transformers
```

### ⚡️ (Optional) Serve Qwen2.5-14B-Instruct via vLLM

To serve Qwen2.5-14B-Instruct locally using [vLLM](https://github.com/vllm-project/vllm) with OpenAI-compatible API:

First, install vLLM inside the uv environment:

```bash
uv add vllm
```

Then, start the server:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --dtype auto \
  --api-key your-api-key
```

> Replace `your-api-key` with a secure token. This should match `RT_RAG_API_KEY` if you are using environment-variable-based config.

📝 **Tip:** For more details, see [vLLM OpenAI-Compatible Server Docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

This is optional and not required for the first environment bootstrap. The default setup is aimed at getting the pipeline running against an OpenAI-compatible API first.

### 🧭 Runtime Config

The repository now keeps two runtime config files:

- `main/config.py`: the recommended lighter-weight runtime config for the fixed MuSiQue 100-question subset and local reproduction workflows
- `main/config_ori.py`: the original heavier baseline-style config kept for comparison, rollback, and reproducing the previous settings

If you are reproducing the current local workflow described in this README, use `main/config.py`. If you want to compare against the earlier, more expensive settings, start from `main/config_ori.py`.

---

## 📦 2. Model Downloads

If you want model downloads and cache files to live in the workspace volume, run the Hugging Face cache setup commands above first.

You can download models manually or use Hugging Face CLI:

### 🔍 Reranker Model

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)

```bash
huggingface-cli download BAAI/bge-reranker-base
```

### 🧠 Language Model (Qwen2.5-14B-Instruct)

- [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)

```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

> Make sure to login if authentication is required:

```bash
huggingface-cli login
```

---

## 🛠️ 3. Data Preparation

The preprocessed corpus is already in the `raw` folder.  
Evaluation and retrieval data are from [LongBench](https://github.com/THUDM/LongBench).

The repository now version-controls benchmark inputs under `data/longbench/`, including the fixed MuSiQue subset used for the current reproduction workflow:

- `data/longbench/musique_100_seed42.jsonl`: fixed 100-question MuSiQue subset
- `data/longbench/musique_100_seed42.meta.json`: metadata describing the source file, sample size, seed, and selected indices

Generated artifacts are still kept local and ignored by git, including:

- `data/embeddings/`
- `output/`
- local virtual environments
- uv cache directories

---

## ✏️ 4. Configure `main/build_dense_index/config.py`

Update your configuration for embedding/index building:

| Parameter       | Description |
|----------------|-------------|
| `raw_path`     | Path to folder containing preprocessed JSON |
| `save_path`    | Where to store FAISS index & metadata |
| `dataset_name` | Filename without `.json` |
| `chunk_size`   | Max words per chunk (e.g., 200) |
| `min_sentence` | Min sentences per chunk (e.g., 2) |
| `overlap`      | Overlapping sentences between chunks (e.g., 2) |
| `base_url`     | Defaults to `RT_RAG_EMBED_BASE_URL` or `RT_RAG_RANKER_URL` |
| `api_key`      | Defaults to `RT_RAG_EMBED_API_KEY`, `RT_RAG_RANKER_KEY`, or `OPENAI_API_KEY` |

---

## 🧱 5. Build the Dense Index

Once `main/build_dense_index/config.py` is ready, build your FAISS index with:

```bash
python main/build_dense_index/dense_build_index.py
```

---

## 🧪 6. Smoke Test

Before running the full dataset, verify the pipeline with a single dense-retrieval question:

```bash
python main/retrieve.py \
  --query "Which film stars Anthony Hopkins as Hannibal Lecter?" \
  --dataset musique \
  --method dense \
  --chunk_size 200 \
  --min_sentence 2 \
  --overlap 2 \
  --topk1 10 \
  --topk2 5 \
  --smoke-test
```

This mode performs one dense retrieval pass and one generation call, which is useful for checking that the index, embedding API, and LLM server are wired correctly.

---

## 🚀 7. Run on the Full Dataset

After the dense index is successfully built:

1. Configure runtime parameters in:

    ```text
    main/config.py
    ```

    Make sure the dataset path, retrieval settings, environment variables, and output paths are correct and aligned with the built index.

2. Run the full dataset through the system:

    ```bash
    uv run python main/load_data.py
    ```

> This step runs the entire dataset through the RT-RAG pipeline: it performs retrieval, reranking, tree generation, and LLM querying.

### ▶️ Partial Runs and Resume

For small smoke runs or interrupted jobs, `main/load_data.py` now supports both `--limit` and `--start-index`:

```bash
python main/load_data.py --limit 10
python main/load_data.py --start-index 66
python main/load_data.py --start-index 66 --limit 10
```

`--start-index` is a 0-based dataset index. This is useful when you want to resume a previous run from a known offset instead of restarting from the beginning.

### ⏱️ Timeout Fallback

The current runtime config applies a whole-question timeout budget of 600 seconds. If tree generation / tree solving does not finish within that budget, the pipeline automatically falls back to `direct_answer(...)` using the existing retrieval path.

Result files now include timeout metadata such as:

- `timeout_triggered`
- `timeout_stage`
- `timing_timeout_budget_seconds`
- `timing_timeout_elapsed_seconds`

---

## 📊 8. Evaluate the Results

Once inference on the full dataset is complete, you can evaluate the generated answers using:

```bash
python main/evaulate.py /path/to/result.txt
```
> Replace `/path/to/result.txt` with the actual path to the output file generated by `main/load_data.py`.

This script will compute metrics on the dataset.

### 🧾 Result File Fields

Recent runs now record both prediction outputs and lightweight runtime metadata. In addition to `qid`, `question`, `predicted_answer`, and `golden_answers`, result files may include:

- `timing_total_seconds`
- `timing_tree_seconds`
- `timing_retrieval_seconds`
- `timing_generation_seconds`
- `timing_refined_query_seconds`
- `timing_direct_fallback_seconds`
- `timeout_triggered`
- `timeout_stage`

### ⚠️ Multi-Run Evaluation Caution

If a dataset is completed across multiple result files because of resume runs, evaluate on unique questions rather than blindly concatenating files with overlapping `qid`s. When a run is resumed incorrectly, duplicate questions can appear across files and distort the final metrics.

## 📈 RT-RAG Performance

The table below summarizes RT-RAG's performance across three benchmark datasets using two different backbone models:

| Model           | Dataset     | F1     | EM     |
|----------------|-------------|--------|--------|
| **GPT-4o-mini** | MuSiQue     | 54.42  | 41.50  |
|                | 2WikiMQA    | 75.08  | 63.00  |
|                | HotpotQA    | 65.26  | 52.50  |
|                | **Average** | **64.92** | **52.33** |
| **Qwen2.5-14B** | MuSiQue     | 50.04  | 39.00  |
|                | 2WikiMQA    | 73.69  | 64.00  |
|                | HotpotQA    | 66.24  | 51.00  |
|                | **Average** | **63.32** | **51.33** |

> RT-RAG consistently outperforms all baselines across diverse multi-hop QA datasets.

