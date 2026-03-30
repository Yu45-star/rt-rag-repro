

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

The repository ignores generated artifacts such as `data/`, `output/`, local virtual environments, and uv cache directories so teammates can keep local runs isolated from versioned source files.

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
uv run python main/build_dense_index/dense_build_index.py
```

---

---

## 🧪 6. Run on the Full Dataset

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



---

---

## 📊 7. Evaluate the Results

Once inference on the full dataset is complete, you can evaluate the generated answers using:

```bash
uv run python main/evaulate.py /path/to/result.txt
```
> Replace `/path/to/result.txt` with the actual path to the output file generated by `main/load_data.py`.

This script will compute metrics on the dataset.

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

