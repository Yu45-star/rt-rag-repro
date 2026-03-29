import os

# Path to the directory containing the original raw dataset JSON files
raw_path = "main/raw"

# Path where processed chunks, FAISS index, and config files will be saved
save_path = "data/embeddings/musique/200_2_2"

# Base URL of the embedding API endpoint
base_url = os.getenv("RT_RAG_EMBED_BASE_URL", os.getenv("RT_RAG_RANKER_URL", "http://localhost:8001/v1"))

# API key for the embedding service
api_key = os.getenv("RT_RAG_EMBED_API_KEY", os.getenv("RT_RAG_RANKER_KEY", os.getenv("OPENAI_API_KEY", "YOUR_KEY")))

# Name of the dataset file (without .json extension)
dataset_name = "musique"

# Maximum number of words per chunk
chunk_size = 200

# Minimum number of sentences required in each chunk
min_sentence = 2

# Number of overlapping sentences between consecutive chunks
overlap = 2
