# config.py

# Path to the directory containing the original raw dataset JSON files
raw_path = "main/raw"

# Path where processed chunks, FAISS index, and config files will be saved
save_path = "data/embeddings/musique/200_2_2"

# Base URL of the OpenAI-compatible API endpoint
base_url = "http://localhost:8001/v1"

# Your OpenAI API key (keep this secure)
api_key = "YOUR_KEY"

# Name of the dataset file (without .json extension)
dataset_name = "musique"

# Maximum number of words per chunk
chunk_size = 200

# Minimum number of sentences required in each chunk
min_sentence = 2

# Number of overlapping sentences between consecutive chunks
overlap = 2
