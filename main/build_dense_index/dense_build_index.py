#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib
import json 
import re  
import os 
import time
import numpy as np
from tqdm import tqdm
import faiss
from openai import OpenAI

def get_word_count(text):
    """Count the number of words, including Chinese characters"""
    regEx = re.compile('[\W]')
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    words = regEx.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([w for w in word_list if len(w.strip()) > 0])

def split_sentences(content, chunk_size, min_sentence, overlap):
    """Split content into chunks based on sentence delimiters and constraints"""
    stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';']
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    sentences = re.split(split_pattern, content)
    
    if len(sentences) == 1:
        return sentences
    
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences) - 1, 2)]
    chunks = []
    temp_text = ''
    sentence_overlap_len = 0
    start_index = 0

    for i, sentence in enumerate(sentences):
        temp_text += sentence
        if get_word_count(temp_text) >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
            if i + 1 > overlap:
                sentence_overlap_len = sum([get_word_count(sentences[j]) for j in range(i+1-overlap, i+1)])
            if chunks:
                if start_index > overlap:
                    start_index -= overlap
            chunk_text = ''.join(sentences[start_index:i+1])
            if not chunks:
                chunks.append(chunk_text)
            elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)
            temp_text = ''
            start_index = i + 1
    
    return chunks

def process_data(file_path, chunk_size, min_sentence, overlap, save_path):
    """Process input JSON file and split text into chunks"""
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_rawid = {}
    processed_chunks = []

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):
        content = item.get("paragraph_text") or item.get("ch_content") or item.get("ch_contenn")
        chunks = split_sentences(content, chunk_size, min_sentence, overlap)
        for i, chunk in enumerate(chunks):
            id_to_rawid[len(processed_chunks) + i] = idx
        processed_chunks.extend(chunks)
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/chunks.json", "w", encoding='utf-8') as fout:
        json.dump(processed_chunks, fout, ensure_ascii=False)
    with open(f"{save_path}/id_to_rawid.json", "w", encoding='utf-8') as fout:
        json.dump(id_to_rawid, fout, ensure_ascii=False)
    
    return processed_chunks

def calculate_openai_embeddings(content, vector_store_path, base_url, api_key, max_retries=3):
    """Calculate OpenAI embeddings and save as FAISS index and NumPy array"""
    print(f"\nSaving FAISS index to: {vector_store_path}")  
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    embeddings = []
    batch_size = 10

    for i in tqdm(range(0, len(content), batch_size), desc="Calculating OpenAI embeddings"):
        batch = content[i:i+batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                print(f"Error in batch {i}-{i+batch_size} (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch embeddings for batch {i}-{i+batch_size} after {max_retries} attempts."
                    ) from e
                time.sleep(5)
    
    embeddings_array = np.array(embeddings).astype('float32')
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, vector_store_path)
    
    embeddings_path = os.path.join(os.path.dirname(vector_store_path), "embeddings.npy")
    print(f"Saving embeddings to: {embeddings_path}")  
    np.save(embeddings_path, embeddings_array)
    
    return embeddings_array

def load_config(config_module_name):
    """Load a build configuration module by name."""
    return importlib.import_module(config_module_name)

def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(description="Build a dense retrieval index for a dataset.")
    parser.add_argument(
        "--config",
        default=os.getenv("RT_RAG_DENSE_CONFIG_MODULE", "config"),
        help="Configuration module name to import from main/build_dense_index (default: config).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config.dataset_name
    raw_file_name = getattr(config, "raw_file_name", dataset_name)
    chunk_size = config.chunk_size
    min_sentence = config.min_sentence
    overlap = config.overlap
    raw_path = config.raw_path
    save_path = config.save_path
    index_name = f"{dataset_name}_chunk{chunk_size}_{min_sentence}_{overlap}"
    vector_store_path = f"{save_path}/{index_name}"
    file_path = f"{raw_path}/{raw_file_name}.json"
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Using config module: {args.config}")
    print(f"Data path: {file_path}")
    print(f"Chunk size: {chunk_size}, Minimum sentences: {min_sentence}, Overlap: {overlap}")
    print(f"Output directory: {save_path}")
    print(f"Index will be saved as: {vector_store_path}")
    print(f"Chunks will be saved to: {save_path}/chunks.json")
    print(f"ID mapping will be saved to: {save_path}/id_to_rawid.json")
    print(f"Config will be saved to: {save_path}/config.json")

    print("\nStarting data processing...")
    os.makedirs(save_path, exist_ok=True)
    content = process_data(file_path, chunk_size, min_sentence, overlap, save_path)

    print("\nCalculating OpenAI embeddings...")
    start_time = time.time()
    embeddings = calculate_openai_embeddings(content, vector_store_path, config.base_url, config.api_key)
    end_time = time.time()
    
    print(f"\nOpenAI embedding generation complete in {end_time - start_time:.2f} seconds.")

    config_path = f"{save_path}/config.json"
    print(f"\nSaving configuration to: {config_path}")
    with open(config_path, "w") as f:
        json.dump({
            "chunk_size": chunk_size,
            "min_sentence": min_sentence,
            "overlap": overlap,
            "embedding_model": "text-embedding-3-small",
            "dataset_name": dataset_name,
            "raw_file_name": raw_file_name,
            "index_name": index_name,
            "input_file": file_path
        }, f, indent=2)
    
    print(f"\nAll files saved successfully!")
    print(f"Index location: {vector_store_path}")
    print(f"Data directory: {save_path}")
    print(f"To run retrieval, execute:")
    print(f"python retrieval.py --index_path {save_path}")

if __name__ == '__main__': 
    main()
