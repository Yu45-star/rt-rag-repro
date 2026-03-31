import re
import os
import uuid
import numpy as np
import json
import math
import time
from typing import List, Dict, Tuple, Any
import spacy
from datetime import datetime
import requests
# from elasticsearch import Elasticsearch
from openai import OpenAI
import argparse
import faiss
import torch
from config import BASE_URL,API_KEY,MODEL_NAME,RANKER_URL,RANKER_KEY,RETRIEVE_TEMPERATURE,SAMPLING_ITERATIONS,EMBEDDING_DATA
from debug_logging import truncate_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
global_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base').to(device)
global_model.eval()

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def generate_response(messages, max_tokens=2000, temperature=0, top_p=1.0, top_k=None, frequency_penalty=0.0, presence_penalty=0.0, debug_collector=None, stage="generate_response", metadata=None):
    
    try:
      
        params = {
            "model": MODEL_NAME, 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
       
        if top_k is not None:
            params["top_k"] = top_k
            
        completion = client.chat.completions.create(**params)
        return completion.choices[0].message.content
    except Exception as e:
        if debug_collector is not None:
            error_details = {"exception_type": type(e).__name__}
            if metadata:
                error_details.update(metadata)
            debug_collector.add_error(stage, str(e), error_details)
            debug_collector.update_summary(generation_failed=True)
        print(f"API_ERROR: {str(e)}")
        return None

# Extract keywords function
def extract_keywords(question: str) -> str:
    doc = nlp(question)
    keywords_with_positions = []
    matched_spans = set()

    name_patterns = [
        re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
        re.compile(r'\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?\b'),
    ]
    
    for ent in doc.ents:
        ent_text = ent.text
        ent_start = question.find(ent_text)
        if ent_start != -1:
            matched_spans.add((ent_start, ent_start + len(ent_text)))
            keywords_with_positions.append((ent_text, ent_start))

    for pattern in name_patterns:
        for match in pattern.finditer(question):
            start, end = match.span()
            if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                matched_spans.add((start, end))
                keywords_with_positions.append((match.group(), start))

    important_pos = {"NOUN", "PROPN", "ADJ", "VERB", "NUM"}
    for token in doc:
        if token.pos_ in important_pos and not token.is_stop:
            token_start = token.idx
            if not any(s <= token_start < e for s, e in matched_spans):
                matched_spans.add((token_start, token_start + len(token.text)))
                keywords_with_positions.append((token.text, token_start))

    keywords_with_positions.sort(key=lambda x: x[1])
    final_keywords = []
    seen_keywords = set()
    for kw, pos in keywords_with_positions:
        if not any(kw in other_kw and kw != other_kw for other_kw in seen_keywords):
            final_keywords.append(kw)
            seen_keywords.add(kw)
    return " ".join(final_keywords)


def generate_answers(question, n=15, temperature=0.5):
    answers = []
    
    
    system_prompt = "Your task is to answer questions directly with a concise response, providing only the answer itself without repeating the question or adding extra text."
    
    for _ in range(n):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        response = generate_response(messages, temperature=temperature)
        if response:
            answers.append(response)
    
    return answers



def direct_answer(question, dataset, method="bm25", chunk_size=200, min_sentence=2, 
                      overlap=2, topk1=45, topk2=15, samples=SAMPLING_ITERATIONS, show_documents=False, debug_collector=None):
    query = extract_keywords(question)
    
    documents = retrieve_documents(query=query, dataset=dataset, method=method, chunk_size=chunk_size, 
                                    min_sentence=min_sentence, overlap=overlap, topk1=topk1, topk2=topk2, 
                                    debug_collector=debug_collector, stage="direct_answer.retrieval")
    if show_documents:
        print(f"\nRetrieved documents:\n{documents}")
    
    answers = []
    for sample_index in range(max(1, samples)):
        response = answer_with_reasoning(
            question,
            documents=documents,
            debug_collector=debug_collector,
            stage="direct_answer.generation",
            metadata={"sample_index": sample_index + 1, "question": question, "documents_length": len(documents)},
        )
        print(response)
        parsed_answer = parse_generated_text(response)['answer']
        answers.append(parsed_answer)
    
    if not answers:
        return "[none]"

    # Count the frequency of each answer
    answer_counts = {}
    for answer in answers:
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1
    
    # Sort answers by frequency (highest to lowest)
    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return the first non-"none" answer with highest frequency
    for answer, count in sorted_answers:
        if "none" not in answer.lower():
            return answer
    
    # If all answers contain "none", return the most frequent one
    return sorted_answers[0][0]


def smoke_test(question, dataset, method="dense", chunk_size=200, min_sentence=2, 
               overlap=2, topk1=10, topk2=5):
    print("\nRunning smoke test: single retrieval + single generation.")
    return direct_answer(
        question=question,
        dataset=dataset,
        method=method,
        chunk_size=chunk_size,
        min_sentence=min_sentence,
        overlap=overlap,
        topk1=topk1,
        topk2=topk2,
        samples=1,
        show_documents=True,
    )

def preprocess_documents_for_llm(documents):
    
   
    processed_docs = []
    current_doc_index = 1  
    
    for orig_doc in documents:
       
        if isinstance(orig_doc, str) and orig_doc.startswith("- doc"):
            content = re.sub(r"^- doc\d+:\s*", "", orig_doc)
        else:
            content = orig_doc
        
        
        segments = content.split("\n\n")
        
       
        for segment in segments:
            if segment.strip():  
                processed_docs.append(f"- doc{current_doc_index}: [ISOLATED DOCUMENT] {segment.strip()} [END ISOLATED DOCUMENT]")
                current_doc_index += 1
    
    return processed_docs

def answer_with_reasoning(question: str, documents: str, max_tokens=10000, temperature=0, debug_collector=None, stage="answer_with_reasoning", metadata=None):
    """
    Function that answers a question based on provided documents, allowing the LLM to use 
    reasoning and its internal knowledge in addition to document content.
    """
    prompt = f"""You are a knowledgeable AI assistant tasked with answering questions.

You will be provided with a question and some documents that might contain relevant information.

INSTRUCTIONS:
1. Read the question and documents carefully
2. You MUST use explicit step-by-step reasoning to arrive at your answer
3. Your reasoning MUST rely on either:
   - Information from the provided documents, OR
   - Your internal knowledge when documents are insufficient
4. Analyze the question from multiple angles and consider different interpretations
5. When the documents contain relevant information, ensure you incorporate it in your reasoning
6. When the documents are incomplete, use your knowledge to fill gaps through explicit reasoning
7. You MUST ALWAYS provide a concrete answer - "I don't know", "None", or similar responses are NOT acceptable
8. If uncertain, provide your best reasoned guess based on available information

YOUR RESPONSE MUST STRICTLY FOLLOW THIS FORMAT:

cot: [Your detailed step-by-step reasoning process using document information or internal knowledge]

so the answer is: [Your final answer with NO additional decorations, explanations, or qualifiers - just the direct, concise answer]

For example:
- If the answer is "Paris", just write "Paris"
- If the answer is a date, just write the date
- If the answer is a person's name, just write the name
- Do NOT add phrases like "I believe", "According to the documents", "The answer would be", etc.

QUESTION: {question}

DOCUMENTS:
{documents}
"""

    # Construct message list
    messages = [
        {"role": "system", "content": 
            "You are a knowledgeable AI assistant that MUST provide answers through explicit reasoning. You MUST perform step-by-step reasoning using either document information or your internal knowledge to reach conclusions. You MUST ALWAYS provide a concrete answer - 'I don't know', 'None', or empty responses are NOT acceptable. If uncertain, provide your best reasoned guess based on available information. You MUST follow the exact output format: 'cot: [detailed reasoning] so the answer is: [direct answer with NO additional text]'"
        },
        {"role": "user", "content": prompt}
    ]
    
    generation_started = time.perf_counter()
    response = generate_response(
        messages, 
        max_tokens=max_tokens, 
        temperature=temperature,
        debug_collector=debug_collector,
        stage=stage,
        metadata=metadata,
    )
    generation_elapsed = time.perf_counter() - generation_started

    if debug_collector is not None:
        debug_collector.add_timing("generation_total_seconds", generation_elapsed)
        debug_collector.increment_counter("generation_call_count")
    
    # Ensure a valid response is returned
    if not response or response.strip() == "" or response.lower() == "none":
        # Fallback response if the model returns None or empty
        response = "cot: The question requires a response even with limited information. Based on the available documents and general knowledge, I must provide my best assessment. so the answer is: [Best possible answer based on limited information]"

    if debug_collector is not None:
        parsed = parse_generated_text(response)
        event_metadata = dict(metadata or {})
        event_metadata["duration_seconds"] = round(generation_elapsed, 4)
        debug_collector.add_generation_event(
            stage=stage,
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_response=response,
            parsed_answer=parsed["answer"],
            metadata=event_metadata,
        )
    
    return response

# Dense retrieval and reranking
def retrieve_and_rerank_chunks(dataset: str, query: str, chunk_size: int = 200, overlap: int = 2, 
                             min_sentence: int = 2, coarse_top_k: int = 45, fine_top_k: int = 15, 
                             debug_collector=None, stage="dense_retrieval") -> List[Dict[str, Any]]:
   
    retrieval_started = time.perf_counter()
    try:
        base_path = f"{EMBEDDING_DATA}/{dataset}/{chunk_size}_{min_sentence}_{overlap}"
        index_file = None

        config_path = os.path.join(base_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                index_config = json.load(f)
            configured_name = index_config.get("index_name")
            if configured_name:
                configured_path = os.path.join(base_path, configured_name)
                if os.path.exists(configured_path):
                    index_file = configured_path

        if index_file is None:
            legacy_index_files = [f for f in os.listdir(base_path) if f.endswith("_index")]
            if legacy_index_files:
                index_file = os.path.join(base_path, legacy_index_files[0])

        if index_file is None:
            raise FileNotFoundError(f"No index file found in {base_path}")

        index = faiss.read_index(index_file)

        with open(os.path.join(base_path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        embedding_client =OpenAI(
            base_url=RANKER_URL,
            api_key=RANKER_KEY
        )
        response = embedding_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, coarse_top_k)

        coarse_results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(chunks):
                coarse_results.append({
                    "score": float(distances[0][i]),
                    "index": int(idx),
                    "content": chunks[idx]
                })

        if not coarse_results:
            retrieval_elapsed = time.perf_counter() - retrieval_started
            if debug_collector is not None:
                debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
                debug_collector.increment_counter("retrieval_call_count")
            return []

        
        tokenizer = global_tokenizer
        model = global_model

        pairs = [[query, item["content"]] for item in coarse_results]
        all_scores = []
        batch_size = 8

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to same device
                scores = model(**inputs).logits.view(-1).float().cpu().tolist()
                all_scores.extend(scores)

        for i, score in enumerate(all_scores):
            coarse_results[i]["rerank_score"] = score

        sorted_results = sorted(coarse_results, key=lambda x: x["rerank_score"], reverse=True)
        final_results = sorted_results[:fine_top_k]
        retrieval_elapsed = time.perf_counter() - retrieval_started
        if debug_collector is not None:
            debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
            debug_collector.increment_counter("retrieval_call_count")
            debug_collector.add_retrieval_event(
                stage=stage,
                query=query,
                method="dense",
                topk1=coarse_top_k,
                topk2=fine_top_k,
                items=final_results,
                metadata={"base_path": base_path, "index_file": index_file, "duration_seconds": round(retrieval_elapsed, 4)},
            )
        return final_results

    except Exception as e:
        retrieval_elapsed = time.perf_counter() - retrieval_started
        if debug_collector is not None:
            debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
            debug_collector.increment_counter("retrieval_call_count")
            debug_collector.add_error(
                stage,
                str(e),
                {"exception_type": type(e).__name__, "query": query, "dataset": dataset, "duration_seconds": round(retrieval_elapsed, 4)},
            )
            debug_collector.update_summary(retrieval_failed=True)
        print(f"Dense retrieval failed: {e}")
        return []

# Search using Elasticsearch BM25
def search_with_bm25(query, dataset, chunk_size, min_sentence, overlap, top_k, debug_collector=None, stage="bm25_retrieval"):
    """Search documents from Elasticsearch index using BM25"""
    retrieval_started = time.perf_counter()
    # Build index name
    index_name = f"{dataset}_chunk{chunk_size}_{min_sentence}_{overlap}"
    
    # Connect to Elasticsearch
    es = Elasticsearch(['localhost:9200'], timeout=100)
    
    # Check connection
    if not es.ping():
        raise ValueError("Unable to connect to Elasticsearch, please ensure Elasticsearch service is running")
    
    # Prepare search query
    search_body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "text"],
                "type": "best_fields",
                "tie_breaker": 0.3
            }
        }
    }
    
    # Execute search
    print(f"Searching in index '{index_name}' for: '{query}'")
    try:
        response = es.search(index=index_name, body=search_body)
    except Exception as e:
        retrieval_elapsed = time.perf_counter() - retrieval_started
        if debug_collector is not None:
            debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
            debug_collector.increment_counter("retrieval_call_count")
            debug_collector.add_error(stage, str(e), {"exception_type": type(e).__name__, "query": query, "duration_seconds": round(retrieval_elapsed, 4)})
            debug_collector.update_summary(retrieval_failed=True)
        print(f"Elasticsearch search failed: {e}")
        return []
    
    # Process results
    hits = response['hits']['hits']
    if not hits:
        retrieval_elapsed = time.perf_counter() - retrieval_started
        if debug_collector is not None:
            debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
            debug_collector.increment_counter("retrieval_call_count")
            debug_collector.update_summary(retrieval_failed=True)
        print("No matching documents found")
        return []
    
    results = []
    for i, hit in enumerate(hits):
        doc = hit['_source']
        score = hit['_score']
        doc_id = hit['_id']
        
        result = {
            'rank': i + 1,
            'id': doc_id,
            'score': score,
            'title': doc.get('title', 'No title'),
            'text': doc.get('text', 'No content'),
        }
        results.append(result)

    retrieval_elapsed = time.perf_counter() - retrieval_started
    if debug_collector is not None:
        debug_collector.add_timing("retrieval_total_seconds", retrieval_elapsed)
        debug_collector.increment_counter("retrieval_call_count")
    
    return results

# Unified retrieval interface
def retrieve_documents(query, dataset, method="bm25", chunk_size=200, min_sentence=2, 
                      overlap=2, topk1=45, topk2=15, debug_collector=None, stage="retrieve_documents"):
    """
    Unified document retrieval interface, supports BM25 and dense retrieval
    
    Parameters:
        query: Query text 
        dataset: Dataset name 
        method: Retrieval method, "bm25" or "dense" 
        chunk_size: Text chunk size
        min_sentence: Minimum number of sentences
        overlap: Text chunk overlap size
        topk1: Number of results returned by coarse ranking (only for dense method)
        topk2: Final number of results returned
    
    Returns:
        Formatted document string
    """
    print(f"Using parameters: dataset={dataset}, method={method}, chunk_size={chunk_size}, min_sentence={min_sentence}, overlap={overlap}, topk1={topk1}, topk2={topk2}")
    
    raw_results = []
    
    if method.lower() == "bm25":
        # BM25 retrieval
        raw_results = search_with_bm25(query, dataset, chunk_size, min_sentence, overlap, topk2, debug_collector=debug_collector, stage="bm25_retrieval")
        if not raw_results:
            return "No relevant documents found"
        
        # Sort by score in ascending order, so more relevant documents appear later
        raw_results.sort(key=lambda x: x['score'])
        if debug_collector is not None:
            debug_collector.add_retrieval_event(
                stage=stage,
                query=query,
                method="bm25",
                topk1=topk2,
                topk2=topk2,
                items=raw_results,
            )
            
    elif method.lower() == "dense":
        # Dense retrieval
        try:
            raw_results = retrieve_and_rerank_chunks(
                dataset=dataset,
                query=query,
                chunk_size=chunk_size,
                overlap=overlap,
                min_sentence=min_sentence,
                coarse_top_k=topk1,
                fine_top_k=topk2,
                debug_collector=debug_collector,
                stage="dense_retrieval",
            )
            
            if not raw_results:
                if debug_collector is not None:
                    debug_collector.update_summary(retrieval_failed=True)
                return "No relevant documents found"
            
            # Sort by rerank_score in ascending order, so more relevant documents appear later
            # For dense retrieval, sort using rerank_score
            raw_results.sort(key=lambda x: x.get('rerank_score', 0))
                
        except Exception as e:
            if debug_collector is not None:
                debug_collector.add_error(
                    stage,
                    f"Dense retrieval failed, trying BM25 fallback: {e}",
                    {"exception_type": type(e).__name__, "query": query},
                )
            print(f"Dense retrieval failed, trying to fallback to BM25: {e}")
            # Fallback to BM25 when dense retrieval fails
            return retrieve_documents(query, dataset, method="bm25", chunk_size=chunk_size, 
                                    min_sentence=min_sentence, overlap=overlap, topk2=topk2, 
                                    debug_collector=debug_collector, stage="bm25_fallback")
    else:
        print(f"Unsupported retrieval method: {method}, using default BM25")
        return retrieve_documents(query, dataset, method="bm25", chunk_size=chunk_size, 
                                min_sentence=min_sentence, overlap=overlap, topk2=topk2, 
                                debug_collector=debug_collector, stage="bm25_default")
    
    # Unified document processing logic - segment processing of retrieval results
    formatted_docs = []
    doc_index = 1  # Used for unified numbering of all segments
    
    for result in raw_results:
        if method.lower() == "bm25":
            text = result['text']
        else:  # dense method
            text = result.get('content', '')
            
        # Split text by paragraphs
        paragraphs = text.split("\n\n")
        
        # Create a new document for each non-empty paragraph
        for paragraph in paragraphs:
            if paragraph.strip() and len(paragraph.strip()) > 1:
                formatted_docs.append(f"- doc{doc_index}: {paragraph.strip()}")
                doc_index += 1
                
    return "\n".join(formatted_docs)



# Call API to generate answer
def call_api_for_answer(question, documents, max_tokens=2000, temperature=0, top_p=0.9, top_k=1, frequency_penalty=0.2, presence_penalty=0.2, debug_collector=None, stage="call_api_for_answer", metadata=None):
    prompt ="""Instructions: For every question, provide a response in the exact format:

**"question: [question] documents: [list of documents] cot: [chain of thought] so the answer is: [answer]"**.


Your task is to extract **explicit** and **direct** information from the provided documents. **You must not assume, infer, or deduce any information that is not explicitly stated.** 

### **Rules for Answering:**
1. **Strictly Use Provided Documents**: You must only use the given documents as your information source. Do not assume or infer beyond the explicit text.
2. **Quote Documents Directly**: In the "cot" section, **quote the exact text from the documents** that supports the reasoning. Every reasoning step must be backed by a direct quote.
3. **No Implicit Assumptions**: If a fact is not explicitly stated in any document, you must return **"[none]"**. Do not assume answers based on indirect clues or general knowledge.
4. **Strict Subject Matching**:
   - **Ensure the subject in the query and the subject in the document match exactly and completely**. 
   - **Do not assume that similar names, partial overlaps, or likely similarities (e.g., shared surnames or initials) indicate the same subject unless explicitly stated in the document**.
   - **Only proceed with reasoning when there is a complete and explicit match between the query subject and the document subject**. If no such match exists, return **"[none]"**.
5. **Resolve Inconsistencies**: If multiple documents provide conflicting information, prioritize:
   - The document that is **most directly relevant** to the question.
   - The document that provides the **most specific and detailed** information.
6. **Use "[none]"** only when necessary: If there is no documentation to provide this information, simply return **"[none]"**.


### **Additional Guidelines to Prevent Assumptions:**
- **Do not infer relationships or connections** between entities unless explicitly stated. For example, do not assume two people with the same last name are related.
- **Do not assume chronological or causal relationships** unless explicitly stated in the documents.
- **Do not use external knowledge** or general facts to fill in gaps. Rely solely on the provided documents.
- **If a document mentions a fact about a subject but does not explicitly link it to the query subject, do not assume they are the same.**
- **When you see the same entity mentioned in different paragraphs, treat each mention as potentially referring to different contexts or time periods.**

** Not allowed to select "nearest option" ** :
Your answer must be based on a clear statement in the document, not on the "most likely option."
- ** If the answer is not mentioned directly in the document, '[none]' ** must be returned, even if there is seemingly relevant information.
- ** It is not allowed to speculate or select the "most likely answer" in the absence of a "clear answer" **

Below are examples:


documents:
- doc1: The Battle of Hastings occurred in 1066.
- doc2: The Norman conquest began in 1066 with the Battle of Hastings.
- doc3: Harold Godwinson died at the Battle of Hastings in October 1066.
question: When did the Battle of Hastings take place?
cot: Doc1 states: "The Battle of Hastings occurred in 1066." Doc2 states: "The Norman conquest began in 1066 with the Battle of Hastings," confirming the year. Doc3 states: "Harold Godwinson died at the Battle of Hastings in October 1066," adding the specific month. Reasoning: The consistent year across all documents is 1066, and doc3 provides the additional detail of October. Thus, the battle occurred in October 1066.
so the answer is: October 1066

documents:
- doc1: Mars is the fourth planet from the Sun in our solar system.
- doc2: Mars has two small moons named Phobos and Deimos.
- doc3: Mars is often called the Red Planet due to its reddish appearance.
question: What is the capital of Mars?
cot: Doc1 states: "Mars is the fourth planet from the Sun in our solar system." Doc2 states: "Mars has two small moons named Phobos and Deimos." Doc3 states: "Mars is often called the Red Planet due to its reddish appearance." Reasoning: All documents describe Mars as a planet, with no mention of cities or capitals. Planets do not have capitals, so no answer can be derived from the documents.
so the answer is: [none]

documents:
- doc1: Wolfgang Amadeus Mozart (January 27, 1756 - December 5, 1791) composed over 600 works during his short life.
- doc2: Mozart began composing at the age of five and continued until his death in December 1791.
- doc3: Born in Salzburg on January 27, 1756, Mozart showed musical talent from a very early age.
question: When was Mozart born?
cot: Doc1 states: "Wolfgang Amadeus Mozart (January 27, 1756 - December 5, 1791)," providing his birth date. Doc3 states: "Born in Salzburg on January 27, 1756," confirming the exact date. Reasoning: Both documents agree on January 27, 1756, as Mozart's birth date, with no conflicting information. Thus, he was born on January 27, 1756.
so the answer is: 27 January 1756

documents:
- doc1: Jane Eyre is a famous novel written by Charlotte Brontë in 1847.
- doc2: Pride is a common theme in many literary works of the 19th century.
- doc3: Prejudice can influence how characters are portrayed in Victorian literature.
question: Who wrote the novel "Pride and Prejudice"?
cot: Doc1 states: "Jane Eyre is a famous novel written by Charlotte Brontë in 1847," referring to a different novel. Doc2 states: "Pride is a common theme in many literary works of the 19th century," and doc3 states: "Prejudice can influence how characters are portrayed in Victorian literature." Reasoning: None of the documents mention "Pride and Prejudice" or its author. The information provided is irrelevant to the question, so no answer can be derived.
so the answer is: [none]

documents:
- doc1: J. Smith is a renowned physicist who contributed to quantum mechanics.
- doc2: John S. is a famous chef known for his innovative culinary techniques.
- doc3: A person named John Smith works as a software engineer in Silicon Valley.
question: What is the profession of John Smith?
cot: Doc1 states: "J. Smith is a renowned physicist." Doc2 states: "John S. is a famous chef." Doc3 states: "A person named John Smith works as a software engineer." Reasoning: The question asks specifically for "John Smith," but the documents mention "J. Smith," "John S.," and "John Smith." However, without explicit confirmation that "J. Smith" or "John S." is the same as "John Smith," we cannot assume they are the same person. Only doc3 explicitly mentions "John Smith" and his profession. Thus, the profession is software engineer.
so the answer is: software engineer

documents:
- doc1: Albert Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
- doc2: In 1939, Albert Einstein was awarded the Copley Medal by the Royal Society for his contributions to scientific knowledge.
- doc3: Albert Einstein was honored with the Presidential Medal of Freedom in 1963 by President John F. Kennedy.
question: What awards did Albert Einstein receive?
cot: Doc1 states: "Albert Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect." Doc2 states: "In 1939, Albert Einstein was awarded the Copley Medal by the Royal Society for his contributions to scientific knowledge." Doc3 states: "Albert Einstein was honored with the Presidential Medal of Freedom in 1963 by President John F. Kennedy." Reasoning: Each document mentions a different award that Albert Einstein received, and all of them are relevant and distinct.
so the answer is: Nobel Prize, Copley Medal, Presidential Medal of Freedom

documents:
- doc1: Bill Gates co-founded Microsoft Corporation in 1975 with Paul Allen. After stepping down as CEO, he focused on global health initiatives through his foundation.
- doc2: William Henry Gates III, commonly known as Bill Gates, established the Bill & Melinda Gates Foundation in 2000, which has become one of the world's largest private charitable foundations with assets of over $50 billion..
- doc3: The Gates Foundation has contributed billions to global health programs, including efforts to eradicate polio and malaria in developing countries.
- doc4:  In 2008, Gates transitioned to a part-time role at Microsoft to devote more time to philanthropy, though he remained chairman of the board until 2014.
question: Which American entrepreneur founded Microsoft and later established a philanthropic foundation?
cot: Doc1 states that "Bill Gates co-founded Microsoft Corporation in 1975" and later focused on philanthropy through his foundation. Doc2 provides his full name: "William Henry Gates III, commonly known as Bill Gates," and confirms he "established the Bill & Melinda Gates Foundation in 2000." Doc3 and Doc4 provide additional context about his foundation and transition from Microsoft to philanthropy. Both forms of his name appear in the documents: "Bill Gates" (partial name) in Doc1, Doc3, and Doc4, while "William Henry Gates III" (full name) appears in Doc2. Since both forms refer to the same person and the documents consistently identify him as the founder of Microsoft who later established a philanthropic foundation, the most complete and accurate answer would prioritize his full name rather than the more commonly used partial name.
so the answer is: William Henry Gates III

documents:
- doc1: Gandhi developed his philosophy of nonviolent resistance, known as Satyagraha, during his time in South Africa before returning to India in 1915 to join the independence movement.
- doc2: Mohandas Karamchand Gandhi, born on October 2, 1869, in Porbandar, India, became the primary leader of India's independence movement against British colonial rule through his advocacy of nonviolent civil disobedience.
- doc3: Widely known as "Mahatma" (great soul), Gandhi inspired movements for civil rights and freedom across the world through his methods of peaceful protest and moral authority.
- doc4: After India gained independence in 1947, Gandhi focused on promoting Hindu-Muslim unity but was assassinated on January 30, 1948, by a Hindu nationalist who opposed his tolerance toward Muslims.
question: Which Indian leader advocated for nonviolent resistance and led the independence movement against British rule?
cot: Doc1 refers to the leader simply as "Gandhi" and mentions his philosophy of nonviolent resistance. Doc2 provides his full name "Mohandas Karamchand Gandhi" and confirms he led India's independence movement against British rule. Doc3 mentions he was known as "Mahatma" and describes his global influence. Doc4 refers to him as "Gandhi" while describing events after India's independence. Different forms of his name appear in the documents: "Gandhi" (partial name) in Doc1 and Doc4, "Mohandas Karamchand Gandhi" (full name) in Doc2, and "Mahatma" (honorific) in Doc3. Since all these names refer to the same Indian leader who advocated nonviolent resistance and led the independence movement, the most complete and accurate answer would prioritize his full name rather than the more commonly used partial name or honorific.
so the answer is: Mohandas Karamchand Gandhi

documents:
- doc1: Mount Kilimanjaro is a dormant volcano located in Tanzania.
- doc2: Tanzania is a country in East Africa.
- doc3: Africa's tallest mountain, Kilimanjaro, draws thousands of tourists each year to Tanzania.
question: Is Mount Kilimanjaro located in South America?
cot: Doc1 states: "Mount Kilimanjaro is a dormant volcano located in Tanzania." Doc2 adds that "Tanzania is a country in East Africa," clarifying the continent. Doc3 refers to Kilimanjaro as "Africa's tallest mountain," reinforcing its location. Reasoning: All documents place Mount Kilimanjaro in Tanzania, which is in East Africa, not South America.
so the answer is: No

documents:
- doc1: The Statue of Liberty stands on Liberty Island in New York Harbor.
- doc2: New York Harbor is located in the United States, along the coast of New York City.
- doc3: The monument was a gift from France to the United States and has become a symbol of American freedom.
question: Is the Statue of Liberty located in Canada?
cot: Doc1 states: "The Statue of Liberty stands on Liberty Island in New York Harbor." Doc2 clarifies that "New York Harbor is located in the United States." Doc3 reinforces the national context, noting it is a symbol of American freedom. Reasoning: All documents consistently place the Statue of Liberty in New York, United States, not in Canada.
so the answer is: No

documents:
- doc1: A standard soccer team is one that plays in an official soccer match.
- doc2: In an official soccer match, a team has 11 on the field.
- doc3: The standard formation for a soccer team includes 11, including one goalkeeper and ten outfield players.
cot: Doc1 explains that a standard soccer team refers to one in an official match. Doc2 confirms that an official team has 11  on the field. Doc3 elaborates on the formation of a standard team, which includes 11 players. Since the question asks "how many" players, the answer should include the unit "players."
question: How many players are on a standard soccer team during a match?
so the answer is: 11 players

documents:
- doc1: Iron Man (Tony Stark) is a superhero in the Marvel Cinematic Universe.
- doc2: He fights using high-tech armor.
question: What is Iron Man's real identity?
cot: The document shows "Iron Man (Tony Stark)" format, where Iron Man is the character/superhero name, and the name in parentheses is the character's real identity.
so the answer is: Tony Stark

documents:
- doc1: The Titanic sank after hitting an iceberg during its maiden voyage.
- doc2: The sinking occurred in the North Atlantic Ocean in April 1912.
- doc3: Over 1,500 people lost their lives when the Titanic went down in 1912.
question: What year did the Titanic sink?
cot: Doc1 mentions the Titanic sank during its maiden voyage. Doc2 specifies that the sinking occurred in April 1912. Doc3 confirms the year again by stating the tragedy happened in 1912.
so the answer is: 1912
### **Now, answer the following question based on the provided documents**###
documents:
{document}
question: {question}
cot: """

    prompt = prompt.format(document=documents, question=question)
    
    # Construct message list
    messages = [
       {
  "role": "system",
  "content": 
  """
You are an AI assistant that must answer questions using only explicit and fully stated information from the provided documents.


### ABSOLUTE RULES:
1. **Do NOT assume or infer**: You may only use what is literally and explicitly written. If the exact answer is not in the text, respond with: [none].
2. **Do NOT match partially**: A subject or phrase in a document must match the question exactly. Near matches, abbreviations, or similar entities do NOT qualify.
3. **Do NOT substitute related facts**: If the question asks "What is A?", and the document only says "A is located at B", you must return: [none].
4. **If multiple answers are valid**, return ALL of them in a list (in plain text), **not just one**. Do not prefer the most prominent or well-known.
5. **Every answer must be fully justified**: In your `cot`, quote exact document sentences that prove each part of your answer.
6.When a question asks for a specific type of information about an entity (such as a codename, alias, location, date, or role), only that exact property type may be returned.
7.If the document mentions a different attribute (e.g., the official name instead of the codename), you must return [none] unless the requested property is explicitly stated.
8.Do not substitute one type of answer for another, even if they refer to the same entity.
9.If any of these rules cannot be satisfied, respond with: [none].
You must follow these instructions with absolute precision, without exceptions.
"""
},

        {"role": "user", "content": prompt}
    ]
    
    generation_started = time.perf_counter()
    response = generate_response(
        messages, 
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        frequency_penalty=frequency_penalty, 
        presence_penalty=presence_penalty,
        debug_collector=debug_collector,
        stage=stage,
        metadata=metadata,
    )
    generation_elapsed = time.perf_counter() - generation_started

    if debug_collector is not None:
        debug_collector.add_timing("generation_total_seconds", generation_elapsed)
        debug_collector.increment_counter("generation_call_count")
    
    if response:
        parsed = parse_generated_text(response)
        if debug_collector is not None:
            event_metadata = dict(metadata or {})
            event_metadata["duration_seconds"] = round(generation_elapsed, 4)
            debug_collector.add_generation_event(
                stage=stage,
                model=MODEL_NAME,
                temperature=temperature,
                max_tokens=max_tokens,
                raw_response=response,
                parsed_answer=parsed["answer"],
                metadata=event_metadata,
            )
        return response

    if debug_collector is not None:
        event_metadata = dict(metadata or {})
        event_metadata["duration_seconds"] = round(generation_elapsed, 4)
        debug_collector.add_generation_event(
            stage=stage,
            model=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens,
            raw_response="Error",
            parsed_answer="Error",
            metadata=event_metadata,
        )
    return "Error"

# Generate optimized query
def generate_refined_query(question, history_queries, debug_collector=None, stage="generate_refined_query", metadata=None):
    """
    Generate an optimized query based on the original question and historical queries.
    """
    # Organize historical queries as a string
    previous_queries = "\n".join(
        f"Query {i+1}: {query}"
        for i, query in enumerate(history_queries)
    )
    
    # Generate optimization query prompt
    new_query_prompt = f"""You are an AI assistant specialized in refining search queries.
Your task is to generate an improved query based on the original question and past queries.

Original Question: {question}

Previous Queries:
{previous_queries}

Please generate a new query that is semantically consistent with the original question but differs in expression. The new query should:
1. Preserve entity names (people, places, titles, etc.) EXACTLY as they appear in the original question
2. Feel free to modify verbs, syntax, grammar, and sentence structure
3. Try different question formats (e.g., active vs. passive voice, direct vs. indirect questions)
4. Experiment with synonyms for non-entity terms (verbs, adjectives, adverbs)
5. Not repeat any of the previous queries

New Query:"""

    # Construct message list
    messages = [
        {"role": "system", "content": 
            "You are an AI specialized in optimizing search queries. Your task is to generate alternative phrasings for the original question to help retrieve more relevant documents."
            "CRITICAL INSTRUCTIONS:"
            "1. The original entity names (people, places, titles, etc.) must be preserved EXACTLY as they appear in the original question"
            "2. FEEL FREE to modify verbs, syntax, grammar, and sentence structure to create alternative phrasings"
            "3. Try different question formats (e.g., active vs. passive voice, direct vs. indirect questions)"
            "4. Experiment with synonyms for non-entity terms (verbs, adjectives, adverbs)"
            "5. Any modification MUST maintain the original semantic meaning - the answer should remain exactly the same"
            "6. If the original entity is not found in the documents, try different phrasings but do NOT switch to a similar entity"
            "7. Return only the query itself, without any explanation"
            "8. The generated query should NOT duplicate any of the previous queries"
            "9. When identifying people, always use the full name of the person or object in your final answer if both the person or object's full name and short name appear in the document. Even though the abbreviation may be mentioned several times during the reasoning process, the final answer must use the most complete form of the name."

            },
        {"role": "user", "content": new_query_prompt}
    ]
    
    refined_query_started = time.perf_counter()
    response = generate_response(messages, max_tokens=50, temperature=0.5, debug_collector=debug_collector, stage=stage, metadata=metadata)
    refined_query_elapsed = time.perf_counter() - refined_query_started

    if debug_collector is not None:
        debug_collector.add_timing("refined_query_total_seconds", refined_query_elapsed)
    
    if response:
        refined_query = response.strip()
        if debug_collector is not None:
            event_metadata = dict(metadata or {})
            event_metadata["duration_seconds"] = round(refined_query_elapsed, 4)
            debug_collector.add_generation_event(
                stage=stage,
                model=MODEL_NAME,
                temperature=0.5,
                max_tokens=50,
                raw_response=response,
                parsed_answer=refined_query,
                metadata=event_metadata,
            )
        return refined_query
    else:
        raise Exception("Error generating new query")

# Parse generated text
def parse_generated_text(generated_text: str) -> Dict[str, str]:
    """
    Parse generated text, robustly handling various formats, including multi-line text and different marker forms.
    
    Args:
        generated_text: Raw text generated by API
        
    Returns:
        Dictionary containing cot and answer
    """
    # Normalize text - handle various line break formats
    text = generated_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Try multiple possible answer markers
    answer_markers = [
        "so the answer is:", 
        "So the answer is:", 
        "the answer is:", 
        "The answer is:",
        "FINAL ANSWER:"
    ]
    
    # Find CoT start position
    cot_markers = ["cot:", "COT:", "REASONING:", "Reasoning:"]
    cot_start = -1
    used_cot_marker = ""
    
    for marker in cot_markers:
        pos = text.find(marker)
        if pos != -1:
            cot_start = pos
            used_cot_marker = marker
            break
    
    # Find answer marker position
    answer_start = -1
    used_marker = ""
    
    for marker in answer_markers:
        # Use case-insensitive search
        pos = text.lower().find(marker.lower())
        if pos != -1:
            answer_start = pos
            used_marker = text[pos:pos+len(marker)]  # Preserve original case
            break
    
    # Process text based on found markers
    if cot_start != -1 and answer_start != -1 and cot_start < answer_start:
        # Found both CoT and answer markers
        cot = text[cot_start + len(used_cot_marker):answer_start].strip()
        answer = text[answer_start + len(used_marker):].strip()
        
        # Handle possible extra text in answer (like code parts)
        code_start = answer.find("import ")
        if code_start != -1:
            answer = answer[:code_start].strip()
    elif answer_start != -1:
        # Only found answer marker
        cot = text[:answer_start].strip()
        answer = text[answer_start + len(used_marker):].strip()
        
        # Handle possible extra text in answer
        code_start = answer.find("import ")
        if code_start != -1:
            answer = answer[:code_start].strip()
    elif cot_start != -1:
        # Only found CoT marker
        cot = text[cot_start + len(used_cot_marker):].strip()
        answer = "[none]"
    else:
        # No markers found, try to extract possible answer directly from text
        lines = text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if non_empty_lines:
            # Check if last line looks like an answer
            last_line = non_empty_lines[-1]
            if len(last_line) < 100 and not last_line.startswith("import"):
                answer = last_line
                cot = '\n'.join(non_empty_lines[:-1])
            else:
                cot = text
                answer = "[none]"
        else:
            cot = text
            answer = "[none]"
    
    # Clean code segments
    if "import " in answer:
        answer = answer.split("import ")[0].strip()
    
    # Clean quotes
    answer = answer.strip('"\'')
    answer = re.sub(r'[*_#]', '', answer)

    # Print debug information
    print(f"[DEBUG] Parse result:\nCoT start position: {cot_start}, Answer start position: {answer_start}")
    print(f"[DEBUG] Extracted answer: '{answer}'")
    
    return {"cot": cot, "answer": answer}

# Format complete response
def format_full_response(question: str, document: str, generated_text: str) -> str:
    result = parse_generated_text(generated_text)
    cot = result["cot"]
    answer = result["answer"]
    return f"question: {question} documents: {document} \ncot: {cot} so the answer is: {answer}"

# Main function: Iteratively answer questions - using updated parameters
def answer_question(question: str, dataset: str, method: str = "bm25", chunk_size: int = 200, 
                  min_sentence: int = 2, overlap: int = 2, topk1: int = 45, topk2: int = 15, 
                  max_iterations: int = 4, debug_collector=None) -> str:
    """
    Answer questions using iterative method
    
    Parameters:
        question: Question
        dataset: Dataset name
        method: Retrieval method ("bm25" or "dense")
        chunk_size: Text chunk size
        min_sentence: Minimum number of sentences
        overlap: Text chunk overlap size
        topk1: Number of results returned by coarse ranking (only for dense method)
        topk2: Final number of results returned
        max_iterations: Maximum number of iterations
        
    Returns:
        Formatted Q&A result
    """
    # Initialize query history
    history_queries = []

    # First round: Use keyword query
    base_query = extract_keywords(question)
    current_query = f"{base_query} "
    history_queries.append(current_query)

    # Use specified retrieval method to get documents
    documents = retrieve_documents(
        query=current_query, 
        dataset=dataset,
        method=method,
        chunk_size=chunk_size,
        min_sentence=min_sentence,
        overlap=overlap,
        topk1=topk1,
        topk2=topk2,
        debug_collector=debug_collector,
        stage="answer_question.iteration_1.retrieval",
    )
    print(f"\nIteration 1 - Retrieved documents (method: {method}):\n{documents}")

    # Generate 7 answers for the same document
    all_responses = []
    for i in range(SAMPLING_ITERATIONS):
        temp = RETRIEVE_TEMPERATURE # Keep consistent temperature parameter
        generated_text = call_api_for_answer(
            question,
            documents,
            temperature=temp,
            debug_collector=debug_collector,
            stage="answer_question.iteration_1.generation",
            metadata={"iteration": 1, "sample_index": i + 1, "question": question, "documents_length": len(documents)},
        )
        all_responses.append(generated_text)
        print(f"\nIteration 1 - Generated response #{i+1}:\n{generated_text}")
    
    # Parse all responses and count answers
    answer_counter = {}
    all_results = []
    
    for response in all_responses:
        result = parse_generated_text(response)
        all_results.append(result)
        answer = result["answer"].strip()
        
        if answer in answer_counter:
            answer_counter[answer] += 1
        else:
            answer_counter[answer] = 1
    
    # Select the most frequent answer
    most_common_answer = max(answer_counter.items(), key=lambda x: x[1]) if answer_counter else ("[none]", 0)
    print(f"\nAnswer statistics: {answer_counter}")
    print(f"Selected most frequent answer: {most_common_answer[0]} (appeared {most_common_answer[1]} times)")
    
    # If the most common answer is not [none], return the corresponding complete response
    if "[none]" not in most_common_answer[0].lower():
        # Find the first response containing this answer
        for i, result in enumerate(all_results):
            if result["answer"].strip() == most_common_answer[0]:
                return format_full_response(question, documents, all_responses[i])
    
    # If the most common answer is [none], continue iteration
    history_docs = documents  # Save previous documents for final result
    best_answer = most_common_answer[0]  # Save current best answer

    for iteration in range(2, max_iterations + 1):
        # Generate new optimized query
        current_query = generate_refined_query(
            question,
            history_queries,
            debug_collector=debug_collector,
            stage="answer_question.refined_query",
            metadata={"iteration": iteration, "history_queries": history_queries},
        )
        history_queries.append(current_query)
        
        print(f"\nIteration {iteration} - Optimized query: {current_query}")
        
        # Continue using the same retrieval method and parameters
        documents = retrieve_documents(
            query=current_query, 
            dataset=dataset,
            method=method,
            chunk_size=chunk_size,
            min_sentence=min_sentence,
            overlap=overlap,
            topk1=topk1,
            topk2=topk2,
            debug_collector=debug_collector,
            stage=f"answer_question.iteration_{iteration}.retrieval",
        )
        history_docs = documents  # Update documents
        
        print(f"\nIteration {iteration} - Retrieved documents (method: {method}):\n{documents}")
        
        # Generate 5 answers for new documents
        all_responses = []
        for i in range(SAMPLING_ITERATIONS):
            temp = RETRIEVE_TEMPERATURE
            generated_text = call_api_for_answer(
                question,
                documents,
                temperature=temp,
                debug_collector=debug_collector,
                stage=f"answer_question.iteration_{iteration}.generation",
                metadata={"iteration": iteration, "sample_index": i + 1, "question": question, "documents_length": len(documents)},
            )
            all_responses.append(generated_text)
            print(f"\nIteration {iteration} - Generated response #{i+1}:\n{generated_text}")
        
        # Parse all responses and count answers
        answer_counter = {}
        all_results = []
        
        for response in all_responses:
            result = parse_generated_text(response)
            all_results.append(result)
            answer = result["answer"].strip()
            
            if answer in answer_counter:
                answer_counter[answer] += 1
            else:
                answer_counter[answer] = 1
        
        # Select the most frequent answer
        most_common_answer = max(answer_counter.items(), key=lambda x: x[1]) if answer_counter else ("[none]", 0)
        print(f"\n7 answer statistics: {answer_counter}")
        print(f"Selected most frequent answer: {most_common_answer[0]} (appeared {most_common_answer[1]} times)")
        
        # If the most common answer is not [none], return the corresponding complete response
        if "[none]" not in most_common_answer[0].lower():
            # Find the first response containing this answer
            for i, result in enumerate(all_results):
                if result["answer"].strip() == most_common_answer[0]:
                    return format_full_response(question, documents, all_responses[i])
        else:
            # Update best answer
            best_answer = most_common_answer[0]
    
    # Reached maximum iterations, use best answer
    modified_cot = "After multiple attempts, no conclusive answer was found in the documents. Providing the most frequent answer from multiple attempts."
    modified_generated_text = f"cot: {modified_cot} so the answer is: {best_answer}"
    
    return format_full_response(question, history_docs, modified_generated_text)

# Test program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Retrieval Q&A System')
    parser.add_argument('--query', type=str, required=True, help='Input question')
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa', help='Dataset name')
    parser.add_argument('--method', type=str, default='bm25', choices=['bm25', 'dense'], help='Retrieval method: bm25 or dense')
    parser.add_argument('--chunk_size', type=int, default=200, help='Text chunk size')
    parser.add_argument('--min_sentence', type=int, default=2, help='Minimum number of sentences')
    parser.add_argument('--overlap', type=int, default=2, help='Text chunk overlap size')
    parser.add_argument('--topk1', type=int, default=45, help='Number of results returned by coarse ranking (only for dense method)')
    parser.add_argument('--topk2', type=int, default=15, help='Final number of results returned')
    parser.add_argument('--max_iterations', type=int, default=4, help='Maximum number of iterations')
    parser.add_argument('--smoke-test', action='store_true', help='Run a minimal single-retrieval, single-generation pipeline check')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Question: {args.query}")
    print(f"Dataset: {args.dataset}")
    print(f"Retrieval method: {args.method}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Minimum sentences: {args.min_sentence}")
    print(f"Overlap size: {args.overlap}")
    print(f"Coarse ranking results: {args.topk1} (only for dense method)")
    print(f"Final results: {args.topk2}")
    print(f"Maximum iterations: {args.max_iterations}")
    print(f"Smoke test mode: {args.smoke_test}")
    print("=" * 80)
    
    if args.smoke_test:
        answer = smoke_test(
            question=args.query,
            dataset=args.dataset,
            method=args.method,
            chunk_size=args.chunk_size,
            min_sentence=args.min_sentence,
            overlap=args.overlap,
            topk1=args.topk1,
            topk2=args.topk2,
        )
    else:
        answer = answer_question(
            question=args.query,
            dataset=args.dataset,
            method=args.method,
            chunk_size=args.chunk_size,
            min_sentence=args.min_sentence,
            overlap=args.overlap,
            topk1=args.topk1,
            topk2=args.topk2,
            max_iterations=args.max_iterations
        )
    
    print("\nFinal answer:")
    print(answer)
