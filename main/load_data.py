import argparse
import json
import random
import os
import asyncio
import time
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from tree_decompose import decompose_and_answer_with_variants
import aiofiles
import config  # Import config.py
from debug_logging import (
    BadCaseDebugCollector,
    classify_failure_reason,
    sanitize_filename,
    utc_timestamp,
)

# Set random seed for reproducibility
random.seed(42)

# Load configuration from config.py
dataset_name = config.DATASET
data_path = config.DATA_PATH
output_dir_root = config.OUTPUT_DIR_ROOT
max_concurrent = config.MAX_CONCURRENT
chunk_size = config.CHUNK_SIZE
min_sentence = config.MIN_SENTENCE
overlap = config.OVERLAP
topk1 = config.TOPK1
topk2 = config.TOPK2
method = config.METHOD
model_name = config.MODEL_NAME
question_timeout_seconds = config.QUESTION_TIMEOUT_SECONDS

# Dynamically construct output directory
output_dir = os.path.join(output_dir_root, dataset_name, f"{method}_chunk{chunk_size}_topk1_{topk1}_topk2_{topk2}")


def format_seconds(value):
    return f"{float(value or 0.0):.4f}"


def extract_result_metrics(collector):
    payload = collector.to_dict()
    timing = payload.get("timing", {})
    summary = payload.get("summary", {})
    return {
        "timing_total_seconds": timing.get("question_total_seconds", 0.0),
        "timing_tree_seconds": timing.get("tree_decompose_total_seconds", 0.0),
        "timing_retrieval_seconds": timing.get("retrieval_total_seconds", 0.0),
        "timing_generation_seconds": timing.get("generation_total_seconds", 0.0),
        "timing_refined_query_seconds": timing.get("refined_query_total_seconds", 0.0),
        "timing_direct_fallback_seconds": timing.get("direct_fallback_seconds", 0.0),
        "timing_timeout_budget_seconds": timing.get("timeout_budget_seconds", 0.0),
        "timing_timeout_elapsed_seconds": timing.get("timeout_elapsed_seconds", 0.0),
        "retry_count": timing.get("retry_count", 0),
        "used_direct_fallback": summary.get("used_direct_fallback", False),
        "timeout_triggered": summary.get("timeout_triggered", False),
        "timeout_stage": summary.get("timeout_stage"),
        "retrieval_call_count": summary.get("retrieval_call_count", 0),
        "generation_call_count": summary.get("generation_call_count", 0),
    }


# Determine next available file name (1.txt, 2.txt, 3.txt, etc.)
def get_next_available_file(output_dir):
    base_file_name = os.path.join(output_dir, "1.txt")
    if not os.path.exists(base_file_name):
        return base_file_name

    file_num = 2
    while os.path.exists(os.path.join(output_dir, f"{file_num}.txt")):
        file_num += 1
    return os.path.join(output_dir, f"{file_num}.txt")


# Get the next available file path
output_file_path = get_next_available_file(output_dir)
debug_output_dir = os.path.join(output_dir, "debug", Path(output_file_path).stem)

# Semaphore to limit concurrency
semaphore = None  # Will be initialized in the main function


async def write_result_to_file(result, file_path):
    async with aiofiles.open(file_path, 'a', encoding='utf-8') as fout:
        await fout.write(f"qid: {result['qid']}\n")

        if result["success"]:
            await fout.write(f"question: {result['question']}\n")
            await fout.write(f"predicted_answer: {result['predicted_answer']}\n")

            if isinstance(result['golden_answers'], str):
                if ',' in result['golden_answers']:
                    golden_answers = [item.strip() for item in result['golden_answers'].split(',')]
                else:
                    golden_answers = [result['golden_answers']]
            elif not isinstance(result['golden_answers'], list):
                golden_answers = [str(result['golden_answers'])]
            else:
                golden_answers = result['golden_answers']

            await fout.write(f"golden_answers: {json.dumps(golden_answers, ensure_ascii=False)}\n")
        else:
            await fout.write(f"error: {result['error']}\n")
            if result.get('question'):
                await fout.write(f"question: {result['question']}\n")

        await fout.write(f"timing_total_seconds: {format_seconds(result.get('timing_total_seconds'))}\n")
        await fout.write(f"timing_tree_seconds: {format_seconds(result.get('timing_tree_seconds'))}\n")
        await fout.write(f"timing_retrieval_seconds: {format_seconds(result.get('timing_retrieval_seconds'))}\n")
        await fout.write(f"timing_generation_seconds: {format_seconds(result.get('timing_generation_seconds'))}\n")
        await fout.write(f"timing_refined_query_seconds: {format_seconds(result.get('timing_refined_query_seconds'))}\n")
        await fout.write(f"timing_direct_fallback_seconds: {format_seconds(result.get('timing_direct_fallback_seconds'))}\n")
        await fout.write(f"timing_timeout_budget_seconds: {format_seconds(result.get('timing_timeout_budget_seconds'))}\n")
        await fout.write(f"timing_timeout_elapsed_seconds: {format_seconds(result.get('timing_timeout_elapsed_seconds'))}\n")
        await fout.write(f"retry_count: {int(result.get('retry_count', 0) or 0)}\n")
        await fout.write(f"used_direct_fallback: {bool(result.get('used_direct_fallback', False))}\n")
        await fout.write(f"timeout_triggered: {bool(result.get('timeout_triggered', False))}\n")
        await fout.write(f"timeout_stage: {result.get('timeout_stage') or 'none'}\n")
        await fout.write(f"retrieval_call_count: {int(result.get('retrieval_call_count', 0) or 0)}\n")
        await fout.write(f"generation_call_count: {int(result.get('generation_call_count', 0) or 0)}\n")
        await fout.write("---\n")
        await fout.flush()


async def write_debug_log(debug_collector, qid, idx):
    os.makedirs(debug_output_dir, exist_ok=True)
    debug_file_path = os.path.join(debug_output_dir, f"{idx}__{sanitize_filename(qid)}.json")
    async with aiofiles.open(debug_file_path, 'w', encoding='utf-8') as fout:
        await fout.write(json.dumps(debug_collector.to_dict(), ensure_ascii=False, indent=2))
        await fout.flush()



def build_debug_collector(qid, idx, question, golden_answers):
    run_metadata = {
        "dataset": dataset_name,
        "data_path": data_path,
        "method": method,
        "chunk_size": chunk_size,
        "min_sentence": min_sentence,
        "overlap": overlap,
        "topk1": topk1,
        "topk2": topk2,
        "model_name": model_name,
        "output_file_path": output_file_path,
        "timestamp": utc_timestamp(),
    }
    sample_metadata = {
        "qid": qid,
        "idx": idx,
        "question": question,
        "golden_answers": golden_answers,
    }
    return BadCaseDebugCollector(run_metadata, sample_metadata)


async def process_example(example, idx, retry_count=0, debug_collector=None):
    qid = example.get("_id", f"unknown_id_{idx}")
    question = example.get("input", "")
    golden_answers = example.get("answers", [])
    collector = debug_collector or build_debug_collector(qid, idx, question, golden_answers)
    current_retry = retry_count
    question_started = None
    collector.set_retry_count(current_retry)
    collector.set_timing("timeout_budget_seconds", question_timeout_seconds)

    try:
        while True:
            async with semaphore:
                if question_started is None:
                    question_started = time.perf_counter()
                question_deadline = question_started + question_timeout_seconds
                loop = asyncio.get_event_loop()
                try:
                    predicted_answer = await loop.run_in_executor(
                        None,
                        lambda: decompose_and_answer_with_variants(
                            question=question,
                            question_started_at=question_started,
                            question_deadline=question_deadline,
                            timeout_budget_seconds=question_timeout_seconds,
                            debug_collector=collector,
                        ),
                    )
                except Exception as e:
                    predicted_answer = f"Error: {str(e)}"
                    collector.add_error(
                        "load_data.executor",
                        str(e),
                        {
                            "exception_type": type(e).__name__,
                            "retry_count": current_retry,
                        },
                    )

            if "Error" not in predicted_answer:
                break

            collector.add_error(
                "load_data.retry",
                predicted_answer,
                {
                    "qid": qid,
                    "retry_count": current_retry + 1,
                },
            )
            print(f"Detected error: {predicted_answer}, retrying question: {qid}")
            current_retry += 1
            collector.set_retry_count(current_retry)

        collector.set_retry_count(current_retry)
        elapsed = 0.0 if question_started is None else time.perf_counter() - question_started
        collector.set_timing("question_total_seconds", elapsed)

        result = {
            "qid": qid,
            "question": question,
            "predicted_answer": predicted_answer,
            "golden_answers": golden_answers,
            "idx": idx,
            "success": True,
            **extract_result_metrics(collector),
        }

        await write_result_to_file(result, output_file_path)

        failure_reason = classify_failure_reason(predicted_answer)
        collector.set_outcome(
            status="failure" if failure_reason else "success",
            predicted_answer=predicted_answer,
            failure_reason=failure_reason,
        )
        if failure_reason:
            await write_debug_log(collector, qid, idx)

        return result

    except Exception as e:
        collector.add_error(
            "load_data.process_example",
            str(e),
            {
                "exception_type": type(e).__name__,
                "retry_count": current_retry,
            },
        )
        collector.set_retry_count(current_retry)
        elapsed = 0.0 if question_started is None else time.perf_counter() - question_started
        collector.set_timing("question_total_seconds", elapsed)
        collector.set_outcome(
            status="failure",
            predicted_answer=None,
            failure_reason="outer_exception",
        )

        error_result = {
            "qid": qid,
            "question": question,
            "error": f"Error during processing: {str(e)}",
            "idx": idx,
            "success": False,
            **extract_result_metrics(collector),
        }

        await write_result_to_file(error_result, output_file_path)
        await write_debug_log(collector, qid, idx)

        return error_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run RT-RAG batch inference")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process at most N examples from the dataset after applying --start-index",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based dataset index to resume from",
    )
    return parser.parse_args()


async def main(limit=None, start_index=0):
    global semaphore

    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"Loading dataset: {dataset_name}")
    data = load_dataset('json', data_files=data_path, split='train')
    print(f"Successfully loaded dataset with {len(data)} examples")

    dataset_size = len(data)
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if start_index > dataset_size:
        raise ValueError(f"--start-index must be <= dataset size ({dataset_size})")

    end_index = dataset_size
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        end_index = min(start_index + limit, dataset_size)
        print(f"Limiting run to {end_index - start_index} examples starting from index {start_index}")
    elif start_index > 0:
        print(f"Resuming run from index {start_index}")

    os.makedirs(output_dir, exist_ok=True)

    tasks = [process_example(data[idx], idx) for idx in range(start_index, end_index)]

    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name} dataset")

    success_count = sum(1 for r in results if r.get("success", False))
    fail_count = len(results) - success_count

    print(f"Processing completed. Main results saved to {output_file_path}")
    print(f"Total examples processed: {len(results)}, Success: {success_count}, Failures: {fail_count}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(limit=args.limit, start_index=args.start_index))
