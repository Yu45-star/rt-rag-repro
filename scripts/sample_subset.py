#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a deterministic subset from a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name stored in the metadata output.")
    parser.add_argument("--input", required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output", required=True, help="Path to the sampled JSONL output.")
    parser.add_argument("--meta", required=True, help="Path to the metadata JSON output.")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed used for sampling.")
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")


def write_metadata(path, metadata):
    meta_path = Path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    records = load_jsonl(args.input)

    if args.sample_size > len(records):
        raise ValueError(
            f"Requested sample_size={args.sample_size}, but source dataset only has {len(records)} records."
        )

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(records)), args.sample_size))
    sampled_records = [records[index] for index in selected_indices]

    write_jsonl(args.output, sampled_records)
    write_metadata(
        args.meta,
        {
            "dataset": args.dataset,
            "source_file": args.input,
            "output_file": args.output,
            "seed": args.seed,
            "sample_size": args.sample_size,
            "source_count": len(records),
            "selected_indices": selected_indices,
        },
    )

    print(f"Wrote {len(sampled_records)} records to {args.output}")
    print(f"Wrote metadata to {args.meta}")


if __name__ == "__main__":
    main()
