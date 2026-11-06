"""Normalize local records or a Hugging Face dataset into a stable manifest."""

import argparse
from pathlib import PurePosixPath

from datasets import load_dataset

from tiger.common import IMAGE_PROMPT, index_records, read_records, sample_id, write_records


def normalize_record(row, position, image_prefix):
    key = sample_id(row.get("id", position))
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        users = [turn["content"] for turn in prompt if turn["role"] == "user"]
        systems = [turn["content"] for turn in prompt if turn["role"] == "system"]
        if not users:
            raise ValueError(f"Sample {key}: prompt has no user turn")
        question = users[-1]
        system_prompt = systems[0] if systems else IMAGE_PROMPT
        answer = row["reward_model"]["ground_truth"]
        target = row.get("target") or []
        solution = target[0]["content"] if target else ""
    else:
        question = row.get("question", row.get("problem"))
        answer = row.get("answer")
        solution = row.get("solution") or ""
        system_prompt = row.get("system_prompt") or IMAGE_PROMPT
        # GSM8K stores the reasoning and final answer in the same field.
        if isinstance(answer, str) and "####" in answer:
            solution = solution or answer
            answer = answer.rsplit("####", 1)[1].strip()
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {key}: missing text question/problem")
    if answer is None or not str(answer).strip():
        raise ValueError(f"Sample {key}: missing ground-truth answer")
    if not isinstance(solution, str) or not isinstance(system_prompt, str):
        raise ValueError(f"Sample {key}: solution and system_prompt must be strings")
    return {
        "id": key,
        "question": question,
        "answer": str(answer),
        "solution": solution,
        "system_prompt": system_prompt,
        "image": str(PurePosixPath(image_prefix) / f"image_{key}.png"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local JSON, JSONL, or Parquet file")
    source.add_argument("--hf-dataset", help="Hugging Face dataset name")
    parser.add_argument("--hf-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="Pin a Hugging Face dataset revision")
    parser.add_argument("--image-prefix", default="images")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.input:
        rows = read_records(args.input)
    else:
        rows = load_dataset(
            args.hf_dataset, args.hf_config, split=args.split, revision=args.revision
        )
    normalized = [
        normalize_record(row, i, args.image_prefix) for i, row in enumerate(rows)
    ]
    index_records(normalized)
    write_records(args.output, normalized)
    print(f"Wrote {len(normalized)} samples to {args.output}")


if __name__ == "__main__":
    main()
