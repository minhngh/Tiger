"""Select text-correct/image-incorrect pairs using verified ground-truth answers."""

import argparse
import json
import random
from collections import Counter

from math_verify import parse, verify

from tiger.common import index_records, read_records, visual_prompt, write_records
from tiger.generation.client import is_complete


def load_generations(paths, samples, expected_mode=None):
    generations = {}
    for path in paths:
        for row in read_records(path):
            key = str(row["id"])
            if key not in samples:
                raise ValueError(f"{path}: unknown sample id {key}")
            mode = row.get("generation_config", {}).get("mode")
            if expected_mode is not None and mode is not None and mode != expected_mode:
                raise ValueError(f"{path}: expected {expected_mode} generations, got {mode}")
            for field in ("question", "answer", "image", "system_prompt"):
                if field in row and str(row[field]) != str(samples[key][field]):
                    raise ValueError(f"{path}: sample {key} has mismatched {field}")
            if is_complete(row):
                # Earlier files/rows win when merging resumed shards or trials.
                generations.setdefault(key, row)
    return generations


def build_pairs(samples, text_rows, image_rows, chosen_source="text"):
    pairs, stats = [], Counter(total=len(samples))
    for key, sample in samples.items():
        if key not in text_rows or key not in image_rows:
            stats["missing_generation"] += 1
            continue
        gold = parse(sample["answer"])
        if not gold:
            stats["unparseable_ground_truth"] += 1
            continue
        text = text_rows[key]["output_text"]
        image = image_rows[key]["output_text"]
        # math_verify expects the gold expression before the prediction.
        text_correct = bool(verify(gold, parse(text)))
        image_correct = bool(verify(gold, parse(image)))
        stats[f"text_{'correct' if text_correct else 'wrong'}_image_"
              f"{'correct' if image_correct else 'wrong'}"] += 1
        if not text_correct or image_correct:
            continue
        chosen = text if chosen_source == "text" else sample["solution"]
        if not chosen.strip():
            stats["missing_solution"] += 1
            continue
        if chosen == image:
            stats["identical_pair"] += 1
            continue
        pairs.append({
            "id": key,
            "image": sample["image"],
            "prompt": visual_prompt(sample),
            "chosen": chosen,
            "rejected": image,
        })
    stats["selected"] = len(pairs)
    return pairs, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--text-generations", nargs="+", required=True)
    parser.add_argument("--image-generations", nargs="+", required=True)
    parser.add_argument("--chosen-source", choices=["text", "solution"], default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = index_records(read_records(args.samples))
    text_rows = load_generations(args.text_generations, samples, expected_mode="text")
    image_rows = load_generations(args.image_generations, samples, expected_mode="image")
    pairs, stats = build_pairs(samples, text_rows, image_rows, args.chosen_source)
    random.Random(args.seed).shuffle(pairs)
    write_records(args.output, pairs)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote DPO pairs to {args.output}")


if __name__ == "__main__":
    main()
