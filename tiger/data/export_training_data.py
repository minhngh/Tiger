"""Export SFT conversations or image-based GRPO JSON/Parquet with ground truth."""

import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq

from tiger.common import (
    image_path,
    index_records,
    output_path,
    read_records,
    visual_prompt,
    write_records,
)


GRPO_REASONING_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. Think about the reasoning process first "
    "and then answer the question. The final answer MUST BE put in \\boxed{}."
)
GRPO_PROMPTS = {
    "qwen2.5vl": (
        "You FIRST think about the reasoning process as an internal monologue and "
        "then provide the final answer. The reasoning process MUST BE enclosed "
        "within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
    ),
    "qwen3vl": GRPO_REASONING_PROMPT,
    "internvl3.5": GRPO_REASONING_PROMPT,
    "llava-onevision-1.5": GRPO_REASONING_PROMPT,
}


def sft_record(sample, pair=None):
    response = pair["chosen"] if pair is not None else sample["solution"]
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Sample {sample['id']}: SFT requires a nonempty solution")
    return {
        "id": str(sample["id"]),
        "image": sample["image"],
        "conversations": [
            {"from": "human", "value": pair["prompt"] if pair else visual_prompt(sample)},
            {"from": "gpt", "value": response},
        ],
    }


def grpo_json_record(sample, model_family):
    return {
        "id": str(sample["id"]),
        "image": sample["image"],
        "conversations": [
            {"from": "human", "value": GRPO_PROMPTS[model_family]},
            {"from": "gpt", "value": "\\boxed{" + sample["answer"] + "}"},
        ],
    }


def write_grpo_parquet(path, samples, image_root, model_family):
    prompt = GRPO_PROMPTS[model_family]
    schema = pa.schema([
        ("images", pa.list_(pa.struct([
            ("bytes", pa.binary()), ("path", pa.string()),
        ]))),
        ("problem", pa.string()),
        ("answer", pa.string()),
    ])
    path = output_path(path)
    # Stream image bytes in small batches instead of holding every PNG in RAM.
    with path.open("xb") as handle, pq.ParquetWriter(handle, schema) as writer:
        for start in range(0, len(samples), 128):
            batch = []
            for sample in samples[start:start + 128]:
                path_to_image = image_path(image_root, sample["image"])
                row = {
                    "images": [{
                        "bytes": path_to_image.read_bytes(),
                        "path": sample["image"],
                    }],
                    "problem": prompt,
                    "answer": sample["answer"],
                }
                batch.append(row)
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="Normalized manifest")
    parser.add_argument("--dpo", help="Restrict to these pairs; use chosen responses for SFT")
    parser.add_argument("--format", required=True,
                        choices=["sft", "grpo-json", "grpo-image"])
    parser.add_argument("--grpo-model", choices=list(GRPO_PROMPTS),
                        help="Required for GRPO exports; selects the model-specific instruction")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.format.startswith("grpo-") and args.grpo_model is None:
        parser.error("--grpo-model is required for GRPO exports")
    parquet = args.format == "grpo-image"
    if parquet and not args.output.endswith(".parquet"):
        parser.error("GRPO Parquet output must end in .parquet")
    if not parquet and not args.output.endswith((".json", ".jsonl")):
        parser.error("Conversation output must end in .json or .jsonl")
    samples = index_records(read_records(args.samples))
    pairs = index_records(read_records(args.dpo)) if args.dpo else None
    selected = []
    for key in pairs if pairs is not None else samples:
        if key not in samples:
            raise ValueError(f"DPO id {key} not found in --samples")
        sample = samples[key]
        if pairs is not None and pairs[key]["image"] != sample["image"]:
            raise ValueError(f"DPO image differs from manifest for sample {key}")
        selected.append(sample)
    random.Random(args.seed).shuffle(selected)
    for sample in selected:
        if not image_path(args.image_root, sample["image"]).is_file():
            raise FileNotFoundError(f"Missing image for sample {sample['id']}")
    if args.format == "sft":
        rows = [
            sft_record(sample, pairs[str(sample["id"])] if pairs is not None else None)
            for sample in selected
        ]
        write_records(args.output, rows)
    elif args.format == "grpo-json":
        write_records(args.output, [
            grpo_json_record(sample, args.grpo_model) for sample in selected
        ])
    else:
        write_grpo_parquet(args.output, selected, args.image_root, args.grpo_model)
    print(f"Wrote {len(selected)} {args.format} samples to {args.output}")


if __name__ == "__main__":
    main()
