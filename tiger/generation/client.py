"""Shared OpenAI-compatible vLLM generation with bounded concurrency and resume."""

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from tiger.common import TEXT_PROMPT, image_path, index_records, output_path, read_records


def messages_for(row, mode, image_root):
    if mode == "text":
        content = [{"type": "text", "text": TEXT_PROMPT + "\nQuestion: " + row["question"]}]
    else:
        path = image_path(image_root, row["image"])
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + encoded},
            },
            {"type": "text", "text": row["system_prompt"]},
        ]
    return [{"role": "user", "content": content}]


def is_complete(row):
    return (
        isinstance(row.get("output_text"), str)
        and bool(row["output_text"].strip())
        and not row.get("error")
        and row.get("finish_reason", "stop") == "stop"
    )


def pending_samples(rows, previous, config):
    indexed = index_records(rows)
    done = set()
    for row in previous:
        key = str(row["id"])
        if key not in indexed:
            raise ValueError(f"Resume output contains unknown sample id {key}")
        if row.get("generation_config") != config:
            raise ValueError("Resume settings differ; use a new --output")
        for field in ("question", "answer", "solution", "image", "system_prompt"):
            if row.get(field) != indexed[key].get(field):
                raise ValueError(f"Resume input changed for sample {key}: {field}")
        if is_complete(row):
            done.add(key)
    return [row for row in rows if str(row["id"]) not in done]


def generate_one(client, row, args, config):
    result = dict(row, generation_config=config)
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=messages_for(row, config["mode"], args.image_root),
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        choice = response.choices[0]
        result.update(
            output_text=choice.message.content,
            finish_reason=choice.finish_reason,
        )
        if not is_complete(result):
            result["error"] = "Empty or incomplete response; retry with --resume"
    except Exception as exc:
        result.update(output_text=None, error=f"{type(exc).__name__}: {exc}")
    return result


def main(mode):
    parser = argparse.ArgumentParser(description=f"Generate {mode}-only reasoning via vLLM")
    parser.add_argument("--input", required=True, help="Normalized manifest")
    parser.add_argument("--output", required=True, help="JSONL output, never silently overwritten")
    parser.add_argument("--model-name", "--model_name", required=True)
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1 or args.max_new_tokens < 1 or args.request_timeout <= 0:
        parser.error("workers, tokens, and timeout must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.max_retries < 0:
        parser.error("temperature/retries must be nonnegative and top-p in (0, 1]")
    if not args.output.endswith(".jsonl"):
        parser.error("--output must end in .jsonl")
    rows = read_records(args.input)
    index_records(rows)
    config = {
        "model": args.model_name,
        "mode": mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    path = Path(args.output)
    if path.exists() and args.resume:
        rows = pending_samples(rows, read_records(path), config)
        # A valid final JSON object without a newline must not be joined to the next row.
        with path.open("rb") as handle:
            handle.seek(0, 2)
            if handle.tell():
                handle.seek(-1, 2)
                needs_newline = handle.read(1) != b"\n"
            else:
                needs_newline = False
        file_mode = "a"
    else:
        path = output_path(path)
        file_mode, needs_newline = "x", False
    if mode == "image":
        for row in rows:
            if not image_path(args.image_root, row["image"]).is_file():
                raise FileNotFoundError(f"Missing image for sample {row['id']}")
    failures = 0
    with path.open(file_mode, encoding="utf-8") as writer, OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    ) as client, ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        if needs_newline:
            writer.write("\n")
        # Bound queued image payloads instead of submitting the entire dataset.
        batch_size = args.num_workers * 2
        completed = 0
        for start in range(0, len(rows), batch_size):
            futures = [
                pool.submit(generate_one, client, row, args, config)
                for row in rows[start:start + batch_size]
            ]
            for future in as_completed(futures):
                result = future.result()
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                writer.flush()
                completed += 1
                if not is_complete(result):
                    failures += 1
                    print(f"Sample {result['id']} failed: {result['error']}", flush=True)
            print(f"Processed {completed}/{len(rows)} samples", flush=True)
    print(f"Wrote {len(rows)} attempts to {path}; failures: {failures}")
    if failures:
        raise SystemExit(1)
