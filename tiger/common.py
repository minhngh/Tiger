"""Shared data formats and file handling for the standalone preparation tools."""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq


IMAGE_PROMPT = (
    "Solve the complex visual logical reasoning problem through step-by-step "
    "reasoning. Think about the reasoning process first, answer the question "
    'and then put the final answer in one "\\boxed{}".'
)
TEXT_PROMPT = (
    "Solve the following logical reasoning problem through step-by-step "
    "reasoning. Think about the reasoning process first, answer the question "
    'and then put the final answer in one "\\boxed{}".'
)


def read_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        records = pq.read_table(path).to_pylist()
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [
                json.loads(line) for line in handle if line.strip()
            ]
    else:
        raise ValueError(f"{path}: use .json, .jsonl, or .parquet")
    if any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: every record must be an object")
    return records


def output_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose a new output")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_records(path, rows):
    if Path(path).suffix not in {".json", ".jsonl"}:
        raise ValueError("JSON output must end in .json or .jsonl")
    path = output_path(path)
    with path.open("x", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif path.suffix == ".json":
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def sample_id(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe sample id {value!r}; use letters, digits, _, -, .")
    return value


def index_records(rows):
    indexed = {}
    for row in rows:
        key = sample_id(row["id"])
        if key in indexed:
            raise ValueError(f"Duplicate sample id: {key}")
        indexed[key] = row
    return indexed


def image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path must stay inside --image-root: {relative}")
    return path


def visual_prompt(row):
    prompt = row.get("system_prompt") or IMAGE_PROMPT
    return "<image>\n" + prompt
