# Seeing Isn't Enough: Unlocking Visual Reasoning by Bridging the Reasoning-Access Gap with TIGER

**TIGER | NeurIPS 2026**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0) [![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) 

Prepare **OpenR1** data from
[Elliott/Openr1-Math-46k-8192](https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192)
for SFT, DPO, and **image-based GRPO**. Start by converting OpenR1 questions
into images, then generate responses and export training data. Model training
uses the external repositories listed below.

## Code Layout

```text
tiger/
  common.py                  # Shared formats, prompts, and file helpers
  data/
    prepare_dataset.py       # OpenR1 manifest preparation
    render_images.py         # Original question-to-image rendering
    build_dpo.py             # Correctness-based preference selection
    export_training_data.py  # SFT and image-GRPO exports
  generation/
    client.py                # Shared vLLM API client and resume handling
    text.py                  # Text-only generation
    image.py                 # Image-only generation
```

Run the commands below from the repository root using `python -m`.

## 1. Convert OpenR1 Questions into Images

Load the training split of `Elliott/Openr1-Math-46k-8192`, then render the
questions into images:

```bash
python -m tiger.data.prepare_dataset \
  --hf-dataset Elliott/Openr1-Math-46k-8192 \
  --split train \
  --image-prefix openr1/single_line_images \
  --output data/samples.jsonl

python -m tiger.data.render_images \
  --input data/samples.jsonl \
  --image-root data
```

Images are saved to `data/openr1/single_line_images/`; the aligned manifest is
`data/samples.jsonl`.

## 2. Generate Text-Only and Image-Only Responses

After Step 1, start a compatible vLLM server in a separate GPU environment:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
```

Adjust the server options for your hardware and model. We use
**temperature 0** for both text-only and image-only data generation:

```bash
export OPENAI_BASE_URL=http://localhost:8001/v1
export OPENAI_API_KEY=EMPTY
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

python -m tiger.generation.text --input data/samples.jsonl \
  --model-name "$MODEL" --output results/text.jsonl \
  --num-workers 8 --max-new-tokens 4096

python -m tiger.generation.image --input data/samples.jsonl --image-root data \
  --model-name "$MODEL" --output results/image.jsonl \
  --num-workers 8 --max-new-tokens 4096
```

Change `MODEL` and `OPENAI_BASE_URL` for your setup. Add `--resume` to the same
generation command to retry incomplete samples. Existing files are not
overwritten; use new output paths when changing settings.

## 3. Build DPO Pairs

```bash
python -m tiger.data.build_dpo --samples data/samples.jsonl \
  --text-generations results/text.jsonl \
  --image-generations results/image.jsonl \
  --output data/dpo.json
```

This selects examples whose text-only answer is correct but image-only answer
is incorrect, verified against OpenR1 ground truth. The text response becomes
`chosen`; the image response becomes `rejected`.

## 4. Export SFT and GRPO

```bash
# SFT: use each selected DPO pair's chosen response.
python -m tiger.data.export_training_data --samples data/samples.jsonl --dpo data/dpo.json \
  --format sft --output data/sft.json

# Image GRPO: use the selected problems and original ground-truth answers.
python -m tiger.data.export_training_data --samples data/samples.jsonl --dpo data/dpo.json \
  --format grpo-json --grpo-model qwen2.5vl --output data/grpo.json
python -m tiger.data.export_training_data --samples data/samples.jsonl --dpo data/dpo.json \
  --format grpo-image --grpo-model qwen2.5vl --output data/grpo_image.parquet
```

- `data/dpo.json`: preference pairs for DPO.
- `data/sft.json`: image-conditioned conversations using the chosen responses.
- `data/grpo_image.parquet`: embedded images, model-specific prompts, and ground-truth answers for EasyR1.
- `data/grpo.json`: optional GRPO conversation export.

GRPO requires `--grpo-model`:

| Model | `--grpo-model` |
| --- | --- |
| Qwen2.5-VL-Instruct | `qwen2.5vl` |
| Qwen3VL | `qwen3vl` |
| InternVL3.5 | `internvl3.5` |
| LLaVA-OneVision-1.5 | `llava-onevision-1.5` |

**Why different instructions?** In our GRPO setup, Qwen2.5-VL-Instruct can
output reasoning enclosed in `<think>...</think>` tags, whereas Qwen3-VL,
InternVL3.5, and LLaVA-OneVision-1.5 do not output this tagged format.
We therefore request tagged reasoning for Qwen2.5-VL-Instruct and untagged
reasoning for the other models. All models are instructed to put the final
answer in `\boxed{}`.

Exact instructions are defined in
[`GRPO_PROMPTS`](tiger/data/export_training_data.py). Configure the trainer's multimodal
template to attach image inputs.

To prepare GRPO on all OpenR1 samples, omit `--dpo`.

### GRPO Format Rewards

Use the corresponding format reward in the GRPO trainer. These functions
score output format only, not answer correctness.

**Qwen2.5-VL-Instruct:** require `<think>...</think>` followed by a boxed answer.

```python
import re


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0
```

**Qwen3-VL, InternVL3.5, and LLaVA-OneVision-1.5:** require a boxed answer.

```python
import re


def format_reward(response: str) -> float:
    pattern = re.compile(r"\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.search(pattern, response)
    return 1.0 if format_match else 0.0
```

## 5. Training with External Repositories

Use these repositories for model training after data preparation:

| Training | Repository | Prepared data |
| --- | --- | --- |
| Image-based GRPO | [EasyR1](https://github.com/hiyouga/EasyR1) | Image GRPO Parquet, such as `data/grpo_image.parquet`. |
| SFT and DPO for QwenVL | [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) | SFT conversations in `data/sft.json` and DPO pairs in `data/dpo.json`. |
| SFT and DPO for InternVL | [InternVL](https://github.com/opengvlab/internvl) | Prepared SFT/DPO examples, adapted to the selected InternVL training recipe. |

For EasyR1, map `images`, `problem`, and `answer` to its dataset configuration
and align the reward function with the selected instruction. For QwenVL, set
`--data_path` to the JSON export and `--image_folder` to `data`. For InternVL,
adapt the records to the chosen training recipe's schema and conversation
template. Follow each upstream repository's installation and training guide.

## 6. Real-World Benchmark Evaluation

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate
models on real-world benchmarks. Follow its documentation for benchmark
setup, model integration, and evaluation commands.

## Acknowledgments

We thank the authors and maintainers of the following open-source projects:

- [EasyR1](https://github.com/hiyouga/EasyR1) for GRPO training.
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) for SFT and DPO training of QwenVL models.
- [InternVL](https://github.com/opengvlab/internvl) for InternVL models and SFT/DPO training workflows.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for real-world benchmark evaluation.


## License

Apache-2.0. See [LICENSE](LICENSE).
