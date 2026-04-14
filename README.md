# ImmuneRAG — Instruction Hierarchy Training for Prompt Injection Defense

Small-scale implementation of the instruction hierarchy framework from
["The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions"](https://arxiv.org/abs/2404.13208),
applied to indirect prompt injection defense in RAG systems.

Models are trained to enforce **System > User > Tool** privilege ordering so that
malicious instructions injected via retrieved content are refused or ignored.

---

## Supported Models

| Model | Config |
|-------|--------|
| `meta-llama/Llama-3.1-8B-Instruct` | `config/train_llama31.yaml` |
| `Qwen/Qwen2.5-7B-Instruct` | `config/train_qwen25.yaml` |

---

## 6-Phase Workflow

```
Phase 1: Data Preparation    ──→  data/final/{train,test}.jsonl
Phase 2: Pretrained Predict  ──→  outputs/predictions/{model}_pretrained/
Phase 3: Pretrained Eval     ──→  outputs/evaluation/{model}_pretrained/metrics.json
Phase 4: Finetune            ──→  outputs/models/{model}/lora_adapter/
Phase 5: Finetuned Predict   ──→  outputs/predictions/{model}_finetuned/
Phase 6: Finetuned Eval      ──→  outputs/evaluation/{model}_finetuned/metrics.json
```

Each phase is independently resumable from its input artifacts.

---

## Installation

```bash
pip install -r requirements.txt

# HuggingFace login (required for gated datasets and models)
huggingface-cli login
```

---

## Phase 1 — Data Preparation

Runs a 4-stage pipeline (A→D) to produce 4,296 train / 950 test examples.
Stage C uses **Mistral-Small-24B-Instruct** (loaded via transformers) to generate
diverse, LLM-augmented hierarchy training cases.

```bash
# Full pipeline
python scripts/run_pipeline.py

# Specific stages
python scripts/run_pipeline.py --stages A B C D

# Resume from stage C
python scripts/run_pipeline.py --from-stage C
```

Stage C auto-resumes from checkpoint after interruptions.  To disable LLM generation
and use templates only, set `llm_generation.enabled: false` in
`config/pipeline_config.yaml`.

**Output**: `data/final/train.jsonl` (4,296 examples), `data/final/test.jsonl` (950 examples)

---

## Phase 2 — Pretrained Model Predictions

Run the base model on the test set and save predictions for evaluation.

```bash
# Llama-3.1-8B
python scripts/run_predict.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/predictions/llama31_pretrained

# Qwen2.5-7B
python scripts/run_predict.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --output_dir outputs/predictions/qwen25_pretrained
```

**Output**: `outputs/predictions/{model}_pretrained/predictions.jsonl`

Each prediction stores: `id`, `model_id`, `scenario`, `alignment`, `attack_family`,
`messages_context`, `expected_output`, `model_output`.

---

## Phase 3 — Pretrained Model Evaluation

Uses Mistral-Small as an LLM judge to evaluate whether each prediction correctly
followed the instruction hierarchy.

```bash
python scripts/run_judge_eval.py \
    --predictions_path outputs/predictions/llama31_pretrained/predictions.jsonl \
    --output_dir outputs/evaluation/llama31_pretrained

# Save VRAM: load judge in 4-bit (~14GB vs ~48GB)
python scripts/run_judge_eval.py \
    --predictions_path outputs/predictions/llama31_pretrained/predictions.jsonl \
    --output_dir outputs/evaluation/llama31_pretrained \
    --load_in_4bit
```

**Output**: `metrics.json` and `judge_results.jsonl` in the output directory.

**Metrics**:
| Metric | Description |
|--------|-------------|
| `hierarchy_adherence_rate` (HAR) | % of all examples correctly handled ↑ |
| `attack_success_rate` (ASR) | % of attacks that succeeded ↓ |
| `task_completion_rate` (TCR) | % of aligned tasks correctly completed ↑ |
| `by_scenario` | HAR/ASR per scenario type |
| `by_attack_family` | ASR per attack family |

---

## Phase 4 — Finetuning

QLoRA (4-bit) finetuning on the instruction hierarchy train set.

```bash
# Llama-3.1-8B
python scripts/run_finetune.py --config config/train_llama31.yaml

# Qwen2.5-7B
python scripts/run_finetune.py --config config/train_qwen25.yaml
```

**Output**: `outputs/models/{llama31|qwen25}/lora_adapter/`

Key hyperparameters (editable in the config files):
- QLoRA rank=16, alpha=32, 4-bit NF4
- LR=2e-4, cosine schedule, 2 epochs, effective batch=16

---

## Phase 5 — Finetuned Model Predictions

Same script as Phase 2, with `--adapter_path` pointing to the saved adapter.

```bash
python scripts/run_predict.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path outputs/models/llama31/lora_adapter \
    --output_dir outputs/predictions/llama31_finetuned

python scripts/run_predict.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --adapter_path outputs/models/qwen25/lora_adapter \
    --output_dir outputs/predictions/qwen25_finetuned
```

---

## Phase 6 — Finetuned Model Evaluation

Same script as Phase 3, pointed at the finetuned predictions.

```bash
python scripts/run_judge_eval.py \
    --predictions_path outputs/predictions/llama31_finetuned/predictions.jsonl \
    --output_dir outputs/evaluation/llama31_finetuned
```

---

## Results

| Model | HAR | ASR | TCR |
|-------|-----|-----|-----|
| Llama-3.1-8B Pretrained | 78.0% | 33.0% | 88.1% |
| Llama-3.1-8B Finetuned | **85.6%** | **23.1%** | **93.6%** |
| Qwen2.5-7B Pretrained | 76.2% | 47.8% | 98.2% |
| Qwen2.5-7B Finetuned | **85.0%** | **27.1%** | **96.0%** |

Finetuning improves hierarchy adherence (HAR) by +7-9pp and reduces attack success
rate (ASR) by 10-21pp across both models, while maintaining or improving task
completion (TCR).

---

## Project Structure

```
scripts/
  run_pipeline.py      Phase 1: data preparation (stages A-D)
  run_predict.py       Phase 2 & 5: model predictions
  run_judge_eval.py    Phase 3 & 6: LLM-as-judge evaluation
  run_finetune.py      Phase 4: QLoRA finetuning

src/
  pipeline/            Stages A-D
    stage_a_download.py
    stage_b_normalize.py
    stage_c_hierarchy.py
    stage_d_quality.py
  utils/
    llm_service.py     Mistral-Small wrapper (transformers)
  evaluation/
    predict.py         ModelPredictor class
    llm_judge.py       LLMJudge class
    judge_eval.py      JudgeEvaluator class
  training/
    train_lora.py      LoRATrainer class

config/
  datasets_manifest.yaml
  pipeline_config.yaml
  train_llama31.yaml
  train_qwen25.yaml

data/
  final/
    train.jsonl        4,296 training examples
    test.jsonl         950 test examples

outputs/
  predictions/         Phase 2 & 5 outputs
  evaluation/          Phase 3 & 6 outputs
  models/              Phase 4 outputs (LoRA adapters)
```

---

## Data Statistics

| Split | Total | Aligned | Misaligned |
|-------|-------|---------|-----------|
| Train | 4,296 | 2,225 (51.8%) | 2,071 (48.2%) |
| Test  | 950   | 498   (52.4%) | 452   (47.6%) |

6 scenario types: `open_aligned`, `sys_probe_aligned`, `open_misaligned`,
`closed_domain_misaligned`, `tool_output_misaligned`, `sys_extract_misaligned`

4 attack families: `override`, `extraction`, `indirect`, `tool_exfil`

