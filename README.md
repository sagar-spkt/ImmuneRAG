# Instruction Hierarchy Training for LLMs

Small-scale implementation of the instruction hierarchy framework from the OpenAI paper ["The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions"](https://arxiv.org/abs/2404.13208).

## Overview

This project trains Llama-3-8B to learn privilege hierarchies (System > User > Tool) for defending against prompt injection attacks, jailbreaks, and system prompt extraction.

### Key Concepts

- **Privilege Hierarchy**: System messages > User messages > Tool/third-party outputs
- **Context Synthesis**: For aligned lower-level instructions → model should follow them
- **Context Ignorance**: For misaligned lower-level instructions → model should ignore or refuse
- **Small Scale**: 4,500 train / 1,000 test examples (vs. full-scale production training)

## Project Structure

```
instruction_hierarchy/
├── config/
│   ├── datasets_manifest.yaml      # Dataset sources and sampling policies
│   ├── pipeline_config.yaml        # Pipeline stage configurations
│   └── training_config.yaml        # LoRA training hyperparameters
├── data/
│   ├── raw/                        # Downloaded datasets (Stage A)
│   ├── intermediate/               # Processed data (Stages B & C)
│   └── final/                      # Train/test splits (Stages D & E)
├── src/
│   ├── pipeline/                   # Data pipeline stages (A-E)
│   ├── training/                   # LoRA training module
│   └── evaluation/                 # Evaluation harness & metrics
├── scripts/
│   └── run_pipeline.py            # Main pipeline orchestrator
└── outputs/                        # Training outputs & checkpoints
```

## Pipeline Stages

### Stage A: Dataset Download
Downloads and caches 7 seed datasets from HuggingFace:
- **Aligned examples**: OASST2, UltraChat (high-quality instruction-following)
- **Adversarial examples**: HackAPrompt, Gandalf, LLMail-Inject, etc.

```bash
python src/pipeline/stage_a_download.py
```

### Stage B: Seed Normalization
Converts diverse formats to canonical `(prompt, response)` seed format with filtering.

```bash
python src/pipeline/stage_b_normalize.py
```

### Stage C: Hierarchy Case Generation
Generates training cases using:
- **Context synthesis** (aligned): Decompose constraints, split across hierarchy levels
- **Context ignorance** (misaligned): Inject attacks, train to ignore/refuse
- Closed-domain, tool-output, and system extraction scenarios

```bash
python src/pipeline/stage_c_hierarchy.py
```

### Stage D: Quality Control & Balancing
- Deduplication (minhash LSH)
- Quality filtering
- Deterministic train/test split (hash-based)
- Balance to target 50/50 aligned/misaligned

```bash
python src/pipeline/stage_d_quality.py
```

### Stage E: Model-Specific Rendering
Renders to Llama-3 chat format with proper role tokens.

```bash
python src/pipeline/stage_e_render.py
```

## Quick Start

### 1. Installation

```bash
# Create environment
conda create -n inst-hierarchy python=3.10
conda activate inst-hierarchy

# Install dependencies
cd instruction_hierarchy
pip install -r requirements.txt
```

### 2. Run Data Pipeline

```bash
# Run all stages
python scripts/run_pipeline.py

# Or run specific stages
python scripts/run_pipeline.py --stages A B C

# Or run from a specific stage
python scripts/run_pipeline.py --from-stage C
```

### 3. Train Model (QLoRA on H100)

```bash
python src/training/train_lora.py --config config/training_config.yaml
```

**Training config highlights**:
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Method: QLoRA (4-bit quantization)
- LoRA rank: 16, alpha: 32
- Learning rate: 2e-4
- Epochs: 2
- Effective batch size: 16

### 4. Evaluate Model

```bash
python src/evaluation/eval_harness.py \
  --model_path outputs/lora_adapter \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --test_data data/final/test_text.jsonl \
  --output outputs/evaluation
```

**Metrics computed**:
- **ASR (Attack Success Rate)**: Lower is better (model resists attacks)
- **System Extraction Rate**: Lower is better (model protects secrets)
- **Over-Refusal Rate**: Lower is better (model doesn't refuse benign requests)
- **Constraint Adherence**: Higher is better (model follows aligned constraints)

## Configuration

### Dataset Manifest (`config/datasets_manifest.yaml`)

Define which datasets to use, sampling policies, and target data budget:

```yaml
datasets:
  oasst2:
    identifier: "OpenAssistant/oasst2"
    license: "Apache-2.0"
    sampling:
      train_samples: 2000
      test_samples: 400

target_budget:
  train_total: 4500
  test_total: 1000
```

### Pipeline Config (`config/pipeline_config.yaml`)

Control each pipeline stage:

```yaml
pipeline:
  stage_c_hierarchy:
    context_synthesis:
      enabled: true  # Generate aligned examples
    context_ignorance:
      enabled: true  # Generate misaligned examples
    closed_domain:
      enabled: true  # Closed-domain tasks (summarization, etc.)
```

### Training Config (`config/training_config.yaml`)

Tune LoRA hyperparameters:

```yaml
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", ...]

training:
  learning_rate: 2.0e-4
  num_train_epochs: 2
```

## Incremental Development

This project is designed for **incremental exploration**:

1. **Start small**: Run pipeline with limited data first
   ```bash
   # Edit config/datasets_manifest.yaml to reduce sample counts
   python scripts/run_pipeline.py
   ```

2. **Inspect outputs**: Check `data/final/stats.json` for data distribution

3. **Iterate on Stage C**: Most customization happens in hierarchy generation
   - Modify `src/pipeline/stage_c_hierarchy.py`
   - Adjust `config/pipeline_config.yaml` (Stage C settings)

4. **Quick training test**: Debug mode in training config
   ```yaml
   debug:
     enabled: true
     max_train_samples: 100
   ```

5. **Scale up**: Increase data budget, run full training

## Data Budget & Mixture

**Target**: 4,500 train / 1,000 test

**Train mixture**:
- 1,800 aligned (open-domain context synthesis)
- 450 aligned (system prompt probing)
- 900 misaligned (open-domain override)
- 600 misaligned (closed-domain injection-in-data)
- 450 misaligned (tool-output injection)
- 300 misaligned (system extraction)

**Test mixture**: Proportional to train (see `config/datasets_manifest.yaml`)

## Expected Results

Based on the paper, a well-trained instruction hierarchy model should achieve:

- **Prompt Injection Defense**: ~30-60% improvement in robustness
- **System Extraction Defense**: ~50-70% reduction in successful extractions
- **Over-Refusal**: <10% on benign aligned instructions
- **Generalization**: Robustness to unseen attack types (jailbreaks, etc.)

Note: Small-scale training may yield lower absolute numbers but should show clear directional improvements.

## Troubleshooting

### Stage A fails with authentication error
Some datasets (e.g., HackAPrompt) are gated. Log in to HuggingFace:
```bash
huggingface-cli login
```

### Out of memory during training
Reduce batch size or enable more aggressive quantization:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Payload library is empty (Stage C)
Stage C placeholder needs implementation. Start by:
1. Reading attack datasets in `_build_payload_library()`
2. Extracting attack strings from each source
3. Categorizing by attack family

## Next Steps

After initial setup:

1. **Implement dataset extractors** (Stage B): Add parsers for each dataset format
2. **Build payload library** (Stage C): Extract real attack strings
3. **Implement hierarchy generators** (Stage C):
   - `_generate_aligned()` → context synthesis logic
   - `_generate_misaligned()` → context ignorance logic
4. **Run full pipeline** and inspect outputs
5. **Train and evaluate**

## References

- **Paper**: [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
- **Proposal**: `../Proposal/execution_plan.md` (detailed implementation plan)
- **Base Model**: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## License

This project implementation is provided for research and educational purposes. Please respect the licenses of individual datasets (Apache-2.0, MIT, etc.) as specified in `config/datasets_manifest.yaml`.
