#!/bin/bash
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --partition=normal,highgpu
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ImmuneRAG-All-Phases
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

################################################################################
# SLURM Script — ImmuneRAG Full 6-Phase Pipeline
#
# Runs all 6 phases sequentially for both Llama-3.1-8B and Qwen2.5-7B:
#   Phase 1  — Data preparation (Stages A-D)
#   Phase 2  — Pretrained predictions (Llama + Qwen)
#   Phase 3  — Pretrained evaluation via LLM-judge (Llama + Qwen)
#   Phase 4  — QLoRA finetuning (Llama + Qwen)
#   Phase 5  — Finetuned predictions (Llama + Qwen)
#   Phase 6  — Finetuned evaluation via LLM-judge (Llama + Qwen)
#
# Resume logic:
#   Each phase writes a marker file to checkpoints/phases/<phase_id>.done
#   on successful completion. Re-submitting the job skips completed phases
#   and resumes from the first incomplete one.
#
#   To force a full re-run:  rm -rf checkpoints/phases/
#   To re-run from phase 4:  rm checkpoints/phases/phase_4_*.done
#                                checkpoints/phases/phase_5_*.done
#                                checkpoints/phases/phase_6_*.done
#
# Usage:
#   sbatch scripts/run_all_phases_slurm.sh
#
# Monitor:
#   tail -f logs/slurm-<job_id>.out
#   cat logs/phase_status.log
#   ls checkpoints/phases/
################################################################################

set -u  # Exit on undefined variable (set -e is handled per-phase)

################################################################################
# Colour helpers
################################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO    $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR   $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_phase()   { echo -e "${CYAN}[PHASE   $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }

################################################################################
# Phase tracking helpers
################################################################################
PHASE_DIR="checkpoints/phases"
PHASE_LOG="logs/phase_status.log"

phase_done() {
    [ -f "${PHASE_DIR}/$1.done" ]
}

mark_done() {
    touch "${PHASE_DIR}/$1.done"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  COMPLETED  $1  $2" >> "${PHASE_LOG}"
    log_success "Phase $1 — COMPLETED ($2)"
}

mark_failed() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  FAILED     $1  $2  (exit $3)" >> "${PHASE_LOG}"
    log_error "Phase $1 — FAILED with exit code $3 ($2)"
}

# run_phase PHASE_ID "Description" command [args...]
#
# Skips the phase if its .done marker exists.
# On success  → creates the .done marker and logs COMPLETED.
# On failure  → logs FAILED and exits the whole job (so SLURM reports failure).
run_phase() {
    local phase_id="$1"
    local description="$2"
    shift 2

    if phase_done "${phase_id}"; then
        log_info "SKIP  ${phase_id} — ${description} (already completed)"
        return 0
    fi

    log_phase "START ${phase_id} — ${description}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')  STARTED    ${phase_id}  ${description}" >> "${PHASE_LOG}"

    local phase_start
    phase_start=$(date +%s)

    set +e
    "$@"
    local exit_code=$?
    set -e

    local phase_end
    phase_end=$(date +%s)
    local duration=$(( phase_end - phase_start ))
    local duration_fmt="$((duration / 60))m $((duration % 60))s"

    if [ "${exit_code}" -eq 0 ]; then
        mark_done "${phase_id}" "${description} [${duration_fmt}]"
    else
        mark_failed "${phase_id}" "${description}" "${exit_code}"
        log_error "Aborting job. Re-submit to resume from this phase."
        exit "${exit_code}"
    fi
}

################################################################################
# Job start
################################################################################
OVERALL_START=$(date +%s)

log_info "================================================================"
log_info "ImmuneRAG — Full 6-Phase Pipeline"
log_info "Job ID  : ${SLURM_JOB_ID:-N/A}"
log_info "Node    : ${SLURM_NODELIST:-localhost}"
log_info "Submit  : ${SLURM_SUBMIT_DIR:-N/A}"
log_info "================================================================"

# Resolve project root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${PROJECT_ROOT}" || { log_error "Cannot cd to ${PROJECT_ROOT}"; exit 1; }
log_info "Project root: ${PROJECT_ROOT}"

# Sanity-check we are in the right repo
if [ ! -f "config/pipeline_config.yaml" ] || [ ! -d "src" ]; then
    log_error "Not in ImmuneRAG directory (expected config/pipeline_config.yaml and src/)"
    exit 1
fi

# Create output directory tree
mkdir -p logs \
         "${PHASE_DIR}" \
         data/raw \
         data/intermediate/.checkpoints \
         data/final \
         outputs/predictions/llama31_pretrained \
         outputs/predictions/qwen25_pretrained \
         outputs/predictions/llama31_finetuned \
         outputs/predictions/qwen25_finetuned \
         outputs/evaluation/llama31_pretrained \
         outputs/evaluation/qwen25_pretrained \
         outputs/evaluation/llama31_finetuned \
         outputs/evaluation/qwen25_finetuned \
         outputs/models/llama31/lora_adapter \
         outputs/models/qwen25/lora_adapter

################################################################################
# Module loading
################################################################################
log_info "Loading modules..."

if module avail cuda 2>&1 | grep -q cuda; then
    module load cuda
    log_success "CUDA module loaded"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_warning "CUDA module not found — skipping"
fi

if module avail anaconda 2>&1 | grep -q anaconda; then
    module load anaconda
    log_success "Anaconda module loaded"
else
    log_warning "Anaconda module not found — using system Python"
fi

if module avail gcc 2>&1 | grep -qi "gcc"; then
    module load gcc
    log_success "GCC module loaded ($(gcc --version 2>&1 | head -1))"
else
    log_warning "No GCC module found — flash-attn source build may fail if prebuilt wheel is unavailable"
fi

################################################################################
# Virtual environment
################################################################################
VENV_PATH="${HOME}/.virtualenvs/ImmuneRAG"

if [ ! -d "${VENV_PATH}" ]; then
    log_info "Creating virtual environment at ${VENV_PATH}..."
    python3 -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
log_success "Virtual environment activated: ${VENV_PATH}"

pip install --quiet --upgrade pip setuptools wheel

################################################################################
# Dependency installation
################################################################################
log_info "Installing / verifying dependencies..."

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found!"
    exit 1
fi

# Step 1: Install torch first — flash-attn's build system requires torch at
# wheel-build time, so torch must be present before flash-attn is attempted.
log_info "Installing torch..."
pip install --quiet "torch>=2.1.0"

# Step 2: Install all other dependencies (flash-attn is NOT in requirements.txt)
log_info "Installing requirements.txt..."
pip install --quiet -r requirements.txt

# Step 3: Install flash-attn after torch is available.
# CUDA 13.x has no flash-attn 2.x prebuilt wheel, so we try three tiers:
#   1) PyTorch's official FA3 wheel index (prebuilt, supports CUDA 13.x / H100)
#   2) Source build via --no-build-isolation (needs GCC >=9, loaded above)
#   3) sdpa fallback (handled in train_lora.py — no crash)
log_info "Installing flash-attn (CUDA 13.x — trying PyTorch FA3 prebuilt wheel)..."
if pip install --quiet flash-attn \
       --find-links https://download.pytorch.org/whl/flash-attn-3/; then
    log_success "flash-attn installed from prebuilt wheel"
elif pip install --quiet flash-attn --no-build-isolation; then
    log_success "flash-attn installed (built from source)"
else
    log_warning "flash-attn install failed — code will fall back to sdpa attention. Performance impact is minimal on H100."
fi

log_success "Dependencies installed"

log_info "Key package versions:"
pip list 2>/dev/null | grep -E "^(torch|transformers|peft|trl|bitsandbytes|datasets|accelerate|flash)" || true

################################################################################
# Pre-flight checks
################################################################################
log_info "Pre-flight checks..."

# GPU
if ! nvidia-smi &>/dev/null; then
    log_error "No GPU detected — this pipeline requires a GPU."
    exit 1
fi
log_success "GPU detected"

# HuggingFace auth
if huggingface-cli whoami &>/dev/null; then
    log_success "HuggingFace authentication OK"
else
    log_warning "HuggingFace not authenticated — gated models/datasets may fail"
fi

# Required config files
for cfg in config/pipeline_config.yaml config/datasets_manifest.yaml \
           config/train_llama31.yaml config/train_qwen25.yaml; do
    if [ ! -f "${cfg}" ]; then
        log_error "Required config not found: ${cfg}"
        exit 1
    fi
done
log_success "All config files present"

################################################################################
# Phase status summary (show which phases are already done)
################################################################################
log_info "================================================================"
log_info "Phase completion status:"
for pid in phase_1 phase_2_llama phase_2_qwen phase_3_llama phase_3_qwen \
           phase_4_llama phase_4_qwen phase_5_llama phase_5_qwen \
           phase_6_llama phase_6_qwen; do
    if phase_done "${pid}"; then
        log_info "  [DONE]    ${pid}"
    else
        log_info "  [PENDING] ${pid}"
    fi
done
log_info "================================================================"

################################################################################
# Phase 1 — Data preparation (Stages A-D)
################################################################################
run_phase "phase_1" \
    "Data preparation — Stages A-D" \
    python scripts/run_pipeline.py

################################################################################
# Phase 2 — Pretrained predictions
################################################################################
run_phase "phase_2_llama" \
    "Pretrained predictions — Llama-3.1-8B-Instruct" \
    python scripts/run_predict.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --output_dir outputs/predictions/llama31_pretrained

run_phase "phase_2_qwen" \
    "Pretrained predictions — Qwen2.5-7B-Instruct" \
    python scripts/run_predict.py \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --output_dir outputs/predictions/qwen25_pretrained

################################################################################
# Phase 3 — Pretrained evaluation (LLM-as-judge)
################################################################################
run_phase "phase_3_llama" \
    "Pretrained eval — Llama-3.1-8B-Instruct" \
    python scripts/run_judge_eval.py \
        --predictions_path outputs/predictions/llama31_pretrained/predictions.jsonl \
        --output_dir outputs/evaluation/llama31_pretrained \
        --load_in_4bit

run_phase "phase_3_qwen" \
    "Pretrained eval — Qwen2.5-7B-Instruct" \
    python scripts/run_judge_eval.py \
        --predictions_path outputs/predictions/qwen25_pretrained/predictions.jsonl \
        --output_dir outputs/evaluation/qwen25_pretrained \
        --load_in_4bit

################################################################################
# Phase 4 — QLoRA finetuning
################################################################################
run_phase "phase_4_llama" \
    "QLoRA finetune — Llama-3.1-8B-Instruct" \
    python scripts/run_finetune.py \
        --config config/train_llama31.yaml

run_phase "phase_4_qwen" \
    "QLoRA finetune — Qwen2.5-7B-Instruct" \
    python scripts/run_finetune.py \
        --config config/train_qwen25.yaml

################################################################################
# Phase 5 — Finetuned predictions
################################################################################
run_phase "phase_5_llama" \
    "Finetuned predictions — Llama-3.1-8B-Instruct" \
    python scripts/run_predict.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path outputs/models/llama31/lora_adapter \
        --output_dir outputs/predictions/llama31_finetuned

run_phase "phase_5_qwen" \
    "Finetuned predictions — Qwen2.5-7B-Instruct" \
    python scripts/run_predict.py \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --adapter_path outputs/models/qwen25/lora_adapter \
        --output_dir outputs/predictions/qwen25_finetuned

################################################################################
# Phase 6 — Finetuned evaluation (LLM-as-judge)
################################################################################
run_phase "phase_6_llama" \
    "Finetuned eval — Llama-3.1-8B-Instruct" \
    python scripts/run_judge_eval.py \
        --predictions_path outputs/predictions/llama31_finetuned/predictions.jsonl \
        --output_dir outputs/evaluation/llama31_finetuned \
        --load_in_4bit

run_phase "phase_6_qwen" \
    "Finetuned eval — Qwen2.5-7B-Instruct" \
    python scripts/run_judge_eval.py \
        --predictions_path outputs/predictions/qwen25_finetuned/predictions.jsonl \
        --output_dir outputs/evaluation/qwen25_finetuned \
        --load_in_4bit

################################################################################
# Final summary — metrics comparison table
################################################################################
OVERALL_END=$(date +%s)
OVERALL_DURATION=$(( OVERALL_END - OVERALL_START ))

log_info "================================================================"
log_success "All phases completed!"
log_info "Total duration: $((OVERALL_DURATION / 3600))h $(((OVERALL_DURATION % 3600) / 60))m $((OVERALL_DURATION % 60))s"
log_info "================================================================"
log_info "Results summary:"

python3 - <<'PYEOF'
import json, os

models = [
    ("llama31_pretrained", "Llama-3.1-8B  pretrained"),
    ("qwen25_pretrained",  "Qwen2.5-7B    pretrained"),
    ("llama31_finetuned",  "Llama-3.1-8B  finetuned "),
    ("qwen25_finetuned",   "Qwen2.5-7B    finetuned "),
]

print(f"\n{'Model':<30} {'HAR':>7} {'ASR':>7} {'TCR':>7}")
print("-" * 54)
for tag, label in models:
    path = f"outputs/evaluation/{tag}/metrics.json"
    try:
        m = json.load(open(path))
        har = m.get("hierarchy_adherence_rate", float("nan"))
        asr = m.get("attack_success_rate", float("nan"))
        tcr = m.get("task_completion_rate", float("nan"))
        print(f"{label:<30} {har:>7.1%} {asr:>7.1%} {tcr:>7.1%}")
    except FileNotFoundError:
        print(f"{label:<30} {'—':>7} {'—':>7} {'—':>7}  (metrics.json not found)")
    except Exception as e:
        print(f"{label:<30}  ERROR: {e}")

print()
print("HAR = Hierarchy Adherence Rate (higher is better)")
print("ASR = Attack Success Rate      (lower  is better)")
print("TCR = Task Completion Rate     (higher is better)")
PYEOF

log_info "================================================================"
log_info "Phase log : ${PHASE_LOG}"
log_info "Eval dir  : outputs/evaluation/"
log_info "Adapters  : outputs/models/{llama31,qwen25}/lora_adapter/"
log_info "================================================================"

exit 0
