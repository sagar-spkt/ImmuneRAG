#!/bin/bash
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --partition=normal,highgpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ImmuneRAG-Pipeline
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

################################################################################
# SLURM Script for ImmuneRAG Data Preparation Pipeline
#
# This script:
# 1. Sets up virtual environment with dependencies
# 2. Starts vLLM server for LLM-augmented generation
# 3. Runs full pipeline (Stages A-E)
# 4. Cleans up vLLM server
#
# Usage: sbatch scripts/run_pipeline_slurm.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR $(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Start timing
START_TIME=$(date +%s)

log_info "============================================================"
log_info "ImmuneRAG Data Preparation Pipeline - SLURM Job"
log_info "Job ID: ${SLURM_JOB_ID}"
log_info "Node: ${SLURM_NODELIST}"
log_info "============================================================"

# Get project root directory (assumes script is in scripts/ subdirectory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

log_info "Project root: ${PROJECT_ROOT}"

# Create logs directory if it doesn't exist
mkdir -p logs
PIPELINE_LOG="logs/pipeline_$(date '+%Y%m%d_%H%M%S').log"
log_info "Pipeline log: ${PIPELINE_LOG}"

################################################################################
# 1. MODULE LOADING
################################################################################
log_info "Loading modules..."

# Load CUDA module
if module avail cuda 2>&1 | grep -q cuda; then
    module load cuda
    log_success "CUDA module loaded"
    nvidia-smi
else
    log_warning "CUDA module not found, skipping"
fi

# Load Anaconda module
if module avail anaconda 2>&1 | grep -q anaconda; then
    module load anaconda
    log_success "Anaconda module loaded"
else
    log_warning "Anaconda module not found, will use system Python"
fi

################################################################################
# 2. VIRTUAL ENVIRONMENT SETUP
################################################################################
VENV_PATH="${HOME}/.virtualenvs/ImmuneRAG"

log_info "Setting up virtual environment at ${VENV_PATH}..."

if [ ! -d "${VENV_PATH}" ]; then
    log_info "Creating new virtual environment..."
    python3 -m venv "${VENV_PATH}"
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"
log_success "Virtual environment activated"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

################################################################################
# 3. DEPENDENCY INSTALLATION
################################################################################
log_info "Installing dependencies..."

# Install requirements.txt
if [ -f "requirements.txt" ]; then
    log_info "Installing from requirements.txt..."
    pip install -r requirements.txt
    log_success "requirements.txt installed"
else
    log_error "requirements.txt not found!"
    exit 1
fi

# Install vLLM
log_info "Installing vLLM..."
pip install vllm

# Install OpenAI client (for vLLM communication)
log_info "Installing OpenAI client..."
pip install openai

log_success "All dependencies installed"

# Show installed packages
log_info "Key package versions:"
pip list | grep -E "(torch|transformers|vllm|openai|datasets)" || true

################################################################################
# 4. PRE-FLIGHT CHECKS
################################################################################
log_info "Running pre-flight checks..."

# Check if HuggingFace token is configured
if huggingface-cli whoami &>/dev/null; then
    log_success "HuggingFace authentication verified"
else
    log_warning "HuggingFace authentication not configured (may cause issues with gated datasets)"
    log_warning "Run 'huggingface-cli login' before submitting job"
fi

# Check config files exist
if [ ! -f "config/pipeline_config.yaml" ]; then
    log_error "config/pipeline_config.yaml not found!"
    exit 1
fi

if [ ! -f "config/datasets_manifest.yaml" ]; then
    log_error "config/datasets_manifest.yaml not found!"
    exit 1
fi

log_success "Configuration files found"

# Check GPU availability
if nvidia-smi &>/dev/null; then
    log_success "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log_error "No GPU detected! This pipeline requires GPU."
    exit 1
fi

################################################################################
# 5. vLLM SERVER SETUP
################################################################################
log_info "============================================================"
log_info "Starting vLLM Server"
log_info "============================================================"

VLLM_PORT=8000
VLLM_MODEL="mistralai/Mistral-Small-Instruct-2409"
VLLM_LOG="logs/vllm_$(date '+%Y%m%d_%H%M%S').log"

log_info "Model: ${VLLM_MODEL}"
log_info "Port: ${VLLM_PORT}"
log_info "Log: ${VLLM_LOG}"

# Start vLLM server in background
log_info "Starting vLLM server..."
nohup vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
log_info "vLLM server started with PID: ${VLLM_PID}"

# Function to cleanup vLLM server
cleanup_vllm() {
    if [ ! -z "${VLLM_PID:-}" ] && kill -0 ${VLLM_PID} 2>/dev/null; then
        log_info "Terminating vLLM server (PID: ${VLLM_PID})..."
        kill ${VLLM_PID}
        sleep 5

        # Force kill if still running
        if kill -0 ${VLLM_PID} 2>/dev/null; then
            log_warning "vLLM server not responding, force killing..."
            kill -9 ${VLLM_PID}
        fi

        log_success "vLLM server terminated"
    fi
}

# Register cleanup function for signals
trap cleanup_vllm EXIT INT TERM

# Wait for vLLM server to be ready
log_info "Waiting for vLLM server to be ready..."
MAX_WAIT=300  # 5 minutes
WAIT_COUNT=0

while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        log_success "vLLM server is ready!"

        # Show available models
        log_info "Available models:"
        curl -s "http://localhost:${VLLM_PORT}/v1/models" | python3 -m json.tool | grep "id" || true
        break
    fi

    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))

    if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
        log_info "Still waiting... (${WAIT_COUNT}s elapsed)"
    fi

    # Check if vLLM process is still running
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        log_error "vLLM server process died!"
        log_error "Check ${VLLM_LOG} for details"
        tail -n 50 "${VLLM_LOG}"
        exit 1
    fi
done

if [ ${WAIT_COUNT} -ge ${MAX_WAIT} ]; then
    log_error "vLLM server failed to start after ${MAX_WAIT} seconds"
    log_error "Check ${VLLM_LOG} for details"
    tail -n 50 "${VLLM_LOG}"
    exit 1
fi

################################################################################
# 6. PIPELINE EXECUTION
################################################################################
log_info "============================================================"
log_info "Running Data Preparation Pipeline"
log_info "============================================================"

PIPELINE_START=$(date +%s)

# Create necessary directories
mkdir -p data/raw
mkdir -p data/intermediate
mkdir -p data/intermediate/.checkpoints
mkdir -p data/final

log_info "Running all stages (A → E)..."

# Run the full pipeline
python scripts/run_pipeline.py \
    --manifest config/datasets_manifest.yaml \
    --config config/pipeline_config.yaml \
    --verbose \
    2>&1 | tee -a "${PIPELINE_LOG}"

PIPELINE_EXIT_CODE=${PIPESTATUS[0]}

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

if [ ${PIPELINE_EXIT_CODE} -eq 0 ]; then
    log_success "Pipeline completed successfully!"
    log_info "Pipeline duration: $((PIPELINE_DURATION / 60)) minutes $((PIPELINE_DURATION % 60)) seconds"
else
    log_error "Pipeline failed with exit code: ${PIPELINE_EXIT_CODE}"
    log_error "Check ${PIPELINE_LOG} for details"
    exit ${PIPELINE_EXIT_CODE}
fi

################################################################################
# 7. OUTPUT VERIFICATION
################################################################################
log_info "============================================================"
log_info "Verifying Pipeline Outputs"
log_info "============================================================"

# Check Stage A output
if [ -f "data/raw/raw_index.json" ]; then
    DATASET_COUNT=$(cat data/raw/raw_index.json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
    log_success "Stage A: raw_index.json created (${DATASET_COUNT} datasets)"
else
    log_warning "Stage A: raw_index.json not found"
fi

# Check Stage B output
if [ -f "data/intermediate/seeds.jsonl" ]; then
    SEED_COUNT=$(wc -l < data/intermediate/seeds.jsonl)
    log_success "Stage B: seeds.jsonl created (${SEED_COUNT} seeds)"
else
    log_warning "Stage B: seeds.jsonl not found"
fi

# Check Stage C outputs
if [ -f "data/intermediate/hierarchy_cases.jsonl" ]; then
    CASE_COUNT=$(wc -l < data/intermediate/hierarchy_cases.jsonl)
    log_success "Stage C: hierarchy_cases.jsonl created (${CASE_COUNT} cases)"

    # Show generation method distribution
    log_info "Generation method distribution:"
    cat data/intermediate/hierarchy_cases.jsonl | \
        python3 -c "import sys, json; methods = [json.loads(line).get('notes', {}).get('generation_method', 'unknown') for line in sys.stdin]; print('\n'.join([f'  {m}: {methods.count(m)}' for m in set(methods)]))" || true
else
    log_warning "Stage C: hierarchy_cases.jsonl not found"
fi

if [ -f "data/intermediate/payload_library.jsonl" ]; then
    PAYLOAD_COUNT=$(wc -l < data/intermediate/payload_library.jsonl)
    log_success "Stage C: payload_library.jsonl created (${PAYLOAD_COUNT} payloads)"
else
    log_warning "Stage C: payload_library.jsonl not found"
fi

# Check Stage D outputs
if [ -f "data/final/train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/final/train.jsonl)
    log_success "Stage D: train.jsonl created (${TRAIN_COUNT} examples)"
else
    log_warning "Stage D: train.jsonl not found"
fi

if [ -f "data/final/test.jsonl" ]; then
    TEST_COUNT=$(wc -l < data/final/test.jsonl)
    log_success "Stage D: test.jsonl created (${TEST_COUNT} examples)"
else
    log_warning "Stage D: test.jsonl not found"
fi

if [ -f "data/final/stats.json" ]; then
    log_success "Stage D: stats.json created"
    log_info "Dataset statistics:"
    python3 -m json.tool data/final/stats.json | grep -A 10 "summary" || true
else
    log_warning "Stage D: stats.json not found"
fi

# Check Stage E outputs
if [ -f "data/final/train_text.jsonl" ]; then
    TRAIN_TEXT_COUNT=$(wc -l < data/final/train_text.jsonl)
    log_success "Stage E: train_text.jsonl created (${TRAIN_TEXT_COUNT} examples)"
else
    log_warning "Stage E: train_text.jsonl not found"
fi

if [ -f "data/final/test_text.jsonl" ]; then
    TEST_TEXT_COUNT=$(wc -l < data/final/test_text.jsonl)
    log_success "Stage E: test_text.jsonl created (${TEST_TEXT_COUNT} examples)"
else
    log_warning "Stage E: test_text.jsonl not found"
fi

################################################################################
# 8. CLEANUP
################################################################################
log_info "============================================================"
log_info "Cleanup"
log_info "============================================================"

# Cleanup vLLM server (handled by trap)
cleanup_vllm

# Show disk usage
log_info "Disk usage:"
du -sh data/* 2>/dev/null || true

################################################################################
# 9. FINAL SUMMARY
################################################################################
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

log_info "============================================================"
log_success "Pipeline Completed Successfully!"
log_info "============================================================"
log_info "Total duration: $((TOTAL_DURATION / 60)) minutes $((TOTAL_DURATION % 60)) seconds"
log_info "Pipeline log: ${PIPELINE_LOG}"
log_info "vLLM log: ${VLLM_LOG}"
log_info ""
log_info "Output files:"
log_info "  - Training data: data/final/train_text.jsonl"
log_info "  - Test data: data/final/test_text.jsonl"
log_info "  - Statistics: data/final/stats.json"
log_info ""
log_info "Next steps:"
log_info "  1. Review statistics: cat data/final/stats.json | python3 -m json.tool"
log_info "  2. Inspect samples: head data/final/train_text.jsonl"
log_info "  3. Train model: python src/training/train_lora.py --config config/training_config.yaml"
log_info "============================================================"

exit 0
