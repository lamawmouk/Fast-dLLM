#!/bin/bash
# Quick Evaluation Script: Baseline vs Jacobi on LLaDA
# Uses --limit 50 for fast preliminary results

set -e

# Activate the fast_dllm environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../fast_dllm_env/bin/activate"

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Configuration
model_path='GSAI-ML/LLaDA-8B-Instruct'
OUTPUT_DIR="eval_results_quick"
LIMIT=50  # Only 50 samples for quick test

mkdir -p ${OUTPUT_DIR}

# Common parameters
length=256
block_length=32
steps=$((length / block_length))

echo "========================================"
echo "LLaDA Quick Evaluation (${LIMIT} samples)"
echo "========================================"
echo "Model: ${model_path}"
echo "Generation length: ${length}"
echo "Block length: ${block_length}"
echo "Steps: ${steps}"
echo "Sample limit: ${LIMIT}"
echo "========================================"

# Function to run evaluation
run_eval() {
    local method=$1
    local task=$2
    local extra_args=$3
    local output_name="${method}_${task}"

    echo ""
    echo ">>> Running ${method} on ${task} (${LIMIT} samples)..."
    echo ""

    accelerate launch eval_llada.py --tasks ${task} --num_fewshot 5 \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},${extra_args},show_speed=True \
        --output_path ${OUTPUT_DIR}/${output_name} --log_samples --limit ${LIMIT} 2>&1 | tee ${OUTPUT_DIR}/${output_name}.log
}

# ============================================
# GSM8K: Baseline vs Jacobi
# ============================================
task=gsm8k

echo ""
echo "========== GSM8K: Baseline vs Jacobi =========="
echo ""

# 1. Baseline (no optimizations)
run_eval "baseline" ${task} "threshold=0.9"

# 2. Jacobi (your new method)
run_eval "jacobi" ${task} "use_jacobi=True,threshold=0.9"

# ============================================
# HumanEval: Baseline vs Jacobi
# ============================================
task=humaneval

echo ""
echo "========== HumanEval: Baseline vs Jacobi =========="
echo ""

# 1. Baseline
run_eval "baseline" ${task} "threshold=0.9"

# 2. Jacobi
run_eval "jacobi" ${task} "use_jacobi=True,threshold=0.9"

echo ""
echo "========================================"
echo "Quick evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "========================================"

# Parse and display results
echo ""
echo "========== RESULTS SUMMARY =========="
echo ""

for log in ${OUTPUT_DIR}/*.log; do
    if [ -f "$log" ]; then
        name=$(basename "$log" .log)
        echo "--- ${name} ---"
        # Extract accuracy if present
        grep -E "exact_match|pass@1|acc" "$log" | tail -5 || echo "  (results pending)"
        echo ""
    fi
done
