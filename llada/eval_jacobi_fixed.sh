#!/bin/bash
# Fixed Jacobi Evaluation Script
# FIXES:
# 1. Use steps=64 so steps_per_block = 8 (allows 8 refinement iterations per block)
# 2. Temperature is now configurable (default 0.1 for Gumbel noise to work)

set -e

# Activate the fast_dllm environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../fast_dllm_env/bin/activate"

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Configuration
model_path='GSAI-ML/LLaDA-8B-Instruct'
OUTPUT_DIR="eval_results_jacobi_fixed"
LIMIT=50  # 50 samples for quick test

mkdir -p ${OUTPUT_DIR}

# Common parameters
length=256
block_length=32
num_blocks=$((length / block_length))  # = 8

# CRITICAL: For Jacobi, we need more steps per block
# Baseline: steps = num_blocks = 8 (1 step per block)
# Jacobi: steps = 64 (8 steps per block for refinement)
steps_baseline=8
steps_jacobi=64

echo "========================================"
echo "LLaDA Jacobi vs Baseline - FIXED CONFIG"
echo "========================================"
echo "Model: ${model_path}"
echo "Generation length: ${length}"
echo "Block length: ${block_length}"
echo "Num blocks: ${num_blocks}"
echo ""
echo "Baseline steps: ${steps_baseline} (${steps_baseline}/${num_blocks} = 1 step/block)"
echo "Jacobi steps: ${steps_jacobi} ($((steps_jacobi / num_blocks)) steps/block)"
echo "Sample limit: ${LIMIT}"
echo "========================================"

# ============================================
# GSM8K Evaluations
# ============================================
task=gsm8k

echo ""
echo "========== GSM8K: Baseline vs Jacobi =========="

# Baseline
echo ""
echo ">>> Running baseline on gsm8k (${LIMIT} samples)..."
echo ""
accelerate launch eval_llada.py --tasks ${task} --num_fewshot 5 \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps_baseline},block_length=${block_length},threshold=0.9,show_speed=True \
    --output_path ${OUTPUT_DIR}/baseline_gsm8k --log_samples --limit ${LIMIT} 2>&1 | tee ${OUTPUT_DIR}/baseline_gsm8k.log

# Jacobi (with fixed config)
echo ""
echo ">>> Running jacobi on gsm8k (${LIMIT} samples)..."
echo ""
accelerate launch eval_llada.py --tasks ${task} --num_fewshot 5 \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${steps_jacobi},block_length=${block_length},use_jacobi=True,jacobi_temperature=0.1,threshold=0.9,show_speed=True \
    --output_path ${OUTPUT_DIR}/jacobi_gsm8k --log_samples --limit ${LIMIT} 2>&1 | tee ${OUTPUT_DIR}/jacobi_gsm8k.log

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "========================================"

# Summary
echo ""
echo "========== RESULTS SUMMARY =========="
echo ""

echo "=== Baseline GSM8K ==="
grep -E "exact_match" ${OUTPUT_DIR}/baseline_gsm8k.log | tail -5 || echo "(parsing...)"
grep "avg nfe" ${OUTPUT_DIR}/baseline_gsm8k.log | tail -1 || echo "(no NFE)"

echo ""
echo "=== Jacobi GSM8K ==="
grep -E "exact_match" ${OUTPUT_DIR}/jacobi_gsm8k.log | tail -5 || echo "(parsing...)"
grep "avg nfe" ${OUTPUT_DIR}/jacobi_gsm8k.log | tail -1 || echo "(no NFE)"
