#!/bin/bash
# Comprehensive Evaluation Script: Baseline vs Jacobi on LLaDA
# This script runs evaluations comparing different decoding methods

set -e

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Configuration
model_path='GSAI-ML/LLaDA-8B-Instruct'
OUTPUT_DIR="eval_results_comparison"
mkdir -p ${OUTPUT_DIR}

# Common parameters
length=256
block_length=32
steps=$((length / block_length))

echo "========================================"
echo "LLaDA Decoding Method Comparison"
echo "========================================"
echo "Model: ${model_path}"
echo "Generation length: ${length}"
echo "Block length: ${block_length}"
echo "Steps: ${steps}"
echo "========================================"

# Function to run evaluation
run_eval() {
    local method=$1
    local task=$2
    local extra_args=$3
    local output_name="${method}_${task}"

    echo ""
    echo ">>> Running ${method} on ${task}..."
    echo ""

    accelerate launch eval_llada.py --tasks ${task} --num_fewshot 5 \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},${extra_args},show_speed=True \
        --output_path ${OUTPUT_DIR}/${output_name} --log_samples 2>&1 | tee ${OUTPUT_DIR}/${output_name}.log
}

# ============================================
# GSM8K Evaluations
# ============================================
task=gsm8k

echo ""
echo "========== GSM8K Evaluations =========="
echo ""

# 1. Baseline (no optimizations)
run_eval "baseline" ${task} "threshold=0.9"

# 2. With Prefix Cache
run_eval "prefix_cache" ${task} "use_cache=True,threshold=0.9"

# 3. With Dual Cache
run_eval "dual_cache" ${task} "use_cache=True,dual_cache=True,threshold=0.9"

# 4. Jacobi (your new method)
run_eval "jacobi" ${task} "use_jacobi=True,threshold=0.9"

# ============================================
# HumanEval Evaluations
# ============================================
task=humaneval

echo ""
echo "========== HumanEval Evaluations =========="
echo ""

# 1. Baseline
run_eval "baseline" ${task} "threshold=0.9"

# 2. With Prefix Cache
run_eval "prefix_cache" ${task} "use_cache=True,threshold=0.9"

# 3. With Dual Cache
run_eval "dual_cache" ${task} "use_cache=True,dual_cache=True,threshold=0.9"

# 4. Jacobi
run_eval "jacobi" ${task} "use_jacobi=True,threshold=0.9"

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "========================================"

# Generate summary
echo ""
echo "Generating summary..."
echo ""

cat > ${OUTPUT_DIR}/summary.md << 'EOF'
# LLaDA Decoding Method Comparison Results

## Methods Compared
1. **Baseline**: Standard confidence-based decoding
2. **Prefix Cache**: With KV-cache for prefix
3. **Dual Cache**: With dual KV-cache
4. **Jacobi**: Fixed Gumbel noise + mismatch detection (NEW)

## Results

Check individual log files for detailed metrics:
- Accuracy
- Tokens per second (TPS)
- Number of forward evaluations (NFE)

EOF

echo "Summary saved to ${OUTPUT_DIR}/summary.md"
