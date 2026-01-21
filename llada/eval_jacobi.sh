#!/bin/bash
# Jacobi Evaluation Script for LLaDA
# This script runs evaluations using your Jacobi forcing approach

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Common parameters
model_path='GSAI-ML/LLaDA-8B-Instruct'
length=256
block_length=32
steps=$((length / block_length))

echo "========================================"
echo "Running Jacobi Evaluation"
echo "========================================"

# GSM8K with Jacobi
echo "Running GSM8K with Jacobi..."
task=gsm8k
accelerate launch eval_llada.py --tasks ${task} --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_jacobi=True,threshold=0.9,show_speed=True

# HumanEval with Jacobi
echo "Running HumanEval with Jacobi..."
task=humaneval
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_jacobi=True,threshold=0.9,show_speed=True \
--output_path evals_results/jacobi/humaneval-ns0-${length} --log_samples

echo "========================================"
echo "Jacobi Evaluation Complete"
echo "========================================"
