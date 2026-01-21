#!/bin/bash
# Jacobi Evaluation Script for Fast-dLLM v2
# This script runs evaluations using your Jacobi forcing approach

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_path=Efficient-Large-Model/Fast_dLLM_v2_7B

echo "========================================"
echo "Running Jacobi Evaluation for Fast-dLLM v2"
echo "========================================"

# MMLU with Jacobi
echo "Running MMLU with Jacobi..."
task=mmlu
accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},use_jacobi=True

# GPQA with Jacobi
echo "Running GPQA with Jacobi..."
task=gpqa_main_n_shot
accelerate launch eval.py --tasks ${task} --batch_size 1 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},use_jacobi=True

# GSM8K with Jacobi
echo "Running GSM8K with Jacobi..."
task=gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},use_jacobi=True,threshold=1,show_speed=True

# MATH with Jacobi
echo "Running MATH with Jacobi..."
task=minerva_math
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},use_jacobi=True,threshold=1,show_speed=True

# IFEval with Jacobi
echo "Running IFEval with Jacobi..."
task=ifeval
accelerate launch eval.py --tasks ${task} --batch_size 32 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},use_jacobi=True,threshold=1,show_speed=True

echo "========================================"
echo "Jacobi Evaluation Complete"
echo "========================================"
