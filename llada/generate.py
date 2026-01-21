# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def generate_fixed_gumbel_noise(shape, device, dtype=torch.float64, seed=None):
    """Generate fixed Gumbel(0,1) noise for deterministic decoding."""
    if seed is not None:
        torch.manual_seed(seed)
    uniform = torch.rand(shape, device=device, dtype=dtype)
    uniform = torch.clamp(uniform, min=1e-10, max=1.0 - 1e-10)
    return -torch.log(-torch.log(uniform))


def select_tokens_with_fixed_gumbel(logits, gumbel_noise, temperature):
    """Select tokens via argmax(logits + temp * gumbel_noise)."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    logits_f64 = logits.to(torch.float64)
    perturbed = logits_f64 + temperature * gumbel_noise
    return torch.argmax(perturbed, dim=-1)


def detect_mismatch_positions(x_prev, x_curr, mask_id, block_start, block_end):
    """Detect first mismatch position between iterations.

    A mismatch is when a previously unmasked token changes value.
    Returns:
        has_mismatch: (B,) bool tensor - whether each batch item has a mismatch
        first_mismatch_pos: (B,) long tensor - absolute position of first mismatch, -1 if none
    """
    B, device = x_prev.shape[0], x_prev.device
    block_prev = x_prev[:, block_start:block_end]
    block_curr = x_curr[:, block_start:block_end]
    was_unmasked = (block_prev != mask_id)
    mismatch = was_unmasked & (block_prev != block_curr)
    positions = torch.arange(block_end - block_start, device=device).unsqueeze(0).expand(B, -1)
    mismatch_positions = torch.where(mismatch, positions, torch.full_like(positions, block_end - block_start))
    first_mismatch_relative = mismatch_positions.min(dim=1).values
    has_mismatch = mismatch.any(dim=1)
    first_mismatch_pos = torch.where(has_mismatch, first_mismatch_relative + block_start,
                                      torch.full((B,), -1, device=device, dtype=torch.long))
    return has_mismatch, first_mismatch_pos


def slide_window_to_mismatch(x, first_mismatch_pos, block_start, block_end, mask_id, has_mismatch):
    """Slide window to first mismatch, re-mask from mismatch to block_end.

    Returns:
        new_block_start: int - updated block start position
        new_block_end: int - unchanged block end position
        x_updated: updated tensor with re-masked positions
    """
    if not has_mismatch.any():
        return block_start, block_end, x
    valid_pos = torch.where(has_mismatch, first_mismatch_pos, torch.full_like(first_mismatch_pos, block_end))
    new_block_start = valid_pos.min().item()
    x_updated = x.clone()
    for b in range(x.shape[0]):
        if has_mismatch[b]:
            x_updated[b, first_mismatch_pos[b].item():block_end] = mask_id
    return new_block_start, block_end, x_updated


def detect_converged_tokens(x_prev, x_curr, mask_id, window_start, window_end):
    """Detect how many tokens from the left of the window have converged.

    A token is converged if:
    1. It's not a mask token in x_curr
    2. It's the same in x_prev and x_curr (stable across iterations)

    Returns:
        num_converged: (B,) tensor - number of consecutive converged tokens from left
    """
    B, device = x_prev.shape[0], x_prev.device
    window_prev = x_prev[:, window_start:window_end]
    window_curr = x_curr[:, window_start:window_end]
    window_size = window_end - window_start

    # A position is converged if it's unmasked and unchanged
    is_unmasked = (window_curr != mask_id)
    is_stable = (window_prev == window_curr)
    is_converged = is_unmasked & is_stable

    # Count consecutive converged from left
    num_converged = torch.zeros(B, device=device, dtype=torch.long)
    for b in range(B):
        for i in range(window_size):
            if is_converged[b, i]:
                num_converged[b] += 1
            else:
                break

    return num_converged


def detect_converged_tokens_batched(x_prev, x_curr, mask_id, window_start, window_end):
    """Batched version: Detect consecutive converged tokens from left of window.

    Returns:
        num_converged: (B,) tensor - number of consecutive converged tokens from left
    """
    B, device = x_prev.shape[0], x_prev.device
    window_prev = x_prev[:, window_start:window_end]
    window_curr = x_curr[:, window_start:window_end]
    window_size = window_end - window_start

    # A position is converged if it's unmasked and unchanged
    is_unmasked = (window_curr != mask_id)
    is_stable = (window_prev == window_curr)
    is_converged = is_unmasked & is_stable  # (B, window_size)

    # Find first non-converged position using cumsum trick
    # If all converged up to position i, cumsum at i equals i+1
    converged_cumsum = is_converged.long().cumsum(dim=1)  # (B, window_size)
    positions = torch.arange(1, window_size + 1, device=device).unsqueeze(0)  # (1, window_size)

    # Position i is part of consecutive run if cumsum[i] == i+1
    is_consecutive = (converged_cumsum == positions)  # (B, window_size)

    # Count consecutive converged (sum of True values, but only consecutive from start)
    # Use the last True position in consecutive run
    num_converged = is_consecutive.long().sum(dim=1)  # (B,)

    return num_converged


@torch.no_grad()
def generate_with_sliding_window_jacobi(
    model, prompt, gen_length=128, window_size=32, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None,
    max_steps=None, min_converged=1, seed=None
):
    """
    Sliding Window Jacobi Decoding - a new approach combining Jacobi iteration with sliding window.

    Key features:
    1. Fixed window size - window always has `window_size` positions
    2. Shift by converged tokens - when leftmost tokens stabilize, shift window right
    3. Append new masks - add mask tokens at right edge to maintain window size
    4. Fixed Gumbel noise - deterministic token selection across iterations
    5. Uses previous predictions as initialization for new window positions

    Algorithm:
    - Window slides from left to right over generation positions
    - At each step, run Jacobi iteration within the window
    - When tokens at left edge converge (stable across iterations), commit them
    - Shift window right and append new masks
    - Continue until all positions are generated

    Args:
        model: Mask predictor model.
        prompt: Input tensor of shape (B, L).
        gen_length: Total length of tokens to generate.
        window_size: Size of the sliding window (fixed).
        temperature: Gumbel noise temperature for sampling.
        remasking: Strategy for confidence computation ('low_confidence' or 'random').
        mask_id: Token ID for [MASK].
        threshold: Confidence threshold for unmasking (if None, unmask all per step).
        max_steps: Maximum total steps (default: 4 * gen_length).
        min_converged: Minimum consecutive converged tokens required to shift window.
        seed: Random seed for reproducible Gumbel noise.

    Returns:
        x: Generated sequence tensor of shape (B, L + gen_length)
        nfe: Number of forward evaluations
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Prompt length
    device = model.device

    if max_steps is None:
        max_steps = 4 * gen_length

    # Initialize output: prompt + gen_length mask tokens
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt

    # Get vocab size and generate fixed Gumbel noise for entire generation
    vocab_size = model.config.vocab_size
    gumbel_noise = generate_fixed_gumbel_noise(
        (B, gen_length, vocab_size), device=device, seed=seed
    )

    nfe = 0

    # Window bounds (in absolute positions)
    # Window covers positions [window_start, window_end) in x
    window_start = Lp  # Start at first generation position
    window_end = min(Lp + window_size, Lp + gen_length)  # End of window

    # Track previous state for convergence detection
    x_prev = x.clone()

    # Number of committed (finalized) generation positions
    committed_pos = 0  # Relative to Lp

    step = 0
    while committed_pos < gen_length and step < max_steps:
        step += 1

        # Current window size (may be smaller at the end)
        curr_window_size = window_end - window_start
        if curr_window_size <= 0:
            break

        # Get Gumbel noise slice for current window (relative to generation start)
        noise_start = window_start - Lp
        noise_end = window_end - Lp
        window_gumbel = gumbel_noise[:, noise_start:noise_end, :]  # (B, curr_window_size, vocab_size)

        # Forward pass through model
        # Option 1: Full forward pass (simpler, no KV cache complexity)
        output = model(x)
        logits = output.logits  # (B, Lp + gen_length, vocab_size)
        nfe += 1

        # Extract logits for current window
        window_logits = logits[:, window_start:window_end, :]  # (B, curr_window_size, vocab_size)

        # Select tokens using fixed Gumbel noise
        x0_window = select_tokens_with_fixed_gumbel(window_logits, window_gumbel, temperature)

        # Compute confidence for transfer selection
        if remasking == "low_confidence":
            p = F.softmax(window_logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0_window.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0_window.shape, device=device, dtype=torch.float64)
        else:
            raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

        # Only update masked positions
        window_mask = (x[:, window_start:window_end] == mask_id)  # (B, curr_window_size)

        # Apply threshold-based or full unmasking
        if threshold is not None:
            # Unmask positions above threshold
            transfer_mask = window_mask & (x0_p >= threshold)
            # Force at least one token to be unmasked (highest confidence)
            masked_confidence = torch.where(window_mask, x0_p,
                                           torch.tensor(float('-inf'), device=device, dtype=x0_p.dtype))
            max_conf_idx = masked_confidence.argmax(dim=1, keepdim=True)
            force_unmask = torch.zeros_like(transfer_mask).scatter_(1, max_conf_idx, True)
            transfer_mask = (transfer_mask | force_unmask) & window_mask
        else:
            # Unmask all masked positions
            transfer_mask = window_mask

        # Update window in x
        window_old = x[:, window_start:window_end]
        window_new = torch.where(transfer_mask, x0_window, window_old)
        x = torch.cat([x[:, :window_start], window_new, x[:, window_end:]], dim=1)

        # Detect converged tokens (stable from left edge)
        num_converged = detect_converged_tokens_batched(x_prev, x, mask_id, window_start, window_end)
        min_batch_converged = num_converged.min().item()

        # Shift window if enough tokens converged
        if min_batch_converged >= min_converged:
            # Commit converged tokens (they're now finalized)
            shift_amount = min_batch_converged
            committed_pos += shift_amount

            # Shift window right
            window_start += shift_amount
            window_end = min(window_start + window_size, Lp + gen_length)

            # New positions in window are already masked (from initialization)
            # The previous predictions serve as initialization for overlapping positions

        # Update previous state
        x_prev = x.clone()

        # Check if all generation positions are unmasked
        if (x[:, Lp:] != mask_id).all():
            break

    return x, nfe


@torch.no_grad()
def generate_with_sliding_window_jacobi_cached(
    model, prompt, gen_length=128, window_size=32, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None,
    max_steps=None, min_converged=1, seed=None
):
    """
    Sliding Window Jacobi Decoding with KV-Cache optimization.

    Same algorithm as generate_with_sliding_window_jacobi but uses KV-cache
    for the prefix (prompt + committed tokens) to reduce computation.

    Args:
        model: Mask predictor model with KV-cache support.
        prompt: Input tensor of shape (B, L).
        gen_length: Total length of tokens to generate.
        window_size: Size of the sliding window (fixed).
        temperature: Gumbel noise temperature for sampling.
        remasking: Strategy for confidence computation.
        mask_id: Token ID for [MASK].
        threshold: Confidence threshold for unmasking.
        max_steps: Maximum total steps.
        min_converged: Minimum consecutive converged tokens to shift window.
        seed: Random seed for reproducible Gumbel noise.

    Returns:
        x: Generated sequence tensor of shape (B, L + gen_length)
        nfe: Number of forward evaluations
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    device = model.device

    if max_steps is None:
        max_steps = 4 * gen_length

    # Initialize output
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt

    vocab_size = model.config.vocab_size
    gumbel_noise = generate_fixed_gumbel_noise(
        (B, gen_length, vocab_size), device=device, seed=seed
    )

    nfe = 0

    # Initialize KV-cache with prompt
    output = model(prompt, use_cache=True)
    past_key_values = output.past_key_values
    nfe += 1

    # Window bounds
    window_start = Lp
    window_end = min(Lp + window_size, Lp + gen_length)

    x_prev = x.clone()
    committed_pos = 0
    cache_end = Lp  # Position up to which KV-cache is valid

    step = 0
    while committed_pos < gen_length and step < max_steps:
        step += 1

        curr_window_size = window_end - window_start
        if curr_window_size <= 0:
            break

        # Get Gumbel noise for current window
        noise_start = window_start - Lp
        noise_end = window_end - Lp
        window_gumbel = gumbel_noise[:, noise_start:noise_end, :]

        # Forward pass with cache
        # We need logits for window positions
        # Pass the window tokens and use replace_position to indicate which positions
        window_tokens = x[:, window_start:window_end]

        # Create replace_position mask
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, window_start:window_end] = True

        output = model(
            window_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            replace_position=replace_position
        )
        window_logits = output.logits  # (B, curr_window_size, vocab_size)
        nfe += 1

        # Select tokens using fixed Gumbel noise
        x0_window = select_tokens_with_fixed_gumbel(window_logits, window_gumbel, temperature)

        # Compute confidence
        if remasking == "low_confidence":
            p = F.softmax(window_logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0_window.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0_window.shape, device=device, dtype=torch.float64)
        else:
            raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

        # Only update masked positions
        window_mask = (x[:, window_start:window_end] == mask_id)

        if threshold is not None:
            transfer_mask = window_mask & (x0_p >= threshold)
            masked_confidence = torch.where(window_mask, x0_p,
                                           torch.tensor(float('-inf'), device=device, dtype=x0_p.dtype))
            max_conf_idx = masked_confidence.argmax(dim=1, keepdim=True)
            force_unmask = torch.zeros_like(transfer_mask).scatter_(1, max_conf_idx, True)
            transfer_mask = (transfer_mask | force_unmask) & window_mask
        else:
            transfer_mask = window_mask

        # Update window
        window_old = x[:, window_start:window_end]
        window_new = torch.where(transfer_mask, x0_window, window_old)
        x = torch.cat([x[:, :window_start], window_new, x[:, window_end:]], dim=1)

        # Detect converged tokens
        num_converged = detect_converged_tokens_batched(x_prev, x, mask_id, window_start, window_end)
        min_batch_converged = num_converged.min().item()

        # Shift window if enough tokens converged
        if min_batch_converged >= min_converged:
            shift_amount = min_batch_converged
            committed_pos += shift_amount

            # Update KV-cache: extend cache with committed tokens
            committed_tokens = x[:, cache_end:cache_end + shift_amount]
            output_cache = model(committed_tokens, past_key_values=past_key_values, use_cache=True)
            past_key_values = output_cache.past_key_values
            cache_end += shift_amount

            # Shift window
            window_start += shift_amount
            window_end = min(window_start + window_size, Lp + gen_length)

        x_prev = x.clone()

        if (x[:, Lp:] != mask_id).all():
            break

    return x, nfe


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe


@torch.no_grad()
def generate_with_jacobi(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    max_retries=3, seed=None
):
    """
    Modified Jacobi decoding with fixed Gumbel noise and mismatch-based window sliding.

    Key modifications from generate_with_dual_cache:
    1. Fixed Gumbel noise - generated once and reused throughout decoding
    2. Modified token selection - argmax(logits + temp * gumbel_noise)
    3. Window sliding on mismatch - move window to first mismatch position, max retries per block

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Total sampling steps.
        gen_length: Generated answer length.
        block_length: Block length for semi-autoregressive generation.
        temperature: Gumbel noise temperature.
        remasking: Remasking strategy ('low_confidence' or 'random').
        mask_id: The token id of [MASK].
        threshold: Confidence threshold for token transfer.
        factor: Dynamic factor for transfer (alternative to threshold).
        max_retries: Maximum retries per block on mismatch.
        seed: Random seed for reproducible Gumbel noise.
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    # Get vocab size from model config
    vocab_size = model.config.vocab_size

    # Generate fixed Gumbel noise for the entire generation length
    gumbel_noise = generate_fixed_gumbel_noise(
        (B, gen_length, vocab_size), device=model.device, seed=seed
    )

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length  # block start (absolute position)
        e = s + block_length         # block end (absolute position)

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # Get the noise slice for this block (relative to gen_length)
        noise_start = nb * block_length
        noise_end = (nb + 1) * block_length
        block_gumbel_noise = gumbel_noise[:, noise_start:noise_end, :]  # (B, block_length, vocab_size)

        # Step 0: initial transfer on the full logits
        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False  # Don't touch beyond current block

        # Select tokens using fixed Gumbel noise
        x0 = select_tokens_with_fixed_gumbel(
            out_full.logits[:, s:e, :], block_gumbel_noise, temperature
        )  # (B, block_length)

        # Compute confidence for transfer selection
        if remasking == "low_confidence":
            p = F.softmax(out_full.logits[:, s:e, :].to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
        else:
            raise NotImplementedError(remasking)

        # Only modify masked spots
        block_mask = global_mask_index[:, s:e]
        x0_full = x[:, s:e].clone()
        x0_full = torch.where(block_mask, x0, x0_full)

        neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
        confidence = torch.where(block_mask, x0_p, neg_inf)

        # Determine which tokens to transfer
        if threshold is not None:
            transfer_index = block_mask & (confidence >= threshold)
            max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
            force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
            transfer_index = (transfer_index | force_mask) & block_mask
        else:
            quota0 = num_transfer_tokens[:, 0]
            quota0 = torch.clamp(quota0, min=0).to(dtype=torch.long, device=confidence.device)
            values, idx = torch.sort(confidence, dim=1, descending=True)
            cols = torch.arange(block_length, device=confidence.device).unsqueeze(0).expand(B, -1)
            k_expanded = quota0.unsqueeze(1).expand(B, -1)
            select_sorted = cols < k_expanded
            transfer_int = torch.zeros(B, block_length, device=confidence.device, dtype=torch.int8)
            transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
            transfer_index = transfer_int.bool() & block_mask

        # Update x with transferred tokens
        blk_new = torch.where(transfer_index, x0_full, x[:, s:e])
        x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

        # Track previous state for mismatch detection
        x_prev = x.clone()
        retries = 0

        # 2) Semi-autoregressive refinement with Jacobi iteration
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            # Get logits for current block with cache
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # (B, block_length, vocab_size)

            nfe += 1

            # Select tokens using fixed Gumbel noise
            x0_blk = select_tokens_with_fixed_gumbel(
                logits_blk, block_gumbel_noise, temperature
            )  # (B, block_length)

            # Compute confidence
            mask_blk = (x[:, s:e] == mask_id)
            if remasking == "low_confidence":
                p_blk = F.softmax(logits_blk.to(torch.float64), dim=-1)
                x0_p_blk = torch.gather(p_blk, dim=-1, index=x0_blk.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p_blk = torch.rand(x0_blk.shape, device=x0_blk.device, dtype=torch.float64)

            x0_blk = torch.where(mask_blk, x0_blk, x[:, s:e])
            confidence_blk = torch.where(mask_blk, x0_p_blk, neg_inf)

            # Determine which tokens to transfer
            if threshold is not None:
                transfer_idx_blk = mask_blk & (confidence_blk >= threshold)
                max_conf_indices = torch.argmax(confidence_blk, dim=1, keepdim=True)
                force_mask = torch.zeros_like(transfer_idx_blk).scatter_(1, max_conf_indices, True)
                transfer_idx_blk = (transfer_idx_blk | force_mask) & mask_blk
            else:
                quota_i = num_transfer_tokens[:, i]
                quota_i = torch.clamp(quota_i, min=0).to(dtype=torch.long, device=confidence_blk.device)
                values, idx = torch.sort(confidence_blk, dim=1, descending=True)
                cols = torch.arange(block_length, device=confidence_blk.device).unsqueeze(0).expand(B, -1)
                k_expanded = quota_i.unsqueeze(1).expand(B, -1)
                select_sorted = cols < k_expanded
                transfer_int = torch.zeros(B, block_length, device=confidence_blk.device, dtype=torch.int8)
                transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
                transfer_idx_blk = transfer_int.bool() & mask_blk

            # Update block
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            # Mismatch detection
            has_mismatch, first_mismatch_pos = detect_mismatch_positions(
                x_prev, x, mask_id, s, e
            )

            if has_mismatch.any() and retries < max_retries:
                # Slide window to mismatch position and re-mask
                new_s, _, x = slide_window_to_mismatch(
                    x, first_mismatch_pos, s, e, mask_id, has_mismatch
                )
                # Update replace_position for the new window
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, new_s:e] = True
                retries += 1
            else:
                # No mismatch or max retries reached, continue normally
                x_prev = x.clone()
                retries = 0

    return x, nfe


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
