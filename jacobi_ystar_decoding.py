"""
Jacobi y* Reference Sequence Decoding for Diffusion Language Models.

Standalone module — no modifications to upstream Fast-dLLM, LLaDA, or Dream code.

Implements the core Jacobi iteration where after each forward pass,
the current prediction y^(k) becomes the new reference sequence y*,
and the next iteration compares against y* to detect convergence and
mismatches.

Algorithm:
    1. Initialize y^(0) = [prompt | MASK MASK ... MASK]
    2. Set y* = y^(0)  (initial reference)
    3. For each iteration k:
        a. Run model forward pass: logits = model(y^(k))
        b. Select tokens via argmax(logits + temp * gumbel_noise)  [fixed noise]
        c. Compute confidence scores per token
        d. Unmask high-confidence tokens -> y^(k+1)
        e. Compare y^(k+1) with y* (reference):
            - Converged tokens: unmasked in both y* and y^(k+1), same value
            - Mismatched tokens: unmasked in y* but changed in y^(k+1)
        f. If mismatch found and retries left:
            - Re-mask from first mismatch to end of block
            - Do NOT update y* (keep old reference for retry)
        g. Else (no mismatch or retries exhausted):
            - y* = y^(k+1)   <-- y* is now the new reference sequence
            - Advance to next step
    4. Return final y*

Provides drop-in interfaces for:
    - LLaDA  : generate_with_jacobi()
    - v2     : JacobiYstarV2.jacobi_sample()   (bind via types.MethodType)
    - Dream  : DreamJacobiMixin._sample_jacobi()
"""

import time
import types
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass


# ===================================================================
#  Gumbel noise utilities
# ===================================================================

def generate_fixed_gumbel_noise(shape, device, dtype=torch.float64, seed=None):
    """Generate fixed Gumbel(0,1) noise, reused across all Jacobi iterations."""
    if seed is not None:
        torch.manual_seed(seed)
    uniform = torch.rand(shape, device=device, dtype=dtype)
    uniform = torch.clamp(uniform, min=1e-10, max=1.0 - 1e-10)
    return -torch.log(-torch.log(uniform))


def select_tokens_with_gumbel(logits, gumbel_noise, temperature):
    """Select tokens via argmax(logits + temperature * gumbel_noise)."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    perturbed = logits.to(torch.float64) + temperature * gumbel_noise
    return torch.argmax(perturbed, dim=-1)


# ===================================================================
#  y* mismatch detection (core of the Jacobi y* mechanism)
# ===================================================================

def detect_ystar_mismatch(y_star, y_current, mask_id, block_start, block_end):
    """
    Compare y^(k+1) against y* to find mismatches.

    A MISMATCH is a position that:
        - was already unmasked in y*  (y*[pos] != mask_id)
        - changed value in y_current  (y*[pos] != y_current[pos])

    Returns:
        has_mismatch: (B,) bool
        first_mismatch_pos: (B,) long — absolute position, -1 if none
    """
    B = y_star.shape[0]
    device = y_star.device
    block_size = block_end - block_start

    ref = y_star[:, block_start:block_end]
    cur = y_current[:, block_start:block_end]

    was_unmasked = (ref != mask_id)
    mismatch = was_unmasked & (ref != cur)
    has_mismatch = mismatch.any(dim=1)

    positions = torch.arange(block_size, device=device).unsqueeze(0).expand(B, -1)
    mm_positions = torch.where(mismatch, positions, torch.full_like(positions, block_size))
    first_mm_rel = mm_positions.min(dim=1).values
    first_mismatch_pos = torch.where(
        has_mismatch,
        first_mm_rel + block_start,
        torch.full((B,), -1, device=device, dtype=torch.long),
    )
    return has_mismatch, first_mismatch_pos


def remask_from_position(y, first_mismatch_pos, block_end, mask_id, has_mismatch):
    """Re-mask from first mismatch to block_end for mismatched batch items."""
    y_out = y.clone()
    for b in range(y.shape[0]):
        if has_mismatch[b]:
            y_out[b, first_mismatch_pos[b].item():block_end] = mask_id
    return y_out


def detect_convergence(y_star, y_current, mask_id, window_start, window_end):
    """Count consecutive converged tokens from left of window (stable vs y*)."""
    B = y_star.shape[0]
    device = y_star.device
    window_size = window_end - window_start

    ref = y_star[:, window_start:window_end]
    cur = y_current[:, window_start:window_end]

    is_converged = (cur != mask_id) & (ref == cur)
    cumsum = is_converged.long().cumsum(dim=1)
    pos_1based = torch.arange(1, window_size + 1, device=device).unsqueeze(0)
    return (cumsum == pos_1based).long().sum(dim=1)


# ===================================================================
#  Transfer token schedule (shared by LLaDA functions)
# ===================================================================

def get_num_transfer_tokens(block_mask_index, steps):
    """Compute per-step transfer counts for a block."""
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode='floor')
    rem = total - base * steps
    cols = torch.arange(steps, device=device).unsqueeze(0)
    return base.unsqueeze(1).expand(-1, steps).long() + (cols < rem.unsqueeze(1)).long()


# ===================================================================
#  LLaDA:  generate_with_jacobi()
#
#  Drop-in replacement for the function previously in llada/generate.py.
#  Signature matches what llada/eval_llada.py expects.
# ===================================================================

@torch.no_grad()
def generate_with_jacobi(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None,
    max_retries=3, seed=None,
):
    """
    Jacobi y* decoding for LLaDA models.

    Returns: (x, nfe) — same contract as generate() / generate_with_dual_cache().
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    device = model.device

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt

    vocab_size = model.config.vocab_size
    gumbel_noise = generate_fixed_gumbel_noise(
        (B, gen_length, vocab_size), device=device, seed=seed
    )

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # ===========================================================
        # y* = x.clone()  — set initial reference for this block
        # ===========================================================
        y_star = x.clone()
        retries = 0

        for step_i in range(steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            # --- forward pass ---
            logits = model(x).logits
            nfe += 1

            # --- token selection with fixed Gumbel noise ---
            block_noise = gumbel_noise[:, nb * block_length:(nb + 1) * block_length, :]
            x0_block = select_tokens_with_gumbel(logits[:, s:e, :], block_noise, temperature)

            # --- confidence ---
            mask_blk = (x[:, s:e] == mask_id)
            if remasking == "low_confidence":
                p = F.softmax(logits[:, s:e, :].to(torch.float64), dim=-1)
                x0_p = torch.gather(p, -1, x0_block.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand(x0_block.shape, device=device, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)

            x0_block = torch.where(mask_blk, x0_block, x[:, s:e])
            neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=device, dtype=x0_p.dtype)
            confidence = torch.where(mask_blk, x0_p, neg_inf)

            # --- determine which tokens to unmask ---
            if threshold is not None:
                transfer = mask_blk & (confidence >= threshold)
                max_idx = confidence.argmax(dim=1, keepdim=True)
                force = torch.zeros_like(transfer).scatter_(1, max_idx, True)
                transfer = (transfer | force) & mask_blk
            elif factor is not None:
                # dynamic factor-based transfer (same logic as get_transfer_index_dynamic)
                transfer = torch.zeros_like(mask_blk)
                num_masked = mask_blk.sum(dim=1, keepdim=True)
                for j in range(B):
                    nt = int(num_masked[j].item())
                    if nt == 0:
                        continue
                    ns = list(range(1, nt + 1))
                    threshs = [1 - factor / (n + 1) for n in ns]
                    threshs[0] = -1  # at least one token
                    sc = torch.sort(confidence[j][mask_blk[j]], descending=True)[0]
                    top_i = 0
                    for top_i in range(len(threshs)):
                        if sc[top_i] < threshs[top_i]:
                            break
                    if top_i == 0 or top_i == len(threshs) - 1:
                        top_i += 1
                    _, sel = torch.topk(confidence[j], k=top_i)
                    transfer[j, sel] = True
                transfer = transfer & mask_blk
            else:
                # top-k schedule
                quota = num_transfer[:, step_i].clamp(min=0).long()
                _, idx = torch.sort(confidence, dim=1, descending=True)
                L = confidence.shape[1]
                cols = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
                sel_sorted = cols < quota.unsqueeze(1).expand(B, L)
                t_int = torch.zeros(B, L, device=device, dtype=torch.int8)
                t_int = t_int.scatter(1, idx, sel_sorted.to(torch.int8))
                transfer = t_int.bool() & mask_blk

            # --- apply: y^(k) → y^(k+1) ---
            blk_new = torch.where(transfer, x0_block, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

            # ===========================================================
            # Compare y^(k+1) against y* (the reference sequence)
            # ===========================================================
            has_mm, first_mm = detect_ystar_mismatch(y_star, x, mask_id, s, e)

            if has_mm.any() and retries < max_retries:
                # MISMATCH: re-mask from first mismatch; do NOT update y*
                x = remask_from_position(x, first_mm, e, mask_id, has_mm)
                retries += 1
            else:
                # NO MISMATCH (or retries exhausted)
                # ==================================================
                # y* = y^(k+1)  — y* is now the new reference sequence
                # ==================================================
                y_star = x.clone()
                retries = 0

    return x, nfe


# ===================================================================
#  LLaDA:  Sliding-window Jacobi y* decoding
# ===================================================================

@torch.no_grad()
def generate_with_sliding_window_jacobi(
    model, prompt, gen_length=128, window_size=32, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None,
    max_steps=None, min_converged=1, seed=None,
):
    """
    Sliding-window variant: window shifts right as tokens converge against y*.

    Returns: (x, nfe)
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    device = model.device

    if max_steps is None:
        max_steps = 4 * gen_length

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt

    vocab_size = model.config.vocab_size
    gumbel_noise = generate_fixed_gumbel_noise(
        (B, gen_length, vocab_size), device=device, seed=seed
    )

    nfe = 0
    window_start = Lp
    window_end = min(Lp + window_size, Lp + gen_length)
    committed = 0

    # y* = x — initial reference
    y_star = x.clone()

    step = 0
    while committed < gen_length and step < max_steps:
        step += 1
        cws = window_end - window_start
        if cws <= 0:
            break

        wg = gumbel_noise[:, window_start - Lp:window_end - Lp, :]
        output = model(x)
        logits = output.logits
        nfe += 1

        wl = logits[:, window_start:window_end, :]
        x0_w = select_tokens_with_gumbel(wl, wg, temperature)

        wmask = (x[:, window_start:window_end] == mask_id)
        if remasking == "low_confidence":
            p = F.softmax(wl.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0_w.unsqueeze(-1)).squeeze(-1)
        else:
            x0_p = torch.rand(x0_w.shape, device=device, dtype=torch.float64)

        if threshold is not None:
            conf = torch.where(wmask, x0_p, torch.tensor(float('-inf'), device=device, dtype=x0_p.dtype))
            tr = wmask & (conf >= threshold)
            mi = conf.argmax(dim=1, keepdim=True)
            force = torch.zeros_like(tr).scatter_(1, mi, True)
            tr = (tr | force) & wmask
        else:
            tr = wmask

        w_old = x[:, window_start:window_end]
        w_new = torch.where(tr, x0_w, w_old)
        x = torch.cat([x[:, :window_start], w_new, x[:, window_end:]], dim=1)

        # Compare against y* for convergence
        nc = detect_convergence(y_star, x, mask_id, window_start, window_end)
        min_conv = nc.min().item()

        if min_conv >= min_converged:
            committed += min_conv
            window_start += min_conv
            window_end = min(window_start + window_size, Lp + gen_length)

        # y* = x — y* is now the new reference sequence
        y_star = x.clone()

        if (x[:, Lp:] != mask_id).all():
            break

    return x, nfe


# ===================================================================
#  Fast-dLLM v2:  JacobiYstarV2.jacobi_sample()
#
#  Bind via: model.mdm_sample = types.MethodType(JacobiYstarV2.jacobi_sample, model)
#  Returns finished_samples dict — same contract as batch_sample().
# ===================================================================

class JacobiYstarV2:
    """Container class whose jacobi_sample method can be bound to a v2 model."""

    @torch.no_grad()
    def jacobi_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        max_retries=3,
        seed=None,
    ):
        """
        Jacobi y* decoding for Fast-dLLM v2.
        Same signature and return type as batch_sample().
        """
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        vocab_size = self.config.vocab_size
        if seed is not None:
            torch.manual_seed(seed)
        uniform = torch.rand((batch_size, max_new_tokens, vocab_size),
                             device=self.device, dtype=torch.float64)
        uniform = torch.clamp(uniform, min=1e-10, max=1.0 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(uniform))

        # KV cache warmup
        if min_len > block_size:
            output = self.forward(
                input_ids=input_ids[:, :(min_len // block_size * block_size)],
                use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                next_token = logits[predict_sample_idx, -1:, :].argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        original_input_length = input_ids.shape[1]

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break

            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()

            # ===========================================================
            # y* = x_t.clone()  — initial reference for this block
            # ===========================================================
            y_star = x_t.clone()
            retries = 0
            block_past_key_values = None
            step = 0

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for si in range(x_t.shape[0]):
                        if finished_flag[si] and seq_len[si] < (block_idx + 1) * block_size:
                            stop_idx = (x_t[si, seq_len[si]:] == stop_token).nonzero()[0][0]
                            x_t[si, seq_len[si] + stop_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(
                        input_ids=x_t[:, -block_size:], use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1
                    break

                for sb_idx in range(num_small_blocks):
                    sb_start = sb_idx * small_block_size
                    sb_end = sb_start + small_block_size
                    start = -block_size + sb_start
                    end = None if block_size == sb_end else -block_size + sb_end

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size + sb_start] == mask_id).any():
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:], use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(
                                    input_ids=x_t[:, start:end], use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False, use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=sb_start).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(
                                input_ids=x_t[:, -block_size:], use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        # Token selection with fixed Gumbel noise
                        gen_offset = x_t.shape[1] - original_input_length - block_size
                        noise_slice = gumbel_noise[:, gen_offset + sb_start:gen_offset + sb_end, :]

                        if temperature == 0:
                            x_1 = torch.argmax(logits, dim=-1)
                        else:
                            x_1 = torch.argmax(
                                logits.to(torch.float64) + temperature * noise_slice, dim=-1)

                        # Confidence
                        logits_temp = logits / temperature if temperature > 0 else logits
                        if top_p is not None and top_p < 1:
                            sorted_l, sorted_i = torch.sort(logits_temp, descending=True)
                            cum_p = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                            remove_sorted = cum_p > top_p
                            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                            remove_sorted[..., 0] = 0
                            rm_mask = torch.zeros_like(logits_temp, dtype=torch.bool)
                            rm_mask = rm_mask.scatter_(-1, sorted_i, remove_sorted)
                            logits_temp = logits_temp.masked_fill(
                                rm_mask, torch.finfo(logits_temp.dtype).min)

                        p_1t = torch.softmax(logits_temp, dim=-1)
                        x1_p = torch.gather(p_1t, -1, x_1.unsqueeze(-1)).squeeze(-1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask[torch.arange(x_1.shape[0], device=self.device), max_prob_idx] = True
                        unmask = unmask & mask_idx[:, start:end]

                        x_t[:, start:end][unmask] = x_1[unmask]

                        finished_row = ((x_1 == stop_token) & unmask).any(dim=1)
                        finished_flag = finished_flag | finished_row

                        step += 1

                        # ===================================================
                        # Compare against y* reference
                        # ===================================================
                        bs_abs = x_t.shape[1] - block_size
                        be_abs = x_t.shape[1]
                        has_mm, first_mm = detect_ystar_mismatch(
                            y_star, x_t, mask_id, bs_abs, be_abs)

                        if has_mm.any() and retries < max_retries:
                            # MISMATCH: re-mask; do NOT update y*
                            positions = torch.arange(block_size, device=self.device).unsqueeze(0).expand(batch_size, -1)
                            mm_pos = torch.where(
                                has_mm.unsqueeze(1).expand(-1, block_size)
                                & (y_star[:, bs_abs:be_abs] != mask_id)
                                & (y_star[:, bs_abs:be_abs] != x_t[:, bs_abs:be_abs]),
                                positions, torch.full_like(positions, block_size))
                            first_rel = mm_pos.min(dim=1).values
                            for b in range(x_t.shape[0]):
                                if has_mm[b]:
                                    x_t[b, bs_abs + first_rel[b].item():be_abs] = mask_id
                            retries += 1
                        else:
                            # ===============================================
                            # y* = x_t  — y* is now the new reference
                            # ===============================================
                            y_star = x_t.clone()
                            retries = 0

            # --- update input_ids for next block ---
            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = \
                            x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            if finished_flag.any():
                for si in range(x_t.shape[0]):
                    if finished_flag[si]:
                        finished_samples[sample_indices[si].item()] = x_t[si:si + 1].clone().squeeze(0)
                keep = ~finished_flag
                sample_indices = sample_indices[keep]
                input_ids = input_ids[keep]
                seq_block_idx = seq_block_idx[keep]
                seq_len = seq_len[keep]
                x_t = x_t[keep]
                gumbel_noise = gumbel_noise[keep]
                y_star = y_star[keep]
                for lid in range(len(past_key_values)):
                    past_key_values.key_cache[lid] = past_key_values.key_cache[lid][keep]
                    past_key_values.value_cache[lid] = past_key_values.value_cache[lid][keep]
                finished_flag = finished_flag[keep]

        if len(finished_samples) < batch_size:
            for si in range(x_t.shape[0]):
                finished_samples[sample_indices[si].item()] = x_t[si:si + 1].clone().squeeze(0)

        assert len(finished_samples) == batch_size
        return finished_samples


# ===================================================================
#  Dream:  Jacobi y* sampling (mixin methods)
#
#  Can be added to a Dream model via:
#      model.diffusion_generate_jacobi = types.MethodType(
#          DreamJacobiMixin.diffusion_generate_jacobi, model)
# ===================================================================

# Re-export Dream helpers needed by the mixin
def _dream_sample_tokens(logits, gumbel_noise, temperature=0.0, top_p=None, top_k=None):
    """Sample tokens using fixed Gumbel noise (Dream-style)."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum_p > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask = mask.scatter_(-1, sorted_i, remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)
    x0 = select_tokens_with_gumbel(logits, gumbel_noise, temperature)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    return confidence, x0


class DreamJacobiMixin:
    """Methods to be bound onto a Dream model for Jacobi y* decoding."""

    @torch.no_grad()
    def diffusion_generate_jacobi(self, inputs=None, generation_config=None,
                                  max_retries=3, seed=None, **kwargs):
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        hook_tok = kwargs.pop("generation_tokens_hook_func", lambda s, x, l: x)
        hook_log = kwargs.pop("generation_logits_hook_func", lambda s, x, l: l)

        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        input_ids_length = input_ids.shape[-1]
        has_default = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default,
            input_ids_length=input_ids_length)
        self._validate_generated_length(generation_config, input_ids_length, has_default)

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids, attention_mask=attention_mask)
        threshold = kwargs.get("threshold", 0.9)

        return self._sample_jacobi(
            input_ids, attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=hook_tok,
            generation_logits_hook_func=hook_log,
            threshold=threshold, max_retries=max_retries, seed=seed)

    def _sample_jacobi(self, input_ids, attention_mask, generation_config,
                        generation_tokens_hook_func, generation_logits_hook_func,
                        threshold=0.9, max_retries=3, seed=None):
        output_history = generation_config.output_history
        return_dict = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict and output_history) else None
        start_time = time.time()

        input_ids_length = input_ids.shape[-1]
        gen_length = max_length - input_ids_length
        x = F.pad(input_ids, (0, gen_length), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, gen_length), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1))
        else:
            tok_idx = None
            attention_mask = "full"

        vocab_size = self.config.vocab_size
        gumbel_noise = generate_fixed_gumbel_noise(
            (x.shape[0], gen_length, vocab_size), device=self.device, seed=seed)

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        x = generation_tokens_hook_func(None, x, None)

        # y* = x — initial reference
        y_star = x.clone()
        retries = 0
        i = 0

        while i < steps:
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s_t = timesteps[i + 1]

            gen_mask = mask_index[:, input_ids_length:]
            confidence, x0 = _dream_sample_tokens(
                mask_logits,
                gumbel_noise[:, :gen_mask.shape[1], :][gen_mask],
                temperature=temperature, top_p=top_p, top_k=top_k)

            if threshold is not None:
                x_ = torch.zeros_like(x, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                full_conf = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_conf[mask_index] = confidence
                tr = (full_conf >= threshold) & mask_index
                max_ci = full_conf.argmax(dim=1, keepdim=True)
                force = torch.zeros_like(tr).scatter_(1, max_ci, True)
                tr = (tr | force) & mask_index
                x[tr] = x_[tr].clone()
            else:
                p_transfer = 1 - s_t / t if i < steps - 1 else 1
                x0_full = torch.zeros_like(x[mask_index], dtype=torch.long) + mask_token_id
                tr_idx = torch.rand(*x0_full.shape, device=self.device) < p_transfer
                x0_full[tr_idx] = x0[tr_idx]
                x[mask_index] = x0_full.clone()

            # Compare against y*
            has_mm, first_mm = detect_ystar_mismatch(
                y_star, x, mask_token_id, input_ids_length, max_length)

            if has_mm.any() and retries < max_retries:
                # MISMATCH: re-mask; do NOT update y*
                x = remask_from_position(x, first_mm, max_length, mask_token_id, has_mm)
                retries += 1
            else:
                # y* = x — y* is now the new reference sequence
                y_star = x.clone()
                retries = 0
                i += 1

            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

        print(f'used steps: {steps}')
        print(f'used time: {time.time() - start_time}')

        from dream.model.generation_utils import DreamModelOutput
        if return_dict:
            return DreamModelOutput(sequences=x, history=histories)
        return x


# ===================================================================
#  Helper: bind Jacobi methods onto models
# ===================================================================

def setup_jacobi_for_v2(model):
    """Bind jacobi_sample onto a Fast-dLLM v2 model."""
    model.jacobi_sample = types.MethodType(JacobiYstarV2.jacobi_sample, model)
    return model


def setup_jacobi_for_dream(model):
    """Bind Jacobi y* methods onto a Dream model."""
    model.diffusion_generate_jacobi = types.MethodType(
        DreamJacobiMixin.diffusion_generate_jacobi, model)
    model._sample_jacobi = types.MethodType(
        DreamJacobiMixin._sample_jacobi, model)
    return model


# ===================================================================
#  Self-test
# ===================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Jacobi y* Reference Sequence Decoding")
    print("=" * 70)
    print()
    print("Provided interfaces:")
    print("  LLaDA : generate_with_jacobi(model, prompt, ...)")
    print("  LLaDA : generate_with_sliding_window_jacobi(model, prompt, ...)")
    print("  v2    : JacobiYstarV2.jacobi_sample(self, input_ids, ...)")
    print("  Dream : DreamJacobiMixin.diffusion_generate_jacobi(self, ...)")
    print()
    print("y* mechanism: after each Jacobi iteration,")
    print("  - if mismatch vs y*: re-mask, retry, do NOT update y*")
    print("  - if no mismatch:    y* = y^(k+1)  (new reference)")
    print()

    # --- unit test ---
    print("Running unit tests...")
    mask_id = 126336
    B, L = 2, 10

    y_star = torch.full((B, L), mask_id, dtype=torch.long)
    y_star[0, 3:6] = torch.tensor([10, 20, 30])
    y_star[1, 3:5] = torch.tensor([40, 50])

    y_cur = y_star.clone()
    y_cur[0, 5] = 99   # mismatch (was 30)
    y_cur[0, 6] = 60   # newly unmasked — not a mismatch
    y_cur[1, 5] = 70   # newly unmasked — not a mismatch

    has_mm, first_mm = detect_ystar_mismatch(y_star, y_cur, mask_id, 3, 8)
    assert has_mm[0].item() is True
    assert has_mm[1].item() is False
    assert first_mm[0].item() == 5
    assert first_mm[1].item() == -1

    nc = detect_convergence(y_star, y_cur, mask_id, 3, 8)
    assert nc[0].item() == 2  # positions 3,4 converged
    assert nc[1].item() == 2

    print("All unit tests passed!")
