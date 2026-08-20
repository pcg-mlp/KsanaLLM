#!/usr/bin/env python3
"""Generate PyTorch baselines for MLA attention kernel unit tests.

Each test case is stored as a directory of .npy files (one per tensor).
The C++ test loads them via a small NPY reader (no zlib/zip dependency).

Output layout:
  mla_baselines/flash_mla_attention/{q.npy, k.npy, v.npy, output.npy, ...}
  mla_baselines/paged_mla_attention/{query.npy, kv_cache.npy, output.npy, ...}
  mla_baselines/flash_mla_with_lse_causal/{q.npy, ..., output.npy, lse.npy}
  mla_baselines/flash_mla_with_lse_noncausal/{q.npy, ..., output.npy, lse.npy}

Usage:
  python3 generate_mla_baselines.py [--outdir mla_baselines]

Requirements: torch >= 2.0, numpy
"""

import argparse
import os
import math
import numpy as np
import torch

# Use float32 on CPU for reference; the C++ test casts to 16bit-fp before kernel call.
# Seeds are fixed so the random inputs match exactly between Python and C++.
TORCH_SEED = 123

# ─── DeepSeek V2 MLA dimensions ────────────────────────────────────────────
QK_NOPE_DIM = 128
QK_ROPE_DIM = 64
QK_HEAD_DIM = QK_NOPE_DIM + QK_ROPE_DIM  # 192
V_HEAD_DIM = 128
KV_LORA_RANK = 512  # for paged MLA (compressed cache)


def save_npy(path, arr):
    """Save a single array as .npy (float32 or int32/int64 depending on dtype)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)
    print(f"  saved: {os.path.basename(path)}  shape={arr.shape}  dtype={arr.dtype}")


def save_case(outdir, name, **arrays):
    """Save all arrays of one test case into outdir/name/."""
    case_dir = os.path.join(outdir, name)
    os.makedirs(case_dir, exist_ok=True)
    print(f"\n[{name}]")
    for key, arr in arrays.items():
        save_npy(os.path.join(case_dir, f"{key}.npy"), arr)


# ─── MLA Flash Attention Reference ─────────────────────────────────────────
def mla_flash_attention_reference(q, k, v, qlod, kvlod, head_num, kv_head_num,
                                  qk_head_dim, v_head_dim, is_causal, scale):
    """q: [total_q, head_num, qk_head_dim]
       k: [total_kv, kv_head_num, qk_head_dim]
       v: [total_kv, kv_head_num, v_head_dim]
       qlod/kvlod: [batch+1] cumulative offsets
       returns: out [total_q, head_num, v_head_dim], lse [total_q, head_num]
    """
    batch = len(qlod) - 1
    total_q = q.shape[0]
    out = torch.zeros(total_q, head_num, v_head_dim, dtype=torch.float32)
    lse = torch.zeros(total_q, head_num, dtype=torch.float32)

    for b in range(batch):
        q_start, q_end = qlod[b], qlod[b + 1]
        kv_start, kv_end = kvlod[b], kvlod[b + 1]
        if q_end <= q_start:
            continue
        q_b = q[q_start:q_end]
        k_b = k[kv_start:kv_end]
        v_b = v[kv_start:kv_end]
        seq_q = q_b.shape[0]
        seq_kv = k_b.shape[0]

        for h in range(head_num):
            kv_h = h if kv_head_num == head_num else 0
            q_h = q_b[:, h, :]
            k_h = k_b[:, kv_h, :]
            v_h = v_b[:, kv_h, :]

            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

            if is_causal:
                mask = torch.triu(
                    torch.full((seq_q, seq_kv), float("-inf"), dtype=torch.float32),
                    diagonal=1,
                )
                scores = scores + mask

            scores_max = torch.max(scores, dim=-1, keepdim=True).values
            scores_exp = torch.exp(scores - scores_max)
            attn = scores_exp / torch.sum(scores_exp, dim=-1, keepdim=True)

            out[q_start:q_end, h, :] = torch.matmul(attn, v_h)
            lse[q_start:q_end, h] = scores_max.squeeze(-1) + torch.log(
                torch.sum(scores_exp, dim=-1))

    return out, lse


# ─── MLA Paged Attention Reference ──────────────────────────────────────────
def mla_paged_attention_reference(query, kv_cache, block_table, context_len,
                                   num_heads, kv_lora_rank, qk_rope_dim,
                                   scale, block_size, max_blocks_per_seq):
    """query: [batch, num_heads, kv_lora_rank + qk_rope_dim]
       kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_dim]
       returns: out [batch, num_heads, kv_lora_rank]
    """
    batch = query.shape[0]
    out = torch.zeros(batch, num_heads, kv_lora_rank, dtype=torch.float32)

    for b in range(batch):
        ctx = int(context_len[b])
        if ctx == 0:
            continue
        kv_tokens = []
        for slot in range(ctx):
            block_idx = int(block_table[b, slot // block_size])
            pos_in_block = slot % block_size
            kv_tokens.append(kv_cache[block_idx, pos_in_block])
        kv_b = torch.stack(kv_tokens, dim=0)  # [ctx, kv_lora_rank + qk_rope_dim]
        q_b = query[b]  # [num_heads, kv_lora_rank + qk_rope_dim]

        for h in range(num_heads):
            q_h = q_b[h]
            scores = torch.matmul(kv_b, q_h) * scale  # [ctx]
            scores_max = torch.max(scores)
            scores_exp = torch.exp(scores - scores_max)
            attn = scores_exp / torch.sum(scores_exp)
            v_b = kv_b[:, :kv_lora_rank]  # [ctx, kv_lora_rank]
            out[b, h, :] = torch.matmul(attn, v_b)

    return out


# ─── Test Case 1: FlashMlaAttention ────────────────────────────────────────
def gen_flash_mla_attention(outdir):
    torch.manual_seed(TORCH_SEED)
    batch, max_seq_q, max_seq_kv = 2, 4, 4
    head_num, kv_head_num = 4, 4
    qk_head_dim, v_head_dim = QK_HEAD_DIM, V_HEAD_DIM
    total_q, total_kv = batch * max_seq_q, batch * max_seq_kv
    scale = 1.0 / math.sqrt(qk_head_dim)

    # Use deterministic generators per tensor so seeds match across runs
    g1, g2, g3 = (torch.Generator().manual_seed(123),
                  torch.Generator().manual_seed(456),
                  torch.Generator().manual_seed(789))
    q = (torch.randn(total_q, head_num, qk_head_dim, generator=g1) * 0.3).float()
    k = (torch.randn(total_kv, kv_head_num, qk_head_dim, generator=g2) * 0.3).float()
    v = (torch.randn(total_kv, kv_head_num, v_head_dim, generator=g3) * 0.3).float()

    qlod = [i * max_seq_q for i in range(batch + 1)]
    kvlod = [i * max_seq_kv for i in range(batch + 1)]

    out_ref, _ = mla_flash_attention_reference(
        q, k, v, qlod, kvlod, head_num, kv_head_num, qk_head_dim, v_head_dim,
        is_causal=True, scale=scale)

    save_case(outdir, "flash_mla_attention",
        q=q.numpy().astype(np.float32),
        k=k.numpy().astype(np.float32),
        v=v.numpy().astype(np.float32),
        qlod=np.array(qlod, dtype=np.int32),
        kvlod=np.array(kvlod, dtype=np.int32),
        output=out_ref.numpy().astype(np.float32),
        scale=np.array(scale, dtype=np.float32),
        batch=np.array(batch, dtype=np.int64),
        max_seq_q=np.array(max_seq_q, dtype=np.int64),
        max_seq_kv=np.array(max_seq_kv, dtype=np.int64),
        head_num=np.array(head_num, dtype=np.int64),
        kv_head_num=np.array(kv_head_num, dtype=np.int64),
        qk_head_dim=np.array(qk_head_dim, dtype=np.int64),
        v_head_dim=np.array(v_head_dim, dtype=np.int64),
    )


# ─── Test Case 2: PagedMlaAttention ────────────────────────────────────────
def gen_paged_mla_attention(outdir):
    torch.manual_seed(TORCH_SEED)
    batch, q_seq_len, num_heads = 2, 1, 4
    kv_lora_rank, qk_rope_dim = KV_LORA_RANK, QK_ROPE_DIM
    q_head_dim = kv_lora_rank + qk_rope_dim  # 576
    block_size, ctx_len, max_blocks_per_seq = 1, 4, 4
    num_blocks = batch * max_blocks_per_seq  # 8
    scale = 1.0 / math.sqrt(q_head_dim)

    g1, g2 = (torch.Generator().manual_seed(456),
              torch.Generator().manual_seed(789))
    query = (torch.randn(batch * q_seq_len, num_heads, q_head_dim, generator=g1) * 0.2).float()
    kv_cache = (torch.randn(num_blocks, block_size, q_head_dim, generator=g2) * 0.2).float()

    block_table = np.zeros((batch, max_blocks_per_seq), dtype=np.int32)
    for b in range(batch):
        for i in range(max_blocks_per_seq):
            block_table[b, i] = b * max_blocks_per_seq + i
    context_len = np.full(batch, ctx_len, dtype=np.int32)

    out_ref = mla_paged_attention_reference(
        query, kv_cache, torch.from_numpy(block_table), context_len,
        num_heads, kv_lora_rank, qk_rope_dim, scale, block_size, max_blocks_per_seq)

    save_case(outdir, "paged_mla_attention",
        query=query.numpy().astype(np.float32),
        kv_cache=kv_cache.numpy().astype(np.float32),
        block_table=block_table,
        context_len=context_len,
        output=out_ref.numpy().astype(np.float32),
        scale=np.array(scale, dtype=np.float32),
        batch=np.array(batch, dtype=np.int64),
        q_seq_len=np.array(q_seq_len, dtype=np.int64),
        num_heads=np.array(num_heads, dtype=np.int64),
        kv_lora_rank=np.array(kv_lora_rank, dtype=np.int64),
        qk_rope_dim=np.array(qk_rope_dim, dtype=np.int64),
        block_size=np.array(block_size, dtype=np.int64),
        ctx_len=np.array(ctx_len, dtype=np.int64),
        max_blocks_per_seq=np.array(max_blocks_per_seq, dtype=np.int64),
    )


# ─── Test Case 3: FlashMlaAttentionWithLse (causal=true) ───────────────────
def gen_flash_mla_with_lse_causal(outdir):
    torch.manual_seed(TORCH_SEED)
    batch, max_seq_q, max_seq_kv = 2, 4, 4
    head_num, kv_head_num = 4, 4
    qk_head_dim, v_head_dim = QK_HEAD_DIM, V_HEAD_DIM
    total_q, total_kv = batch * max_seq_q, batch * max_seq_kv
    scale = 1.0 / math.sqrt(qk_head_dim)

    g1, g2, g3 = (torch.Generator().manual_seed(789),
                  torch.Generator().manual_seed(321),
                  torch.Generator().manual_seed(654))
    q = (torch.randn(total_q, head_num, qk_head_dim, generator=g1) * 0.3).float()
    k = (torch.randn(total_kv, kv_head_num, qk_head_dim, generator=g2) * 0.3).float()
    v = (torch.randn(total_kv, kv_head_num, v_head_dim, generator=g3) * 0.3).float()

    qlod = [i * max_seq_q for i in range(batch + 1)]
    kvlod = qlod  # self-attention

    out_ref, lse_ref = mla_flash_attention_reference(
        q, k, v, qlod, kvlod, head_num, kv_head_num, qk_head_dim, v_head_dim,
        is_causal=True, scale=scale)

    save_case(outdir, "flash_mla_with_lse_causal",
        q=q.numpy().astype(np.float32),
        k=k.numpy().astype(np.float32),
        v=v.numpy().astype(np.float32),
        qlod=np.array(qlod, dtype=np.int32),
        kvlod=np.array(kvlod, dtype=np.int32),
        output=out_ref.numpy().astype(np.float32),
        lse=lse_ref.numpy().astype(np.float32),
        scale=np.array(scale, dtype=np.float32),
        batch=np.array(batch, dtype=np.int64),
        max_seq_q=np.array(max_seq_q, dtype=np.int64),
        max_seq_kv=np.array(max_seq_kv, dtype=np.int64),
        head_num=np.array(head_num, dtype=np.int64),
        kv_head_num=np.array(kv_head_num, dtype=np.int64),
        qk_head_dim=np.array(qk_head_dim, dtype=np.int64),
        v_head_dim=np.array(v_head_dim, dtype=np.int64),
    )


# ─── Test Case 4: FlashMlaAttentionWithLse (causal=false) ──────────────────
def gen_flash_mla_with_lse_noncausal(outdir):
    torch.manual_seed(TORCH_SEED)
    batch, max_seq_q, max_seq_kv = 2, 4, 6
    head_num, kv_head_num = 4, 4
    qk_head_dim, v_head_dim = QK_HEAD_DIM, V_HEAD_DIM
    total_q, total_kv = batch * max_seq_q, batch * max_seq_kv
    scale = 1.0 / math.sqrt(qk_head_dim)

    g1, g2, g3 = (torch.Generator().manual_seed(111),
                  torch.Generator().manual_seed(222),
                  torch.Generator().manual_seed(333))
    q = (torch.randn(total_q, head_num, qk_head_dim, generator=g1) * 0.3).float()
    k = (torch.randn(total_kv, kv_head_num, qk_head_dim, generator=g2) * 0.3).float()
    v = (torch.randn(total_kv, kv_head_num, v_head_dim, generator=g3) * 0.3).float()

    qlod = [i * max_seq_q for i in range(batch + 1)]
    kvlod = [i * max_seq_kv for i in range(batch + 1)]

    out_ref, lse_ref = mla_flash_attention_reference(
        q, k, v, qlod, kvlod, head_num, kv_head_num, qk_head_dim, v_head_dim,
        is_causal=False, scale=scale)

    save_case(outdir, "flash_mla_with_lse_noncausal",
        q=q.numpy().astype(np.float32),
        k=k.numpy().astype(np.float32),
        v=v.numpy().astype(np.float32),
        qlod=np.array(qlod, dtype=np.int32),
        kvlod=np.array(kvlod, dtype=np.int32),
        output=out_ref.numpy().astype(np.float32),
        lse=lse_ref.numpy().astype(np.float32),
        scale=np.array(scale, dtype=np.float32),
        batch=np.array(batch, dtype=np.int64),
        max_seq_q=np.array(max_seq_q, dtype=np.int64),
        max_seq_kv=np.array(max_seq_kv, dtype=np.int64),
        head_num=np.array(head_num, dtype=np.int64),
        kv_head_num=np.array(kv_head_num, dtype=np.int64),
        qk_head_dim=np.array(qk_head_dim, dtype=np.int64),
        v_head_dim=np.array(v_head_dim, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate MLA attention baselines for Kunlun unit tests")
    parser.add_argument("--outdir", default="mla_baselines",
                        help="output directory for .npy baseline files")
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    print(f"Output directory: {outdir}")

    gen_flash_mla_attention(outdir)
    gen_paged_mla_attention(outdir)
    gen_flash_mla_with_lse_causal(outdir)
    gen_flash_mla_with_lse_noncausal(outdir)

    print("\nAll baselines generated successfully.")
    print("Each test case directory contains .npy files for input tensors and the")
    print("expected output. By default the C++ test regenerates baselines into")
    print("<source_dir>/mla_baselines/ (next to the test .cpp), independent of cwd.")


if __name__ == "__main__":
    main()
