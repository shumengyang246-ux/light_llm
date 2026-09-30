"""Rotary position embedding (RoPE) applied to q and k in a fused Triton kernel.

One launch rotates every (head, position) pair in place using the cos/sin
tables; head padding keeps the block mask aligned for any head count.

Usage:
    rope_emb_forward(q, k, cos, sin)
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope_emb(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_b_stride,
    cos_s_stride,
    sin,
    sin_b_stride,
    sin_s_stride,
    sl,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // sl # 句子长度
    cos_row_idx = pid % sl

    # 定位q，k行起点
    q_ptr += pid * q_row_stride # 实际上是token维度的步长
    k_ptr += pid * k_row_stride

    # 定位到cos，sin对应的batch id的cos_row_idx行
    cos_ptr = cos + batch_id * cos_b_stride + cos_row_idx * cos_s_stride
    sin_ptr = sin + batch_id * sin_b_stride + cos_row_idx * sin_s_stride

    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=0.0)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0.0)

    # 计算 head 和 dim的偏移
    # 沿 head 维遍历全部 head（pad 到 2 的幂），沿 dim 维只取前半，用于 rotate-half
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]

    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
        tl.arange(0, pad_hd // 2)[None, :] < hd // 2
    )

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def _rows_are_contiguous(tensor) -> bool:
    """Whether ``head * head_dim + channel`` addresses a token's row correctly.

    The kernel walks tokens with ``stride(0)`` and heads with ``head_dim``, so anything
    whose heads are adjacent within a row qualifies — including a slice of a fused
    ``[q | k | v]`` output, whose row stride is simply wider than its row.
    """
    return tensor.stride(2) == 1 and tensor.stride(1) == tensor.shape[2]


def rope_emb_forward(q, k, cos, sin):
    """Rotate ``q`` and ``k`` by the positions encoded in ``cos``/``sin``.

    Args:
        q: ``(batch_size * seq_len, n_q_heads, head_dim)`` queries.
        k: ``(batch_size * seq_len, n_k_heads, head_dim)`` keys.
        cos: ``(batch_size, seq_len, head_dim)`` rotation table. The batch and
            sequence geometry the kernel indexes with comes from here rather
            than from separate arguments: passing it twice only creates a way
            for the two to disagree, and the kernel would read ``cos`` with the
            caller's numbers either way.
        sin: Same shape as ``cos``.

    Returns:
        ``(q, k)`` rotated in place; a tensor whose heads are not adjacent
        within a row is materialised first, in which case the copy is what
        comes back.
    """
    N, n_qh, HEAD_DIM = q.shape
    _, n_kh, _ = k.shape
    batch_size, seq_len = cos.shape[0], cos.shape[1]
    if batch_size * seq_len != N:
        raise ValueError(f"cos/sin describe {batch_size}x{seq_len} positions but q has {N} tokens")

    pad_hd = triton.next_power_of_2(HEAD_DIM)
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    BLOCK_SIZE = max(pad_n_qh, pad_n_kh)

    if HEAD_DIM >= 128:
        num_warps = 8
    else:
        num_warps = 4

    q = q if _rows_are_contiguous(q) else q.contiguous()
    k = k if _rows_are_contiguous(k) else k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_rope_emb[(N,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        seq_len,
        n_qh,
        n_kh,
        HEAD_DIM,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, k

#=======torch的实现===============
def _rotate_half_torch(x, cos, sin):
    """Llama / GPT-NeoX 风格 rotate-half，与 kernel 公式一致。

    x: ``[N, n_heads, head_dim]``
    cos/sin: ``[B, S, head_dim]``，kernel 只使用前 ``head_dim // 2`` 个通道。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_h = cos[..., :half].reshape(-1, 1, half).to(dtype=x.dtype)
    sin_h = sin[..., :half].reshape(-1, 1, half).to(dtype=x.dtype)
    return torch.cat([x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h], dim=-1)


def apply_rope_torch(q, k, cos, sin):
    return _rotate_half_torch(q, cos, sin), _rotate_half_torch(k, cos, sin)


def _make_cos_sin(batch, seq_len, head_dim, device, dtype, theta=10000.0):
    """按标准 RoPE 频率生成 cos/sin 表，后半维复制前半（HF Llama 布局）。"""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [S, half]
    cos_h = freqs.cos()[None, :, :].expand(batch, seq_len, half)
    sin_h = freqs.sin()[None, :, :].expand(batch, seq_len, half)
    cos = torch.cat([cos_h, cos_h], dim=-1).to(dtype)
    sin = torch.cat([sin_h, sin_h], dim=-1).to(dtype)
    return cos, sin


def _report(name, triton_t, torch_t, atol, rtol):
    diff = (triton_t.float() - torch_t.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = (not torch.isnan(triton_t).any()) and torch.allclose(
        triton_t.float(), torch_t.float(), atol=atol, rtol=rtol
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {name}: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}"
        f"  shape={tuple(triton_t.shape)} dtype={triton_t.dtype}"
    )
    if not ok:
        flat = diff.view(-1)
        idx = int(flat.argmax().item())
        print(
            f"        worst triton={triton_t.view(-1)[idx].item():.6f}"
            f"  torch={torch_t.view(-1)[idx].item():.6f}"
        )
    return ok, max_diff


def _run_case(name, batch, seq_len, n_qh, n_kh, head_dim, dtype=torch.float32, fused_qkv=False):
    device = torch.device("cuda")
    N = batch * seq_len
    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = atol
    print(
        f"\n== {name} ==  B={batch} S={seq_len} n_qh={n_qh} n_kh={n_kh} "
        f"hd={head_dim} dtype={dtype} fused_qkv={fused_qkv}"
    )

    cos, sin = _make_cos_sin(batch, seq_len, head_dim, device, dtype)
    if fused_qkv:
        n_vh = n_kh
        qkv = torch.randn(N, n_qh + n_kh + n_vh, head_dim, device=device, dtype=dtype)
        q = qkv[:, :n_qh]
        k = qkv[:, n_qh : n_qh + n_kh]
    else:
        q = torch.randn(N, n_qh, head_dim, device=device, dtype=dtype)
        k = torch.randn(N, n_kh, head_dim, device=device, dtype=dtype)

    q_ref = q.clone()
    k_ref = k.clone()
    torch_q, torch_k = apply_rope_torch(q_ref, k_ref, cos, sin)

    q_in, k_in = q.clone(), k.clone()
    out_q, out_k = rope_emb_forward(q_in, k_in, cos, sin)

    # pos=0 的 cos=1/sin=0，旋转是恒等；取一个靠后的 token 才能看出数值变化
    pos0 = 0
    pos_rot = min(seq_len - 1, max(seq_len // 2, 1))
    for pos, tag in ((pos0, "pos=0 (应近似不变)"), (pos_rot, f"pos={pos_rot} (应旋转)")):
        print(f"  sample q[{pos},0,:4] {tag}")
        print("    before ", q[pos, 0, :4].float().cpu().tolist())
        print("    triton ", out_q[pos, 0, :4].float().cpu().tolist())
        print("    torch  ", torch_q[pos, 0, :4].float().cpu().tolist())

    ok_q, _ = _report("q", out_q, torch_q, atol, rtol)
    ok_k, _ = _report("k", out_k, torch_k, atol, rtol)
    # 就地写入：返回的应是同一块存储（非连续 fused 切片时也是原 storage）
    inplace_q = out_q.data_ptr() == q_in.data_ptr()
    inplace_k = out_k.data_ptr() == k_in.data_ptr()
    print(f"  inplace q={inplace_q} k={inplace_k}")
    if not (ok_q and ok_k):
        raise AssertionError(f"RoPE 正确性失败: {name}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("RoPE Triton kernel 需要 CUDA")

    torch.manual_seed(0)
    print("RoPE 正确性对照 (Triton vs PyTorch rotate-half)")

    _run_case("mha fp32", batch=2, seq_len=16, n_qh=8, n_kh=8, head_dim=64)
    _run_case("gqa fp32", batch=2, seq_len=32, n_qh=32, n_kh=8, head_dim=128)
    _run_case("pad heads (非 2 的幂)", batch=1, seq_len=8, n_qh=6, n_kh=3, head_dim=64)
    _run_case("fp16", batch=2, seq_len=64, n_qh=16, n_kh=16, head_dim=64, dtype=torch.float16)
    _run_case(
        "fused qkv slice",
        batch=2,
        seq_len=8,
        n_qh=8,
        n_kh=2,
        head_dim=64,
        fused_qkv=True,
    )

    print("\n全部用例通过。")


if __name__ == "__main__":
    main()
