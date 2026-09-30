"""Numerical regression checks for the dense-model Triton path."""
import pytest
import torch
import torch.nn.functional as F

from light_llm.kernels.ops.attention.flashattention2_nopad import flash_attention2_no_pad
from light_llm.kernels.ops.attention.flashdecoding import flash_decoding, adaptive_partition_size
from light_llm.kernels.ops.kvcache.update_kv_buffer import update_kv_buffer
from light_llm.kernels.ops.layernorm.skip_rmsnorm import skip_rmsnorm, fused_add_rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_residual_rmsnorm_matches_model_dtype_rounding(dtype):
    torch.manual_seed(21)
    x = torch.randn(7, 1536, dtype=dtype, device='cuda')
    residual = torch.randn_like(x)
    weight = torch.randn(1536, dtype=dtype, device='cuda')
    added = x + residual
    ref = (added.float() * torch.rsqrt(added.float().square().mean(-1, keepdim=True) + 1e-6)).to(dtype) * weight
    for fn in (skip_rmsnorm, fused_add_rmsnorm):
        actual, actual_residual = fn(x, residual.clone(), weight, 1e-6)
        torch.testing.assert_close(actual_residual, added, rtol=0, atol=0)
        torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.01)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_ragged_prefill_and_permuted_kv_decode_match_sdpa(dtype):
    torch.manual_seed(42)
    lengths = [7, 37]
    total = sum(lengths)
    q = torch.randn(total, 12, 128, device='cuda', dtype=dtype)
    k = torch.randn(total, 2, 128, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    starts = torch.tensor([0, lengths[0]], device='cuda', dtype=torch.int32)
    lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    actual = flash_attention2_no_pad(q, k, v, 128**-0.5, starts, lens, max(lengths))
    refs = []
    offset = 0
    for length in lengths:
        qs = q[offset:offset+length].transpose(0, 1)[None]
        ks = k[offset:offset+length].repeat_interleave(6, dim=1).transpose(0, 1)[None]
        vs = v[offset:offset+length].repeat_interleave(6, dim=1).transpose(0, 1)[None]
        refs.append(F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)[0].transpose(0, 1))
        offset += length
    torch.testing.assert_close(actual, torch.cat(refs), rtol=0.02, atol=0.02)

    # Scatter into noncontiguous slots, and make request ids differ from batch rows.
    slots = torch.randperm(total, device='cuda').to(torch.int32)
    cache = torch.zeros(total, 4, 128, device='cuda', dtype=dtype)
    update_kv_buffer(k, v, slots, cache)
    torch.testing.assert_close(cache[slots.long(), :2], k, rtol=0, atol=0)
    torch.testing.assert_close(cache[slots.long(), 2:], v, rtol=0, atol=0)
    table = torch.zeros(4, max(lengths), device='cuda', dtype=torch.int32)
    requests = torch.tensor([3, 1], device='cuda', dtype=torch.int32)
    table[3, :lengths[0]] = slots[:lengths[0]]
    table[1, :lengths[1]] = slots[lengths[0]:]
    last_queries = q[torch.tensor([lengths[0]-1, total-1], device='cuda')]
    decoded = flash_decoding(last_queries, cache[:, :2], cache[:, 2:], 128**-0.5,
                             table, requests, lens, max(lengths))
    torch.testing.assert_close(decoded, torch.stack([r[-1] for r in refs]), rtol=0.02, atol=0.02)
    assert adaptive_partition_size(2, 2, max(lengths), 108) % 16 == 0
