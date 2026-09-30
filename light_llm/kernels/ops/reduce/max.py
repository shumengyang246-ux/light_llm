"""Max as a Triton kernel. 这里的最大值算子采用了轴向规约的方式

Usage:
    out = max(inp)
"""

import torch
import triton
import triton.language as tl
import logging
import math

@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 用于记录每一行的最大值
    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=-float("inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int32, value=-1)

    for start_n in range(0, N, BLOCK_N):
        # 确定处N维上的偏移量
        n_offset = start_n + tl.arange(0, BLOCK_N)
        # 确定整个偏移量
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # 确定当前掩码
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        # 对m行，第start_n个分块的所有数据进行加载
        inp_ptrs = inp + offset
        # 加载当前块内元素
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        local_max, local_argmax = tl.max(inp_vals, axis=1, return_index=True)

        # 和全局的最大值进行比较，如果当前的最小值小于全局的最小值，则更新全局的最小值
        update = local_max < max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    # 确定输出张量的偏移量
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    # 将结果存储到输出张量中
    tl.store(out_value_ptrs, max_values, mask=mask1)
    tl.store(out_index_ptrs, argmax_values, mask=mask1)

def max_triton(input_tensor, dim=1):
    # 确保输入是2D张量
    assert input_tensor.dim() == 2, "输入张量必须是2D的"
    # 确保在dim=1上求最大值
    assert dim == 1, "当前只支持在dim=1上求最大值"

    M, N = input_tensor.shape
    K = 1  # 因为我们只处理2D张量

    # 分配输出张量
    max_values = torch.empty((M, K), dtype=torch.float32, device=input_tensor.device)
    max_indices = torch.empty((M, K), dtype=torch.int64, device=input_tensor.device)

    # 定义block大小
    BLOCK_M = 128
    BLOCK_N = 128

    # 计算grid大小
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        K,
    )

    # 调用kernel
    max_kernel[grid](
        input_tensor,
        max_values,
        max_indices,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return max_values.squeeze(1), max_indices.squeeze(1)

if __name__ == '__main__':
    input_tensor = torch.randn(100, 200, device='cuda')
    max_values, max_indices = max_triton(input_tensor, dim=1)
    print('-')