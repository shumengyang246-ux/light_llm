"""Min as a Triton kernel. 这里的最小值算子采用了分级规约的方式

Usage:
    out = min(inp)
"""

import torch
import triton
import triton.language as tl
import logging
import math 

@triton.jit
def min_kernel_1(
        inp,
        mid,
        M,
        BLOCK_SIZE: tl.constexpr,
):
    # BLOCK_SIZE是一个块的大小，根据画板中的例子，BLOCK_SIZE
    pid = tl.program_id(0)
    # 确定当前一个块中所有元素的偏移量。确定当前块处理数据的范围
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     
    inp_ptrs = inp + offset
    mask = offset < M
    # 加载当前块中的所有元素
    inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    # 计算当前块中的最小值
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    # 确定中间结果中所有数据的偏移量
    mid_ptrs = mid + offset
    mask = offset < mid_size
    # 对中间结果的所有数据进行加载
    mid_val = tl.load(mid_ptrs, mask=mask, other=float("inf"))
    # 计算中间结果中的最小值
    min_val = tl.min(mid_val)
    # 将最小值存储到输出张量中
    tl.store(out, min_val)

def min(inp):
    logging.debug("GEMS MIN")
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    min_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out