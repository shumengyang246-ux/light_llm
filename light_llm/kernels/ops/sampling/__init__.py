"""Sampling domain: turning logits into token ids.

Registers the domain's spec rows; the stochastic draw itself is served
by the native sampler or an external backend's sampling wrapper.

Usage:
    from light_llm.kernels.ops import SampleOp
"""

from light_llm.kernels.backend.flashinfer import CUDA_SM75
from light_llm.kernels.dispatcher import (
    UNMEASURED,
    GoldenRecord,
    KernelSpec,
    register,
)

register(
    KernelSpec(
        name="flashinfer/sample",
        op="sample",
        backend="flashinfer",
        target="light_llm.kernels.backend.flashinfer.sample:sample",
        available="light_llm.kernels.backend.flashinfer:available",
        capability=CUDA_SM75,
        dtypes=("bf16", "fp16", "fp32"),
        # Sampling is compared on the greedy path (argmax parity), where the
        # two implementations must agree exactly.
        golden=GoldenRecord(verified=True, max_abs_diff=0.0, baseline="greedy argmax parity"),
        priority=UNMEASURED,
    )
)