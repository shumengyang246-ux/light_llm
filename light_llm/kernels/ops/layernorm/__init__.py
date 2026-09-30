"""Register the bundled layernorm implementations without optional backends."""

from light_llm.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/skip_rmsnorm",
        op="rmsnorm",
        backend="native",
        target="light_llm.kernels.ops.layernorm.skip_rmsnorm:skip_rmsnorm",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
