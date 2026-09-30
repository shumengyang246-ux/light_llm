"""Register the bundled gemm implementations without optional backends."""

from light_llm.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/linear_torch",
        op="linear",
        backend="native",
        target="light_llm.kernels.ops.gemm.linear:linear_torch",
        schemes=("unquantized",),
        golden=NATIVE_BASELINE,
    )
)
