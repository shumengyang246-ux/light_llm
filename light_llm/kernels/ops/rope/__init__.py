"""Register the bundled rope implementations without optional backends."""

from light_llm.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/rope_emb_forward",
        op="rope",
        backend="native",
        target="light_llm.kernels.ops.rope.rope_emb:rope_emb_forward",
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)
