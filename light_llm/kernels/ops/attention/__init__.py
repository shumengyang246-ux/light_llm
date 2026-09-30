"""Register the bundled attention implementations without optional backends."""

from light_llm.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/flash_attention2_no_pad",
        op="attention.prefill",
        backend="native",
        target="light_llm.kernels.ops.attention.flashattention2_nopad:flash_attention2_no_pad",
        # fp32 has no path through the kernel: it casts inputs to fp16.
        dtypes=("bf16", "fp16"),
        golden=NATIVE_BASELINE,
    )
)

register(
    KernelSpec(
        name="native/flash_attention2_chunked",
        op="attention.chunked_prefill",
        backend="native",
        target="light_llm.kernels.ops.attention.flashattention2_nopad:flash_attention2_chunked",
        # fp8_kv rides the same kernel as the unquantised cache: uint8 rows are
        # e4m3 bytes dequantised on load, so no separate row is warranted.
        dtypes=("bf16", "fp16"),
        schemes=("unquantized", "fp8_kv"),
        golden=NATIVE_BASELINE,
    )
)

register(
    KernelSpec(
        name="native/flash_decoding",
        op="attention.decode",
        backend="native",
        target="light_llm.kernels.ops.attention.flashdecoding:flash_decoding",
        dtypes=("bf16", "fp16"),
        # fp8_kv rides the same kernel: K/V arrive as uint8 rows and are
        # dequantised inside, so no separate row is warranted.
        schemes=("unquantized", "fp8_kv"),
        layout=PAGED_KV,
        golden=NATIVE_BASELINE,
    )
)
