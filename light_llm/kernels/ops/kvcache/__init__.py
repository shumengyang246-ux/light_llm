"""KV-cache domain: writing new tokens into the paged buffer.

Registers the domain's spec rows and re-exports the two write kernels:
:func:`~light_llm.kernels.ops.kvcache.update_kv_buffer.update_kv_buffer`
(scatter rows) and ``update_kv_index`` (record their locations).

Usage:
    from light_llm.kernels import update_kv_buffer
"""

from light_llm.kernels.dispatcher import NATIVE_BASELINE, PAGED_KV, KernelSpec, register

register(
    KernelSpec(
        name="native/update_kv_buffer",
        op="kv_write",
        backend="native",
        target="light_llm.kernels.ops.kvcache.update_kv_buffer:update_kv_buffer",
        layout=PAGED_KV,
        golden=NATIVE_BASELINE,
    )
)