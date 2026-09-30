"""Minimal quantization facade for the unquantised (default) path.

Only the unquantised scheme is wired up; advanced runtime schemes (fp8/awq/
gptq/...) raise on request because the minimal build does not carry their
method classes.
"""
from __future__ import annotations

from typing import Any

from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    run_quant_linear,
)
from .unquant import (
    UnquantizedConfig,
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from .parameter import RawParameter
from .utils import adapt_packed_checkpoint


#: Runtime quantisation schemes accepted by ``--quantization`` (minimal build:
#: only the unquantised path is wired).
RUNTIME_SCHEMES: dict[str, type[QuantizationConfig]] = {
    "unquantized": UnquantizedConfig,
}

#: Checkpoint suffix of the per-block dequantisation scale (used by the loader).
SCALE_SUFFIX = "weight_scale_inv"

#: FP8 block size used when repeating per-block scales during load.
FP8_BLOCK = 128


def get_quant_config_from_hf(hf_config) -> "QuantizationConfig | None":
    """Return None for every checkpoint in the minimal (unquantised) build."""
    raw = getattr(hf_config, "quantization_config", None)
    if not raw:
        return None
    # A quantised checkpoint would need its method classes, which the minimal
    # build omits; surface a clear error instead of silently mishandling it.
    raise ValueError(
        "minimal build only supports unquantised checkpoints; "
        f"found quantization_config={raw}"
    )


def for_runtime_scheme(name: str) -> "QuantizationConfig":
    """Build a QuantizationConfig for ``--quantization <name>``."""
    cls = RUNTIME_SCHEMES.get((name or "").lower())
    if cls is None:
        raise ValueError(
            f"unknown runtime quantisation {name!r}; "
            f"supported in the minimal build: {sorted(RUNTIME_SCHEMES)}"
        )
    return cls()


__all__ = [
    "adapt_packed_checkpoint",
    "QuantizationConfig",
    "QuantizeMethodBase",
    "LinearMethodBase",
    "FusedMoEMethodBase",
    "run_quant_linear",
    "UnquantizedConfig",
    "UnquantizedLinearMethod",
    "UnquantizedFusedMoEMethod",
    "RawParameter",
    "RUNTIME_SCHEMES",
    "for_runtime_scheme",
    "get_quant_config_from_hf",
    "SCALE_SUFFIX",
    "FP8_BLOCK",
]
