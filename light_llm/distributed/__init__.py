"""Parallelism state: the TP rank grid and its collectives (minimal build).

Only the tensor-parallel / sequence-parallel primitives the minimal inference
framework needs are re-exported; data-parallel attention (dp_attention) is
omitted. Rank queries and group setup resolve straight from the submodules.
"""

from .parallel_state import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_size,
    init_parallel,
    init_tensor_parallel,
    destroy_parallel,
    destroy_tensor_parallel,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_max,
    tensor_model_parallel_all_reduce_min,
    tensor_model_parallel_broadcast,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_ranks_agree,
    data_parallel_all_reduce,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_group,
    expert_parallel_enabled,
    dp_attention_enabled,
    warmup_collectives,
)
from .sequence_parallel import (
    SequenceParallelPass,
    SequenceParallelRegion,
    sequence_parallel_enabled,
    sequence_parallel_region,
    sp_active,
)

__all__ = [
    "SequenceParallelPass",
    "SequenceParallelRegion",
    "data_parallel_all_reduce",
    "destroy_parallel",
    "destroy_tensor_parallel",
    "divide",
    "dp_attention_enabled",
    "expert_parallel_enabled",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_world_size",
    "init_parallel",
    "init_tensor_parallel",
    "sequence_parallel_enabled",
    "sequence_parallel_region",
    "sp_active",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_reduce_max",
    "tensor_model_parallel_all_reduce_min",
    "tensor_model_parallel_broadcast",
    "tensor_model_parallel_ranks_agree",
    "warmup_collectives",
]
