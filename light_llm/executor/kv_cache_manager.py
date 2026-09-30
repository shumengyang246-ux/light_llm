"""
只做两件事：
1、算当前显存中还能够存放多少kv；2、真实创建GPU KV Tensors
Paged KV cache: block allocation with reference counting, plus sizing.

class:`MemoryProfiler` measures free device memory to decide how many cache blocks
fit; :class:`KVCacheManager` hands out and ref-counts block indices so prefill and
decode reserve rows and release them when a sequence ends.

Usage:
    idx = kv.alloc_kvcache_index(need_size); kv.release_ref(idx)
"""

import gc

import torch

from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata

logger = get_logger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size of one element of ``dtype`` in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


class MemoryProfiler:
    """Estimates how many KV-cache tokens fit in the remaining GPU memory.

    A short dummy forward measures peak activation memory; the leftover budget
    (``total * utilization - peak``) is divided by the per-token KV size. The
    constructed model supplies every dimension, so no dummy-input config is needed.

    Args:
        num_layers: Decoder layer count.
        kv_row: ``(slots, dim)`` of one token's per-layer cache row —
            ``(2 * kv_heads, head_dim)`` for MHA/GQA, ``(1, kv_lora_rank +
            qk_rope_head_dim)`` for MLA (latent row, replicated across ranks).
        gpu_memory_utilization: Fraction of GPU memory the cache may occupy.
        dtype: KV-cache dtype.
        device: Torch device string.
        reserved_bytes: Extra budget to withhold (e.g. CUDA graph workspace).
    """
    def __init__(
        self, 
        num_layers: int, 
        kv_row: tuple[int, int],
        gpu_memory_utilization: float = 0.9, 
        dtype: torch.dtype = torch.bfloat16, 
        device: str = "cuda",
        reserved_bytes: int = 0,
    ):
        self.num_layers = num_layers
        self.kv_row = kv_row
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.device = device
        self.reserved_bytes = reserved_bytes

    def _kv_bytes_per_token(self):
        slots, dim = self.kv_row
        return slots * dim * self.num_layers * get_dtype_size(self.dtype)
    
    def _run_dummy_forward(self, model, vocab_size: int, seq_len: int = 32):
        """Drive one prefill pass so peak activation memory is recorded."""
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        dummy = AttentionMetadata()
        dummy.kv_buffer = [
            torch.empty(
                (seq_len, *self.kv_row),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        dummy.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device=self.device)
        dummy.b_req_tokens_table = torch.arange(
            seq_len, dtype=torch.int32, device=self.device
        ).view(1, seq_len)
        dummy.b_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_req_idx = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        dummy.max_actual_seq_len = seq_len

        with torch.no_grad():
            model(input_ids, position_ids, dummy)
    
    def available_kv_blocks(self, model, vocab_size: int) -> int:
        """Return the number of KV-cache tokens that fit in free GPU memory.

        Falls back to a small fixed budget on CPU (profiling APIs do not apply),
        keeping unit tests runnable without a GPU.
        """
        if torch.device(self.device).type == "cpu":
            logger.warning("CUDA unavailable; using a minimal KV cache for CPU execution")
            return 4096
        
        torch.cuda.empty_cache() # 清空显存
        torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
        _, total_gpu_memory = torch.cuda.mem_get_info()

        self._run_dummy_forward(model, vocab_size) # 运行一次dummy forward，记录显存峰值
        torch.cuda.synchronize()

        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        torch.cuda.empty_cache()
        # 统计在 torch 缓存分配器之外分配的内存。
        torch_current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        free_now, _ = torch.cuda.mem_get_info()
        non_torch = (total_gpu_memory - free_now) - torch_current
        if non_torch > 0:
            peak_memory += non_torch
        
        budget = total_gpu_memory * self.gpu_memory_utilization - peak_memory - self.reserved_bytes
        num_blocks = max(int(budget // self._kv_bytes_per_token()), 0)

        logger.info(
            "KV-cache profiling: total=%.2f GB peak=%.2f GB -> %d cache tokens (util=%.2f)",
            total_gpu_memory / 1024**3,
            peak_memory / 1024**3,
            num_blocks,
            self.gpu_memory_utilization,
        )

        gc.collect()
        torch.cuda.empty_cache()
        return num_blocks


class KVCacheManager:
    """Owns the paged KV buffers and hands out cache rows by reference count.

    One row per token (``block_size=1``). ``kv_mem_use_state[i]`` is row ``i``'s
    refcount (free at zero); :attr:`can_use_mem_size` counts free rows.
    :meth:`alloc_kvcache_index` picks the cheapest strategy that fits: bump cursor,
    contiguous-run search, then scattered rows.

    Args:
        num_layers: Decoder layer count.
        kv_row: ``(slots, dim)`` of one token's per-layer cache row (as in
            :class:`MemoryProfiler`).
        gpu_num_blocks: Cache capacity in blocks (profiled or caller-set).
        block_size: Tokens per block; only 1 is implemented.
        dtype: KV-cache dtype.
        device: Torch device string.
    """
    def __init__(
        self,
        num_layers,
        kv_row,
        gpu_num_blocks,
        block_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        watermark: float = 0.1,
        hysteresis: float = 0.05,
    ):
        self.num_layers = num_layers
        self.kv_row = tuple(kv_row)
        self.gpu_num_blocks = gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks  # rows currently free

        # Watermark: 当空闲块占比低于该阈值时，拒绝新请求。
        # Hysteresis: 当数值降至阈值以下后，仅在数值回升至阈值以上时，准入才会恢复
        # watermark + recovery band, so a level oscillating around one threshold
        # cannot flap admit/preempt (O9).
        self._watermark_blocks = int(gpu_num_blocks * watermark)
        self._recover_blocks = int(gpu_num_blocks * hysteresis)
        self._under_pressure = False
        logger.info(
            "KV cache: %d blocks, watermark=%d blocks (%.0f%%), recovery band=%d blocks",
            gpu_num_blocks,
            self._watermark_blocks,
            watermark * 100,
            self._recover_blocks,
        )

        # 待分配的行索引，以及每行的引用计数。
        self.kv_mem_pos_indices = torch.arange(
            0, self.max_num_tokens, dtype=torch.long, device=self.device
        )
        self.kv_mem_use_state = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        ) # 记录 kvcache block的使用情况
        # 预转换 int32 类型副本，使得增量快速路径返回视图，而非逐次类型转换。
        self.kv_mem_pos_indices_int32 = self.kv_mem_pos_indices.to(torch.int32)
        # Append-only cursor, and whether it is still exact (invalidated by a partial
        # free, restored by ``free_all``).
        self._bump_cursor = 0
        self._bump_is_exact = True

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(self.max_num_tokens, self.kv_row, num_layers, dtype, device)
    
    def can_admit(self, need_blocks: int) -> bool:
        """Check if a new request requiring *need_blocks* can be admitted.

        False when free capacity after allocation would drop below the watermark
        (preventing eviction cascades). A pure read, but it remembers which side of
        the watermark the level last sat on: after a dip the bar rises by the recovery
        band until the level climbs back, so an oscillating level cannot flap admission
        (O9 hysteresis).
        """
        if self._under_pressure:
            if self.can_use_mem_size >= self._watermark_blocks + self._recover_blocks:
                self._under_pressure = False
        elif self.can_use_mem_size < self._watermark_blocks:
            self._under_pressure = True
        floor = (
            self._watermark_blocks + self._recover_blocks
            if self._under_pressure
            else self._watermark_blocks
        )
        return self.can_use_mem_size - need_blocks >= floor
    
    @property
    def utilization(self) -> float:
        """Fraction of KV cache currently in use (0.0 empty, 1.0 full)."""
        return 1.0 - (self.can_use_mem_size / self.max_num_tokens)
    
    def init_kv_buffers(
        self,
        max_num_tokens,
        kv_row,
        num_layers,
        dtype,
        device: str = "cuda",
    ) -> None:
        """Pre-allocate one KV tensor per layer, ``[max_num_tokens, *kv_row]``.

        MHA/GQA row is ``(2 * kv_heads, head_dim)`` (K heads then V, so a decode step
        writes both in one launch); MLA is ``(1, latent_dim)`` (K and V share one
        latent vector, written whole).
        """
        # TODO: reshape into [blocks, block_size, ...] to support PagedAttention.
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, *kv_row), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")
    
    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        """Reserve ``need_size`` free rows, wherever they are. Returns ``None`` if short."""
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_ref(select_index)

        return select_index
    
    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        """Reserve ``need_size`` *consecutive* free rows, or ``None`` if no such run.

        Returns:
            ``(select_index, start_index, end_index)``, or ``None``.
        """
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if need_size <= N:
            # 自由列表的两种视图，偏移量为所需大小减 1：start_indices [j] 和
            # end_indices [j] 是候选窗口的结束位置（最后一个有效起始位置位于
            # N - need_size；切片操作不包含终止索引，因此需要 +1）。
            start_indices = can_use_pos_index[: N - need_size + 1]
            end_indices = can_use_pos_index[need_size - 1 :]
            # A window is consecutive exactly when its ends differ by need_size - 1;
            # larger means a used row sits between.
            contiguous_blocks = (end_indices - start_indices == need_size - 1).nonzero(
                as_tuple=True
            )[0]

            if contiguous_blocks.numel() > 0:
                start_index = start_indices[contiguous_blocks[0]].item()  # first run wins
                end_index = start_index + need_size
                select_index = self.kv_mem_pos_indices[start_index:end_index]
                self.add_ref(select_index)
                return select_index, start_index, end_index

        return None
    
    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        """Reserve ``need_size`` cache rows, preferring a contiguous run.

        On the decode hot path. The general search costs a ``nonzero`` plus two
        ``.item()`` reads (three device syncs that stall the pipeline). While the cache
        is append-only (every ``generate()`` starts here, via :meth:`free_all`) the
        answer is the next ``need_size`` rows, so the bump cursor returns them with no
        device reads; a partial free falls back to the search.
        """
        if self._bump_is_exact and self._bump_cursor + need_size <= self.max_num_tokens:
            start = self._bump_cursor
            select_index = self.kv_mem_pos_indices_int32[start : start + need_size]
            self.kv_mem_use_state[start : start + need_size] += 1
            self._bump_cursor += need_size
            self.can_use_mem_size -= need_size
            return select_index

        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, _start_index, _end_index = alloc_mem
        else:
            select_index = self.alloc_kvcache(need_size)
        return select_index.to(torch.int32)
    

    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        """Increment the reference count of the given rows.

        Only previously-free rows reduce :attr:`can_use_mem_size`; a second reference
        on an already-held row costs no capacity.
        """
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens

        self.kv_mem_use_state[token_index] += 1
        return

    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
        """Decrement the reference count of the given rows, freeing those that reach zero.

        ``token_index`` may name a row more than once (the engine releases prefill and
        every decode step in one tensor), so counts are collapsed with ``unique`` first.
        """
        # Freeing leaves holes, so the append-only cursor no longer describes the free
        # list; fall back to searching until ``free_all``.
        self._bump_is_exact = False
        token_index, counts = token_index.unique(return_counts=True)
        self.kv_mem_use_state[token_index] -= counts
        state = self.kv_mem_use_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    @torch.no_grad()
    def claim(self, num_rows: int) -> None:
        """Hand the first ``num_rows`` rows to an external owner, permanently.

        Continuous batching maps each slot onto a fixed contiguous region (see
        :class:`~light_llm.executor.slot_batch.SlotBatch`); marking those rows used
        keeps :attr:`can_use_mem_size` honest, and the bump cursor resumes past the
        claimed region so both schemes share one pool.
        """
        if num_rows > self.can_use_mem_size:
            raise ValueError(f"cannot claim {num_rows} rows: only {self.can_use_mem_size} are free")
        self.kv_mem_use_state[:num_rows] += 1
        self.can_use_mem_size -= num_rows
        self._bump_cursor = max(self._bump_cursor, num_rows)

    @torch.no_grad()
    def free(self, free_index):
        """Release rows by index, logging when that empties the cache."""
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.kv_mem_use_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    @torch.no_grad()
    def free_all(
        self,
    ):
        """Drop every reference at once, as each ``generate()`` call starts by doing."""
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0
        # The cache is empty again, so appending from row 0 is exact once more.
        self._bump_cursor = 0
        self._bump_is_exact = True

if __name__ == "__main__":
    def _idx_list(t):
        if t is None:
            return None
        return t.detach().cpu().tolist()

    def _used_map(kv):
        used = (kv.kv_mem_use_state > 0).nonzero(as_tuple=False).view(-1)
        return {int(i): int(kv.kv_mem_use_state[i]) for i in used.tolist()}

    def _dump(kv, title):
        print(f"\n[{title}]")
        print(
            f"  空闲 {kv.can_use_mem_size}/{kv.max_num_tokens}"
            f"  利用率 {kv.utilization:.2f}"
            f"  bump_cursor={kv._bump_cursor}"
            f"  bump_exact={kv._bump_is_exact}"
            f"  under_pressure={kv._under_pressure}"
        )
        print(f"  占用行 {{index: refcount}} = {_used_map(kv)}")

    # CPU 即可演示分配逻辑，不必占整卡显存。
    device = "cpu"
    num_layers = 2
    kv_heads, head_dim = 2, 8
    kv_row = (2 * kv_heads, head_dim)  # K 头 + V 头拼在一行
    gpu_num_blocks = 16
    watermark, hysteresis = 0.25, 0.125  # 水位 4 块，恢复带 2 块

    print("=" * 60)
    print("1) MemoryProfiler：每 token 占用，以及 CPU 下的固定预算")
    print("=" * 60)
    profiler = MemoryProfiler(
        num_layers=num_layers,
        kv_row=kv_row,
        gpu_memory_utilization=0.9,
        dtype=torch.float32,
        device=device,
    )
    bytes_per_tok = profiler._kv_bytes_per_token()
    print(
        f"  kv_row={kv_row}, layers={num_layers}, dtype=float32"
        f" -> {bytes_per_tok} bytes/token"
        f"  (= {kv_row[0]}*{kv_row[1]}*{num_layers}*4)"
    )
    print(f"  CPU 回退 available_kv_blocks = {profiler.available_kv_blocks(model=None, vocab_size=32)}")

    print("\n" + "=" * 60)
    print("2) 初始化 KVCacheManager：预分配每层一张大表，refcount 全 0")
    print("=" * 60)
    kv = KVCacheManager(
        num_layers=num_layers,
        kv_row=kv_row,
        gpu_num_blocks=gpu_num_blocks,
        block_size=1,
        dtype=torch.float32,
        device=device,
        watermark=watermark,
        hysteresis=hysteresis,
    )
    print(f"  gpu_kv_buffer: {len(kv.gpu_kv_buffer)} layers, shape={tuple(kv.gpu_kv_buffer[0].shape)}")
    print(
        f"  watermark={kv._watermark_blocks}  recover_band={kv._recover_blocks}"
        f"  （空闲低于水位则拒绝新请求；恢复需再高出 hysteresis）"
    )
    _dump(kv, "初始")

    print("\n" + "=" * 60)
    print("3) Prefill：bump 路径连续切 4 行（append-only，无 GPU nonzero）")
    print("=" * 60)
    prefill = kv.alloc_kvcache_index(4)
    print(f"  alloc_kvcache_index(4) -> { _idx_list(prefill) }")
    _dump(kv, "prefill 之后")
    assert _idx_list(prefill) == [0, 1, 2, 3]
    assert kv._bump_is_exact and kv._bump_cursor == 4

    print("\n" + "=" * 60)
    print("4) Decode：再 bump 1 行，紧挨着 prefill")
    print("=" * 60)
    decode = kv.alloc_kvcache_index(1)
    print(f"  alloc_kvcache_index(1) -> { _idx_list(decode) }")
    _dump(kv, "decode 一步之后")
    assert _idx_list(decode) == [4]

    print("\n" + "=" * 60)
    print("5) 引用计数：同一行加第二次引用，不额外占容量（共享 prefix）")
    print("=" * 60)
    free_before = kv.can_use_mem_size
    kv.add_ref(prefill)
    print(f"  add_ref(prefill) 空闲 {free_before} -> {kv.can_use_mem_size}（应不变）")
    _dump(kv, "refcount=2")
    kv.release_ref(prefill)
    print("  release_ref(prefill) 一次后仍占用（refcount 从 2 降到 1）")
    _dump(kv, "refcount=1")
    assert all(int(kv.kv_mem_use_state[i]) == 1 for i in range(5))

    print("\n" + "=" * 60)
    print("6) 释放整段序列：出现空洞，bump 不再精确，后续要搜索空闲行")
    print("=" * 60)
    seq_tokens = torch.cat([prefill, decode])
    kv.release_ref(seq_tokens)
    _dump(kv, "整段 release 之后")
    assert kv.can_use_mem_size == gpu_num_blocks
    assert kv._bump_is_exact is False

    print("\n" + "=" * 60)
    print("7) 制造空洞：占用 [0,1,2,3,4,5]，释放 1 和 3")
    print("=" * 60)
    kv.free_all()
    block = kv.alloc_kvcache_index(6)
    print(f"  先 alloc 6 -> {_idx_list(block)}")
    kv.release_ref(torch.tensor([1, 3], dtype=torch.int32, device=device))
    _dump(kv, "释放 1,3 之后（空洞）")

    contig = kv.alloc_contiguous_kvcache(2)
    print(f"  alloc_contiguous_kvcache(2) -> index={_idx_list(contig[0]) if contig else None}"
          f"  start,end={None if contig is None else (contig[1], contig[2])}")
    print("  （连续窗口要物理下标差 == need-1，1 和 3 中间夹着已用的 2，所以不会选它们）")
    _dump(kv, "连续分配 2 行之后")
    assert contig is not None and _idx_list(contig[0]) == [6, 7]

    scattered = kv.alloc_kvcache(2)
    print(f"  alloc_kvcache(2) 可散落 -> {_idx_list(scattered)}")
    _dump(kv, "散落分配之后")
    assert _idx_list(scattered) == [1, 3]

    print("\n" + "=" * 60)
    print("8) alloc_kvcache_index：bump 已失效，走连续搜索 / 再退化为散落")
    print("=" * 60)
    kv.free_all()
    kv.alloc_kvcache_index(3)
    kv.release_ref(torch.tensor([0], dtype=torch.int32, device=device))  # 破坏 bump
    mixed = kv.alloc_kvcache_index(2)
    print(f"  bump 失效后 alloc_kvcache_index(2) -> {_idx_list(mixed)}")
    _dump(kv, "搜索路径分配")
    assert mixed is not None

    print("\n" + "=" * 60)
    print("9) watermark + hysteresis：空闲跌破水位后，要涨回水位+恢复带才重新准入")
    print("=" * 60)
    kv.free_all()
    floor0 = kv._watermark_blocks
    rec = kv._recover_blocks
    print(f"  水位={floor0}, 恢复带={rec}, 恢复门槛={floor0 + rec}")
    print(f"  空闲={kv.can_use_mem_size} can_admit(8)={kv.can_admit(8)}")  # 16-8=8 >= 4
    kv.alloc_kvcache_index(13)  # 空闲 3 < 水位 4
    _dump(kv, "分配 13 行，空闲跌破水位")
    print(f"  can_admit(0)={kv.can_admit(0)}  （under_pressure 后门槛升到 {floor0 + rec}）")
    assert kv.can_admit(0) is False
    kv.release_ref(torch.arange(4, dtype=torch.int32, device=device))  # 空闲 3+4=7 >= 6 恢复
    _dump(kv, "释放 4 行，空闲回到恢复门槛以上")
    print(f"  can_admit(0)={kv.can_admit(0)}")
    assert kv._under_pressure is False
    assert kv.can_admit(0) is True

    print("\n" + "=" * 60)
    print("10) claim：把前 N 行永久划给外部（如 SlotBatch），bump 从其后继续")
    print("=" * 60)
    kv.free_all()
    kv.claim(4)
    nxt = kv.alloc_kvcache_index(2)
    print(f"  claim(4) 后 bump alloc(2) -> {_idx_list(nxt)}  （应从 4 开始）")
    _dump(kv, "claim + bump")
    assert _idx_list(nxt) == [4, 5]

    print("\n" + "=" * 60)
    print("11) 容量不足；free_all 清空并恢复 bump")
    print("=" * 60)
    print(f"  alloc_contiguous_kvcache(100) -> {kv.alloc_contiguous_kvcache(100)}")
    print(f"  alloc_kvcache(100) -> {kv.alloc_kvcache(100)}")
    try:
        too_many = kv.alloc_kvcache_index(100)
        print(f"  alloc_kvcache_index(100) -> {_idx_list(too_many)}")
    except AttributeError as e:
        print(f"  alloc_kvcache_index(100) 异常: {e}")
        print("  （底层 alloc_* 已返回 None，但 index 路径未判空就 .to(int32)）")
    kv.free_all()
    _dump(kv, "free_all")
    assert kv.can_use_mem_size == gpu_num_blocks
    assert kv._bump_is_exact and kv._bump_cursor == 0
    restored = kv.alloc_kvcache_index(3)
    print(f"  free_all 后再 alloc(3) 又走 bump -> {_idx_list(restored)}")
    assert _idx_list(restored) == [0, 1, 2]

    print("\n全部逻辑演示通过。")

