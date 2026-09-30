"""Physical KV block pool: reference counting, LRU eviction, hash lookup.

:class:`BlockPool` owns every physical block of the KV cache and is the only
thing that decides which block a request gets. Blocks carry a reference count
(a block several requests share is held by all of them) and a chained block
hash (a cached block is reusable by any prompt with the same prefix), and free
blocks sit in :class:`FreeBlockQueue` in eviction order.

Usage:
    pool = BlockPool(num_blocks=1024, block_size=16)
    blocks = pool.get_new_blocks(4); pool.free_blocks(reversed(blocks))
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: Reserved block id. Its rows exist but hold nothing anyone reads: a sliding
#: window group points the table entries of positions below its window here, so
#: those entries stay in-range without pinning a real block.
NULL_BLOCK_ID = 0


@dataclass
class KVCacheBlock:
    """One physical block of the KV cache, plus the two ledgers over it.

    Attributes:
        block_id: Index of this block; its cache rows are
            ``[block_id * block_size, (block_id + 1) * block_size)``.
        ref_cnt: Live requests holding this block. A shared prefix block is
            held by every request sharing it, and a block is only reusable for
            something else at zero.
        block_hash: Chained hash of this block *and* every block before it,
            or ``None`` for a block whose tokens are not all computed yet.
            ``None`` also means "not in the hash index", so a partially filled
            block can never be handed to another prompt as a prefix hit.
        prev_free_block: Predecessor in :class:`FreeBlockQueue`, or ``None``.
        next_free_block: Successor in :class:`FreeBlockQueue`, or ``None``.
    """
    block_id: int
    ref_cnt: int = 0 # ref_cnt=0代表了当前没有活跃的请求依赖他
    block_hash: int | None = None
    # The free list is threaded through the blocks themselves rather than kept
    # as a separate deque: removing an arbitrary block (a hit on a block that
    # was sitting free) has to be O(1), and a deque cannot do that.
    prev_free_block: KVCacheBlock | None = field(default=None, repr=False, compare=False)
    next_free_block: KVCacheBlock | None = field(default=None, repr=False, compare=False)

    def reset_hash(self) -> None:
        """Forget this block's hash, so nothing can match it any more."""
        self.block_hash = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"KVCacheBlock(id={self.block_id}, ref_cnt={self.ref_cnt}, hash={self.block_hash})"

class FreeBlockQueue:
    """Free blocks in eviction order: front is evicted first, back last.

    A doubly linked list threaded through the blocks, with sentinel head and
    tail nodes so every operation is branch-free at the ends. The point of the
    list — rather than a deque — is :meth:`remove`: a cached block sitting free
    can be hit at any moment, and pulling it out of the middle must not cost a
    scan of the free list.

    Two insertion ends encode the eviction policy. A block with no hash holds
    nothing worth keeping, so it goes to the *front* and is reused first; a
    cached block goes to the *back*, so the queue orders cached blocks
    least-recently-freed first. Handing out from the front therefore consumes
    worthless blocks before it starts evicting real cache entries.

    Args:
        blocks: Initially free blocks, in the order they should be handed out.
    """

    def __init__(self, blocks: Sequence[KVCacheBlock]) -> None:
        self._head = KVCacheBlock(block_id=-1)
        self._tail = KVCacheBlock(block_id=-1)
        self._head.next_free_block = self._tail
        self._tail.prev_free_block = self._head
        self.num_free_blocks = 0
        for block in blocks:
            self.append(block)
    
    def append(self, block: KVCacheBlock) -> None:
        """Add a block to the end of the free list."""
        prev = self._tail.prev_free_block
        assert prev is not None
        prev.next_free_block = block
        block.prev_free_block = prev
        block.next_free_block = self._tail
        self._tail.prev_free_block = block
        self.num_free_blocks += 1
    

    def prepend(self, block: KVCacheBlock) -> None:
        """Add a block to the front of the free list."""
        nxt = self._head.next_free_block
        assert nxt is not None
        self._head.next_free_block = block
        block.prev_free_block = self._head
        block.next_free_block = nxt
        nxt.prev_free_block = block
        self.num_free_blocks += 1
    
    def popleft(self) -> KVCacheBlock:
        """Remove and return the front block.

        Raises:
            IndexError: The queue is empty.
        """
        block = self._head.next_free_block
        if block is None or block is self._tail:
            raise IndexError("no free block")
        self.remove(block)
        return block
    
    def remove(self, block: KVCacheBlock) -> None:
        """Unlink *block* wherever it sits, in constant time.

        Raises:
            ValueError: The block is not currently in this queue.
        """
        prev, nxt = block.prev_free_block, block.next_free_block
        if prev is None or nxt is None:
            raise ValueError(f"block {block.block_id} is not in the free queue")
        prev.next_free_block = nxt
        nxt.prev_free_block = prev
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1
    
    def __iter__(self) -> Iterable[KVCacheBlock]:
        """Walk the queue front to back, i.e. in eviction order."""
        block = self._head.next_free_block
        while block is not None and block is not self._tail:
            yield block
            block = block.next_free_block


@dataclass
class BlockPoolStats:
    """Cumulative counters describing what the pool did.

    Attributes:
        evictions: Cached blocks dropped from the hash index to be reused for
            something else — the pressure signal a capacity plot wants.
        resets: Successful :meth:`BlockPool.reset_prefix_cache` calls.
    """

    evictions: int = 0
    resets: int = 0


class BlockPool:
    """Every physical KV block, handed out by reference count and LRU order.

    The pool knows nothing about tokens, prompts or attention: it maps block
    hashes to blocks, hands out free blocks, and takes them back. What a block
    *means* is decided one layer up, by the KV cache group that allocated it.

    Args:
        num_blocks: Total blocks, including the reserved null block.
        block_size: Cache rows per block; the pool only needs it to translate a
            block id into rows for callers that ask.
        enable_caching: When False, blocks are still ref-counted and recycled
            but never indexed by hash, so nothing is ever reused across
            requests. This is the "prefix caching off" configuration, and it
            takes the same code path rather than a parallel one.

    Raises:
        ValueError: Fewer than two blocks (one is reserved as the null block),
            or a non-positive block size.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_caching: bool = True) -> None:
        if num_blocks < 2:
            raise ValueError(f"need at least 2 blocks (1 is the null block), got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_caching = enable_caching
        self.stats = BlockPoolStats()

        self.blocks: list[KVCacheBlock] = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        #: Never handed out, never freed: a permanent reference keeps every
        #: accounting path (which only ever tests ``ref_cnt``) away from it.
        self.null_block = self.blocks[NULL_BLOCK_ID]
        self.null_block.ref_cnt = 1
        self.free_block_queue = FreeBlockQueue(self.blocks[NULL_BLOCK_ID + 1 :])

        #: block hash -> block id -> block. Two blocks can carry the same hash
        #: for a moment: one being freed while another request has already
        #: cached the same prefix under a fresh block. Keying by id too keeps
        #: eviction from deleting the *other* one's index entry.
        self.cached_block_hash_to_block: dict[int, dict[int, KVCacheBlock]] = {}

    # ------------------------------------------------------------------ views #
    @property
    def num_free_blocks(self) -> int:
        """Blocks available right now, cached-but-unreferenced ones included."""
        return self.free_block_queue.num_free_blocks

    @property
    def num_used_blocks(self) -> int:
        """Blocks a live request holds, plus the null block."""
        return self.num_blocks - self.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        """Blocks currently reachable through the hash index."""
        return sum(len(by_id) for by_id in self.cached_block_hash_to_block.values())

    def rows_of(self, block_id: int) -> range:
        """Cache rows backing *block_id*."""
        return range(block_id * self.block_size, (block_id + 1) * self.block_size)
    
    # ------------------------------------------------------------------ lookup #
    def get_cached_block(self, block_hash: int) -> KVCacheBlock | None:
        by_id = self.cached_block_hash_to_block.get(block_hash)
        if by_id is None:
            return None
        return next(iter(by_id.values()))

    # ------------------------------------------------------------------ allocation #
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock] | None:
        """Take *num_blocks* free blocks, evicting cached ones if it must.

        Returns ``None`` rather than raising when the pool is short: the caller
        (admission, or a decode step crossing a block boundary) has a policy for
        that — preempt someone, or leave the request waiting — and an exception
        would make every call site wrap this in a try block.

        The blocks come off the front of the free queue, which is ordered so
        that hash-less blocks go before cached ones; a cached block that does
        get taken is dropped from the hash index here, because its rows are
        about to be overwritten.
        """
        if num_blocks < 0:
            raise ValueError(f"cannot allocate {num_blocks} blocks")
        if num_blocks > self.num_free_blocks:
            return None
        
        taken: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self.free_block_queue.popleft()
            assert block.ref_cnt == 0, f"free block {block.block_id} still referenced"
            self._maybe_evict_cached_block(block)
            block.ref_cnt = 1
            taken.append(block)
        return taken
    
    def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
        """Take a reference on cached blocks a new request is about to share.

        A block sitting free is still in the free queue; the first reference has
        to pull it out, or the next allocation would hand out rows this request
        is now reading.

        The null block is skipped entirely: it is never in the free queue and is
        never freed, so counting references on it would only make the pool's
        accounting drift.
        """
        for block in blocks:
            if block is self.null_block:
                continue
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
    
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Drop one reference per block, returning those that hit zero.

        Order matters and is the caller's to choose: blocks are returned to the
        free queue in the order given, and the queue evicts from its front, so
        a caller that frees a request's blocks *tail first* gets the tail
        evicted first — the head of a prompt is the part another request is
        likely to share.

        The null block is skipped: it is never allocated, so it is never freed.
        """
        for block in ordered_blocks:
            if block is self.null_block:
                continue
            if block.ref_cnt <= 0:
                raise ValueError(f"block {block.block_id} freed more times than referenced")
            block.ref_cnt -= 1
            if block.ref_cnt > 0:
                continue
            # A block that holds nothing worth reusing goes to the front, so the
            # next allocation consumes it before it starts evicting real cache.
            if block.block_hash is None:
                self.free_block_queue.prepend(block)
            else:
                self.free_block_queue.append(block)

    # ------------------------------------------------------------ hash index #
    def cache_full_blocks(
        self,
        blocks: Sequence[KVCacheBlock],
        block_hashes: Sequence[int],
    ) -> None:
        """Index *blocks* under their hashes so later prompts can hit them.

        Callers must only pass blocks whose tokens are *computed* — the model
        has written their K/V — because indexing a block advertises its rows as
        readable. A block that already carries a hash is left alone, so calling
        this repeatedly as a prompt fills in is cheap and idempotent.

        Args:
            blocks: The request's blocks, from its first block onwards.
            block_hashes: Chained hash per block, parallel to ``blocks``.

        Raises:
            ValueError: The two sequences describe different block counts.
        """
        if not self.enable_caching:
            return
        if len(blocks) != len(block_hashes):
            raise ValueError("blocks and block_hashes must describe the same blocks")
        for block, block_hash in zip(blocks, block_hashes, strict=True):
            if block is self.null_block or block.block_hash is not None:
                continue
            block.block_hash = block_hash
            self.cached_block_hash_to_block.setdefault(block_hash, {})[block.block_id] = block

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> None:
        """Drop *block* from the hash index, if it was in it."""
        block_hash = block.block_hash
        if block_hash is None:
            return
        by_id = self.cached_block_hash_to_block.get(block_hash)
        if by_id is not None:
            by_id.pop(block.block_id, None)
            if not by_id:
                del self.cached_block_hash_to_block[block_hash]
        block.reset_hash()
        self.stats.evictions += 1

    def reset_prefix_cache(self) -> bool:
        """Forget every cached block, or refuse if any is still in use.

        Returns False (and changes nothing) while a live request holds a block:
        clearing the index under a running request would leave its blocks
        unreachable but referenced, so they could never be reused *or* hit —
        a slow leak that only shows up as a mysterious capacity loss. Only the
        null block's permanent reference may remain.
        """
        if self.num_used_blocks != 1:
            return False
        for block in self.blocks:
            block.reset_hash()
        self.cached_block_hash_to_block.clear()
        self.stats.resets += 1
        return True


def _snapshot(pool: BlockPool, label: str) -> None:
    """Print a one-line pool snapshot for demos."""
    free_ids = [b.block_id for b in pool.free_block_queue]
    print(
        f"  [{label}] free={pool.num_free_blocks} used={pool.num_used_blocks} "
        f"cached={pool.num_cached_blocks} evictions={pool.stats.evictions} "
        f"free_order={free_ids}"
    )


if __name__ == "__main__":
    # Small pool: block 0 is the null block; blocks 1..5 are allocatable.
    pool = BlockPool(num_blocks=6, block_size=16, enable_caching=True)
    print("=" * 60)
    print("BlockPool demo (num_blocks=6, block_size=16, null=block 0)")
    print("=" * 60)
    _snapshot(pool, "init")
    print(f"  null_block={pool.null_block} rows={list(pool.rows_of(NULL_BLOCK_ID))}")

    # ------------------------------------------------------------------ #
    # Case 1: allocate → use → free (no hash → prepend, reused first)
    # ------------------------------------------------------------------ #
    print("\n--- Case 1: allocate / free without hashing ---")
    req_a = pool.get_new_blocks(3)
    assert req_a is not None
    print(f"  allocated req_a: {req_a}")
    _snapshot(pool, "after alloc 3")

    # Free tail-first so the prompt head stays farther from eviction.
    pool.free_blocks(reversed(req_a))
    print(f"  freed req_a (tail-first); blocks now: {req_a}")
    _snapshot(pool, "after free (no hash → front)")

    # ------------------------------------------------------------------ #
    # Case 2: cache full blocks, free → append (LRU back); next alloc
    #         consumes hash-less blocks before evicting cached ones.
    # ------------------------------------------------------------------ #
    print("\n--- Case 2: prefix caching + LRU free-queue order ---")
    req_b = pool.get_new_blocks(2)
    assert req_b is not None
    print(f"  allocated req_b: {req_b}")
    pool.cache_full_blocks(req_b, block_hashes=[0xA1, 0xA2])
    print(f"  cached under hashes [0xA1, 0xA2]: {req_b}")
    _snapshot(pool, "after cache")

    pool.free_blocks(reversed(req_b))
    print("  freed req_b (cached → append to back)")
    _snapshot(pool, "after free cached")

    # Hash-less blocks sit at the front; they are handed out before cached ones.
    filler = pool.get_new_blocks(3)
    assert filler is not None
    print(f"  allocated 3 (prefer uncached): {filler}")
    _snapshot(pool, "prefer uncached")
    # Leave filler held so only the two cached blocks remain free.
    print("  (hold filler; free queue is only the two cached blocks)")
    _snapshot(pool, "only cached free")

    # Force eviction: next alloc must take a cached free block and drop its hash.
    before_evictions = pool.stats.evictions
    victim = pool.get_new_blocks(1)
    assert victim is not None
    print(
        f"  allocated 1 (evict cached): {victim}; "
        f"evictions {before_evictions} → {pool.stats.evictions}"
    )
    _snapshot(pool, "after eviction")
    pool.free_blocks(reversed(filler))
    pool.free_blocks(victim)

    # ------------------------------------------------------------------ #
    # Case 3: touch — share a cached prefix; first ref pulls it off free list
    # ------------------------------------------------------------------ #
    print("\n--- Case 3: touch / shared prefix (ref_cnt) ---")
    # Re-cache two blocks so they sit free with hashes.
    shared = pool.get_new_blocks(2)
    assert shared is not None
    pool.cache_full_blocks(shared, block_hashes=[0xB1, 0xB2])
    pool.free_blocks(reversed(shared))
    print(f"  shared prefix sitting free: {shared}")
    _snapshot(pool, "shared free")

    # New request hits the prefix: touch bumps ref_cnt and removes from free queue.
    pool.touch(shared)
    print(f"  after touch (req hits prefix): {shared}")
    _snapshot(pool, "after touch")

    # A second request shares the same prefix.
    pool.touch(shared)
    print(f"  after 2nd touch: {shared}")
    _snapshot(pool, "ref_cnt=2")

    # Each free drops one ref; only at zero does it re-enter the free queue.
    pool.free_blocks(reversed(shared))
    print(f"  after 1st free (still held): {shared}")
    _snapshot(pool, "ref_cnt=1")
    pool.free_blocks(reversed(shared))
    print(f"  after 2nd free (back to free): {shared}")
    _snapshot(pool, "ref_cnt=0")

    # ------------------------------------------------------------------ #
    # Case 4: allocation failure when pool is exhausted
    # ------------------------------------------------------------------ #
    print("\n--- Case 4: get_new_blocks returns None when short ---")
    held = pool.get_new_blocks(pool.num_free_blocks)
    assert held is not None
    print(f"  took all free blocks ({len(held)}); next alloc → None")
    print(f"  get_new_blocks(1) = {pool.get_new_blocks(1)!r}")
    pool.free_blocks(reversed(held))

    # ------------------------------------------------------------------ #
    # Case 5: reset_prefix_cache only when nothing live holds blocks
    # ------------------------------------------------------------------ #
    print("\n--- Case 5: reset_prefix_cache ---")
    busy = pool.get_new_blocks(1)
    assert busy is not None
    pool.cache_full_blocks(busy, block_hashes=[0xC1])
    print(f"  reset while busy → {pool.reset_prefix_cache()} (refused)")
    pool.free_blocks(busy)
    # Leave a cached free block, then reset when only null is "used".
    leftover = pool.get_new_blocks(1)
    assert leftover is not None
    pool.cache_full_blocks(leftover, block_hashes=[0xC2])
    pool.free_blocks(leftover)
    _snapshot(pool, "before reset")
    print(f"  reset while idle → {pool.reset_prefix_cache()}")
    _snapshot(pool, "after reset")
    print(f"  leftover hash cleared: {leftover}")

    print("\n" + "=" * 60)
    print("demo done")
    print("=" * 60)
