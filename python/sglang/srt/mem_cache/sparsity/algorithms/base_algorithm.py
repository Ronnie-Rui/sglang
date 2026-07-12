from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class _RetrievalPlan:
    """Layer-invariant inputs for one sparse decode forward."""

    forward_batch: "ForwardBatch"
    batch_size: int
    device: torch.device
    seq_lens: torch.Tensor
    seq_lens_cpu: list[int] | None
    req_pool_indices: torch.Tensor
    sparse_mask: torch.Tensor
    num_pages: torch.Tensor
    num_pages_cpu: list[int] | None
    max_num_pages: int
    page_idx: torch.Tensor
    physical_pages: torch.Tensor
    valid_page_mask: torch.Tensor
    active_mask: torch.Tensor
    recent_start: torch.Tensor
    history_page_mask: torch.Tensor
    k_per_req: torch.Tensor
    max_k: int
    score_order_required: bool
    recent_idx: torch.Tensor
    recent_valid: torch.Tensor


class BaseSparseAlgorithm(ABC):
    """
    Abstract base class for sparse attention algorithms.

    This class provides a unified interface for implementing various retrievable KVCache
    compression algorithms. Token-wise sparsity is treated as page-wise with page_size=1.

    References:
        - ChunkKV: https://arxiv.org/abs/2502.00299
        - Quest: https://arxiv.org/pdf/2406.10774
        - PQCache: https://arxiv.org/abs/2407.12820
        - SnapKV: https://arxiv.org/pdf/2404.14469
        - Look-ahead QCache: https://arxiv.org/pdf/2505.20334
        - and more...
    """

    def __init__(self, config, device: torch.device, **kwargs):
        self.config = config
        self.device = device
        self.req_to_token_pool = None
        self.states = None

    def begin_forward(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Prepare algorithm state shared by every layer in one forward."""

    def should_update_representations(self, forward_batch: "ForwardBatch") -> bool:
        """Return whether this decode forward may have completed a page."""
        return True

    def get_selected_physical_pages(
        self, selected_indices: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a cached physical mapping when the algorithm has one."""
        return None

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        """
        Initialize algorithm-specific representation pool and set context.

        Called once during SparseCoordinator initialization. Algorithms allocate
        their own representation tensors and store references to context.

        Algorithm-specific implementations:
            - ChunkKV: Allocate chunk scores [num_chunks, 1] for tracking semantic chunk importance
            - Quest: Allocate page representations [num_pages, repr_dim] via key pooling
            - PQCache: Allocate centroids [n_subvec, n_centroids, subvec_dim] and token codes [num_tokens, n_subvec]
            - SnapKV: Allocate voting scores [num_tokens] and selected positions mask for retention strategy
            - Look-ahead QCache: Allocate importance scores [num_tokens], eviction mask, and optional pseudo query cache [cache_size, hidden_dim]
        """
        pass

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """
        Construct initial representations during prefill phase.

        Called at every layer during forward pass. Algorithm internally decides
        whether to perform construction.
        Typically only constructs once per request during prefill/extend phase.

        Algorithm-specific implementations:
            - ChunkKV: Compute chunk importance scores via aggregated key L2 norms within semantic chunks
            - Quest: Compute page representations via mean pooling of keys within each page
            - PQCache: Run K-means clustering to generate centroids and assign each token to nearest centroid
            - SnapKV: Select observation window (recent tokens), compute attention weights, aggregate via voting to identify important prefix positions, apply 1D pooling to preserve context
            - Look-ahead QCache: Generate pseudo lookahead query (e.g., mean of last k queries), compute KV importance scores, mark low-importance KVs for eviction
        """
        pass

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """
        Incrementally update representations during decode phase.

        Called at every layer during forward pass. Algorithm internally decides
        whether to update based on:
        - self.states.repr_constructed[req_id]: Whether initial construction done
        - self.states.last_constructed_page[req_id]: Last constructed page index
        - Current seq_lens: To detect new tokens/pages

        Algorithm-specific implementations:
            - ChunkKV: Incrementally compute importance scores for newly generated chunks during decode
            - Quest: Incrementally compute representations for newly generated pages during decode
            - PQCache: Assign new tokens to existing centroids (no centroid update during decode)
            - SnapKV: Optional: periodically re-run voting with sliding observation window (typically static after prefill)
            - Look-ahead QCache: Periodically regenerate pseudo queries and re-evaluate importance scores to adapt to generation dynamics
        """
        pass

    @abstractmethod
    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Retrieve top-k important KV indices for sparse attention.

        Called before attention computation at each layer. Uses current query
        and pre-computed representations to select the most important subset
        of KV cache for attention computation.

        Args:
            queries: [bs, num_heads, head_dim] Current query vectors
            layer_id: Current layer index
            req_pool_indices: [bs] Request pool indices
            sparse_mask: [bs] bool, which requests need sparse attention
            attn_metadata: Attention metadata (contains seq_lens, etc.)
            **kwargs: Algorithm-specific arguments

        Returns:
            selected_indices: [bs, max_selected] Selected page/token indices, padded with -1
            valid_lengths: [bs] Actual number of selected indices per request

        Note:
            - Indices are logical positions. Implementations may expose a cached
              physical mapping to BackendAdaptor via get_selected_physical_pages().

        Algorithm-specific implementations:
            - ChunkKV: Select top-k chunks based on pre-computed importance scores with layer-wise index reuse
            - Quest: Compute query-page similarity using current query and stored page representations, select top-k pages
            - PQCache: Calculate query-centroid similarity, use centroid scores to rank tokens, select top-k tokens
            - SnapKV: Return union of voted important prefix positions (with clustered neighbors) and observation window tokens
            - Look-ahead QCache: Return KVs not marked for eviction (eviction based on pseudo query importance evaluation)
        """
        pass


class BaseSparseAlgorithmImpl(BaseSparseAlgorithm):
    """
    Implementation base class for sparse attention algorithms.

    Provides common infrastructure for algorithms that operate at page/chunk granularity
    (token-wise is simply page_size=1):
    - Generic construct/update flow with state tracking
    - TopK retrieval with recent page retention (can be overridden)

    Subclasses need to implement:
    - _initialize_representation_pools(): Initialize algorithm-specific representation pools
    - _compute_page_representations(): Compute page scores/representations
    - _retrieve_page_scores(): Retrieve page scores for TopK selection

    Subclasses can also override any method for specialized behavior
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.sparsity_ratio = config.sparse_extra_config.get("sparsity_ratio", 0.7)
        self.num_recent_pages = config.sparse_extra_config.get("num_recent_pages", 4)
        self.page_size = config.page_size
        self._retrieval_plan: _RetrievalPlan | None = None
        self._representation_update_batch = None
        self._representation_update_due: bool | None = None

    def begin_forward(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Cache decode metadata that is identical across attention layers."""
        self._representation_update_batch = forward_batch
        if self.req_to_token_pool is None:
            self._retrieval_plan = None
            self._representation_update_due = self._decode_page_boundary_reached(
                forward_batch
            )
            return
        self._retrieval_plan = self._build_retrieval_plan(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
        )
        self._representation_update_due = self._has_completed_page(
            self._retrieval_plan.seq_lens_cpu
        )

    def should_update_representations(self, forward_batch: "ForwardBatch") -> bool:
        if self._representation_update_batch is not forward_batch:
            self._representation_update_batch = forward_batch
            self._representation_update_due = self._decode_page_boundary_reached(
                forward_batch
            )
        # A missing CPU mirror means the cheap host gate is unavailable. Let
        # the existing device-side valid mask make the final decision.
        return self._representation_update_due is not False

    def get_selected_physical_pages(
        self, selected_indices: torch.Tensor
    ) -> torch.Tensor | None:
        plan = self._retrieval_plan
        if (
            plan is None
            or selected_indices.ndim != 2
            or plan.physical_pages.shape[0] != selected_indices.shape[0]
        ):
            return None

        if plan.physical_pages.shape[1] == 0:
            return torch.zeros_like(selected_indices, dtype=torch.int32)

        logical_pages = selected_indices.to(torch.long)
        physical_pages = torch.gather(
            plan.physical_pages,
            1,
            logical_pages.clamp(min=0, max=plan.physical_pages.shape[1] - 1),
        )
        return torch.where(
            logical_pages >= 0,
            physical_pages,
            torch.zeros_like(physical_pages),
        ).to(torch.int32)

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.states = states

        total_num_tokens = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
        total_num_pages = (total_num_tokens + self.page_size - 1) // self.page_size

        # Initialize algorithm-specific representation pools
        self._initialize_representation_pools(start_layer, end_layer, total_num_pages)

    def construct_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:

        if not forward_batch.forward_mode.is_extend():
            return

        if getattr(forward_batch, "extend_prefix_lens", None) is not None:
            new_req_mask = forward_batch.extend_prefix_lens == 0
            if new_req_mask.any():
                new_req_indices = req_pool_indices[new_req_mask]
                self.states.repr_constructed[new_req_indices] = False
                self.states.prompt_lens[new_req_indices] = 0
                self.states.last_constructed_page[new_req_indices] = 0

        prompt_lens = self.states.prompt_lens[req_pool_indices]
        self.states.prompt_lens[req_pool_indices] = torch.maximum(prompt_lens, seq_lens)

        num_pages = seq_lens // self.page_size
        start_page = torch.where(
            self.states.repr_constructed[req_pool_indices],
            self.states.last_constructed_page[req_pool_indices],
            torch.zeros_like(num_pages),
        )
        valid_mask = (seq_lens >= self.states.prompt_lens[req_pool_indices]) & (
            num_pages > start_page
        )

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            num_pages[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_constructed_page[success_indices] = num_pages[valid_mask]

    def update_representations(
        self,
        layer_id,
        req_pool_indices,
        seq_lens,
        k_buffer,
        forward_batch,
    ) -> torch.Tensor:
        if not forward_batch.forward_mode.is_decode():
            return

        if not self.should_update_representations(forward_batch):
            return

        end_page = seq_lens // self.page_size
        constructed = self.states.repr_constructed[req_pool_indices]
        start_page = torch.where(
            constructed,
            self.states.last_constructed_page[req_pool_indices],
            torch.zeros_like(end_page),
        )
        valid_mask = start_page < end_page

        if not valid_mask.any():
            return

        # Compute page representations by subclass
        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

        # Update tracking states
        if layer_id == self.end_layer - 1:
            success_indices = req_pool_indices[valid_mask]
            self.states.repr_constructed[success_indices] = True
            self.states.last_constructed_page[success_indices] = end_page[valid_mask]

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Default TopK retrieval: score-based selection + recent pages.
        Subclasses can override for query-dependent retrieval.

        TODO:
            1. Using triton kernel to speed up this function
            2. Support CUDA Graph
        """
        bs, device = queries.shape[0], queries.device

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for TopK retrieval"
            )
        plan = self._retrieval_plan
        if (
            plan is None
            or plan.forward_batch is not seq_lens_source
            or plan.batch_size != bs
            or plan.device != device
        ):
            plan = self._build_retrieval_plan(
                seq_lens_source,
                req_pool_indices,
                sparse_mask,
                device,
            )
            self._retrieval_plan = plan

        if bs == 1:
            return self._retrieve_topk_single(
                queries,
                layer_id,
                plan,
            )

        return self._retrieve_topk_batched(
            queries,
            layer_id,
            plan,
        )

    def _build_retrieval_plan(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
    ) -> _RetrievalPlan:
        """Build the page layout and selection sizes once per forward."""
        bs = req_pool_indices.numel()
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
        seq_lens_cpu = self._get_seq_lens_cpu(forward_batch, bs)
        req_pool_indices = req_pool_indices.to(device=device, dtype=torch.long)
        sparse_mask = sparse_mask.to(device=device, dtype=torch.bool)
        num_pages = (seq_lens + self.page_size - 1) // self.page_size

        if seq_lens_cpu is not None:
            num_pages_cpu = [
                max((seq_len + self.page_size - 1) // self.page_size, 0)
                for seq_len in seq_lens_cpu
            ]
            max_num_pages = max(num_pages_cpu, default=0)
        else:
            num_pages_cpu = None
            max_num_pages = int(num_pages.max().item()) if bs > 0 else 0

        page_idx = torch.arange(max_num_pages, device=device, dtype=torch.long)
        valid_page_mask = page_idx.unsqueeze(0) < num_pages.unsqueeze(1)
        req_to_token = self.req_to_token_pool.req_to_token
        if max_num_pages > 0:
            page_starts = (page_idx * self.page_size).clamp(
                0, req_to_token.shape[1] - 1
            )
            physical_pages = (
                req_to_token[
                    req_pool_indices[:, None],
                    page_starts[None, :],
                ].to(torch.long)
                // self.page_size
            )
        else:
            physical_pages = torch.empty((bs, 0), device=device, dtype=torch.long)

        active_mask = sparse_mask & (num_pages > self.num_recent_pages)
        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        history_page_mask = page_idx.unsqueeze(0) < recent_start.unsqueeze(1)
        history_pages = recent_start.clamp(min=1)
        k_per_req = (history_pages.to(torch.float32) * self.sparsity_ratio).to(
            torch.long
        )
        k_per_req = torch.maximum(k_per_req, torch.ones_like(k_per_req))
        k_per_req = torch.minimum(k_per_req, history_pages)
        k_per_req = torch.where(active_mask, k_per_req, torch.zeros_like(k_per_req))

        if num_pages_cpu is not None:
            k_per_req_cpu = []
            for count in num_pages_cpu:
                if count <= self.num_recent_pages:
                    k_per_req_cpu.append(0)
                    continue
                history_count = count - self.num_recent_pages
                k_per_req_cpu.append(
                    min(max(int(history_count * self.sparsity_ratio), 1), history_count)
                )
            max_k = max(k_per_req_cpu, default=0)
            positive_k = {k for k in k_per_req_cpu if k > 0}
            score_order_required = len(positive_k) > 1
        else:
            max_k = int(k_per_req.max().item()) if bs > 0 else 0
            score_order_required = True

        recent_offsets = torch.arange(
            self.num_recent_pages, device=device, dtype=torch.long
        )
        recent_idx = recent_start.unsqueeze(1) + recent_offsets.unsqueeze(0)
        recent_valid = active_mask.unsqueeze(1) & (recent_idx < num_pages.unsqueeze(1))
        return _RetrievalPlan(
            forward_batch=forward_batch,
            batch_size=bs,
            device=device,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            num_pages=num_pages,
            num_pages_cpu=num_pages_cpu,
            max_num_pages=max_num_pages,
            page_idx=page_idx,
            physical_pages=physical_pages,
            valid_page_mask=valid_page_mask,
            active_mask=active_mask,
            recent_start=recent_start,
            history_page_mask=history_page_mask,
            k_per_req=k_per_req,
            max_k=max_k,
            score_order_required=score_order_required,
            recent_idx=recent_idx,
            recent_valid=recent_valid,
        )

    @staticmethod
    def _get_seq_lens_cpu(forward_batch, bs: int):
        """Return the existing host mirror without copying a device tensor."""
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return None

        if torch.is_tensor(seq_lens_cpu):
            if seq_lens_cpu.device.type != "cpu" or seq_lens_cpu.numel() != bs:
                return None
            values = seq_lens_cpu.reshape(-1).tolist()
        else:
            try:
                values = list(seq_lens_cpu)
            except TypeError:
                return None
            if len(values) != bs:
                return None

        return [int(seq_len) for seq_len in values]

    def _decode_page_boundary_reached(
        self, forward_batch: "ForwardBatch"
    ) -> bool | None:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            bs = seq_lens.numel()
        else:
            host_seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
            if torch.is_tensor(host_seq_lens):
                bs = host_seq_lens.numel()
            else:
                try:
                    bs = len(host_seq_lens)
                except TypeError:
                    return None
        seq_lens_cpu = self._get_seq_lens_cpu(forward_batch, bs)
        return self._has_completed_page(seq_lens_cpu)

    def _has_completed_page(self, seq_lens_cpu: list[int] | None) -> bool | None:
        if seq_lens_cpu is None:
            return None
        return any(
            seq_len > 0 and seq_len % self.page_size == 0 for seq_len in seq_lens_cpu
        )

    def _retrieve_topk_single(
        self,
        queries: torch.Tensor,
        layer_id: int,
        plan: _RetrievalPlan,
    ) -> tuple:
        """Low-overhead path for the latency-sensitive single-request case."""
        device = queries.device
        num_pages = plan.max_num_pages
        if num_pages <= self.num_recent_pages:
            return self._empty_retrieval(1, device)

        scores = self._retrieve_page_scores(
            layer_id,
            plan.physical_pages,
            plan.req_pool_indices,
            queries,
        )

        recent_start = num_pages - self.num_recent_pages
        k = plan.max_k
        topk_scores, topk_idx = torch.topk(
            scores[:, :recent_start], k=k, dim=1, sorted=False
        )
        active = plan.sparse_mask.view(1, 1)
        topk_valid = active & torch.isfinite(topk_scores)

        return self._finalize_selected_pages(
            torch.cat([topk_idx.to(torch.long), plan.recent_idx], dim=1),
            torch.cat([topk_valid, plan.recent_valid], dim=1),
            num_pages,
        )

    def _retrieve_topk_batched(
        self,
        queries: torch.Tensor,
        layer_id: int,
        plan: _RetrievalPlan,
    ) -> tuple:
        """Vectorized retrieval for batches, including ragged sequence lengths."""
        bs, device = queries.shape[0], queries.device
        if plan.max_num_pages <= self.num_recent_pages:
            return self._empty_retrieval(bs, device)

        scores = self._retrieve_page_scores(
            layer_id,
            plan.physical_pages,
            plan.req_pool_indices,
            queries,
        )

        score_mask = (
            plan.active_mask.unsqueeze(1)
            & plan.valid_page_mask
            & plan.history_page_mask
        )
        scores = torch.where(score_mask, scores, torch.full_like(scores, float("-inf")))

        if plan.max_k <= 0:
            return self._empty_retrieval(bs, device)

        topk_scores, topk_idx = torch.topk(
            scores,
            k=plan.max_k,
            dim=1,
            sorted=plan.score_order_required,
        )
        topk_idx = topk_idx.to(torch.long)
        topk_rank = torch.arange(plan.max_k, device=device, dtype=torch.long)
        topk_valid = (
            topk_rank.unsqueeze(0) < plan.k_per_req.unsqueeze(1)
        ) & torch.isfinite(topk_scores)

        combined_idx = torch.cat([topk_idx, plan.recent_idx], dim=1)
        combined_valid = torch.cat([topk_valid, plan.recent_valid], dim=1)

        return self._finalize_selected_pages(
            combined_idx, combined_valid, plan.max_num_pages
        )

    @staticmethod
    def _empty_retrieval(bs: int, device: torch.device) -> tuple:
        return (
            torch.full((bs, 1), -1, dtype=torch.int32, device=device),
            torch.zeros(bs, dtype=torch.int32, device=device),
        )

    @staticmethod
    def _finalize_selected_pages(
        combined_idx: torch.Tensor,
        combined_valid: torch.Tensor,
        sentinel: int,
    ) -> tuple:
        """Compact valid logical pages, sort them, and leave a -1 suffix."""
        sortable_idx = torch.where(
            combined_valid, combined_idx, torch.full_like(combined_idx, sentinel)
        )
        sorted_idx = torch.sort(sortable_idx, dim=1)[0].to(torch.int32)
        out_indices = torch.where(
            sorted_idx == sentinel,
            torch.full_like(sorted_idx, -1),
            sorted_idx,
        )
        out_lengths = combined_valid.sum(dim=1).to(torch.int32)
        return out_indices, out_lengths

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        """Initialize algorithm-specific representation pools for all layers."""
        raise NotImplementedError

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        """Compute and store page representations for given page range."""
        raise NotImplementedError

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve page scores for TopK selection."""
        raise NotImplementedError
