from abc import ABC, abstractmethod
from ctypes import c_float
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _float32_scaled_count(count: int, ratio: float) -> int:
    """Match the device float32 multiply followed by integer truncation."""
    count_f32 = c_float(count).value
    ratio_f32 = c_float(ratio).value
    return int(c_float(count_f32 * ratio_f32).value)


def _selection_count(
    history_count: int, ratio: float, history_page_cap: int | None = None
) -> int:
    count = min(
        max(_float32_scaled_count(history_count, ratio), 1),
        history_count,
    )
    return min(count, history_page_cap) if history_page_cap is not None else count


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
    sparse_mask_cpu: list[bool] | None
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
    recent_idx_i32: torch.Tensor | None
    recent_valid: torch.Tensor
    fixed_capacity: bool


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
        fixed_capacity: bool | int = False,
    ) -> None:
        """Prepare state shared by every sparse layer in one forward."""

    def begin_dense_forward(self, forward_batch: "ForwardBatch") -> None:
        """Prepare representation tracking without building a retrieval plan."""

    def prepare_graph_forward(self) -> None:
        """Prepare algorithm lifecycle state before graph replay."""

    def should_update_representations(self, forward_batch: "ForwardBatch") -> bool:
        return True

    def finalize_forward(self, forward_batch: "ForwardBatch") -> None:
        """Finalize representation trackers after all layers have run."""

    def should_finalize_graph_forward(self, forward_batch: "ForwardBatch") -> bool:
        return False

    def get_layer_sparsity_ratio(self, layer_id: int) -> float:
        return self.sparsity_ratio

    def get_history_page_selection_cap(self) -> int | None:
        return None

    def should_update_metadata_lengths(self, layer_id: int) -> bool:
        return layer_id == getattr(self, "start_layer", layer_id)

    def get_selected_physical_pages(
        self, selected_indices: torch.Tensor
    ) -> torch.Tensor | None:
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
            - Indices are logical positions that will be mapped to physical KV cache by BackendAdaptor

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
        self._retrieval_plans_by_ratio: dict[float, _RetrievalPlan] = {}
        self._representation_update_batch = None
        self._representation_update_due: bool | None = None

    def begin_forward(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
        fixed_capacity: bool | int = False,
    ) -> None:
        self._retrieval_plans_by_ratio.clear()
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
            fixed_capacity=fixed_capacity,
        )
        self._retrieval_plans_by_ratio[self.sparsity_ratio] = self._retrieval_plan
        self._representation_update_due = self._has_completed_page(
            self._retrieval_plan.seq_lens_cpu
        )

    def begin_dense_forward(self, forward_batch: "ForwardBatch") -> None:
        self._retrieval_plan = None
        self._retrieval_plans_by_ratio.clear()
        self._representation_update_batch = forward_batch
        self._representation_update_due = self._decode_page_boundary_reached(
            forward_batch
        )

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
        """Select important history pages and retain the recent window."""
        bs, device = queries.shape[0], queries.device

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for TopK retrieval"
            )
        plan = self._retrieval_plan
        if (
            bs == 1
            and self._get_num_pages_cpu(seq_lens_source, 1) is not None
            and not getattr(self, "use_lazy_page_update_score_kernel", False)
            and (plan is None or not plan.fixed_capacity)
        ):
            return self._retrieve_topk_single(
                queries,
                layer_id,
                req_pool_indices,
                sparse_mask,
                seq_lens_source,
            )

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
            self._retrieval_plans_by_ratio = {self.sparsity_ratio: plan}
        plan = self._get_retrieval_plan_for_ratio(
            plan, self.get_layer_sparsity_ratio(layer_id)
        )
        if plan.max_num_pages <= self.num_recent_pages or plan.max_k <= 0:
            return self._empty_retrieval(bs, device)

        scores = self._retrieve_page_scores_batched(layer_id, queries, plan)
        topk_scores, topk_idx = torch.topk(
            scores,
            k=plan.max_k,
            dim=1,
            sorted=plan.score_order_required,
        )

        direct_result = self._try_finalize_to_flashattention_metadata(
            topk_scores,
            topk_idx,
            plan,
            kwargs.get("attn_metadata"),
            layer_id,
        )
        if direct_result is not None:
            return direct_result
        return self._finalize_topk_with_recent(topk_scores, topk_idx, plan)

    def _try_finalize_to_flashattention_metadata(
        self,
        topk_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        plan: _RetrievalPlan,
        attn_metadata,
        layer_id: int,
    ) -> tuple | None:
        if (
            not getattr(self, "use_direct_fa_metadata_kernel", False)
            or attn_metadata is None
            or not plan.fixed_capacity
            or not topk_scores.is_cuda
            or torch.version.hip is not None
            or topk_scores.dtype != torch.float32
            or topk_idx.dtype not in (torch.int32, torch.int64)
            or plan.k_per_req.dtype not in (torch.int32, torch.int64)
        ):
            return None

        required_attrs = ("page_table", "cache_seqlens_int32", "cu_seqlens_k")
        if not all(hasattr(attn_metadata, attr) for attr in required_attrs):
            return None

        recent_indices = (
            plan.recent_idx_i32 if plan.recent_idx_i32 is not None else plan.recent_idx
        )
        combined_width = topk_scores.shape[1] + recent_indices.shape[1]
        from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
            QUEST_DIRECT_METADATA_MAX_WIDTH,
            quest_finalize_to_flashattention_metadata_,
        )

        if (
            combined_width > QUEST_DIRECT_METADATA_MAX_WIDTH
            or attn_metadata.page_table.shape[0] != plan.batch_size
            or attn_metadata.page_table.shape[1] < combined_width
            or not attn_metadata.page_table.is_cuda
            or not attn_metadata.cache_seqlens_int32.is_cuda
            or not attn_metadata.cu_seqlens_k.is_cuda
        ):
            return None

        valid_lengths = torch.empty(
            plan.batch_size, dtype=torch.int32, device=topk_scores.device
        )
        quest_finalize_to_flashattention_metadata_(
            topk_scores=topk_scores,
            topk_indices=topk_idx,
            k_per_req=plan.k_per_req,
            recent_indices=recent_indices,
            recent_valid=plan.recent_valid,
            valid_lengths=valid_lengths,
            sparse_mask=plan.sparse_mask,
            seq_lens=plan.seq_lens,
            req_pool_indices=plan.req_pool_indices,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_table=attn_metadata.page_table,
            cache_seqlens_int32=attn_metadata.cache_seqlens_int32,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            page_size=self.page_size,
            update_lengths=self.should_update_metadata_lengths(layer_id),
        )
        return attn_metadata.page_table[:, :combined_width], valid_lengths, True

    def _finalize_topk_with_recent(
        self,
        topk_scores: torch.Tensor,
        topk_idx: torch.Tensor,
        plan: _RetrievalPlan,
    ) -> tuple:
        combined_width = topk_scores.shape[1] + plan.recent_idx.shape[1]
        if plan.fixed_capacity and topk_scores.is_cuda and torch.version.hip is None:
            from sglang.srt.mem_cache.sparsity.kernels.quest_finalize import (
                QUEST_FINALIZE_MAX_WIDTH,
                quest_finalize_selected_pages,
            )

            if combined_width <= QUEST_FINALIZE_MAX_WIDTH:
                return quest_finalize_selected_pages(
                    topk_scores,
                    topk_idx,
                    plan.k_per_req,
                    plan.recent_idx,
                    plan.recent_valid,
                )

        device = topk_scores.device
        topk_idx = topk_idx.to(torch.long)
        topk_rank = torch.arange(plan.max_k, device=device, dtype=torch.long)
        topk_valid = (
            topk_rank.unsqueeze(0) < plan.k_per_req.unsqueeze(1)
        ) & torch.isfinite(topk_scores)
        return self._finalize_selected_pages(
            torch.cat([topk_idx, plan.recent_idx], dim=1),
            torch.cat([topk_valid, plan.recent_valid], dim=1),
            plan.max_num_pages,
        )

    def _retrieve_topk_single(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        forward_batch: "ForwardBatch",
    ) -> tuple:
        """Keep the low-overhead historical path for single-request decode."""
        device = queries.device
        sparse_mask_cpu = self._get_bool_mask_cpu(
            sparse_mask, 1, forward_batch=forward_batch
        )
        if sparse_mask_cpu is not None and not sparse_mask_cpu[0]:
            return self._empty_retrieval(1, device)

        num_pages_cpu = self._get_num_pages_cpu(forward_batch, 1)
        assert num_pages_cpu is not None
        num_pages = num_pages_cpu[0]
        if num_pages <= self.num_recent_pages:
            return self._empty_retrieval(1, device)

        req_to_token = self.req_to_token_pool.req_to_token
        page_idx = torch.arange(num_pages, device=device, dtype=torch.long)
        page_starts = (page_idx * self.page_size).clamp(0, req_to_token.shape[1] - 1)
        single_req_pool_indices = req_pool_indices.to(device=device, dtype=torch.long)
        physical_pages = (
            req_to_token[
                single_req_pool_indices,
                page_starts.unsqueeze(0),
            ].to(torch.long)
            // self.page_size
        )
        scores = self._retrieve_page_scores(
            layer_id,
            physical_pages,
            single_req_pool_indices,
            queries,
        )

        recent_start = num_pages - self.num_recent_pages
        history_pages = max(recent_start, 1)
        k = _selection_count(
            history_pages,
            self.get_layer_sparsity_ratio(layer_id),
            self.get_history_page_selection_cap(),
        )
        topk_idx = torch.topk(
            scores[:, :recent_start], k=k, dim=1, sorted=False
        ).indices.squeeze(0)
        recent_idx = torch.arange(
            recent_start, num_pages, device=device, dtype=torch.long
        )
        selected = torch.cat([topk_idx, recent_idx]).sort()[0].to(torch.int32)
        active_mask = sparse_mask.to(device=device, dtype=torch.bool).reshape(1)
        out_indices = torch.where(
            active_mask.unsqueeze(1),
            selected.unsqueeze(0),
            torch.full((1, selected.numel()), -1, dtype=torch.int32, device=device),
        )
        lengths = torch.where(
            active_mask,
            torch.full((1,), selected.numel(), dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )
        return out_indices, lengths

    def _build_retrieval_plan(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        device: torch.device,
        *,
        fixed_capacity: bool | int = False,
    ) -> _RetrievalPlan:
        """Build page mapping and selection widths once per forward."""
        batch_size = req_pool_indices.numel()
        seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
        seq_lens_cpu = (
            None
            if fixed_capacity
            else self._get_seq_lens_cpu(forward_batch, batch_size)
        )
        req_pool_indices = req_pool_indices.to(device=device, dtype=torch.long)
        sparse_mask_source = sparse_mask
        sparse_mask = sparse_mask.to(device=device, dtype=torch.bool)
        num_pages = (seq_lens + self.page_size - 1) // self.page_size

        num_pages_cpu = (
            [
                max((seq_len + self.page_size - 1) // self.page_size, 0)
                for seq_len in seq_lens_cpu
            ]
            if seq_lens_cpu is not None
            else None
        )
        sparse_mask_cpu = (
            self._get_bool_mask_cpu(
                sparse_mask_source,
                batch_size,
                forward_batch=forward_batch,
            )
            if num_pages_cpu is not None
            else None
        )
        if fixed_capacity:
            max_context_len = self.req_to_token_pool.max_context_len
            pool_max_pages = getattr(
                self,
                "cuda_graph_max_num_pages",
                max(max_context_len // self.page_size, 1),
            )
            max_num_pages = (
                pool_max_pages
                if isinstance(fixed_capacity, bool)
                else min(fixed_capacity, pool_max_pages)
            )
        elif num_pages_cpu is not None:
            max_num_pages = max(num_pages_cpu, default=0)
        else:
            max_num_pages = int(num_pages.max().item()) if batch_size > 0 else 0
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
            physical_pages = torch.where(
                valid_page_mask,
                physical_pages,
                torch.full_like(physical_pages, -1),
            )
        else:
            physical_pages = torch.empty(
                (batch_size, 0), device=device, dtype=torch.long
            )

        active_mask = sparse_mask & (num_pages > self.num_recent_pages)
        recent_start = (num_pages - self.num_recent_pages).clamp(min=0)
        history_page_mask = page_idx.unsqueeze(0) < recent_start.unsqueeze(1)
        k_per_req, max_k, score_order_required = self._build_selection_budget(
            active_mask=active_mask,
            recent_start=recent_start,
            num_pages_cpu=num_pages_cpu,
            sparse_mask_cpu=sparse_mask_cpu,
            max_num_pages=max_num_pages,
            fixed_capacity=bool(fixed_capacity),
            ratio=self.sparsity_ratio,
        )

        recent_offsets = torch.arange(
            self.num_recent_pages, device=device, dtype=torch.long
        )
        recent_idx = recent_start.unsqueeze(1) + recent_offsets.unsqueeze(0)
        recent_idx_i32 = (
            recent_idx.to(torch.int32)
            if k_per_req.dtype == torch.int32 and bool(fixed_capacity)
            else None
        )
        recent_valid = active_mask.unsqueeze(1) & (recent_idx < num_pages.unsqueeze(1))
        return _RetrievalPlan(
            forward_batch=forward_batch,
            batch_size=batch_size,
            device=device,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            sparse_mask_cpu=sparse_mask_cpu,
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
            recent_idx_i32=recent_idx_i32,
            recent_valid=recent_valid,
            fixed_capacity=bool(fixed_capacity),
        )

    def _build_selection_budget(
        self,
        *,
        active_mask: torch.Tensor,
        recent_start: torch.Tensor,
        num_pages_cpu: list[int] | None,
        sparse_mask_cpu: list[bool] | None,
        max_num_pages: int,
        fixed_capacity: bool,
        ratio: float,
    ) -> tuple[torch.Tensor, int, bool]:
        history_page_cap = self.get_history_page_selection_cap()
        if history_page_cap is not None:
            history_page_cap = min(history_page_cap, max_num_pages)

        history_pages = recent_start.clamp(min=1)
        k_per_req = (history_pages.to(torch.float32) * ratio).to(torch.int32)
        k_per_req = torch.maximum(k_per_req, torch.ones_like(k_per_req))
        k_per_req = torch.minimum(k_per_req, history_pages.to(torch.int32))
        if history_page_cap is not None:
            k_per_req = torch.clamp(k_per_req, max=history_page_cap)
        k_per_req = torch.where(active_mask, k_per_req, torch.zeros_like(k_per_req))

        if fixed_capacity:
            history_capacity = max(max_num_pages - self.num_recent_pages, 0)
            max_k = (
                _selection_count(history_capacity, ratio, history_page_cap)
                if history_capacity > 0
                else 0
            )
            score_order_required = True
        elif num_pages_cpu is not None:
            k_per_req_cpu = []
            for row, count in enumerate(num_pages_cpu):
                is_sparse = (
                    sparse_mask_cpu[row] if sparse_mask_cpu is not None else True
                )
                history_count = max(count - self.num_recent_pages, 0)
                k_per_req_cpu.append(
                    _selection_count(history_count, ratio, history_page_cap)
                    if is_sparse and history_count > 0
                    else 0
                )
            max_k = max(k_per_req_cpu, default=0)
            positive_k = {value for value in k_per_req_cpu if value > 0}
            score_order_required = len(positive_k) > 1
        else:
            max_k = int(k_per_req.max().item()) if k_per_req.numel() > 0 else 0
            score_order_required = True

        return k_per_req, max_k, score_order_required

    def _get_retrieval_plan_for_ratio(
        self, plan: _RetrievalPlan, ratio: float
    ) -> _RetrievalPlan:
        if ratio == self.sparsity_ratio:
            return plan

        cached_plan = self._retrieval_plans_by_ratio.get(ratio)
        if cached_plan is not None:
            return cached_plan

        k_per_req, max_k, score_order_required = self._build_selection_budget(
            active_mask=plan.active_mask,
            recent_start=plan.recent_start,
            num_pages_cpu=plan.num_pages_cpu,
            sparse_mask_cpu=plan.sparse_mask_cpu,
            max_num_pages=plan.max_num_pages,
            fixed_capacity=plan.fixed_capacity,
            ratio=ratio,
        )
        cached_plan = replace(
            plan,
            k_per_req=k_per_req,
            max_k=max_k,
            score_order_required=score_order_required,
        )
        self._retrieval_plans_by_ratio[ratio] = cached_plan
        return cached_plan

    @staticmethod
    def _get_bool_mask_cpu(
        mask: torch.Tensor,
        batch_size: int,
        *,
        forward_batch=None,
    ) -> list[bool] | None:
        if not torch.is_tensor(mask) or mask.numel() != batch_size:
            raise ValueError("sparse_mask must contain one value per request")

        host_mask = getattr(forward_batch, "sparse_mask_cpu", None)
        if host_mask is None:
            if mask.device.type != "cpu":
                return None
            host_mask = mask

        if torch.is_tensor(host_mask):
            if host_mask.device.type != "cpu" or host_mask.numel() != batch_size:
                return None
            values = host_mask.detach().reshape(-1).tolist()
        else:
            try:
                values = list(host_mask)
            except TypeError:
                return None
            if len(values) != batch_size:
                return None
        return [bool(value) for value in values]

    def _get_num_pages_cpu(
        self, forward_batch: "ForwardBatch", batch_size: int
    ) -> list[int] | None:
        values = self._get_seq_lens_cpu(forward_batch, batch_size)
        if values is None:
            return None
        return [
            max((value + self.page_size - 1) // self.page_size, 0) for value in values
        ]

    @staticmethod
    def _get_seq_lens_cpu(forward_batch: "ForwardBatch", batch_size: int):
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            return None
        if torch.is_tensor(seq_lens_cpu):
            if seq_lens_cpu.device.type != "cpu" or seq_lens_cpu.numel() != batch_size:
                return None
            values = seq_lens_cpu.reshape(-1).tolist()
        else:
            try:
                values = list(seq_lens_cpu)
            except TypeError:
                return None
            if len(values) != batch_size:
                return None
        return [int(value) for value in values]

    def should_update_representations(self, forward_batch: "ForwardBatch") -> bool:
        if self._representation_update_batch is not forward_batch:
            self._representation_update_batch = forward_batch
            self._representation_update_due = self._decode_page_boundary_reached(
                forward_batch
            )
        return self._representation_update_due is not False

    def _decode_page_boundary_reached(
        self, forward_batch: "ForwardBatch"
    ) -> bool | None:
        """Skip only proven single-token decodes that cannot complete a page."""
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None:
            batch_size = seq_lens.numel()
        else:
            host_seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
            try:
                batch_size = (
                    host_seq_lens.numel()
                    if torch.is_tensor(host_seq_lens)
                    else len(host_seq_lens)
                )
            except TypeError:
                return None
        values = self._get_seq_lens_cpu(forward_batch, batch_size)
        if values is None:
            return None
        if self._may_process_multiple_decode_tokens(forward_batch, batch_size):
            return True
        return self._has_completed_page(values)

    def _has_completed_page(self, seq_lens_cpu: list[int] | None) -> bool | None:
        if seq_lens_cpu is None:
            return None
        return any(
            seq_len > 0 and seq_len % self.page_size == 0 for seq_len in seq_lens_cpu
        )

    @staticmethod
    def _may_process_multiple_decode_tokens(forward_batch, batch_size: int) -> bool:
        """Detect known multi-token/speculative layouts and keep the safe path."""
        if getattr(forward_batch, "spec_info", None) is not None:
            return True

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            try:
                if any(int(length) != 1 for length in extend_seq_lens_cpu):
                    return True
            except TypeError:
                return True

        num_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if num_tokens is not None and int(num_tokens) != batch_size:
            return True

        positions = getattr(forward_batch, "positions", None)
        if torch.is_tensor(positions) and positions.numel() != batch_size:
            return True

        # Upstream ForwardMode.DECODE is one token per request by contract.
        # Out-of-tree multi-token callers must expose one of the indicators
        # above, or omit seq_lens_cpu to force the conservative device check.
        return False

    def _retrieve_page_scores_batched(
        self, layer_id: int, queries: torch.Tensor, plan: _RetrievalPlan
    ) -> torch.Tensor:
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
        return torch.where(score_mask, scores, torch.full_like(scores, float("-inf")))

    @staticmethod
    def _empty_retrieval(batch_size: int, device: torch.device) -> tuple:
        return (
            torch.full((batch_size, 1), -1, dtype=torch.int32, device=device),
            torch.zeros(batch_size, dtype=torch.int32, device=device),
        )

    @staticmethod
    def _finalize_selected_pages(
        combined_idx: torch.Tensor,
        combined_valid: torch.Tensor,
        sentinel: int,
    ) -> tuple:
        sortable_idx = torch.where(
            combined_valid, combined_idx, torch.full_like(combined_idx, sentinel)
        )
        sorted_idx = torch.sort(sortable_idx, dim=1)[0].to(torch.int32)
        out_indices = torch.where(
            sorted_idx == sentinel,
            torch.full_like(sorted_idx, -1),
            sorted_idx,
        )
        return out_indices, combined_valid.sum(dim=1).to(torch.int32)

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
