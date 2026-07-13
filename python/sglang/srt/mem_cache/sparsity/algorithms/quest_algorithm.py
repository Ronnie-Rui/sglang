"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
"""

import logging

import torch

from sglang.srt.arg_groups.hisparse_hook import (
    QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION,
    resolve_quest_page_bounds_dtype,
)
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class QuestAlgorithm(BaseSparseAlgorithmImpl):
    """Quest page-wise sparse attention using bounding-box criticality."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.use_triton_score_kernel = config.sparse_extra_config.get(
            "use_triton_score_kernel", True
        )
        self.use_fused_score_mask_kernel = config.sparse_extra_config.get(
            "use_fused_score_mask_kernel", True
        )
        self.enable_cuda_graph_retrieval = config.sparse_extra_config.get(
            "enable_cuda_graph_retrieval", True
        )
        self.use_jit_topk_kernel = config.sparse_extra_config.get(
            "use_jit_topk_kernel", True
        )
        self.use_triton_page_update_kernel = config.sparse_extra_config.get(
            "use_triton_page_update_kernel", True
        )
        self.use_direct_fa_metadata_kernel = config.sparse_extra_config.get(
            "use_direct_fa_metadata_kernel", True
        )
        self.use_fused_topk_fa_metadata_kernel = config.sparse_extra_config.get(
            "use_fused_topk_fa_metadata_kernel", False
        )
        self.use_lazy_page_update_score_kernel = config.sparse_extra_config.get(
            "use_lazy_page_update_score_kernel", False
        )
        self.use_native_page_bounds_dtype = config.sparse_extra_config.get(
            QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION, False
        )
        self.layer_selection_reuse_interval = config.sparse_extra_config.get(
            "layer_selection_reuse_interval", 1
        )
        self.layer_page_budget = tuple(
            (
                budget_range["start_layer"],
                budget_range["end_layer"],
                float(budget_range["scale"]),
            )
            for budget_range in config.sparse_extra_config.get("layer_page_budget", ())
        )
        self._selection_cache = None
        self._selection_cache_group = None
        self._selection_cache_layer = None
        self._actual_selection_anchors = set()
        self._metadata_length_updates = {}
        self._last_metadata_layer = None
        self._lazy_page_update_active = False
        self._lazy_page_update_graph_states = {}
        self.page_k_min = {}
        self.page_k_max = {}
        self.page_valid = {}

    def begin_forward(self, *args, **kwargs) -> None:
        # Selection reuse is only valid among consecutive layers of one decode
        # forward. The first local layer calls this before every eager run or
        # graph capture; graph replay executes the captured anchor kernels.
        self._selection_cache = None
        self._selection_cache_group = None
        self._selection_cache_layer = None
        self._actual_selection_anchors.clear()
        self._metadata_length_updates.clear()
        self._last_metadata_layer = None
        super().begin_forward(*args, **kwargs)
        self._lazy_page_update_active = self._can_enable_lazy_page_update()
        fixed_capacity = kwargs.get(
            "fixed_capacity", args[4] if len(args) > 4 else False
        )
        if isinstance(fixed_capacity, int) and not isinstance(fixed_capacity, bool):
            self._lazy_page_update_graph_states[int(fixed_capacity)] = (
                self._lazy_page_update_active,
                self._retrieval_plan.max_num_pages,
            )

    def _get_lazy_page_update_state(self, forward_batch=None) -> tuple[bool, int]:
        capacity = getattr(forward_batch, "runtime_sparse_page_capacity", None)
        if isinstance(capacity, int) and not isinstance(capacity, bool):
            graph_state = self._lazy_page_update_graph_states.get(capacity)
            if graph_state is not None:
                return graph_state

        plan = self._retrieval_plan
        return (
            self._lazy_page_update_active,
            plan.max_num_pages if plan is not None else 0,
        )

    def should_finalize_graph_forward(self, forward_batch) -> bool:
        capacity = getattr(forward_batch, "runtime_sparse_page_capacity", None)
        if isinstance(capacity, int) and not isinstance(capacity, bool):
            return capacity in self._lazy_page_update_graph_states
        return False

    def _get_layer_budget_scale(self, layer_id: int) -> float:
        for start_layer, end_layer, scale in self.layer_page_budget:
            if start_layer <= layer_id < end_layer:
                return scale
        return 1.0

    def get_layer_sparsity_ratio(self, layer_id: int) -> float:
        return self.sparsity_ratio * self._get_layer_budget_scale(layer_id)

    def _selection_group(self, layer_id: int) -> tuple[int, float]:
        local_layer_id = layer_id - self.start_layer
        return (
            local_layer_id // self.layer_selection_reuse_interval,
            self._get_layer_budget_scale(layer_id),
        )

    def _is_selection_anchor(self, layer_id: int) -> bool:
        return layer_id == self.start_layer or self._selection_group(
            layer_id
        ) != self._selection_group(layer_id - 1)

    def _is_actual_selection_anchor(self, layer_id: int) -> bool:
        # Direct representation-update callers do not pass through retrieval.
        # Preserve their static-anchor behavior while making live forwards use
        # the layers that really recomputed selection.
        return (
            layer_id in self._actual_selection_anchors
            if self._actual_selection_anchors
            else self._is_selection_anchor(layer_id)
        )

    def _can_enable_lazy_page_update(self) -> bool:
        plan = self._retrieval_plan
        if (
            not self.use_lazy_page_update_score_kernel
            or not self.use_triton_score_kernel
            or plan is None
            or plan.max_num_pages <= self.num_recent_pages
            or torch.device(self.device).type != "cuda"
            or torch.version.hip is not None
            or self.req_to_token_pool is None
            or self.token_to_kv_pool is None
            or self.states is None
            or self.page_size <= 0
            or self.page_size > 32
        ):
            return False

        key_buffer = self.token_to_kv_pool.get_key_buffer(self.start_layer)
        if (
            key_buffer.ndim != 3
            or key_buffer.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or key_buffer.shape[0] <= 0
            or key_buffer.shape[1] <= 0
            or key_buffer.shape[2] <= 0
            or key_buffer.shape[2] > 256
        ):
            return False

        tensors = (
            self.req_to_token_pool.req_to_token,
            self.states.repr_constructed,
            self.states.last_constructed_page,
            self.page_k_min[self.start_layer],
            self.page_k_max[self.start_layer],
            self.page_valid[self.start_layer],
        )
        return all(tensor.device == key_buffer.device for tensor in tensors)

    def should_update_metadata_lengths(self, layer_id: int) -> bool:
        cached = self._metadata_length_updates.get(layer_id)
        if cached is not None:
            return cached

        previous_layer = self._last_metadata_layer
        update_lengths = previous_layer is None or self._get_layer_budget_scale(
            layer_id
        ) != self._get_layer_budget_scale(previous_layer)
        self._metadata_length_updates[layer_id] = update_lengths
        self._last_metadata_layer = layer_id
        return update_lengths

    def should_update_representations(self, forward_batch) -> bool:
        # Lazy retrieval updates each actual selection anchor while scoring.
        # Keep graph forward_end active so it reaches the shared finalizer.
        if self._get_lazy_page_update_state(forward_batch)[0]:
            return True
        return super().should_update_representations(forward_batch)

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        # The coordinator asks before retrieval while fused metadata paths ask
        # again inside retrieval. Cache one decision for both call sites.
        self.should_update_metadata_lengths(layer_id)
        group = self._selection_group(layer_id)
        previous_layer = self._selection_cache_layer
        layer_order_is_contiguous = (
            previous_layer is None or layer_id == previous_layer + 1
        )
        can_reuse = (
            self._selection_cache is not None
            and self._selection_cache_group == group
            and self._selection_cache_layer is not None
            and layer_order_is_contiguous
            and layer_id != self.start_layer
        )
        if can_reuse:
            self._selection_cache_layer = layer_id
            selected_indices, valid_lengths = self._selection_cache
            # The anchor layer already rewrote the shared FA metadata. Marking
            # it prepared avoids remapping or rewriting the same pages.
            return selected_indices, valid_lengths, True

        self._actual_selection_anchors.add(layer_id)
        result = super().retrieve_topk(
            queries,
            layer_id,
            req_pool_indices,
            sparse_mask,
            **kwargs,
        )
        self._selection_cache = result[:2]
        self._selection_cache_group = group
        self._selection_cache_layer = layer_id
        return result

    def _can_use_triton_page_update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
    ) -> bool:
        if (
            not self.use_triton_page_update_kernel
            or not k_buffer.is_cuda
            or torch.version.hip is not None
            or k_buffer.ndim != 3
            or k_buffer.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or self.page_size <= 0
            or self.page_size > 128
            or k_buffer.shape[-1] <= 0
            or k_buffer.shape[-1] > 256
        ):
            return False

        tensors = (
            req_pool_indices,
            seq_lens,
            self.req_to_token_pool.req_to_token,
            self.states.repr_constructed,
            self.states.last_constructed_page,
        )
        return all(tensor.device == k_buffer.device for tensor in tensors)

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        if not forward_batch.forward_mode.is_decode():
            return
        lazy_page_update_active, _ = self._get_lazy_page_update_state(forward_batch)
        if lazy_page_update_active:
            return
        if not self.should_update_representations(forward_batch):
            return

        if not self._is_actual_selection_anchor(layer_id):
            return

        if not self._can_use_triton_page_update(req_pool_indices, seq_lens, k_buffer):
            return self._update_representations_without_tracker_advance(
                layer_id,
                req_pool_indices,
                seq_lens,
                k_buffer,
                forward_batch,
            )

        # Reused layers never consume their own decode-time Quest bounds. Every
        # actual anchor writes against the same pre-forward tracker snapshot;
        # finalize_forward advances it only after all anchors have completed.
        from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
            quest_update_page_representations_,
        )

        quest_update_page_representations_(
            req_pool_indices,
            seq_lens,
            self.req_to_token_pool.req_to_token,
            k_buffer,
            self.states.repr_constructed,
            self.states.last_constructed_page,
            self.page_k_min[layer_id],
            self.page_k_max[layer_id],
            self.page_valid[layer_id],
            self.page_size,
            advance_trackers=False,
        )

    def _update_representations_without_tracker_advance(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Portable representation update that leaves shared trackers unchanged."""
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

        self._compute_page_representations(
            layer_id,
            req_pool_indices[valid_mask],
            seq_lens[valid_mask],
            start_page[valid_mask],
            end_page[valid_mask],
            k_buffer,
        )

    def finalize_forward(self, forward_batch) -> None:
        if not forward_batch.forward_mode.is_decode():
            return

        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if req_pool_indices is None or seq_lens is None:
            return

        lazy_page_update_active, lazy_max_pages = self._get_lazy_page_update_state(
            forward_batch
        )
        if lazy_page_update_active:
            if lazy_max_pages <= 0:
                return
            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_advance_lazy_page_trackers_,
            )

            quest_advance_lazy_page_trackers_(
                req_pool_indices,
                seq_lens,
                self.states.repr_constructed,
                self.states.last_constructed_page,
                self.page_size,
                max_pages=lazy_max_pages,
            )
            return

        if not self.should_update_representations(forward_batch):
            return

        if req_pool_indices.is_cuda and torch.version.hip is None:
            from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
                quest_advance_page_trackers_,
            )

            quest_advance_page_trackers_(
                req_pool_indices,
                seq_lens,
                self.states.repr_constructed,
                self.states.last_constructed_page,
                self.page_size,
            )
            return

        end_page = seq_lens // self.page_size
        constructed = self.states.repr_constructed[req_pool_indices]
        last_page = self.states.last_constructed_page[req_pool_indices]
        start_page = torch.where(constructed, last_page, torch.zeros_like(end_page))
        update_mask = start_page < end_page
        self.states.repr_constructed[req_pool_indices] = constructed | update_mask
        self.states.last_constructed_page[req_pool_indices] = torch.where(
            update_mask, end_page, last_page
        )

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]
        bounds_dtype = resolve_quest_page_bounds_dtype(
            key_buf.dtype, self.use_native_page_bounds_dtype
        )

        for layer_id in range(start_layer, end_layer):
            self.page_k_min[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=bounds_dtype,
                device=self.device,
            )
            self.page_k_max[layer_id] = torch.zeros_like(self.page_k_min[layer_id])
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized Quest page reps: %d pages, %d layers, head_num=%d, "
            "head_dim=%d, bounds_dtype=%s",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
            bounds_dtype,
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        keys = k_buffer[phys_tok].to(self.page_k_min[layer_id].dtype)

        if bool((end_page * self.page_size <= seq_lens).all().item()):
            page_min = keys.amin(dim=2)
            page_max = keys.amax(dim=2)
        else:
            tok_mask = (
                tok_pos
                < (tok_start + self.page_size)
                .clamp(max=seq_lens.unsqueeze(1))
                .unsqueeze(2)
            ) & pg_mask.unsqueeze(2)
            mask = tok_mask.unsqueeze(-1).unsqueeze(-1)
            page_min = torch.where(
                mask, keys, torch.full_like(keys, float("inf"))
            ).amin(dim=2)
            page_max = torch.where(
                mask, keys, torch.full_like(keys, float("-inf"))
            ).amax(dim=2)

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_k_min[layer_id].shape[0] - 1
        )
        self.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
        self.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

    def _can_use_triton_score_kernel(self, queries: torch.Tensor) -> bool:
        return (
            self.use_triton_score_kernel
            and queries.is_cuda
            and torch.version.hip is None
            and self.page_k_min
            and next(iter(self.page_k_min.values())).shape[-1] <= 256
        )

    def _retrieve_page_scores_batched(self, layer_id, queries, plan) -> torch.Tensor:
        if getattr(
            self, "_lazy_page_update_active", False
        ) and self._can_use_triton_score_kernel(queries):
            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_lazy_update_page_scores,
            )

            return quest_lazy_update_page_scores(
                queries=queries,
                page_k_min=self.page_k_min[layer_id],
                page_k_max=self.page_k_max[layer_id],
                page_valid=self.page_valid[layer_id],
                physical_pages=plan.physical_pages,
                req_pool_indices=plan.req_pool_indices,
                seq_lens=plan.seq_lens,
                req_to_token=self.req_to_token_pool.req_to_token,
                k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
                repr_constructed=self.states.repr_constructed,
                last_constructed_page=self.states.last_constructed_page,
                page_size=self.page_size,
                active_mask=plan.active_mask,
                history_page_counts=plan.recent_start,
                advance_trackers=False,
            )

        if self.use_fused_score_mask_kernel and self._can_use_triton_score_kernel(
            queries
        ):
            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_page_scores,
            )

            return quest_page_scores(
                queries,
                self.page_k_min[layer_id],
                self.page_k_max[layer_id],
                self.page_valid[layer_id],
                plan.physical_pages,
                active_mask=plan.active_mask,
                history_page_counts=plan.recent_start,
            )
        return super()._retrieve_page_scores_batched(layer_id, queries, plan)

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        physical_pages = phys_pages

        if getattr(
            self, "_lazy_page_update_active", False
        ) and self._can_use_triton_score_kernel(queries):
            plan = self._retrieval_plan
            if plan is None:
                raise RuntimeError("Quest lazy page update requires a retrieval plan")

            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_lazy_update_page_scores,
            )

            return quest_lazy_update_page_scores(
                queries=queries,
                page_k_min=self.page_k_min[layer_id],
                page_k_max=self.page_k_max[layer_id],
                page_valid=self.page_valid[layer_id],
                physical_pages=physical_pages,
                req_pool_indices=req_pool_indices,
                seq_lens=plan.seq_lens,
                req_to_token=self.req_to_token_pool.req_to_token,
                k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
                repr_constructed=self.states.repr_constructed,
                last_constructed_page=self.states.last_constructed_page,
                page_size=self.page_size,
                advance_trackers=False,
            )

        if self._can_use_triton_score_kernel(queries):
            from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
                quest_page_scores,
            )

            return quest_page_scores(
                queries,
                self.page_k_min[layer_id],
                self.page_k_max[layer_id],
                self.page_valid[layer_id],
                physical_pages,
            )

        # Clamp pages only for the portable torch fallback. The Triton kernel
        # handles invalid physical page ids without this allocation.
        phys_pages_clamped = phys_pages.clamp(0, self.page_k_min[layer_id].shape[0] - 1)
        k_min = self.page_k_min[layer_id][phys_pages_clamped].to(torch.float32)
        k_max = self.page_k_max[layer_id][phys_pages_clamped].to(torch.float32)
        valid_mask = self.page_valid[layer_id][phys_pages_clamped] & (
            (physical_pages >= 0)
            & (physical_pages < self.page_k_min[layer_id].shape[0])
        )
        # Align query shape to KV heads.
        head_dim = k_min.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Quest query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.reshape(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

        kv_heads = k_min.shape[-2]
        q_heads = q.shape[1]
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
            )

        # FlashAttention uses one page table per request, shared by all query
        # heads. Compute each head's bounding-box upper bound independently,
        # then use the largest bound as the conservative shared-page score.
        group = q_heads // kv_heads
        q = q.view(q.shape[0], kv_heads, group, head_dim)
        q = q.to(torch.float32).unsqueeze(1)
        k_min = k_min.unsqueeze(3)
        k_max = k_max.unsqueeze(3)
        per_head_bound = torch.where(q >= 0, q * k_max, q * k_min).sum(dim=-1)
        criticality = per_head_bound.amax(dim=(2, 3))
        criticality = torch.where(
            valid_mask, criticality, torch.full_like(criticality, float("-inf"))
        )

        return criticality
