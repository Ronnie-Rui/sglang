"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection. For each KV page, it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.
"""

import logging
from dataclasses import dataclass

import torch

from sglang.srt.arg_groups.hisparse_hook import (
    QUEST_DECODE_TOKEN_SELECTION_REUSE_INTERVAL_OPTION,
    QUEST_MAX_SELECTED_TOKENS_OPTION,
    QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION,
    resolve_quest_page_bounds_dtype,
)
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


@dataclass
class _DecodeSelectionCacheState:
    request_signature: tuple
    sequence_lengths: tuple[int, ...]
    page_counts: tuple[int, ...]
    selections: dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor]]
    age: int


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
        quest_max_selected_tokens = config.sparse_extra_config.get(
            QUEST_MAX_SELECTED_TOKENS_OPTION
        )
        self.quest_max_selected_pages = (
            None
            if quest_max_selected_tokens is None
            else quest_max_selected_tokens // self.page_size
        )
        self.layer_selection_reuse_interval = config.sparse_extra_config.get(
            "layer_selection_reuse_interval", 1
        )
        self.decode_token_selection_reuse_interval = config.sparse_extra_config.get(
            QUEST_DECODE_TOKEN_SELECTION_REUSE_INTERVAL_OPTION, 1
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
        self._decode_selection_cache_state = None
        self._pending_decode_selection_cache_state = None
        self._decode_selection_cache_mode = None
        self._decode_selection_cache_touched = False
        self._decode_selection_graph_bypass_logged = False
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
        self._discard_pending_decode_selection_cache()
        super().begin_forward(*args, **kwargs)
        self._lazy_page_update_active = self._can_enable_lazy_page_update()
        forward_batch = kwargs.get("forward_batch", args[0] if args else None)
        fixed_capacity = kwargs.get(
            "fixed_capacity", args[4] if len(args) > 4 else False
        )
        if isinstance(fixed_capacity, int) and not isinstance(fixed_capacity, bool):
            self._lazy_page_update_graph_states[int(fixed_capacity)] = (
                self._lazy_page_update_active,
                self._retrieval_plan.max_num_pages,
            )
        self._prepare_decode_selection_cache(forward_batch, fixed_capacity)

    @staticmethod
    def _host_int_tuple(value, expected_size: int) -> tuple[int, ...] | None:
        if torch.is_tensor(value):
            if value.device.type != "cpu" or value.numel() != expected_size:
                return None
            values = value.reshape(-1).tolist()
        else:
            try:
                values = list(value)
            except TypeError:
                return None
            if len(values) != expected_size:
                return None
        return tuple(int(item) for item in values)

    def _decode_selection_signatures(self, forward_batch, fixed_capacity):
        if (
            self.decode_token_selection_reuse_interval <= 1
            or forward_batch is None
            or fixed_capacity is not False
            or getattr(forward_batch, "spec_info", None) is not None
            or getattr(forward_batch, "runtime_sparse_page_capacity", None) is not None
        ):
            return None

        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_decode = getattr(forward_mode, "is_decode", None)
        if not callable(is_decode) or not is_decode():
            return None

        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            is_in_breakable_cuda_graph,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            is_in_tc_piecewise_cuda_graph,
        )

        if is_in_breakable_cuda_graph() or is_in_tc_piecewise_cuda_graph():
            if not self._decode_selection_graph_bypass_logged:
                logger.info(
                    "Quest decode-token selection reuse is disabled inside CUDA Graph "
                    "execution."
                )
                self._decode_selection_graph_bypass_logged = True
            return None
        if (
            torch.device(self.device).type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            return None

        plan = self._retrieval_plan
        if plan is None or plan.fixed_capacity:
            return None
        request_slots = self._host_int_tuple(
            getattr(forward_batch, "req_pool_indices_cpu", None), plan.batch_size
        )
        sequence_lengths = self._host_int_tuple(plan.seq_lens_cpu, plan.batch_size)
        page_counts = self._host_int_tuple(plan.num_pages_cpu, plan.batch_size)
        rids = getattr(forward_batch, "rids", None)
        if (
            request_slots is None
            or sequence_lengths is None
            or page_counts is None
            or not isinstance(rids, (list, tuple))
            or len(rids) != plan.batch_size
        ):
            return None
        request_ids = tuple(rids)
        try:
            hash(request_ids)
        except TypeError:
            return None
        return (request_ids, request_slots), sequence_lengths, page_counts

    def _prepare_decode_selection_cache(self, forward_batch, fixed_capacity) -> None:
        signatures = self._decode_selection_signatures(forward_batch, fixed_capacity)
        if signatures is None:
            self._invalidate_decode_selection_cache()
            return

        request_signature, sequence_lengths, page_counts = signatures
        committed = self._decode_selection_cache_state
        sequential = (
            committed is not None
            and len(committed.sequence_lengths) == len(sequence_lengths)
            and all(
                current == previous + 1
                for current, previous in zip(
                    sequence_lengths, committed.sequence_lengths
                )
            )
        )
        completed_page = any(
            seq_len > 0 and seq_len % self.page_size == 0
            for seq_len in sequence_lengths
        )
        can_reuse = (
            committed is not None
            and bool(committed.selections)
            and committed.request_signature == request_signature
            and committed.page_counts == page_counts
            and sequential
            and not completed_page
            and committed.age < self.decode_token_selection_reuse_interval - 1
        )
        self._decode_selection_cache_mode = "reuse" if can_reuse else "refresh"
        self._decode_selection_cache_touched = False
        self._pending_decode_selection_cache_state = _DecodeSelectionCacheState(
            request_signature=request_signature,
            sequence_lengths=sequence_lengths,
            page_counts=page_counts,
            selections=(dict(committed.selections) if can_reuse else {}),
            age=committed.age + 1 if can_reuse else 0,
        )

    def _discard_pending_decode_selection_cache(self) -> None:
        self._pending_decode_selection_cache_state = None
        self._decode_selection_cache_mode = None
        self._decode_selection_cache_touched = False

    def _invalidate_decode_selection_cache(self) -> None:
        self._decode_selection_cache_state = None
        self._discard_pending_decode_selection_cache()

    def _commit_pending_decode_selection_cache(self) -> None:
        pending = self._pending_decode_selection_cache_state
        if (
            pending is not None
            and self._decode_selection_cache_touched
            and pending.selections
        ):
            self._decode_selection_cache_state = pending
        else:
            self._decode_selection_cache_state = None
        self._discard_pending_decode_selection_cache()

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

    def get_history_page_selection_cap(self) -> int | None:
        if self.quest_max_selected_pages is None:
            return None
        return self.quest_max_selected_pages - self.num_recent_pages

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

        pending = self._pending_decode_selection_cache_state
        cached_selection = (
            pending.selections.get(group)
            if pending is not None
            and self._decode_selection_cache_mode == "reuse"
            and self._is_selection_anchor(layer_id)
            and layer_order_is_contiguous
            else None
        )
        if cached_selection is not None:
            self._actual_selection_anchors.add(layer_id)
            self._decode_selection_cache_touched = True
            self._selection_cache = cached_selection
            self._selection_cache_group = group
            self._selection_cache_layer = layer_id
            selected_physical_pages, valid_lengths = cached_selection
            # Cross-token hits must rebuild metadata for the current token.
            # The explicit physical-pages field prevents a second logical map.
            return (
                selected_physical_pages,
                valid_lengths,
                False,
                selected_physical_pages,
            )

        if pending is not None and self._decode_selection_cache_mode == "reuse":
            # A previously unseen/non-static group makes the remainder of this
            # forward a refresh. Publish only newly owned results at finalize.
            pending.selections.clear()
            pending.age = 0
            self._decode_selection_cache_mode = "refresh"

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
        pending = self._pending_decode_selection_cache_state
        if pending is not None:
            metadata_prepared = len(result) >= 3 and bool(result[2])
            if metadata_prepared:
                selected_physical_pages = result[0].detach().clone()
            else:
                selected_physical_pages = self.get_selected_physical_pages(result[0])
                if selected_physical_pages is not None:
                    selected_physical_pages = selected_physical_pages.detach().clone()
            if selected_physical_pages is not None:
                pending.selections[group] = (
                    selected_physical_pages,
                    result[1].detach().clone(),
                )
                self._decode_selection_cache_touched = True
        return result

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        if forward_batch.forward_mode.is_extend():
            self._invalidate_decode_selection_cache()
        return super().construct_representations(
            layer_id,
            req_pool_indices,
            seq_lens,
            k_buffer,
            forward_batch,
        )

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
            self._invalidate_decode_selection_cache()
            return

        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if req_pool_indices is None or seq_lens is None:
            self._discard_pending_decode_selection_cache()
            return

        try:
            self._finalize_representation_trackers(
                forward_batch, req_pool_indices, seq_lens
            )
        except Exception:
            self._discard_pending_decode_selection_cache()
            raise
        self._commit_pending_decode_selection_cache()

    def _finalize_representation_trackers(
        self, forward_batch, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
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
