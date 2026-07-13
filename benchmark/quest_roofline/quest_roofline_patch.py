"""Process-local Quest roofline ablations for SGLang benchmark servers.

This module is loaded through ``sitecustomize``.  It deliberately patches only
benchmark processes and keeps the production runtime free of experiment flags.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType

ENV_NAME = "SGLANG_QUEST_ROOFLINE_MODE"
FIXED_SPARSE = "fixed-sparse"
RETRIEVAL_DENSE = "retrieval-dense"
RUNTIME_DENSE = "runtime-dense"
SUPPORTED_MODES = (FIXED_SPARSE, RETRIEVAL_DENSE, RUNTIME_DENSE)

_QUEST_MODULE = "sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm"
_COORDINATOR_MODULE = "sglang.srt.mem_cache.sparsity.core.sparse_coordinator"
_ADAPTOR_MODULE = "sglang.srt.mem_cache.sparsity.backend.backend_adaptor"
_TARGET_MODULES = frozenset({_QUEST_MODULE, _COORDINATOR_MODULE, _ADAPTOR_MODULE})
_installed_mode: str | None = None
_marked_patches: set[tuple[str, str]] = set()
_runtime_markers: set[tuple[str, str]] = set()


def _mark_runtime(mode: str, component: str) -> None:
    marker = (mode, component)
    if marker in _runtime_markers:
        return
    _runtime_markers.add(marker)
    sys.stderr.write(f"[quest-roofline] active={mode} component={component}\n")
    sys.stderr.flush()


def _is_quest(algorithm) -> bool:
    cls = type(algorithm)
    return cls.__name__ == "QuestAlgorithm" and cls.__module__ == _QUEST_MODULE


def _host_seq_lens(algorithm, forward_batch, batch_size: int) -> list[int]:
    seq_lens_cpu = algorithm._get_seq_lens_cpu(forward_batch, batch_size)
    if seq_lens_cpu is not None:
        return seq_lens_cpu

    # This fallback is intentionally visible in profiles. Real SGLang decode
    # batches provide seq_lens_cpu, so a sync here indicates a broken benchmark
    # setup rather than an optimization hidden by the ablation.
    return [int(value) for value in forward_batch.seq_lens.detach().cpu().tolist()]


def build_fixed_page_plan(
    algorithm,
    queries,
    sparse_mask,
    forward_batch,
):
    """Return deterministic logical pages with Quest-equivalent page counts.

    The plan keeps the first ``k`` history pages plus all recent pages. Page
    identity is irrelevant to the compute roofline; matching Quest's selected
    width is what keeps the FA3 workload comparable.
    """
    import torch

    from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
        _float32_scaled_count,
    )

    batch_size = queries.shape[0]
    device = queries.device
    seq_lens = _host_seq_lens(algorithm, forward_batch, batch_size)

    page_counts = [
        max((seq_len + algorithm.page_size - 1) // algorithm.page_size, 0)
        for seq_len in seq_lens
    ]
    history_counts = [
        max(page_count - algorithm.num_recent_pages, 0) for page_count in page_counts
    ]
    history_kept_host = [
        (
            min(
                max(
                    _float32_scaled_count(history_count, algorithm.sparsity_ratio),
                    1,
                ),
                history_count,
            )
            if page_count > algorithm.num_recent_pages
            else 0
        )
        for page_count, history_count in zip(page_counts, history_counts)
    ]
    if batch_size == 1:
        history_kept = history_kept_host
    else:
        history_tensor = torch.tensor(history_counts, dtype=torch.float32)
        history_kept_device = (history_tensor * algorithm.sparsity_ratio).to(torch.long)
        history_kept_device.clamp_(min=1)
        history_kept_device = torch.minimum(
            history_kept_device, history_tensor.to(torch.long)
        )
        max_k = max(history_kept_host, default=0)
        history_kept = [
            min(int(kept), max_k) if page_count > algorithm.num_recent_pages else 0
            for kept, page_count in zip(history_kept_device.tolist(), page_counts)
        ]
    selected_counts = [
        (
            kept + algorithm.num_recent_pages
            if page_count > algorithm.num_recent_pages
            else 0
        )
        for page_count, kept in zip(page_counts, history_kept)
    ]
    width = max(max(selected_counts, default=0), 1)

    columns = torch.arange(width, dtype=torch.long, device=device).unsqueeze(0)
    kept = torch.tensor(history_kept, dtype=torch.long, device=device).unsqueeze(1)
    recent_starts = torch.tensor(
        history_counts, dtype=torch.long, device=device
    ).unsqueeze(1)
    counts = torch.tensor(selected_counts, dtype=torch.long, device=device).unsqueeze(1)

    logical_pages = torch.where(
        columns < kept,
        columns,
        recent_starts + (columns - kept),
    )
    active = sparse_mask.to(device=device, dtype=torch.bool).unsqueeze(1)
    valid = active & (columns < counts)
    selected_indices = torch.where(
        valid,
        logical_pages,
        torch.full_like(logical_pages, -1),
    ).to(torch.int32)
    valid_lengths = valid.sum(dim=1).to(torch.int32)
    return selected_indices, valid_lengths


def _fixed_retrieve_topk(
    self,
    queries,
    layer_id,
    req_pool_indices,
    sparse_mask,
    **kwargs,
):
    del req_pool_indices
    forward_batch = kwargs.get("forward_batch")
    if forward_batch is None:
        raise ValueError("fixed-sparse roofline mode requires forward_batch")
    return build_fixed_page_plan(self, queries, sparse_mask, forward_batch)


def _fixed_initialize_representation_pools(
    self, start_layer: int, end_layer: int, total_num_pages: int
):
    del start_layer, end_layer, total_num_pages
    # Fixed selection never reads Quest min/max representations. Avoiding the
    # allocation and construction is part of the attention-only roofline.
    self.page_k_min.clear()
    self.page_k_max.clear()
    self.page_valid.clear()


def _fixed_begin_forward(
    self,
    forward_batch,
    req_pool_indices,
    sparse_mask,
    device,
    fixed_capacity: bool | int = False,
):
    del forward_batch, req_pool_indices, sparse_mask, device, fixed_capacity
    # Newer Quest baselines build a reusable logical/physical page plan before
    # layer 0. The fixed roofline must bypass that retrieval preprocessing too,
    # including when production passes its CUDA graph page-bucket capacity.
    self._retrieval_plan = None


def _fixed_get_selected_physical_pages(self, selected_indices):
    del self, selected_indices
    # Let the adaptor map the fixed logical plan once at layer 0. Later layers
    # reuse that already-installed page table in _patch_coordinator_module.
    return None


def _patch_quest_module(module: ModuleType) -> None:
    cls = module.QuestAlgorithm
    if getattr(cls, "_quest_roofline_patched", False):
        return
    cls._initialize_representation_pools = _fixed_initialize_representation_pools
    cls.begin_forward = _fixed_begin_forward
    cls.get_selected_physical_pages = _fixed_get_selected_physical_pages
    cls.retrieve_topk = _fixed_retrieve_topk
    cls._quest_roofline_patched = True


def _patch_coordinator_module(module: ModuleType, mode: str) -> None:
    cls = module.SparseCoordinator
    if getattr(cls, "_quest_roofline_patched", False):
        return

    original_handle = cls._handle_sparse_retrieve
    original_mask = cls._compute_sparse_mask
    original_attention_end = cls.attention_end
    original_forward_end = cls.forward_end

    def roofline_handle(self, query, layer, forward_batch, attn_metadata, **kwargs):
        if not _is_quest(self.algorithm):
            return original_handle(
                self, query, layer, forward_batch, attn_metadata, **kwargs
            )
        _mark_runtime(mode, "coordinator")
        if mode == RUNTIME_DENSE:
            return attn_metadata
        if layer.layer_id != self.start_layer:
            # The first layer installs a layer-invariant fixed page table. The
            # same metadata tensor remains live for all later FA3 layer calls.
            return attn_metadata
        return original_handle(
            self, query, layer, forward_batch, attn_metadata, **kwargs
        )

    def fixed_mask(self, req_pool_indices):
        if not _is_quest(self.algorithm) or mode != FIXED_SPARSE:
            return original_mask(self, req_pool_indices)
        min_prompt_len = self.config.min_sparse_prompt_len or 0
        return self.states.prompt_lens[req_pool_indices] >= min_prompt_len

    def no_representation_attention_end(self, output, layer, forward_batch):
        if _is_quest(self.algorithm):
            return None
        return original_attention_end(self, output, layer, forward_batch)

    def no_representation_forward_end(self, forward_batch):
        if _is_quest(self.algorithm):
            return None
        return original_forward_end(self, forward_batch)

    cls._handle_sparse_retrieve = roofline_handle
    cls._compute_sparse_mask = fixed_mask
    cls.attention_end = no_representation_attention_end
    cls.forward_end = no_representation_forward_end
    cls._quest_roofline_patched = True


def _patch_adaptor_module(module: ModuleType, mode: str) -> None:
    cls = module.FlashAttentionAdaptor
    if getattr(cls, "_quest_roofline_patched", False):
        return

    def preserve_dense_metadata(self, metadata):
        del metadata
        self._original_metadata = None
        self._reset_forward_state()

    def keep_dense_attention(
        self,
        selected_indices,
        valid_lengths,
        sparse_mask,
        current_metadata,
        forward_batch,
        req_to_token,
        page_size,
        layer_id,
        **kwargs,
    ):
        del (
            selected_indices,
            valid_lengths,
            sparse_mask,
            forward_batch,
            req_to_token,
            page_size,
            layer_id,
            kwargs,
        )
        _mark_runtime(mode, "adaptor")
        return current_metadata

    cls.save_original_metadata = preserve_dense_metadata
    cls.adapt_for_attn_metadata = keep_dense_attention
    cls._quest_roofline_patched = True


def _patch_loaded_module(fullname: str, module: ModuleType, mode: str) -> None:
    applied = False
    if mode == FIXED_SPARSE:
        if fullname == _QUEST_MODULE:
            _patch_quest_module(module)
            applied = True
        elif fullname == _COORDINATOR_MODULE:
            _patch_coordinator_module(module, mode)
            applied = True
    elif mode == RETRIEVAL_DENSE:
        if fullname == _ADAPTOR_MODULE:
            _patch_adaptor_module(module, mode)
            applied = True
    elif mode == RUNTIME_DENSE:
        if fullname == _QUEST_MODULE:
            _patch_quest_module(module)
            applied = True
        elif fullname == _COORDINATOR_MODULE:
            _patch_coordinator_module(module, mode)
            applied = True
        elif fullname == _ADAPTOR_MODULE:
            _patch_adaptor_module(module, mode)
            applied = True

    marker = (mode, fullname)
    if applied and marker not in _marked_patches:
        _marked_patches.add(marker)
        sys.stderr.write(f"[quest-roofline] applied={mode} module={fullname}\n")
        sys.stderr.flush()


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, loader, fullname: str, mode: str):
        self._loader = loader
        self._fullname = fullname
        self._mode = mode

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_loaded_module(self._fullname, module, self._mode)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mode: str):
        self.mode = mode

    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGET_MODULES:
            return None

        # PathFinder bypasses meta_path and therefore cannot recurse back into
        # this finder while locating the real SGLang module.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchLoader(spec.loader, fullname, self.mode)
        return spec


def install(mode: str | None = None) -> None:
    """Install delayed patches for the requested benchmark ablation."""
    global _installed_mode

    mode = mode or os.environ.get(ENV_NAME)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"{ENV_NAME} must be one of {SUPPORTED_MODES}, got {mode!r}")
    if _installed_mode is not None:
        if _installed_mode != mode:
            raise RuntimeError(
                f"Quest roofline patch already installed as {_installed_mode!r}"
            )
        return

    _installed_mode = mode
    finder = _PatchFinder(mode)
    sys.meta_path.insert(0, finder)

    # This also makes install() safe for wrappers that run after an SGLang
    # module has already been imported.
    for fullname in _TARGET_MODULES:
        module = sys.modules.get(fullname)
        if module is not None:
            _patch_loaded_module(fullname, module, mode)


def mode_description(mode: str) -> str:
    if mode == FIXED_SPARSE:
        return (
            "fixed sparse pages; Quest retrieval and representation lifecycle disabled"
        )
    if mode == RETRIEVAL_DENSE:
        return "Quest retrieval enabled; FA3 metadata remains dense"
    if mode == RUNTIME_DENSE:
        return "Quest retrieval disabled; FA3 metadata remains dense"
    raise ValueError(f"Unknown Quest roofline mode: {mode!r}")
