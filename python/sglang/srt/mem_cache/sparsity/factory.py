import json
import logging
from typing import Optional

import torch

from sglang.srt.arg_groups.hisparse_hook import (
    QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_INTERVAL_OPTION,
    QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_MIN_PAGES_OPTION,
    QUEST_DECODE_TOKEN_SELECTION_REUSE_INTERVAL_OPTION,
    QUEST_DENSE_FALLBACK_MAX_SEQ_LEN_OPTION,
    QUEST_MAX_SELECTED_TOKENS_OPTION,
    QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION,
    QUEST_SUPERPAGE_OVERSAMPLE_OPTION,
    QUEST_SUPERPAGE_SIZE_OPTION,
)
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(
    server_args,
    *,
    default_algorithm: Optional[str] = None,
    default_backend: Optional[str] = None,
    default_page_size: Optional[int] = None,
    default_min_sparse_prompt_len: Optional[int] = None,
) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
        if not isinstance(extra_config, dict):
            raise ValueError(
                f"hisparse_config must be a JSON object, got {type(extra_config).__name__}."
            )
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", default_algorithm)
    backend = extra_config.pop("backend", default_backend)
    if isinstance(algorithm, str):
        algorithm = algorithm.strip().lower()
    if isinstance(backend, str):
        backend = backend.strip().lower()
    min_sparse_prompt_len = extra_config.pop(
        "min_sparse_prompt_len", default_min_sparse_prompt_len
    )
    page_size = extra_config.pop("page_size", default_page_size)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def parse_runtime_sparse_config(server_args) -> SparseConfig:
    """Parse runtime sparse attention config from --hisparse-config."""
    from sglang.srt.arg_groups.overrides import attention_backends_of

    prefill_backend, decode_backend = attention_backends_of(server_args)
    runtime_page_size = server_args.page_size
    config = _parse_sparse_config(
        server_args,
        default_algorithm="quest",
        default_backend=decode_backend or prefill_backend,
        default_page_size=runtime_page_size,
        default_min_sparse_prompt_len=0,
    )
    if not config.algorithm:
        raise ValueError("Sparse runtime config requires an algorithm.")
    if not config.backend:
        raise ValueError("Sparse runtime config requires an attention backend.")
    if not isinstance(config.page_size, int) or isinstance(config.page_size, bool):
        raise ValueError(
            f"Sparse runtime config page_size must be an integer, got {config.page_size!r}."
        )
    if config.page_size <= 0:
        raise ValueError(
            f"Sparse runtime config page_size must be positive, got {config.page_size}."
        )
    if runtime_page_size is not None and config.page_size != runtime_page_size:
        raise ValueError(
            "Sparse runtime config page_size must match the runtime KV cache "
            f"--page-size ({runtime_page_size}), got {config.page_size}."
        )
    if config.min_sparse_prompt_len is not None and (
        not isinstance(config.min_sparse_prompt_len, int)
        or isinstance(config.min_sparse_prompt_len, bool)
        or config.min_sparse_prompt_len < 0
    ):
        raise ValueError(
            "Sparse runtime config min_sparse_prompt_len must be a non-negative integer, "
            f"got {config.min_sparse_prompt_len!r}."
        )
    sparsity_ratio = config.sparse_extra_config.get("sparsity_ratio")
    if sparsity_ratio is not None and (
        not isinstance(sparsity_ratio, (int, float))
        or isinstance(sparsity_ratio, bool)
        or not 0 < sparsity_ratio <= 1
    ):
        raise ValueError(
            "Sparse runtime config sparsity_ratio must be in the range (0, 1], "
            f"got {sparsity_ratio!r}."
        )
    num_recent_pages = config.sparse_extra_config.get("num_recent_pages")
    if num_recent_pages is not None and (
        not isinstance(num_recent_pages, int)
        or isinstance(num_recent_pages, bool)
        or num_recent_pages <= 0
    ):
        raise ValueError(
            "Sparse runtime config num_recent_pages must be a positive integer, "
            f"got {num_recent_pages!r}."
        )

    quest_max_selected_tokens = config.sparse_extra_config.get(
        QUEST_MAX_SELECTED_TOKENS_OPTION
    )
    if quest_max_selected_tokens is not None:
        if (
            not isinstance(quest_max_selected_tokens, int)
            or isinstance(quest_max_selected_tokens, bool)
            or quest_max_selected_tokens <= 0
        ):
            raise ValueError(
                "Sparse runtime config quest_max_selected_tokens must be a "
                "positive integer or null, "
                f"got {quest_max_selected_tokens!r}."
            )
        if quest_max_selected_tokens % config.page_size != 0:
            raise ValueError(
                "Sparse runtime config quest_max_selected_tokens must be a "
                f"multiple of page_size ({config.page_size}), "
                f"got {quest_max_selected_tokens}."
            )
        effective_recent_pages = num_recent_pages if num_recent_pages is not None else 4
        minimum_selected_tokens = (effective_recent_pages + 1) * config.page_size
        if quest_max_selected_tokens < minimum_selected_tokens:
            raise ValueError(
                "Sparse runtime config quest_max_selected_tokens must include "
                "all recent pages and at least one history page; expected at "
                f"least {minimum_selected_tokens}, got {quest_max_selected_tokens}."
            )

    layer_selection_reuse_interval = config.sparse_extra_config.get(
        "layer_selection_reuse_interval"
    )
    if layer_selection_reuse_interval is not None and (
        not isinstance(layer_selection_reuse_interval, int)
        or isinstance(layer_selection_reuse_interval, bool)
        or layer_selection_reuse_interval <= 0
    ):
        raise ValueError(
            "Sparse runtime config layer_selection_reuse_interval must be a "
            "positive integer, "
            f"got {layer_selection_reuse_interval!r}."
        )

    context_adaptive_interval_present = (
        QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_INTERVAL_OPTION
        in config.sparse_extra_config
    )
    context_adaptive_min_pages_present = (
        QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_MIN_PAGES_OPTION
        in config.sparse_extra_config
    )
    if context_adaptive_interval_present != context_adaptive_min_pages_present:
        raise ValueError(
            "Sparse runtime config context-adaptive layer selection reuse "
            "requires both context_adaptive_layer_selection_reuse_interval and "
            "context_adaptive_layer_selection_reuse_min_pages."
        )
    if context_adaptive_interval_present:
        context_adaptive_interval = config.sparse_extra_config[
            QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_INTERVAL_OPTION
        ]
        context_adaptive_min_pages = config.sparse_extra_config[
            QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_MIN_PAGES_OPTION
        ]
        for option, value in (
            (
                QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_INTERVAL_OPTION,
                context_adaptive_interval,
            ),
            (
                QUEST_CONTEXT_ADAPTIVE_LAYER_SELECTION_REUSE_MIN_PAGES_OPTION,
                context_adaptive_min_pages,
            ),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"Sparse runtime config {option} must be a positive integer, "
                    f"got {value!r}."
                )

        base_interval = layer_selection_reuse_interval or 1
        if context_adaptive_interval <= base_interval:
            raise ValueError(
                "Sparse runtime config "
                "context_adaptive_layer_selection_reuse_interval must be greater "
                f"than layer_selection_reuse_interval ({base_interval}), got "
                f"{context_adaptive_interval}."
            )
        if (
            config.sparse_extra_config.get("use_lazy_page_update_score_kernel")
            is not True
            or config.sparse_extra_config.get("use_triton_score_kernel", True)
            is not True
        ):
            raise ValueError(
                "Sparse runtime config context-adaptive layer selection reuse "
                "requires use_lazy_page_update_score_kernel=true and "
                "use_triton_score_kernel=true so restored anchors can materialize "
                "missed page bounds before scoring."
            )

    if QUEST_DECODE_TOKEN_SELECTION_REUSE_INTERVAL_OPTION in config.sparse_extra_config:
        decode_token_selection_reuse_interval = config.sparse_extra_config[
            QUEST_DECODE_TOKEN_SELECTION_REUSE_INTERVAL_OPTION
        ]
        if (
            not isinstance(decode_token_selection_reuse_interval, int)
            or isinstance(decode_token_selection_reuse_interval, bool)
            or decode_token_selection_reuse_interval <= 0
        ):
            raise ValueError(
                "Sparse runtime config decode_token_selection_reuse_interval "
                "must be a positive integer, "
                f"got {decode_token_selection_reuse_interval!r}."
            )

    quest_superpage_size = config.sparse_extra_config.get(
        QUEST_SUPERPAGE_SIZE_OPTION, 1
    )
    if (
        not isinstance(quest_superpage_size, int)
        or isinstance(quest_superpage_size, bool)
        or quest_superpage_size not in (1, 2, 4, 8, 16)
    ):
        raise ValueError(
            "Sparse runtime config quest_superpage_size must be one of "
            f"1, 2, 4, 8, or 16; got {quest_superpage_size!r}."
        )

    quest_superpage_oversample = config.sparse_extra_config.get(
        QUEST_SUPERPAGE_OVERSAMPLE_OPTION, 2
    )
    if (
        not isinstance(quest_superpage_oversample, int)
        or isinstance(quest_superpage_oversample, bool)
        or quest_superpage_oversample <= 0
    ):
        raise ValueError(
            "Sparse runtime config quest_superpage_oversample must be a positive "
            f"integer, got {quest_superpage_oversample!r}."
        )

    if QUEST_DENSE_FALLBACK_MAX_SEQ_LEN_OPTION in config.sparse_extra_config:
        dense_fallback_max_seq_len = config.sparse_extra_config[
            QUEST_DENSE_FALLBACK_MAX_SEQ_LEN_OPTION
        ]
        if (
            not isinstance(dense_fallback_max_seq_len, int)
            or isinstance(dense_fallback_max_seq_len, bool)
            or dense_fallback_max_seq_len < 0
        ):
            raise ValueError(
                "Sparse runtime config dense_fallback_max_seq_len must be a "
                "non-negative integer, "
                f"got {dense_fallback_max_seq_len!r}."
            )

    layer_page_budget = config.sparse_extra_config.get("layer_page_budget")
    if layer_page_budget is not None:
        if not isinstance(layer_page_budget, list):
            raise ValueError(
                "Sparse runtime config layer_page_budget must be a list, "
                f"got {layer_page_budget!r}."
            )

        normalized_ranges = []
        required_keys = {"start_layer", "end_layer", "scale"}
        for index, budget_range in enumerate(layer_page_budget):
            if not isinstance(budget_range, dict) or set(budget_range) != required_keys:
                raise ValueError(
                    "Each layer_page_budget entry must be an object with exactly "
                    "start_layer, end_layer, and scale; "
                    f"entry {index} is {budget_range!r}."
                )

            start_layer = budget_range["start_layer"]
            end_layer = budget_range["end_layer"]
            scale = budget_range["scale"]
            if (
                not isinstance(start_layer, int)
                or isinstance(start_layer, bool)
                or start_layer < 0
                or not isinstance(end_layer, int)
                or isinstance(end_layer, bool)
                or end_layer <= start_layer
            ):
                raise ValueError(
                    "layer_page_budget ranges must use non-negative, half-open "
                    "integer layer bounds with end_layer > start_layer; "
                    f"entry {index} is {budget_range!r}."
                )
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not 0 < scale <= 1
            ):
                raise ValueError(
                    "layer_page_budget scale must be in the range (0, 1], "
                    f"got {scale!r} in entry {index}."
                )
            normalized_ranges.append((start_layer, end_layer))

        normalized_ranges.sort()
        for previous, current in zip(normalized_ranges, normalized_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    "layer_page_budget ranges must not overlap, "
                    f"got [{previous[0]}, {previous[1]}) and "
                    f"[{current[0]}, {current[1]})."
                )

    boolean_options = (
        "use_triton_score_kernel",
        "use_fused_score_mask_kernel",
        "enable_cuda_graph_retrieval",
        "use_jit_topk_kernel",
        "use_triton_page_update_kernel",
        "use_direct_fa_metadata_kernel",
        "use_fused_topk_fa_metadata_kernel",
        "use_lazy_page_update_score_kernel",
        QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION,
    )
    for option in boolean_options:
        if option not in config.sparse_extra_config:
            continue
        value = config.sparse_extra_config[option]
        if not isinstance(value, bool):
            raise ValueError(
                f"Sparse runtime config {option} must be a boolean, got {value!r}."
            )
    cuda_graph_context_buckets = config.sparse_extra_config.get(
        "cuda_graph_context_buckets"
    )
    if cuda_graph_context_buckets is not None and (
        not isinstance(cuda_graph_context_buckets, list)
        or not cuda_graph_context_buckets
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in cuda_graph_context_buckets
        )
    ):
        raise ValueError(
            "Sparse runtime config cuda_graph_context_buckets must be a "
            "non-empty list of positive integers, "
            f"got {cuda_graph_context_buckets!r}."
        )
    return config


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    max_context_len: Optional[int] = None,
    **kwargs,
) -> SparseCoordinator:
    config = parse_runtime_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
        max_context_len=max_context_len,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
