from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")
RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM = {
    "quest": {"fa3", "flashattention"},
}
RUNTIME_SPARSE_ALGORITHMS = set(RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM)
RUNTIME_SPARSE_ATTENTION_BACKEND_ALIASES = {"flashattention": "fa3"}
QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION = "use_native_page_bounds_dtype"


def _load_hisparse_config(server_args: ServerArgs) -> dict:
    if not server_args.hisparse_config:
        return {}
    try:
        config = json.loads(server_args.hisparse_config)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    if not isinstance(config, dict):
        raise ValueError(
            f"hisparse_config must be a JSON object, got {type(config).__name__}."
        )
    return config


def use_native_quest_page_bounds_dtype(server_args: ServerArgs) -> bool:
    return (
        _load_hisparse_config(server_args).get(
            QUEST_NATIVE_PAGE_BOUNDS_DTYPE_OPTION, False
        )
        is True
    )


def resolve_quest_page_bounds_dtype(kv_cache_dtype, use_native_page_bounds_dtype: bool):
    import torch

    if use_native_page_bounds_dtype and kv_cache_dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return kv_cache_dtype
    return torch.float32


def get_hisparse_algorithm(server_args: ServerArgs) -> str | None:
    config = _load_hisparse_config(server_args)
    algorithm = config.get("algorithm")
    return algorithm.strip().lower() if isinstance(algorithm, str) else None


def get_hisparse_backend(server_args: ServerArgs) -> str | None:
    config = _load_hisparse_config(server_args)
    backend = config.get("backend")
    if isinstance(backend, str):
        return backend.strip().lower()
    return None


def get_hisparse_page_size(server_args: ServerArgs):
    return _load_hisparse_config(server_args).get("page_size")


def get_hisparse_attention_backend(server_args: ServerArgs) -> str | None:
    backend = get_hisparse_backend(server_args)
    if backend is None:
        return None
    return RUNTIME_SPARSE_ATTENTION_BACKEND_ALIASES.get(backend, backend)


def get_runtime_sparse_backends(server_args: ServerArgs) -> set[str]:
    algorithm = get_hisparse_algorithm(server_args)
    if algorithm is None:
        return set()
    return RUNTIME_SPARSE_BACKENDS_BY_ALGORITHM.get(algorithm, set())


def use_runtime_sparse_attention(server_args: ServerArgs) -> bool:
    return (
        server_args.enable_hisparse
        and get_hisparse_algorithm(server_args) in RUNTIME_SPARSE_ALGORITHMS
    )


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use --dsa-{label}-backend="
            f"{_hisparse_default_backend(kv_cache_dtype)} "
            "or omit it."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    if use_runtime_sparse_attention(server_args):
        from sglang.srt.arg_groups.overrides import attention_backends_of
        from sglang.srt.configs.linear_attn_model_registry import (
            resolve_mambaish_config,
        )
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.mem_cache.sparsity import parse_runtime_sparse_config
        from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

        sparse_config = parse_runtime_sparse_config(server_args)
        sparse_algorithm = sparse_config.algorithm
        sparse_algorithm_subject = f"Sparse attention algorithm {sparse_algorithm!r}"
        sparse_algorithm_label = f"sparse attention algorithm {sparse_algorithm!r}"
        supported_backends = get_runtime_sparse_backends(server_args)
        sparse_backend = sparse_config.backend
        if _is_hip():
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support ROCm; runtime "
                "Quest currently requires the NVIDIA CUDA FA3 path."
            )
        if sparse_backend is not None and sparse_backend not in supported_backends:
            raise ValueError(
                f"{sparse_algorithm_subject} currently supports backend values "
                f"{sorted(supported_backends)}, but got {sparse_backend!r}."
            )
        if not server_args.disable_radix_cache:
            raise ValueError(
                f"{sparse_algorithm_subject} currently requires --disable-radix-cache."
            )
        prefill_backend, decode_backend = attention_backends_of(server_args)
        unsupported = [
            backend
            for backend in (prefill_backend, decode_backend)
            if backend not in supported_backends
        ]
        if unsupported:
            raise ValueError(
                f"{sparse_algorithm_subject} currently supports FlashAttention "
                f"backends {sorted(supported_backends)}, but got "
                f"prefill={prefill_backend}, decode={decode_backend}. "
                f"Please set --attention-backend fa3 for {sparse_algorithm_label}."
            )

        if server_args.speculative_algorithm is not None:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support speculative "
                "decoding because the target verify path does not run "
                "query-dependent sparse retrieval."
            )
        if server_args.dllm_algorithm is not None:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support diffusion LLM "
                "inference because its multi-token forward modes do not run "
                "decode-only sparse retrieval."
            )
        if server_args.enable_pdmux:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support PD multiplexing "
                "because sparse metadata must follow the per-stream attention backend."
            )
        if server_args.disaggregation_mode != "null":
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support PD disaggregation "
                "because Quest page representations are not transferred between "
                "prefill and decode workers."
            )
        if server_args.enable_two_batch_overlap:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support two-batch overlap "
                "because sparse attention metadata is not isolated between TBO "
                "sub-batches."
            )
        if server_args.enable_mixed_chunk:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support mixed chunked "
                "prefill because query-dependent sparse retrieval requires a "
                "decode-only batch."
            )
        if server_args.enable_dp_attention:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support DP attention "
                "because cross-rank forward-mode alignment can relabel decode "
                "batches before query-dependent sparse retrieval."
            )
        if server_args.enable_prefill_cp:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support prefill context "
                "parallelism because Quest page representations are not merged "
                "across context-parallel ranks."
            )
        if server_args.attn_cp_size > 1:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support attention "
                "context parallelism because Quest page representations and "
                "physical-page mappings are rank-local."
            )
        if server_args.dcp_size > 1:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support decode context "
                "parallelism because DCP changes the KV page mapping used by "
                "Quest retrieval."
            )

        model_config = server_args.get_model_config()
        if model_config.attention_arch != AttentionArch.MHA:
            raise ValueError(
                f"{sparse_algorithm_subject} currently supports standard MHA/GQA "
                "models only; MLA is not supported."
            )
        if getattr(model_config, "is_encoder_decoder", False) is True:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support encoder-decoder "
                "models."
            )
        if getattr(model_config, "is_multimodal", False) is True:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support multimodal models."
            )
        if getattr(model_config, "is_generation", True) is False:
            raise ValueError(
                f"{sparse_algorithm_subject} is only supported for generation models."
            )
        sliding_window_size = getattr(model_config, "sliding_window_size", None)
        has_sliding_window = isinstance(sliding_window_size, (int, float)) and (
            sliding_window_size > -1
        )
        if (
            getattr(model_config, "is_hybrid_swa", False) is True
            or has_sliding_window
            or getattr(model_config, "attention_chunk_size", None) is not None
        ):
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support sliding-window "
                "or local-attention layers."
            )
        if resolve_mambaish_config(model_config.hf_config) is not None:
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support hybrid linear-"
                "attention models."
            )
        num_kv_shared_layers = getattr(
            model_config.hf_text_config, "num_kv_shared_layers", 0
        )
        if (
            isinstance(num_kv_shared_layers, int)
            and not isinstance(num_kv_shared_layers, bool)
            and num_kv_shared_layers > 0
        ):
            raise ValueError(
                f"{sparse_algorithm_subject} does not yet support cross-layer KV "
                "sharing."
            )

        locked = getattr(server_args, "_cuda_graph_config_locked", set())
        decode_backend = server_args.cuda_graph_config.decode.backend
        decode_will_use_default = (
            decode_backend == Backend.FULL and (Phase.DECODE, "backend") not in locked
        )
        if not decode_will_use_default and decode_backend not in (
            Backend.BREAKABLE,
            Backend.DISABLED,
        ):
            raise ValueError(
                f"{sparse_algorithm_subject} supports decode CUDA graph only with "
                "the breakable backend, because sparse page retrieval is "
                "query-dependent and must run at graph breaks. Please use "
                "--cuda-graph-backend-decode breakable or disable decode CUDA graph."
            )

        prefill_backend = server_args.cuda_graph_config.prefill.backend
        prefill_will_use_default = (
            prefill_backend != Backend.DISABLED
            and (Phase.PREFILL, "backend") not in locked
        )
        if not prefill_will_use_default and prefill_backend != Backend.DISABLED:
            raise ValueError(
                f"{sparse_algorithm_subject} builds page representations during "
                "prefill, so prefill CUDA graph is not supported. "
                "Please use --cuda-graph-backend-prefill disabled."
            )
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = _is_hip()
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)


def apply_runtime_sparse_cuda_graph_defaults(server_args: ServerArgs) -> None:
    """Apply default-only CUDA graph adjustments after pure validation passes."""
    if not use_runtime_sparse_attention(server_args):
        return

    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    sparse_algorithm = get_hisparse_algorithm(server_args)
    sparse_algorithm_subject = f"Sparse attention algorithm {sparse_algorithm!r}"
    locked = getattr(server_args, "_cuda_graph_config_locked", set())
    if (
        server_args.cuda_graph_config.decode.backend == Backend.FULL
        and (Phase.DECODE, "backend") not in locked
    ):
        logger.warning(
            "%s uses breakable decode CUDA graph. Graph-safe retrieval is "
            "captured when supported; other paths retain eager graph breaks.",
            sparse_algorithm_subject,
        )
        server_args.cuda_graph_config.decode.backend = Backend.BREAKABLE

    if (
        server_args.cuda_graph_config.prefill.backend != Backend.DISABLED
        and (Phase.PREFILL, "backend") not in locked
    ):
        logger.warning(
            "Prefill CUDA graph is disabled for sparse attention algorithm %r "
            "because it builds page representations during prefill.",
            sparse_algorithm,
        )
        server_args.cuda_graph_config.prefill.backend = Backend.DISABLED
