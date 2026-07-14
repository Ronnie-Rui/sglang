import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.sparsity.kernels.quest_dtype import (
    validate_quest_page_bounds_dtype,
    validate_quest_page_bounds_k_dtype,
)


@triton.jit
def _quest_page_score_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    candidate_superpages_ptr,
    repair_mask_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    APPLY_RETRIEVAL_MASK: tl.constexpr,
    FILTER_SUPERPAGES: tl.constexpr,
    REPAIR_ONLY: tl.constexpr,
    SUPERPAGE_SIZE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    page_is_selected = page_in_bounds
    if APPLY_RETRIEVAL_MASK:
        request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
        history_page_count = tl.load(history_page_counts_ptr + batch_idx)
        page_is_selected &= request_is_active & (page_idx < history_page_count)
    if FILTER_SUPERPAGES:
        num_superpages = tl.cdiv(num_pages, SUPERPAGE_SIZE)
        superpage_idx = page_idx // SUPERPAGE_SIZE
        is_candidate = tl.load(
            candidate_superpages_ptr + batch_idx * num_superpages + superpage_idx
        ).to(tl.int1)
        if REPAIR_ONLY:
            repair_request = tl.load(repair_mask_ptr + batch_idx).to(tl.int1)
            page_is_selected &= repair_request & ~is_candidate
        else:
            page_is_selected &= is_candidate
    physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    page_is_valid = tl.load(
        page_valid_ptr + physical_page, mask=page_is_selected, other=0
    )

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = (dim_offsets < HEAD_DIM) & page_is_selected & page_is_valid
    best_bound = -float("inf")

    for kv_head in range(KV_HEADS):
        key_offsets = (
            physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dim_offsets
        )
        key_min = tl.load(page_k_min_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        key_max = tl.load(page_k_max_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )

        for group_idx in range(GROUP_SIZE):
            query_head = kv_head * GROUP_SIZE + group_idx
            query_offsets = (
                batch_idx * Q_HEADS * HEAD_DIM + query_head * HEAD_DIM + dim_offsets
            )
            query = tl.load(queries_ptr + query_offsets, mask=dim_mask, other=0.0).to(
                tl.float32
            )
            bound_keys = tl.where(query >= 0, key_max, key_min)
            head_bound = tl.sum(query * bound_keys, axis=0)
            best_bound = tl.maximum(best_bound, head_bound)

    score = tl.where(page_is_selected & page_is_valid, best_bound, -float("inf"))
    if REPAIR_ONLY:
        tl.store(
            output_ptr + batch_idx * num_pages + page_idx,
            score,
            mask=page_is_selected,
        )
    else:
        tl.store(output_ptr + batch_idx * num_pages + page_idx, score)


@triton.jit
def _quest_superpage_score_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    SUPERPAGE_SIZE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    superpage_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
    history_page_count = tl.load(history_page_counts_ptr + batch_idx)

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_in_bounds = dim_offsets < HEAD_DIM
    best_bound = -float("inf")
    superpage_has_valid_page = False
    superpage_has_nonfinite_bound = False

    for kv_head in range(KV_HEADS):
        superpage_min = tl.full((BLOCK_D,), float("inf"), tl.float32)
        superpage_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        head_has_valid_page = False
        head_has_nonfinite_bound = False

        for page_offset in range(SUPERPAGE_SIZE):
            page_idx = superpage_idx * SUPERPAGE_SIZE + page_offset
            logical_page_valid = (
                request_is_active
                & (page_idx < num_pages)
                & (page_idx < history_page_count)
            )
            physical_page_raw = tl.load(
                physical_pages_ptr + batch_idx * num_pages + page_idx,
                mask=page_idx < num_pages,
                other=-1,
            )
            page_in_bounds = (
                logical_page_valid
                & (physical_page_raw >= 0)
                & (physical_page_raw < num_pool_pages)
            )
            physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
            page_is_valid = tl.load(
                page_valid_ptr + physical_page,
                mask=page_in_bounds,
                other=0,
            ).to(tl.int1)
            include_page = page_in_bounds & page_is_valid
            key_offsets = (
                physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dim_offsets
            )
            key_min = tl.load(
                page_k_min_ptr + key_offsets,
                mask=dim_in_bounds & include_page,
                other=float("inf"),
            ).to(tl.float32)
            key_max = tl.load(
                page_k_max_ptr + key_offsets,
                mask=dim_in_bounds & include_page,
                other=-float("inf"),
            ).to(tl.float32)
            nonfinite_bound = (
                dim_in_bounds
                & include_page
                & (
                    (key_min != key_min)
                    | (key_max != key_max)
                    | (tl.abs(key_min) == float("inf"))
                    | (tl.abs(key_max) == float("inf"))
                )
            )
            superpage_min = tl.minimum(superpage_min, key_min)
            superpage_max = tl.maximum(superpage_max, key_max)
            head_has_valid_page |= include_page
            head_has_nonfinite_bound |= tl.sum(nonfinite_bound.to(tl.int32), axis=0) > 0

        superpage_has_valid_page |= head_has_valid_page
        superpage_has_nonfinite_bound |= head_has_nonfinite_bound
        query_mask = dim_in_bounds & head_has_valid_page
        for group_idx in range(GROUP_SIZE):
            query_head = kv_head * GROUP_SIZE + group_idx
            query_offsets = (
                batch_idx * Q_HEADS * HEAD_DIM + query_head * HEAD_DIM + dim_offsets
            )
            query = tl.load(
                queries_ptr + query_offsets,
                mask=query_mask,
                other=0.0,
            ).to(tl.float32)
            bound_keys = tl.where(query >= 0, superpage_max, superpage_min)
            head_bound = tl.sum(query * bound_keys, axis=0)
            best_bound = tl.maximum(best_bound, head_bound)

    score = tl.where(superpage_has_valid_page, best_bound, -float("inf"))
    score = tl.where(superpage_has_nonfinite_bound, float("inf"), score)
    num_superpages = tl.cdiv(num_pages, SUPERPAGE_SIZE)
    tl.store(output_ptr + batch_idx * num_superpages + superpage_idx, score)


@triton.jit
def _quest_lazy_update_page_score_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    k_buffer_ptr,
    repr_constructed_ptr,
    last_constructed_page_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    num_k_tokens,
    req_pool_indices_stride,
    seq_lens_stride,
    req_to_token_stride_r,
    req_to_token_stride_t,
    k_buffer_stride_t,
    k_buffer_stride_h,
    k_buffer_stride_d,
    APPLY_RETRIEVAL_MASK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    constructed = tl.load(repr_constructed_ptr + req_idx)
    tracked_page = tl.load(last_constructed_page_ptr + req_idx).to(tl.int64)
    start_page = tl.where(constructed, tracked_page, 0)

    # attention_begin runs before the backend writes this token's K. Only pages
    # completed by the preceding token are safe to materialize here.
    ready_end_page = tl.minimum(tl.maximum((seq_len - 1) // PAGE_SIZE, 0), num_pages)
    needs_update = (page_idx >= start_page) & (page_idx < ready_end_page)

    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    score_physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    update_physical_page = tl.minimum(
        tl.maximum(physical_page_raw, 0), num_pool_pages - 1
    )

    page_is_selected = page_in_bounds
    if APPLY_RETRIEVAL_MASK:
        request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
        history_page_count = tl.load(history_page_counts_ptr + batch_idx)
        page_is_selected &= request_is_active & (page_idx < history_page_count)

    old_page_valid = tl.load(
        page_valid_ptr + score_physical_page, mask=page_is_selected, other=0
    )
    dim_offsets = tl.arange(0, BLOCK_D)
    token_offsets = tl.arange(0, BLOCK_T)
    dim_in_bounds = dim_offsets < HEAD_DIM
    token_in_bounds = token_offsets < PAGE_SIZE
    best_bound = -float("inf")

    # This is a program-uniform branch. Keeping the PAGE_SIZE x HEAD_DIM
    # reduction behind it is essential: almost every page program only scores
    # an already materialized representation.
    if needs_update:
        logical_tokens = page_idx * PAGE_SIZE + token_offsets
        physical_tokens = tl.load(
            req_to_token_ptr
            + req_idx * req_to_token_stride_r
            + logical_tokens * req_to_token_stride_t,
            mask=token_in_bounds,
            other=0,
        ).to(tl.int64)
        safe_physical_tokens = tl.minimum(
            tl.maximum(physical_tokens, 0), num_k_tokens - 1
        )
        update_load_mask = token_in_bounds[:, None] & dim_in_bounds[None, :]
        for kv_head in range(KV_HEADS):
            update_key_offsets = (
                safe_physical_tokens[:, None] * k_buffer_stride_t
                + kv_head * k_buffer_stride_h
                + dim_offsets[None, :] * k_buffer_stride_d
            )
            page_keys = tl.load(
                k_buffer_ptr + update_key_offsets,
                mask=update_load_mask,
                other=0.0,
            ).to(tl.float32)
            key_min = tl.min(
                tl.where(update_load_mask, page_keys, float("inf")), axis=0
            )
            key_max = tl.max(
                tl.where(update_load_mask, page_keys, -float("inf")), axis=0
            )
            update_pool_offsets = (
                update_physical_page * KV_HEADS * HEAD_DIM
                + kv_head * HEAD_DIM
                + dim_offsets
            )
            tl.store(
                page_k_min_ptr + update_pool_offsets,
                key_min,
                mask=dim_in_bounds,
            )
            tl.store(
                page_k_max_ptr + update_pool_offsets,
                key_max,
                mask=dim_in_bounds,
            )

            if page_is_selected:
                for group_idx in range(GROUP_SIZE):
                    query_head = kv_head * GROUP_SIZE + group_idx
                    query_offsets = (
                        batch_idx * Q_HEADS * HEAD_DIM
                        + query_head * HEAD_DIM
                        + dim_offsets
                    )
                    query = tl.load(
                        queries_ptr + query_offsets,
                        mask=dim_in_bounds,
                        other=0.0,
                    ).to(tl.float32)
                    bound_keys = tl.where(query >= 0, key_max, key_min)
                    head_bound = tl.sum(query * bound_keys, axis=0)
                    best_bound = tl.maximum(best_bound, head_bound)
    else:
        score_dim_mask = dim_in_bounds & page_is_selected & old_page_valid
        for kv_head in range(KV_HEADS):
            pool_offsets = (
                score_physical_page * KV_HEADS * HEAD_DIM
                + kv_head * HEAD_DIM
                + dim_offsets
            )
            key_min = tl.load(
                page_k_min_ptr + pool_offsets, mask=score_dim_mask, other=0.0
            ).to(tl.float32)
            key_max = tl.load(
                page_k_max_ptr + pool_offsets, mask=score_dim_mask, other=0.0
            ).to(tl.float32)
            for group_idx in range(GROUP_SIZE):
                query_head = kv_head * GROUP_SIZE + group_idx
                query_offsets = (
                    batch_idx * Q_HEADS * HEAD_DIM + query_head * HEAD_DIM + dim_offsets
                )
                query = tl.load(
                    queries_ptr + query_offsets,
                    mask=score_dim_mask,
                    other=0.0,
                ).to(tl.float32)
                bound_keys = tl.where(query >= 0, key_max, key_min)
                head_bound = tl.sum(query * bound_keys, axis=0)
                best_bound = tl.maximum(best_bound, head_bound)

    tl.store(page_valid_ptr + update_physical_page, 1, mask=needs_update)
    score = tl.where(
        page_is_selected & (old_page_valid | needs_update),
        best_bound,
        -float("inf"),
    )
    tl.store(output_ptr + batch_idx * num_pages + page_idx, score)


@triton.jit
def _quest_advance_lazy_page_trackers_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    repr_constructed_ptr,
    last_constructed_page_ptr,
    req_pool_indices_stride,
    seq_lens_stride,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    ready_end_page = tl.minimum(tl.maximum((seq_len - 1) // PAGE_SIZE, 0), MAX_PAGES)
    constructed = tl.load(repr_constructed_ptr + req_idx)
    tracked_page = tl.load(last_constructed_page_ptr + req_idx).to(tl.int64)
    start_page = tl.where(constructed, tracked_page, 0)
    update = start_page < ready_end_page
    tl.store(repr_constructed_ptr + req_idx, 1, mask=update)
    tl.store(last_constructed_page_ptr + req_idx, ready_end_page, mask=update)


def quest_advance_lazy_page_trackers_(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    repr_constructed: torch.Tensor,
    last_constructed_page: torch.Tensor,
    page_size: int,
    *,
    max_pages: int,
) -> None:
    """Advance lazy Quest trackers only through pages safe to materialize."""
    if not req_pool_indices.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest lazy tracker update requires NVIDIA CUDA tensors")
    if req_pool_indices.ndim != 1 or seq_lens.shape != req_pool_indices.shape:
        raise ValueError("Quest request indices and sequence lengths must be 1D")
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest request indices must use int32 or int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest sequence lengths must use int32 or int64")
    if (
        repr_constructed.ndim != 1
        or repr_constructed.dtype != torch.bool
        or not repr_constructed.is_contiguous()
    ):
        raise ValueError("Quest constructed tracker must be a contiguous bool vector")
    if (
        last_constructed_page.shape != repr_constructed.shape
        or last_constructed_page.dtype not in (torch.int32, torch.int64)
        or not last_constructed_page.is_contiguous()
    ):
        raise ValueError(
            "Quest last-page tracker must be a matching contiguous integer vector"
        )
    if page_size <= 0 or page_size > 32:
        raise ValueError("Quest lazy tracker page_size must be in [1, 32]")
    if max_pages < 0:
        raise ValueError(
            f"Quest lazy tracker max_pages must be non-negative, got {max_pages}"
        )
    if any(
        tensor.device != req_pool_indices.device
        for tensor in (seq_lens, repr_constructed, last_constructed_page)
    ):
        raise ValueError("Quest lazy tracker tensors must share one CUDA device")

    batch_size = req_pool_indices.numel()
    if batch_size == 0:
        return
    _quest_advance_lazy_page_trackers_kernel[(batch_size,)](
        req_pool_indices,
        seq_lens,
        repr_constructed,
        last_constructed_page,
        req_pool_indices.stride(0),
        seq_lens.stride(0),
        PAGE_SIZE=page_size,
        MAX_PAGES=max_pages,
        num_warps=1,
    )


def quest_page_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    history_page_counts: torch.Tensor | None = None,
    candidate_superpages: torch.Tensor | None = None,
    superpage_size: int = 1,
    repair_mask: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Quest's conservative per-page GQA bound without intermediates.

    Out-of-range physical pages are masked to ``-inf``. When ``active_mask``
    and ``history_page_counts`` are provided, inactive requests and logical
    recent/padding pages are masked in the same kernel. The representation
    tensors are persistent contiguous pools in Quest; they are deliberately
    not copied here because doing so would dominate scoring. ``candidate_superpages``
    optionally restricts scoring to logical page groups. Supplying ``repair_mask``
    updates only omitted pages for requests whose exactness certificate failed.
    """
    if not queries.is_cuda:
        raise ValueError("quest_page_scores requires CUDA tensors")
    if page_k_min.dim() != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError(
            "Quest page min/max tensors must have matching [pages, heads, dim] shapes"
        )
    validate_quest_page_bounds_dtype(page_k_min, page_k_max)
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest page min/max tensors must be contiguous")
    if (
        page_valid.dim() != 1
        or page_valid.shape[0] != page_k_min.shape[0]
        or page_valid.dtype != torch.bool
    ):
        raise ValueError("Quest page validity must be a bool tensor of shape [pages]")
    if not page_valid.is_contiguous():
        raise ValueError("Quest page validity tensor must be contiguous")
    if physical_pages.dim() != 2 or physical_pages.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "Quest physical pages must be an int32/int64 [batch, pages] tensor"
        )
    if any(
        tensor.device != queries.device
        for tensor in (page_k_min, page_k_max, page_valid, physical_pages)
    ):
        raise ValueError("Quest score tensors must be on the same CUDA device")

    apply_retrieval_mask = active_mask is not None or history_page_counts is not None
    if apply_retrieval_mask:
        if active_mask is None or history_page_counts is None:
            raise ValueError(
                "Quest active mask and history page counts must be provided together"
            )
        if (
            active_mask.shape != (physical_pages.shape[0],)
            or active_mask.dtype != torch.bool
        ):
            raise ValueError("Quest active mask must be bool with shape [batch]")
        if history_page_counts.shape != (
            physical_pages.shape[0],
        ) or history_page_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "Quest history page counts must be int32/int64 with shape [batch]"
            )
        if (
            active_mask.device != queries.device
            or history_page_counts.device != queries.device
        ):
            raise ValueError("Quest retrieval masks must share the score tensor device")

    filter_superpages = candidate_superpages is not None
    repair_only = repair_mask is not None
    if repair_only and not filter_superpages:
        raise ValueError("Quest superpage repair requires a candidate mask")
    if filter_superpages:
        if (
            not isinstance(superpage_size, int)
            or isinstance(superpage_size, bool)
            or superpage_size <= 1
        ):
            raise ValueError("Quest superpage size must be an integer greater than one")
        expected_superpages = (
            physical_pages.shape[1] + superpage_size - 1
        ) // superpage_size
        if (
            candidate_superpages.shape != (physical_pages.shape[0], expected_superpages)
            or candidate_superpages.dtype != torch.bool
        ):
            raise ValueError(
                "Quest candidate superpages must be bool with shape "
                "[batch, ceil(pages / superpage_size)]"
            )
        if candidate_superpages.device != queries.device:
            raise ValueError("Quest candidate superpages must share the score device")
        if not candidate_superpages.is_contiguous():
            candidate_superpages = candidate_superpages.contiguous()
    if repair_only:
        if (
            repair_mask.shape != (physical_pages.shape[0],)
            or repair_mask.dtype != torch.bool
        ):
            raise ValueError("Quest repair mask must be bool with shape [batch]")
        if repair_mask.device != queries.device:
            raise ValueError("Quest repair mask must share the score device")
        if not repair_mask.is_contiguous():
            repair_mask = repair_mask.contiguous()

    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("Quest page representations require positive head dimensions")
    if queries.dim() == 2:
        batch_size, hidden_size = queries.shape
        if hidden_size % head_dim != 0:
            raise ValueError(
                f"Quest query hidden size {hidden_size} not divisible by "
                f"head_dim {head_dim}"
            )
        query_heads = hidden_size // head_dim
        queries = queries.reshape(batch_size, query_heads, head_dim)
    elif queries.dim() == 3:
        batch_size, query_heads, query_head_dim = queries.shape
        if query_head_dim != head_dim:
            raise ValueError(
                f"Quest query head_dim {query_head_dim} does not match {head_dim}"
            )
    else:
        raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

    if query_heads <= 0:
        raise ValueError("Quest queries require at least one attention head")
    if physical_pages.shape[0] != batch_size:
        raise ValueError(
            f"Quest physical page batch {physical_pages.shape[0]} does not match "
            f"query batch {batch_size}"
        )
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"Query heads {query_heads} not divisible by KV heads {kv_heads}"
        )
    if head_dim > 256:
        raise ValueError(
            f"Quest Triton score kernel supports head_dim <= 256, got {head_dim}"
        )
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not physical_pages.is_contiguous():
        physical_pages = physical_pages.contiguous()
    if apply_retrieval_mask:
        if not active_mask.is_contiguous():
            active_mask = active_mask.contiguous()
        if not history_page_counts.is_contiguous():
            history_page_counts = history_page_counts.contiguous()

    num_pages = physical_pages.shape[1]
    expected_output_shape = (batch_size, num_pages)
    if output is None:
        if repair_only:
            raise ValueError("Quest repair requires the candidate score output")
        output = torch.empty(
            expected_output_shape, dtype=torch.float32, device=queries.device
        )
    elif (
        output.shape != expected_output_shape
        or output.dtype != torch.float32
        or output.device != queries.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Quest score output must be contiguous float32 with shape [batch, pages]"
        )
    if batch_size == 0 or num_pages == 0:
        return output
    if num_pool_pages == 0:
        raise ValueError("Quest page representation pool cannot be empty")

    block_d = triton.next_power_of_2(head_dim)
    active_mask_arg = active_mask if apply_retrieval_mask else physical_pages
    history_page_counts_arg = (
        history_page_counts if apply_retrieval_mask else physical_pages
    )
    candidate_superpages_arg = (
        candidate_superpages if filter_superpages else physical_pages
    )
    repair_mask_arg = repair_mask if repair_only else physical_pages
    _quest_page_score_kernel[(num_pages, batch_size)](
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask_arg,
        history_page_counts_arg,
        candidate_superpages_arg,
        repair_mask_arg,
        output,
        num_pages,
        num_pool_pages,
        APPLY_RETRIEVAL_MASK=apply_retrieval_mask,
        FILTER_SUPERPAGES=filter_superpages,
        REPAIR_ONLY=repair_only,
        SUPERPAGE_SIZE=superpage_size,
        Q_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output


def quest_superpage_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    active_mask: torch.Tensor,
    history_page_counts: torch.Tensor,
    superpage_size: int,
) -> torch.Tensor:
    """Score conservative logical superpage bounds on NVIDIA CUDA.

    Each superpage takes the elementwise minimum/maximum over its valid member
    pages. Its Quest score therefore upper-bounds every member page score.
    """
    if not queries.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest superpage scoring requires NVIDIA CUDA tensors")
    if (
        not isinstance(superpage_size, int)
        or isinstance(superpage_size, bool)
        or not 2 <= superpage_size <= 16
    ):
        raise ValueError("Quest superpage size must be an integer in [2, 16]")
    if page_k_min.ndim != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError(
            "Quest page min/max tensors must have matching [pages, heads, dim] shapes"
        )
    validate_quest_page_bounds_dtype(page_k_min, page_k_max)
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest page min/max tensors must be contiguous")
    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if (
        page_valid.shape != (num_pool_pages,)
        or page_valid.dtype != torch.bool
        or not page_valid.is_contiguous()
    ):
        raise ValueError("Quest page validity must be a contiguous bool vector")
    if physical_pages.ndim != 2 or physical_pages.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "Quest physical pages must be an integer [batch, pages] tensor"
        )
    if (
        active_mask.shape != (physical_pages.shape[0],)
        or active_mask.dtype != torch.bool
    ):
        raise ValueError("Quest active mask must be bool with shape [batch]")
    if history_page_counts.shape != (
        physical_pages.shape[0],
    ) or history_page_counts.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest history page counts must be integer with shape [batch]")
    if any(
        tensor.device != queries.device
        for tensor in (
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask,
            history_page_counts,
        )
    ):
        raise ValueError("Quest superpage tensors must share one CUDA device")
    if kv_heads <= 0 or not 0 < head_dim <= 256:
        raise ValueError("Quest superpage representations require head_dim in [1, 256]")

    if queries.ndim == 2:
        batch_size, hidden_size = queries.shape
        if hidden_size % head_dim != 0:
            raise ValueError(
                f"Quest query hidden size {hidden_size} not divisible by head_dim {head_dim}"
            )
        query_heads = hidden_size // head_dim
        queries = queries.reshape(batch_size, query_heads, head_dim)
    elif queries.ndim == 3:
        batch_size, query_heads, query_head_dim = queries.shape
        if query_head_dim != head_dim:
            raise ValueError(
                f"Quest query head_dim {query_head_dim} does not match {head_dim}"
            )
    else:
        raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")
    if physical_pages.shape[0] != batch_size:
        raise ValueError("Quest physical page batch must match the query batch")
    if query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("Quest query heads must be a positive multiple of KV heads")

    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not physical_pages.is_contiguous():
        physical_pages = physical_pages.contiguous()
    if not active_mask.is_contiguous():
        active_mask = active_mask.contiguous()
    if not history_page_counts.is_contiguous():
        history_page_counts = history_page_counts.contiguous()

    num_pages = physical_pages.shape[1]
    num_superpages = (num_pages + superpage_size - 1) // superpage_size
    output = torch.empty(
        (batch_size, num_superpages), dtype=torch.float32, device=queries.device
    )
    if batch_size == 0 or num_superpages == 0:
        return output
    if num_pool_pages == 0:
        raise ValueError("Quest page representation pool cannot be empty")

    _quest_superpage_score_kernel[(num_superpages, batch_size)](
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
        output,
        num_pages,
        num_pool_pages,
        SUPERPAGE_SIZE=superpage_size,
        Q_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def quest_exact_superpage_page_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    active_mask: torch.Tensor,
    history_page_counts: torch.Tensor,
    k_per_req: torch.Tensor,
    *,
    max_k: int,
    superpage_size: int,
    oversample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune page scoring with a device-side exactness certificate and repair.

    Candidate pages are sufficient only when the candidate kth page score is
    strictly greater than every omitted superpage upper bound. Requests that
    fail this certificate repair all omitted page scores before returning.
    """
    if not isinstance(max_k, int) or isinstance(max_k, bool) or max_k <= 0:
        raise ValueError("Quest superpage max_k must be a positive integer")
    if (
        not isinstance(oversample, int)
        or isinstance(oversample, bool)
        or oversample <= 0
    ):
        raise ValueError("Quest superpage oversample must be a positive integer")
    if k_per_req.shape != (physical_pages.shape[0],) or k_per_req.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest k_per_req must be integer with shape [batch]")
    if k_per_req.device != queries.device:
        raise ValueError("Quest k_per_req must share the score device")
    if max_k > physical_pages.shape[1]:
        raise ValueError("Quest superpage max_k cannot exceed the page width")

    superpage_scores = quest_superpage_scores(
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
        superpage_size,
    )
    num_superpages = superpage_scores.shape[1]
    minimum_groups = (max_k + superpage_size - 1) // superpage_size
    candidate_group_count = min(num_superpages, max(1, minimum_groups * oversample))
    candidate_group_scores, candidate_group_indices = torch.topk(
        superpage_scores,
        k=candidate_group_count,
        dim=1,
        sorted=False,
    )
    candidate_superpages = torch.zeros_like(superpage_scores, dtype=torch.bool)
    candidate_superpages.scatter_(
        1, candidate_group_indices, torch.isfinite(candidate_group_scores)
    )

    candidate_scores = quest_page_scores(
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask=active_mask,
        history_page_counts=history_page_counts,
        candidate_superpages=candidate_superpages,
        superpage_size=superpage_size,
    )
    candidate_top_scores = torch.topk(
        candidate_scores, k=max_k, dim=1, sorted=True
    ).values
    kth_ranks = torch.clamp(k_per_req.to(torch.long) - 1, min=0, max=max_k - 1)
    candidate_threshold = candidate_top_scores.gather(
        1, kth_ranks.unsqueeze(1)
    ).squeeze(1)
    omitted_upper_bound = torch.where(
        candidate_superpages,
        torch.full_like(superpage_scores, float("-inf")),
        superpage_scores,
    ).amax(dim=1)
    needs_selection = active_mask & (k_per_req > 0)
    certified = (~needs_selection) | (
        torch.isfinite(candidate_threshold)
        & (candidate_threshold > omitted_upper_bound)
    )
    repair_mask = needs_selection & ~certified
    quest_page_scores(
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask=active_mask,
        history_page_counts=history_page_counts,
        candidate_superpages=candidate_superpages,
        superpage_size=superpage_size,
        repair_mask=repair_mask,
        output=candidate_scores,
    )
    return candidate_scores, certified


def quest_lazy_update_page_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    k_buffer: torch.Tensor,
    repr_constructed: torch.Tensor,
    last_constructed_page: torch.Tensor,
    page_size: int,
    *,
    active_mask: torch.Tensor | None = None,
    history_page_counts: torch.Tensor | None = None,
    advance_trackers: bool = False,
) -> torch.Tensor:
    """Lazily materialize ready Quest pages, then score them in one launch.

    ``attention_begin`` precedes the current token's KV-cache write, so this
    experimental path only updates pages below ``(seq_len - 1) // page_size``.
    The caller must set ``advance_trackers`` only on the final layer that
    performs page selection in the forward; prefill construction stays on the
    existing eager path.
    """
    if not queries.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest lazy score update requires NVIDIA CUDA tensors")
    if queries.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Quest lazy score queries must use fp16, bf16, or fp32")
    if page_k_min.ndim != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError("Quest page min/max tensors must have matching 3D shapes")
    validate_quest_page_bounds_dtype(page_k_min, page_k_max)
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest page min/max tensors must be contiguous")

    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if num_pool_pages <= 0 or kv_heads <= 0 or head_dim <= 0 or head_dim > 256:
        raise ValueError("Quest lazy score received unsupported representation shape")
    if (
        page_valid.shape != (num_pool_pages,)
        or page_valid.dtype != torch.bool
        or not page_valid.is_contiguous()
    ):
        raise ValueError("Quest page validity must be a contiguous bool vector")
    if physical_pages.ndim != 2 or physical_pages.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest physical pages must be a 2D integer tensor")

    if queries.ndim == 2:
        batch_size, hidden_size = queries.shape
        if hidden_size % head_dim != 0:
            raise ValueError("Quest query hidden size must be divisible by head_dim")
        query_heads = hidden_size // head_dim
        queries = queries.reshape(batch_size, query_heads, head_dim)
    elif queries.ndim == 3:
        batch_size, query_heads, query_head_dim = queries.shape
        if query_head_dim != head_dim:
            raise ValueError("Quest query and representation head dimensions differ")
    else:
        raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")
    if query_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError("Quest query heads must be a positive multiple of KV heads")
    if physical_pages.shape[0] != batch_size:
        raise ValueError("Quest physical page batch does not match queries")

    if (
        req_pool_indices.shape != (batch_size,)
        or req_pool_indices.dtype not in (torch.int32, torch.int64)
        or seq_lens.shape != (batch_size,)
        or seq_lens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Quest request indices and sequence lengths must be integer [batch]"
        )
    if req_to_token.ndim != 2 or req_to_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest req_to_token must be a 2D integer tensor")
    if k_buffer.ndim != 3 or k_buffer.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Quest K cache must be a 3D floating-point tensor")
    validate_quest_page_bounds_k_dtype(page_k_min, k_buffer)
    if k_buffer.shape[0] <= 0 or k_buffer.shape[1:] != (kv_heads, head_dim):
        raise ValueError("Quest K cache and representation dimensions must match")
    if repr_constructed.ndim != 1 or repr_constructed.dtype != torch.bool:
        raise ValueError("Quest constructed tracker must be a bool vector")
    if (
        last_constructed_page.shape != repr_constructed.shape
        or last_constructed_page.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Quest last-page tracker must be a matching integer vector")
    if page_size <= 0 or page_size > 32:
        raise ValueError("Quest lazy score page_size must be in [1, 32]")
    if req_to_token.shape[1] < page_size:
        raise ValueError("Quest req_to_token width must hold at least one page")

    apply_retrieval_mask = active_mask is not None or history_page_counts is not None
    if apply_retrieval_mask:
        if active_mask is None or history_page_counts is None:
            raise ValueError("Quest active mask and history page counts are paired")
        if active_mask.shape != (batch_size,) or active_mask.dtype != torch.bool:
            raise ValueError("Quest active mask must be bool with shape [batch]")
        if history_page_counts.shape != (
            batch_size,
        ) or history_page_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Quest history page counts must be integer [batch]")

    tensors = (
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        req_pool_indices,
        seq_lens,
        req_to_token,
        k_buffer,
        repr_constructed,
        last_constructed_page,
    )
    if apply_retrieval_mask:
        tensors += (active_mask, history_page_counts)
    if any(tensor.device != queries.device for tensor in tensors):
        raise ValueError("Quest lazy score tensors must share one CUDA device")

    queries = queries.contiguous()
    physical_pages = physical_pages.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    seq_lens = seq_lens.contiguous()
    if apply_retrieval_mask:
        active_mask = active_mask.contiguous()
        history_page_counts = history_page_counts.contiguous()

    num_pages = physical_pages.shape[1]
    output = torch.empty(
        (batch_size, num_pages), dtype=torch.float32, device=queries.device
    )
    if batch_size == 0 or num_pages == 0:
        return output

    block_t = triton.next_power_of_2(page_size)
    block_d = triton.next_power_of_2(head_dim)
    active_mask_arg = active_mask if apply_retrieval_mask else physical_pages
    history_page_counts_arg = (
        history_page_counts if apply_retrieval_mask else physical_pages
    )
    _quest_lazy_update_page_score_kernel[(num_pages, batch_size)](
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask_arg,
        history_page_counts_arg,
        req_pool_indices,
        seq_lens,
        req_to_token,
        k_buffer,
        repr_constructed,
        last_constructed_page,
        output,
        num_pages,
        num_pool_pages,
        k_buffer.shape[0],
        req_pool_indices.stride(0),
        seq_lens.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        APPLY_RETRIEVAL_MASK=apply_retrieval_mask,
        PAGE_SIZE=page_size,
        Q_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    if advance_trackers:
        _quest_advance_lazy_page_trackers_kernel[(batch_size,)](
            req_pool_indices,
            seq_lens,
            repr_constructed,
            last_constructed_page,
            req_pool_indices.stride(0),
            seq_lens.stride(0),
            PAGE_SIZE=page_size,
            MAX_PAGES=num_pages,
            num_warps=1,
        )
    return output
