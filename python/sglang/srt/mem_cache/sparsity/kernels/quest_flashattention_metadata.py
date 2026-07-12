import torch
import triton
import triton.language as tl


@triton.jit
def _quest_update_flashattention_metadata_kernel(
    selected_indices_ptr,
    selected_indices_stride_b,
    selected_indices_stride_p,
    valid_lengths_ptr,
    valid_lengths_stride_b,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    page_table_ptr,
    page_table_stride_b,
    page_table_stride_p,
    cache_seqlens_ptr,
    cache_seqlens_stride_b,
    cu_seqlens_ptr,
    cu_seqlens_stride_b,
    max_selected,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_offsets = tl.program_id(1) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)

    valid_length = tl.load(valid_lengths_ptr + batch_idx * valid_lengths_stride_b).to(
        tl.int32
    )
    use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b)
    active = use_sparse & (valid_length > 0)
    write_mask = active & (page_offsets < max_selected) & (page_offsets < valid_length)

    logical_pages = tl.load(
        selected_indices_ptr
        + batch_idx * selected_indices_stride_b
        + page_offsets * selected_indices_stride_p,
        mask=write_mask,
        other=-1,
    ).to(tl.int64)
    req_idx = tl.load(
        req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b,
        mask=active,
        other=0,
    ).to(tl.int64)
    nonnegative_page = logical_pages >= 0
    token_offsets = tl.maximum(logical_pages, 0) * PAGE_SIZE
    first_tokens = tl.load(
        req_to_token_ptr
        + req_idx * req_to_token_stride_b
        + token_offsets * req_to_token_stride_t,
        mask=write_mask & nonnegative_page,
        other=0,
    ).to(tl.int64)
    physical_pages = tl.where(nonnegative_page, first_tokens // PAGE_SIZE, 0)
    tl.store(
        page_table_ptr
        + batch_idx * page_table_stride_b
        + page_offsets * page_table_stride_p,
        physical_pages.to(tl.int32),
        mask=write_mask,
    )

    # One program computes the small batch prefix sum. Page-table programs are
    # independent, and the following attention launch provides stream ordering.
    if UPDATE_LENGTHS and batch_idx == 0 and tl.program_id(1) == 0:
        cumulative_length = 0
        for idx in range(BATCH_SIZE):
            seq_len = tl.load(seq_lens_ptr + idx * seq_lens_stride_b).to(tl.int64)
            row_valid_length = tl.load(
                valid_lengths_ptr + idx * valid_lengths_stride_b
            ).to(tl.int64)
            row_sparse = tl.load(sparse_mask_ptr + idx * sparse_mask_stride_b)
            row_active = row_sparse & (row_valid_length > 0)
            last_page_length = tl.where(seq_len > 0, (seq_len - 1) % PAGE_SIZE + 1, 0)
            sparse_seq_len = (row_valid_length - 1) * PAGE_SIZE + last_page_length
            cache_seq_len = tl.where(row_active, sparse_seq_len, seq_len).to(tl.int32)

            tl.store(
                cache_seqlens_ptr + idx * cache_seqlens_stride_b,
                cache_seq_len,
            )
            tl.store(
                cu_seqlens_ptr + idx * cu_seqlens_stride_b,
                cumulative_length,
            )
            cumulative_length += cache_seq_len

        tl.store(
            cu_seqlens_ptr + BATCH_SIZE * cu_seqlens_stride_b,
            cumulative_length,
        )


def quest_update_flashattention_metadata_(
    selected_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    *,
    update_lengths: bool,
) -> None:
    """Map Quest pages and update fixed-address FA metadata in place."""
    if not selected_indices.is_cuda:
        raise ValueError("Quest FlashAttention metadata kernel requires CUDA tensors")
    if selected_indices.ndim != 2:
        raise ValueError("selected_indices must have shape [batch, pages]")

    batch_size, max_selected = selected_indices.shape
    if valid_lengths.shape != (batch_size,):
        raise ValueError("valid_lengths must have shape [batch]")
    if sparse_mask.shape != (batch_size,):
        raise ValueError("sparse_mask must have shape [batch]")
    if seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have shape [batch]")
    if req_pool_indices.shape != (batch_size,):
        raise ValueError("req_pool_indices must have shape [batch]")
    if cache_seqlens_int32.shape != (batch_size,):
        raise ValueError("cache_seqlens_int32 must have shape [batch]")
    if cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch + 1]")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, pages]")
    if page_table.shape[1] < max_selected:
        raise ValueError(
            f"page_table width {page_table.shape[1]} is smaller than "
            f"selection width {max_selected}"
        )
    if req_to_token.ndim != 2:
        raise ValueError("req_to_token must be a two-dimensional tensor")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_indices must use int32 or int64")
    if valid_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid_lengths must use int32 or int64")
    if sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must use bool")
    if req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_to_token must use int32 or int64")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must use int32")
    if cache_seqlens_int32.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("FlashAttention sequence metadata must use int32")

    tensors = (
        valid_lengths,
        sparse_mask,
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        cache_seqlens_int32,
        cu_seqlens_k,
    )
    if any(tensor.device != selected_indices.device for tensor in tensors):
        raise ValueError("Quest FlashAttention metadata tensors must share one device")
    if batch_size == 0:
        return

    block_pages = 128
    page_blocks = max(triton.cdiv(max_selected, block_pages), 1)
    _quest_update_flashattention_metadata_kernel[(batch_size, page_blocks)](
        selected_indices,
        selected_indices.stride(0),
        selected_indices.stride(1),
        valid_lengths,
        valid_lengths.stride(0),
        sparse_mask,
        sparse_mask.stride(0),
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        cache_seqlens_int32,
        cache_seqlens_int32.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        max_selected,
        BATCH_SIZE=batch_size,
        PAGE_SIZE=page_size,
        UPDATE_LENGTHS=update_lengths,
        BLOCK_PAGES=block_pages,
        num_warps=4,
    )
