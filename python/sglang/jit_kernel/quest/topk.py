from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_quest_topk_module() -> Module:
    return load_jit(
        "quest_radix_topk",
        cuda_files=["quest/topk.cuh"],
        cuda_wrappers=[
            ("quest_topk_out", "QuestTopKKernel::run"),
            (
                "quest_topk_to_flashattention_metadata_out",
                "QuestTopKToMetadataKernel::run",
            ),
        ],
    )


def quest_topk_out(
    scores: torch.Tensor,
    k_per_req: torch.Tensor,
    output_scores: torch.Tensor,
    output_indices: torch.Tensor,
) -> None:
    """Run Quest's per-row radix top-k into preallocated output tensors."""
    module = _jit_quest_topk_module()
    module.quest_topk_out(scores, k_per_req, output_scores, output_indices)


def quest_topk(
    scores: torch.Tensor,
    k_per_req: torch.Tensor,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unsorted per-row top-k scores and indices with fixed-width padding."""
    output_scores = scores.new_empty(
        (scores.shape[0], output_width), dtype=torch.float32
    )
    output_indices = scores.new_empty(
        (scores.shape[0], output_width), dtype=torch.int32
    )
    quest_topk_out(scores, k_per_req, output_scores, output_indices)
    return output_scores, output_indices


def quest_topk_to_flashattention_metadata_out(
    scores: torch.Tensor,
    k_per_req: torch.Tensor,
    recent_indices: torch.Tensor,
    recent_valid: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    valid_lengths: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk_width: int,
    page_size: int,
    *,
    update_lengths: bool,
) -> None:
    """Select Quest pages and write fixed-address FA metadata in one JIT op.

    This destination-passing prototype preserves the existing score tensor so
    scoring can retain page-parallel execution. It fuses exact row-wise top-k,
    finite filtering, recent-page merge, logical ordering, physical mapping,
    and FlashAttention metadata mutation.
    """
    if recent_valid.dtype != torch.bool or sparse_mask.dtype != torch.bool:
        raise ValueError("Quest recent validity and sparse mask must use bool")

    module = _jit_quest_topk_module()
    module.quest_topk_to_flashattention_metadata_out(
        scores,
        k_per_req,
        recent_indices,
        # TVM-FFI exposes torch.bool with the DLPack bool code. The CUDA entry
        # consumes the same byte storage as uint8 to keep tensor matching
        # portable across supported TVM-FFI versions.
        recent_valid.view(torch.uint8),
        sparse_mask.view(torch.uint8),
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        valid_lengths,
        cache_seqlens_int32,
        cu_seqlens_k,
        topk_width,
        page_size,
        update_lengths,
    )
