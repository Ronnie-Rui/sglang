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
        cuda_wrappers=[("quest_topk_out", "QuestTopKKernel::run")],
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
