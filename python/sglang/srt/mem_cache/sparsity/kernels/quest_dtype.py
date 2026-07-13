import torch

_SUPPORTED_PAGE_BOUNDS_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)


def validate_quest_page_bounds_dtype(
    page_k_min: torch.Tensor, page_k_max: torch.Tensor
) -> None:
    if (
        page_k_min.dtype != page_k_max.dtype
        or page_k_min.dtype not in _SUPPORTED_PAGE_BOUNDS_DTYPES
    ):
        raise ValueError(
            "Quest page min/max tensors must share fp16, bf16, or fp32 dtype"
        )


def validate_quest_page_bounds_k_dtype(
    page_k_min: torch.Tensor, k_buffer: torch.Tensor
) -> None:
    if page_k_min.dtype != torch.float32 and page_k_min.dtype != k_buffer.dtype:
        raise ValueError("Quest fp16/bf16 page bounds must match the K-cache dtype")
