import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_ENABLE_ASYNC_ASSERT = envs.SGLANG_ENABLE_ASYNC_ASSERT.get()

logger = logging.getLogger(__name__)


class BackendAdaptor(ABC):
    """Base class for attention backend adaptors."""

    def __init__(self, device: torch.device):
        self.device = device
        self._original_metadata = None

    def save_original_metadata(self, metadata: Any) -> None:
        """Save original metadata in the beginning of the forward pass."""
        pass

    @abstractmethod
    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        selected_physical_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """
        Adapt attention metadata for sparse KVCache access.

        Transforms sparse retrieval results (logical indices of important KV pages/tokens)
        into backend-specific attention metadata format.

        Returns:
            Modified attention metadata compatible with the backend
        """
        pass


class DSABackendAdaptor(BackendAdaptor):
    """Adaptor for DSA (DeepSeek Sparse Attention) backend."""

    def __init__(
        self,
        device: torch.device,
        req_to_token_pool,
    ):
        super().__init__(device)
        self.req_to_token_pool = req_to_token_pool

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        selected_physical_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """
        Transform logical page indices to physical device indices for DSA backend.
        """
        # TODO: Implement DSA backend adaptor logic
        pass


class FlashAttentionAdaptor(BackendAdaptor):
    """Adaptor for FlashAttention backend."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._metadata_prepared = False
        self._page_table_update_mask = None
        self._max_selected = None
        self._valid_lengths = None

    def _reset_forward_state(self) -> None:
        self._metadata_prepared = False
        self._page_table_update_mask = None
        self._max_selected = None
        self._valid_lengths = None

    def save_original_metadata(self, metadata: Any) -> None:
        # The adaptor is reused across forwards (and BCG replays), while the
        # layer-invariant masks below are valid for exactly one forward.
        self._reset_forward_state()

        required_attrs = (
            "page_table",
            "cache_seqlens_int32",
            "cu_seqlens_k",
        )
        if metadata is None or not all(
            hasattr(metadata, attr) for attr in required_attrs
        ):
            self._original_metadata = None
            return

        self._original_metadata = {
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "max_seq_len_k": metadata.max_seq_len_k,
        }
        # Runtime sparse attention rewrites cache_seqlens after FA3's dense
        # metadata initialization, so a dense scheduler plan would be stale.
        # Clear it once per forward rather than once per layer.
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = None

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        selected_physical_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """
        Adapt FlashAttention metadata for sparse KVCache access.

        Modifies page_table, cache_seqlens, and related metadata to redirect
        FlashAttention to only process selected sparse pages.

        # TODO: Optimize performance
        """
        if self._original_metadata is None:
            return current_metadata

        physical_pages = selected_physical_indices
        if physical_pages is None:
            physical_pages = self._logical_to_physical_pages_batch(
                selected_indices,
                forward_batch.req_pool_indices,
                req_to_token,
                page_size,
            )

        max_selected = physical_pages.shape[1]
        if not self._metadata_prepared:
            active_sparse_mask = sparse_mask & (valid_lengths > 0)
            valid_mask = torch.arange(
                max_selected, device=physical_pages.device
            ).unsqueeze(0) < valid_lengths.unsqueeze(1)
            self._page_table_update_mask = active_sparse_mask.unsqueeze(1) & valid_mask
            self._max_selected = max_selected
            # Keep the first layer's immutable result by reference. Cloning it
            # would launch a GPU copy even when async assertions are disabled.
            self._valid_lengths = valid_lengths

            seq_lens = forward_batch.seq_lens
            positions_in_page = (seq_lens - 1) % page_size
            diff = page_size - positions_in_page - 1
            sparse_seq_lens = (valid_lengths * page_size - diff).to(torch.int32)

            current_metadata.cache_seqlens_int32.copy_(
                torch.where(
                    active_sparse_mask,
                    sparse_seq_lens,
                    self._original_metadata["cache_seqlens_int32"],
                )
            )

            current_metadata.cu_seqlens_k[0].zero_()
            current_metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(
                    current_metadata.cache_seqlens_int32,
                    dim=0,
                    dtype=torch.int32,
                )
            )
            # Keep a safe upper bound for mixed sparse/dense batches without a
            # GPU reduction or D2H sync. FA3 scheduler metadata is disabled for
            # runtime sparse attention, so this value is not on the decode hot
            # path today.
            current_metadata.max_seq_len_k = max(
                self._original_metadata["max_seq_len_k"],
                max_selected * page_size,
            )
            self._metadata_prepared = True
        elif max_selected != self._max_selected:
            raise ValueError(
                "Sparse selection width changed within one forward: "
                f"expected {self._max_selected}, got {max_selected}."
            )
        elif _ENABLE_ASYNC_ASSERT:
            torch._assert_async(
                (valid_lengths == self._valid_lengths).all(),
                "Sparse valid lengths changed between layers in one forward.",
            )

        page_table = current_metadata.page_table[:, :max_selected]
        page_table.copy_(
            torch.where(
                self._page_table_update_mask,
                physical_pages,
                page_table,
            )
        )
        return current_metadata

    def _logical_to_physical_pages_batch(
        self,
        logical_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        bs, max_pages = logical_pages.shape

        page_starts = logical_pages * page_size
        page_starts_clamped = page_starts.clamp(min=0)

        req_indices_expanded = req_pool_indices.unsqueeze(1).expand(-1, max_pages)
        first_tokens = req_to_token[req_indices_expanded, page_starts_clamped]

        physical_pages = first_tokens // page_size
        physical_pages = torch.where(
            logical_pages >= 0, physical_pages, torch.zeros_like(physical_pages)
        )

        return physical_pages.to(torch.int32)
