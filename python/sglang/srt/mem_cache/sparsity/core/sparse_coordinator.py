import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_DEFAULT_CUDA_GRAPH_CONTEXT_BUCKETS = (10 * 1024, 33 * 1024)


class RequestTrackers:
    """State tracker for sparse attention requests."""

    def __init__(
        self,
        max_pool_size: int,
        device: torch.device,
        num_layers: int,
        min_sparse_prompt_len: int,
        max_context_len: int,
    ):
        self.device = device
        self.num_layers = num_layers

        self.repr_constructed = torch.zeros(
            max_pool_size, dtype=torch.bool, device=device
        )
        self.prompt_lens = torch.zeros(max_pool_size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(
            max_pool_size, dtype=torch.int64, device=device
        )

        # TODO: Add more trackers for hierarchical KVCache management

    def register(self, idx: int, prompt_len: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = prompt_len
        self.last_constructed_page[idx] = 0

    def clear(self, idx: int) -> None:
        self.repr_constructed[idx] = False
        self.prompt_lens[idx] = 0
        self.last_constructed_page[idx] = 0


@dataclass
class SparseConfig:
    """Configuration for sparse attention."""

    top_k: int = 2048
    device_buffer_size: int = 4096
    host_to_device_ratio: int = 2
    swap_in_block_size: int = 960
    algorithm: Optional[str] = None
    backend: Optional[str] = None
    page_size: Optional[int] = None
    min_sparse_prompt_len: Optional[int] = None
    sparse_extra_config: dict = field(
        default_factory=dict
    )  # Algorithm-specific config, parsed by each algorithm


class SparseCoordinator:
    """
    Coordinator for sparse attention with retrievable KV cache compression.

    This coordinator framework is designed for decode-phase retrievable algorithms
    (e.g., Quest, PQCache, SnapKV) that dynamically select important KV cache entries
    based on current queries. It manages the lifecycle of sparse attention including
    representation construction, sparse retrieval, and token offloading.

    Request Lifecycle and API Calls:
        1. Request Start:
           - on_request_begin(req) -> Register request and initialize state

        2. Prefill Phase:
           - attention_end(...)    -> Construct representations

        3. Decode Phase:
           - forward_begin(batch)  -> Wait for pending KVCache offloading
           - attention_begin(...)  -> Identify important KV, load offloaded KVCache, adapt attention metadata
           - attention_end(...)    -> Construct/update representations
           - forward_end(batch)    -> Trigger KVCache offloading

        4. Request End:
           - on_request_end(req) -> Clean up state and resources
    """

    def __init__(
        self,
        config: SparseConfig,
        algorithm: BaseSparseAlgorithm,
        backend_adaptor: Optional[BackendAdaptor],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        start_layer: int,
        end_layer: int,
        device: torch.device,
        max_context_len: Optional[int] = None,
    ):
        self.config = config
        self.algorithm = algorithm
        self.backend_adaptor = backend_adaptor
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.page_size = config.page_size
        self.cuda_graph_max_num_pages = (
            (max_context_len + self.page_size - 1) // self.page_size
            if max_context_len is not None
            else max(self.req_to_token_pool.max_context_len // self.page_size, 1)
        )
        self.algorithm.cuda_graph_max_num_pages = self.cuda_graph_max_num_pages
        self.enable_cuda_graph_retrieval = bool(
            getattr(algorithm, "enable_cuda_graph_retrieval", False)
            and config.backend in ("fa3", "flashattention")
            and torch.device(device).type == "cuda"
            and torch.version.hip is None
            and max_context_len is not None
        )
        self.cuda_graph_page_buckets = self._build_cuda_graph_page_buckets()

        self.states = RequestTrackers(
            req_to_token_pool.req_to_token.shape[0],
            device,
            end_layer - start_layer + 1,
            self.config.min_sparse_prompt_len,
            self.req_to_token_pool.max_context_len,
        )

        # Initialize algorithm representation pool and context
        self.algorithm.initialize_representation_pool(
            start_layer,
            end_layer,
            self.token_to_kv_pool,
            self.req_to_token_pool,
            self.states,
        )
        self._forward_sparse_mask = None
        self._last_sparse_layer_id = None
        self._forward_started = False

        logger.info(
            f"SparseCoordinator initialized with sparse algorithm={type(algorithm).__name__}"
        )
        if self.cuda_graph_page_buckets:
            logger.info(
                "Runtime sparse CUDA graph page buckets: %s",
                self.cuda_graph_page_buckets,
            )

    def _build_cuda_graph_page_buckets(self) -> tuple[int, ...]:
        if not self.enable_cuda_graph_retrieval:
            return ()

        max_pages = self.cuda_graph_max_num_pages
        context_buckets = self.config.sparse_extra_config.get(
            "cuda_graph_context_buckets",
            _DEFAULT_CUDA_GRAPH_CONTEXT_BUCKETS,
        )
        page_buckets = {
            min((context_len + self.page_size - 1) // self.page_size, max_pages)
            for context_len in context_buckets
            if context_len > 0
        }
        page_buckets.add(max_pages)
        return tuple(sorted(page_buckets))

    def select_cuda_graph_page_capacity(self, seq_lens_cpu) -> Optional[int]:
        """Choose a captured page bucket without synchronizing a device tensor."""
        if not self.cuda_graph_page_buckets:
            return None

        if torch.is_tensor(seq_lens_cpu):
            if seq_lens_cpu.device.type != "cpu":
                return self.cuda_graph_page_buckets[-1]
            if seq_lens_cpu.numel() == 0:
                max_seq_len = 0
            else:
                max_seq_len = int(seq_lens_cpu.max().item())
        else:
            try:
                max_seq_len = max((int(value) for value in seq_lens_cpu), default=0)
            except TypeError:
                return self.cuda_graph_page_buckets[-1]

        required_pages = (max_seq_len + self.page_size - 1) // self.page_size
        return next(
            (
                capacity
                for capacity in self.cuda_graph_page_buckets
                if capacity >= required_pages
            ),
            None,
        )

    def on_request_begin(self, req: "Req") -> None:
        """
        Handle request begin event. Called when a new request is created.

        Registers the request in the state tracker to enable sparse attention processing.
        """
        if req.req_pool_idx is not None:
            self.states.register(req.req_pool_idx, len(req.origin_input_ids))

    def on_request_end(self, req: "Req") -> None:
        """
        Handle request end event. Called when a request is completed or aborted.
        Cleans up request-specific state and releases resources.
        """
        if req.req_pool_idx is None:
            return

        self.states.clear(req.req_pool_idx)

        # TODO: Implement request end handling
        # - Release host indices if any were allocated for offloading

    def forward_begin(
        self,
        forward_batch: "ForwardBatch",
        *,
        fixed_capacity: bool | int = False,
    ) -> None:
        """
        Handle forward pass begin event. Called before each forward pass starts.

        Wait for pending KVCache offloading operations to complete before forward pass.
        Ensures memory consistency for subsequent sparse attention operations.
        """
        req_pool_indices = forward_batch.req_pool_indices
        if req_pool_indices is None:
            self._forward_sparse_mask = None
            return

        if not isinstance(fixed_capacity, int) or isinstance(fixed_capacity, bool):
            # A scheduler ForwardBatch can be reused after graph replay. Do not
            # let its published graph bucket override the new eager plan.
            forward_batch.runtime_sparse_page_capacity = None

        self._forward_sparse_mask = self._compute_sparse_mask(req_pool_indices)
        self.algorithm.begin_forward(
            forward_batch=forward_batch,
            req_pool_indices=req_pool_indices,
            sparse_mask=self._forward_sparse_mask,
            device=forward_batch.seq_lens.device,
            fixed_capacity=fixed_capacity,
        )

    def forward_end(self, forward_batch: "ForwardBatch") -> None:
        """
        Handle forward pass end event. Called after each forward pass completes.

        Update decode representations after a captured model replay. Scheduling
        this once per forward keeps the per-layer BCG break limited to retrieval
        and metadata mutation while preserving KV-write-before-update ordering.
        """
        if not forward_batch.forward_mode.is_decode():
            return
        captured_forward = self.algorithm.should_finalize_graph_forward(forward_batch)
        if not getattr(self, "_forward_started", False) and not captured_forward:
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        if (
            req_pool_indices is not None
            and seq_lens is not None
            and self.algorithm.should_update_representations(forward_batch)
        ):
            for layer_id in range(self.start_layer, self.end_layer):
                self.algorithm.update_representations(
                    layer_id=layer_id,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
                    forward_batch=forward_batch,
                )

        # Replay does not execute the Python attention hooks that mark an eager
        # forward as started. The captured retrieval did run, so force the
        # matching post-replay tracker finalization.
        self.finalize_forward(forward_batch, force=True)

    def prepare_graph_forward(self) -> None:
        """Clear capture-time lifecycle state before one graph replay."""
        self._forward_started = False
        self._last_sparse_layer_id = None

    def finalize_forward(
        self, forward_batch: "ForwardBatch", *, force: bool = False
    ) -> None:
        """Finalize tracker state after eager execution or graph replay."""
        if not force and not self._forward_started:
            return

        try:
            self.algorithm.finalize_forward(forward_batch)
        finally:
            self._forward_started = False
            self._last_sparse_layer_id = None

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        fixed_capacity: bool | int = False,
        **kwargs,
    ) -> Optional[Any]:
        """
        Handle attention begin event. Called before each attention pass starts.

        Identify important KV entries via sparse algorithm, load offloaded KVCache if needed,
        and adapt attention metadata for the attention backend.
        """
        layer_id = layer.layer_id
        if self._last_sparse_layer_id is None or layer_id <= self._last_sparse_layer_id:
            self.forward_begin(forward_batch, fixed_capacity=fixed_capacity)
            self.backend_adaptor.save_original_metadata(attn_metadata)
            self._forward_started = True
        self._last_sparse_layer_id = layer_id

        return self._handle_sparse_retrieve(
            query, layer, forward_batch, attn_metadata, **kwargs
        )

    def attention_end(
        self,
        output: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ) -> None:
        """
        Handle attention end event. Called after each attention pass completes.

        Maybe construct and update sparse representations.
        """
        layer_id = layer.layer_id

        # Maybe construct representations
        self.algorithm.construct_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

        # Maybe update representations
        self.algorithm.update_representations(
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer_id),
            forward_batch=forward_batch,
        )

    def _handle_sparse_retrieve(
        self,
        query: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_metadata: Optional[Any],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        req_pool_indices = forward_batch.req_pool_indices
        layer_id = layer.layer_id

        # Compute Topk
        sparse_mask = self._forward_sparse_mask
        if sparse_mask is None:
            sparse_mask = self._compute_sparse_mask(req_pool_indices)
        update_metadata_lengths = self.algorithm.should_update_metadata_lengths(
            layer_id
        )
        retrieval_result = self.algorithm.retrieve_topk(
            queries=query,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        metadata_prepared = False
        selected_physical_indices = None
        if len(retrieval_result) == 4:
            (
                selected_indices,
                valid_lengths,
                metadata_prepared,
                selected_physical_indices,
            ) = retrieval_result
        elif len(retrieval_result) == 3:
            selected_indices, valid_lengths, metadata_prepared = retrieval_result
        elif len(retrieval_result) == 2:
            selected_indices, valid_lengths = retrieval_result
        else:
            raise ValueError(
                "Sparse retrieval must return 2, 3, or 4 values; "
                f"got {len(retrieval_result)}."
            )
        if (
            selected_physical_indices is None
            and self.backend_adaptor.requires_selected_physical_indices
        ):
            selected_physical_indices = self.algorithm.get_selected_physical_pages(
                selected_indices
            )

        # Adapt Attention Metadata
        return self.backend_adaptor.adapt_for_attn_metadata(
            selected_indices=selected_indices,
            valid_lengths=valid_lengths,
            sparse_mask=sparse_mask,
            current_metadata=attn_metadata,
            forward_batch=forward_batch,
            req_to_token=self.req_to_token_pool.req_to_token,
            page_size=self.page_size,
            layer_id=layer.layer_id,
            selected_physical_indices=selected_physical_indices,
            metadata_prepared=metadata_prepared,
            update_metadata_lengths=update_metadata_lengths,
        )

    def _compute_sparse_mask(self, req_pool_indices):
        min_sparse_prompt_len = self.config.min_sparse_prompt_len or 0
        mask = self.states.repr_constructed[req_pool_indices] & (
            self.states.prompt_lens[req_pool_indices] >= min_sparse_prompt_len
        )

        return mask
