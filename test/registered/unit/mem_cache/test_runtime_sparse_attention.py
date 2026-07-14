import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseCoordinator
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.utils import enable_fused_set_kv_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionAdaptor(unittest.TestCase):
    def test_uses_precomputed_physical_pages_without_remapping(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
        )
        adaptor.save_original_metadata(metadata)

        with patch.object(
            adaptor,
            "_logical_to_physical_pages_batch",
            side_effect=AssertionError("unexpected remap"),
        ):
            adaptor.adapt_for_attn_metadata(
                selected_indices=torch.tensor([[0, 1]], dtype=torch.int32),
                valid_lengths=torch.tensor([2], dtype=torch.int32),
                sparse_mask=torch.tensor([True]),
                current_metadata=metadata,
                forward_batch=forward_batch,
                req_to_token=torch.arange(8, dtype=torch.int64).view(1, 8),
                page_size=4,
                layer_id=0,
                selected_physical_indices=torch.tensor([[7, 3]], dtype=torch.int32),
            )

        self.assertEqual(metadata.page_table.tolist(), [[7, 3]])

    def test_explicit_physical_retrieval_result_bypasses_algorithm_remap(self):
        coordinator = object.__new__(SparseCoordinator)
        logical_pages = torch.tensor([[0, 1]], dtype=torch.int32)
        physical_pages = torch.tensor([[7, 3]], dtype=torch.int32)
        valid_lengths = torch.tensor([2], dtype=torch.int32)
        coordinator._forward_sparse_mask = torch.tensor([True])
        coordinator.page_size = 4
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [[16, 17, 18, 19, 24, 25, 26, 27]], dtype=torch.int64
            )
        )
        coordinator.algorithm = SimpleNamespace(
            should_update_metadata_lengths=Mock(return_value=True),
            retrieve_topk=Mock(
                return_value=(
                    logical_pages,
                    valid_lengths,
                    False,
                    physical_pages,
                )
            ),
            get_selected_physical_pages=Mock(
                side_effect=AssertionError("explicit physical pages were remapped")
            ),
        )
        coordinator.backend_adaptor = SimpleNamespace(
            requires_selected_physical_indices=True,
            adapt_for_attn_metadata=Mock(return_value="adapted"),
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
        )

        result = coordinator._handle_sparse_retrieve(
            query=torch.empty((1, 1)),
            layer=SimpleNamespace(layer_id=0),
            forward_batch=forward_batch,
            attn_metadata=object(),
        )

        self.assertEqual(result, "adapted")
        coordinator.algorithm.get_selected_physical_pages.assert_not_called()
        call = coordinator.backend_adaptor.adapt_for_attn_metadata.call_args
        self.assertIs(call.kwargs["selected_indices"], logical_pages)
        self.assertIs(call.kwargs["selected_physical_indices"], physical_pages)

    def test_rewrites_flashattention_metadata_in_place(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([7, 6], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
            max_seq_len_k=7,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens = metadata.cu_seqlens_k
        adaptor.save_original_metadata(metadata)

        # Runtime page_size=4. Request 0's second logical page starts at
        # physical token 8, hence FlashAttention physical page 2. Request 1
        # remains dense because sparse_mask[1] is false.
        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [4, 5, 6, 7, 12, 13, 14, 15],
            ],
            dtype=torch.int64,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([7, 6], dtype=torch.int64),
        )

        result = adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertIs(result, metadata)
        self.assertIs(metadata.page_table, page_table)
        self.assertIs(metadata.cache_seqlens_int32, cache_seqlens)
        self.assertIs(metadata.cu_seqlens_k, cu_seqlens)
        self.assertEqual(metadata.page_table.tolist(), [[2, 2], [1, 3]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 3, 9])
        # Request 1 remains dense, so the batch max must stay a safe upper
        # bound for its six-token cache rather than shrinking to one page.
        self.assertEqual(metadata.max_seq_len_k, 7)
        self.assertIsNone(metadata.scheduler_metadata)

        cache_seqlens_version = cache_seqlens._version
        cu_seqlens_version = cu_seqlens._version
        page_table_version = page_table._version

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
        )
        self.assertEqual(metadata.page_table.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(cache_seqlens._version, cache_seqlens_version)
        self.assertEqual(cu_seqlens._version, cu_seqlens_version)
        self.assertGreater(page_table._version, page_table_version)

    def test_ragged_mixed_batch_reuses_layer_invariant_metadata(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32
            ),
            cache_seqlens_int32=torch.tensor([10, 7, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10, 17, 20], dtype=torch.int32),
            max_seq_len_k=10,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27],
                [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31],
                [8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([10, 7, 3], dtype=torch.int64),
        )
        adaptor.save_original_metadata(metadata)

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor(
                [[2, 0], [1, -1], [0, -1]], dtype=torch.int32
            ),
            valid_lengths=torch.tensor([2, 1, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True, True]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertEqual(
            metadata.page_table.tolist(), [[6, 0, 12], [4, 21, 22], [30, 31, 32]]
        )
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [6, 3, 3])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 6, 9, 12])
        self.assertEqual(metadata.max_seq_len_k, 10)

        cache_seqlens_version = metadata.cache_seqlens_int32._version
        cu_seqlens_version = metadata.cu_seqlens_k._version
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor(
                [[1, 2], [0, -1], [2, -1]], dtype=torch.int32
            ),
            valid_lengths=torch.tensor([2, 1, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True, True]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
        )

        self.assertEqual(
            metadata.page_table.tolist(), [[3, 6, 12], [1, 21, 22], [30, 31, 32]]
        )
        self.assertEqual(metadata.cache_seqlens_int32._version, cache_seqlens_version)
        self.assertEqual(metadata.cu_seqlens_k._version, cu_seqlens_version)

    def test_budget_boundary_updates_dynamic_width_and_lengths_in_place(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([10, 7], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10, 17], dtype=torch.int32),
            max_seq_len_k=10,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        pointers = (
            metadata.page_table.data_ptr(),
            metadata.cache_seqlens_int32.data_ptr(),
            metadata.cu_seqlens_k.data_ptr(),
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([10, 7], dtype=torch.int64),
        )
        req_to_token = torch.arange(24, dtype=torch.int64).view(2, 12)
        sparse_mask = torch.tensor([True, False])
        adaptor.save_original_metadata(metadata)

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0, 2], [0, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 1], dtype=torch.int32),
            sparse_mask=sparse_mask,
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
            update_metadata_lengths=True,
        )
        self.assertEqual(metadata.page_table.tolist(), [[0, 2, 2], [3, 4, 5]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [6, 7])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 6, 13])

        versions = (
            metadata.page_table._version,
            metadata.cache_seqlens_int32._version,
            metadata.cu_seqlens_k._version,
        )
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0, 2], [0, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 1], dtype=torch.int32),
            sparse_mask=sparse_mask,
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
            metadata_prepared=True,
            update_metadata_lengths=False,
        )
        self.assertEqual(
            (
                metadata.page_table._version,
                metadata.cache_seqlens_int32._version,
                metadata.cu_seqlens_k._version,
            ),
            versions,
        )

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=sparse_mask,
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
            update_metadata_lengths=True,
        )
        self.assertEqual(metadata.page_table.tolist(), [[1, 2, 2], [3, 4, 5]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [2, 7])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 2, 9])

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0, 2], [0, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 1], dtype=torch.int32),
            sparse_mask=sparse_mask,
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=2,
            update_metadata_lengths=True,
        )
        self.assertEqual(metadata.page_table.tolist(), [[0, 2, 2], [3, 4, 5]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [6, 7])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 6, 13])
        self.assertEqual(metadata.max_seq_len_k, 10)
        self.assertIsNone(metadata.scheduler_metadata)
        self.assertEqual(
            (
                metadata.page_table.data_ptr(),
                metadata.cache_seqlens_int32.data_ptr(),
                metadata.cu_seqlens_k.data_ptr(),
            ),
            pointers,
        )

    def test_direct_metadata_state_accepts_budget_width_change(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([10], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10], dtype=torch.int32),
            max_seq_len_k=10,
            scheduler_metadata=None,
        )
        page_table_ptr = metadata.page_table.data_ptr()
        adaptor.save_original_metadata(metadata)
        common_kwargs = {
            "sparse_mask": torch.tensor([True]),
            "current_metadata": metadata,
            "forward_batch": SimpleNamespace(
                req_pool_indices=torch.tensor([0]),
                seq_lens=torch.tensor([10]),
            ),
            "req_to_token": torch.arange(12).view(1, 12),
            "page_size": 4,
            "metadata_prepared": True,
            "update_metadata_lengths": True,
        }

        adaptor.adapt_for_attn_metadata(
            selected_indices=metadata.page_table[:, :2],
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            layer_id=0,
            **common_kwargs,
        )
        adaptor.adapt_for_attn_metadata(
            selected_indices=metadata.page_table[:, :1],
            valid_lengths=torch.tensor([1], dtype=torch.int32),
            layer_id=1,
            **common_kwargs,
        )

        self.assertEqual(adaptor._max_selected, 1)
        self.assertEqual(adaptor._valid_lengths.tolist(), [1])
        self.assertEqual(metadata.page_table.data_ptr(), page_table_ptr)

    def test_no_selection_keeps_dense_metadata(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[9, 10]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
            max_seq_len_k=5,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        adaptor.save_original_metadata(metadata)

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[-1]], dtype=torch.int32),
            valid_lengths=torch.tensor([0], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([5], dtype=torch.int64),
            ),
            req_to_token=torch.arange(8, dtype=torch.int64).view(1, 8),
            page_size=4,
            layer_id=0,
        )

        self.assertEqual(metadata.page_table.tolist(), [[9, 10]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [5])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 5])
        self.assertEqual(metadata.max_seq_len_k, 5)
        self.assertIsNone(metadata.scheduler_metadata)

    def test_debug_assert_enforces_layer_invariant_lengths(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([8], dtype=torch.int64),
        )
        req_to_token = torch.arange(8, dtype=torch.int64).view(1, 8)
        adaptor.save_original_metadata(metadata)
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0, 1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        with (
            patch(
                "sglang.srt.mem_cache.sparsity.backend.backend_adaptor."
                "_ENABLE_ASYNC_ASSERT",
                True,
            ),
            self.assertRaisesRegex(RuntimeError, "valid lengths changed"),
        ):
            adaptor.adapt_for_attn_metadata(
                selected_indices=torch.tensor([[0, -1]], dtype=torch.int32),
                valid_lengths=torch.tensor([1], dtype=torch.int32),
                sparse_mask=torch.tensor([True]),
                current_metadata=metadata,
                forward_batch=forward_batch,
                req_to_token=req_to_token,
                page_size=4,
                layer_id=1,
            )

    def test_save_original_metadata_resets_plan_for_next_replay(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([9], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 9], dtype=torch.int32),
            max_seq_len_k=9,
            scheduler_metadata=None,
        )
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens = metadata.cu_seqlens_k
        req_to_token = torch.tensor(
            [[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]], dtype=torch.int64
        )

        adaptor.save_original_metadata(metadata)
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[2, 0]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([9], dtype=torch.int64),
            ),
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )
        self.assertEqual(metadata.page_table.tolist(), [[4, 0, 2]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [5])

        # BCG replay refills the same fixed-address metadata buffers with the
        # next dense batch before the start-layer hook runs again.
        metadata.page_table.copy_(torch.tensor([[0, 1, 2]], dtype=torch.int32))
        metadata.cache_seqlens_int32.copy_(torch.tensor([6], dtype=torch.int32))
        metadata.cu_seqlens_k.copy_(torch.tensor([0, 6], dtype=torch.int32))
        metadata.max_seq_len_k = 6
        metadata.scheduler_metadata = torch.ones(1, dtype=torch.int32)

        adaptor.save_original_metadata(metadata)
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1]], dtype=torch.int32),
            valid_lengths=torch.tensor([1], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([6], dtype=torch.int64),
            ),
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertIs(metadata.page_table, page_table)
        self.assertIs(metadata.cache_seqlens_int32, cache_seqlens)
        self.assertIs(metadata.cu_seqlens_k, cu_seqlens)
        self.assertEqual(metadata.page_table.tolist(), [[2, 1, 2]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [2])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 2])
        self.assertEqual(metadata.max_seq_len_k, 6)
        self.assertIsNone(metadata.scheduler_metadata)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required for the Quest metadata kernel",
)
class TestQuestFlashAttentionMetadataKernel(unittest.TestCase):
    @staticmethod
    def _make_inputs():
        device = torch.device("cuda")
        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27],
                [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31],
                [8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
            device=device,
        )
        return {
            "selected_indices": torch.tensor(
                [[2, 0], [1, -1], [0, -1]],
                dtype=torch.int32,
                device=device,
            ),
            "valid_lengths": torch.tensor([2, 1, 0], dtype=torch.int32, device=device),
            "sparse_mask": torch.tensor(
                [True, True, True], dtype=torch.bool, device=device
            ),
            "seq_lens": torch.tensor([10, 7, 3], dtype=torch.int64, device=device),
            "req_pool_indices": torch.tensor(
                [0, 1, 2], dtype=torch.int64, device=device
            ),
            "req_to_token": req_to_token,
            "page_table": torch.tensor(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                dtype=torch.int32,
                device=device,
            ),
            "cache_seqlens_int32": torch.tensor(
                [10, 7, 3], dtype=torch.int32, device=device
            ),
            "cu_seqlens_k": torch.tensor(
                [0, 10, 17, 20], dtype=torch.int32, device=device
            ),
        }

    @staticmethod
    def _run_kernel(inputs, *, update_lengths, selected_indices_are_physical=False):
        from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
            quest_update_flashattention_metadata_,
        )

        quest_update_flashattention_metadata_(
            **inputs,
            page_size=4,
            update_lengths=update_lengths,
            selected_indices_are_physical=selected_indices_are_physical,
        )

    def test_updates_ragged_mixed_metadata_in_place(self):
        inputs = self._make_inputs()
        page_table_ptr = inputs["page_table"].data_ptr()
        cache_seqlens_ptr = inputs["cache_seqlens_int32"].data_ptr()
        cu_seqlens_ptr = inputs["cu_seqlens_k"].data_ptr()

        self._run_kernel(inputs, update_lengths=True)
        torch.cuda.synchronize()

        self.assertEqual(
            inputs["page_table"].cpu().tolist(),
            [[6, 0, 12], [4, 21, 22], [30, 31, 32]],
        )
        self.assertEqual(inputs["cache_seqlens_int32"].cpu().tolist(), [6, 3, 3])
        self.assertEqual(inputs["cu_seqlens_k"].cpu().tolist(), [0, 6, 9, 12])

        inputs["selected_indices"].copy_(
            torch.tensor([[1, 2], [0, -1], [2, -1]], device="cuda")
        )
        self._run_kernel(inputs, update_lengths=False)
        torch.cuda.synchronize()

        self.assertEqual(
            inputs["page_table"].cpu().tolist(),
            [[3, 6, 12], [1, 21, 22], [30, 31, 32]],
        )
        self.assertEqual(inputs["cache_seqlens_int32"].cpu().tolist(), [6, 3, 3])
        self.assertEqual(inputs["cu_seqlens_k"].cpu().tolist(), [0, 6, 9, 12])
        self.assertEqual(inputs["page_table"].data_ptr(), page_table_ptr)
        self.assertEqual(inputs["cache_seqlens_int32"].data_ptr(), cache_seqlens_ptr)
        self.assertEqual(inputs["cu_seqlens_k"].data_ptr(), cu_seqlens_ptr)

    def test_cuda_graph_replays_with_new_values_and_stable_addresses(self):
        inputs = self._make_inputs()
        self._run_kernel(inputs, update_lengths=True)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_kernel(inputs, update_lengths=True)

        page_table_ptr = inputs["page_table"].data_ptr()
        cache_seqlens_ptr = inputs["cache_seqlens_int32"].data_ptr()
        cu_seqlens_ptr = inputs["cu_seqlens_k"].data_ptr()
        dense_page_table = torch.tensor(
            [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
            dtype=torch.int32,
            device="cuda",
        )

        inputs["selected_indices"].copy_(
            torch.tensor([[1, 2], [2, -1], [0, -1]], device="cuda")
        )
        inputs["valid_lengths"].copy_(torch.tensor([2, 1, 0], device="cuda"))
        inputs["sparse_mask"].copy_(torch.tensor([True, False, True], device="cuda"))
        inputs["seq_lens"].copy_(torch.tensor([9, 6, 3], device="cuda"))
        inputs["page_table"].copy_(dense_page_table)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(
            inputs["page_table"].cpu().tolist(),
            [[3, 6, 12], [20, 21, 22], [30, 31, 32]],
        )
        self.assertEqual(inputs["cache_seqlens_int32"].cpu().tolist(), [5, 6, 3])
        self.assertEqual(inputs["cu_seqlens_k"].cpu().tolist(), [0, 5, 11, 14])

        inputs["selected_indices"].copy_(
            torch.tensor([[0, 1], [0, -1], [1, -1]], device="cuda")
        )
        inputs["valid_lengths"].copy_(torch.tensor([2, 1, 1], device="cuda"))
        inputs["sparse_mask"].copy_(torch.tensor([True, True, True], device="cuda"))
        inputs["seq_lens"].copy_(torch.tensor([8, 5, 4], device="cuda"))
        inputs["page_table"].copy_(dense_page_table)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(
            inputs["page_table"].cpu().tolist(),
            [[0, 3, 12], [1, 21, 22], [5, 31, 32]],
        )
        self.assertEqual(inputs["cache_seqlens_int32"].cpu().tolist(), [8, 1, 4])
        self.assertEqual(inputs["cu_seqlens_k"].cpu().tolist(), [0, 8, 9, 13])
        self.assertEqual(inputs["page_table"].data_ptr(), page_table_ptr)
        self.assertEqual(inputs["cache_seqlens_int32"].data_ptr(), cache_seqlens_ptr)
        self.assertEqual(inputs["cu_seqlens_k"].data_ptr(), cu_seqlens_ptr)

    def test_physical_pages_bypass_mapping_during_cuda_graph_replay(self):
        inputs = self._make_inputs()
        inputs["selected_indices"].copy_(
            torch.tensor([[17, 13], [9, -1], [5, -1]], device="cuda")
        )
        self._run_kernel(
            inputs,
            update_lengths=True,
            selected_indices_are_physical=True,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_kernel(
                inputs,
                update_lengths=True,
                selected_indices_are_physical=True,
            )

        page_table_ptr = inputs["page_table"].data_ptr()
        cache_seqlens_ptr = inputs["cache_seqlens_int32"].data_ptr()
        cu_seqlens_ptr = inputs["cu_seqlens_k"].data_ptr()
        inputs["selected_indices"].copy_(
            torch.tensor([[7, 11], [15, -1], [19, -1]], device="cuda")
        )
        inputs["valid_lengths"].copy_(torch.tensor([2, 1, 1], device="cuda"))
        inputs["seq_lens"].copy_(torch.tensor([9, 6, 3], device="cuda"))
        inputs["page_table"].copy_(
            torch.tensor(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                dtype=torch.int32,
                device="cuda",
            )
        )

        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(
            inputs["page_table"].cpu().tolist(),
            [[7, 11, 12], [15, 21, 22], [19, 31, 32]],
        )
        self.assertEqual(inputs["cache_seqlens_int32"].cpu().tolist(), [5, 2, 3])
        self.assertEqual(inputs["cu_seqlens_k"].cpu().tolist(), [0, 5, 7, 10])
        self.assertEqual(inputs["page_table"].data_ptr(), page_table_ptr)
        self.assertEqual(inputs["cache_seqlens_int32"].data_ptr(), cache_seqlens_ptr)
        self.assertEqual(inputs["cu_seqlens_k"].data_ptr(), cu_seqlens_ptr)


class TestQuestScoring(unittest.TestCase):
    def test_gqa_heads_are_scored_without_sign_cancellation(self):
        algorithm = QuestAlgorithm.__new__(QuestAlgorithm)
        algorithm.use_triton_score_kernel = False
        algorithm.page_k_min = {
            0: torch.tensor([[[-10.0]], [[0.0]]], dtype=torch.float32)
        }
        algorithm.page_k_max = {
            0: torch.tensor([[[1.0]], [[5.0]]], dtype=torch.float32)
        }
        algorithm.page_valid = {0: torch.tensor([True, True])}

        scores = algorithm._retrieve_page_scores(
            layer_id=0,
            phys_pages=torch.tensor([[0, 1]], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            queries=torch.tensor([[[1.0], [-1.0]]], dtype=torch.float32),
        )

        self.assertEqual(scores.tolist(), [[10.0, 5.0]])


class TestRuntimeSparseKvWritePolicy(unittest.TestCase):
    def test_disables_fused_kv_prewrite(self):
        with forward_context(
            ForwardContext(
                attn_backend=None,
                runtime_sparse_coordinator=object(),
            )
        ):
            self.assertFalse(enable_fused_set_kv_buffer(SimpleNamespace()))


class _ForwardMode:
    def __init__(self, *, extend=False, decode=False):
        self._extend = extend
        self._decode = decode

    def is_extend(self):
        return self._extend

    def is_decode(self):
        return self._decode


class _RecordingAlgorithm(BaseSparseAlgorithmImpl):
    def __init__(self):
        config = SimpleNamespace(
            page_size=4,
            sparse_extra_config={"sparsity_ratio": 0.5, "num_recent_pages": 1},
        )
        super().__init__(config, torch.device("cpu"))
        self.calls = []
        self.end_layer = 1
        self.states = SimpleNamespace(
            repr_constructed=torch.tensor([True]),
            prompt_lens=torch.tensor([99], dtype=torch.int64),
            last_constructed_page=torch.tensor([9], dtype=torch.int64),
        )

    def _compute_page_representations(
        self, layer_id, reqs, seq_lens, start_page, end_page, k_buffer
    ):
        self.calls.append(
            (
                layer_id,
                reqs.clone(),
                seq_lens.clone(),
                start_page.clone(),
                end_page.clone(),
            )
        )

    def _initialize_representation_pools(self, start_layer, end_layer, total_num_pages):
        pass

    def _retrieve_page_scores(self, layer_id, phys_pages, req_pool_indices, queries):
        raise NotImplementedError


class TestDenseFallbackDispatch(unittest.TestCase):
    @staticmethod
    def _make_forward_batch(seq_lens_cpu):
        batch_size = len(seq_lens_cpu) if seq_lens_cpu is not None else 2
        seq_lens = (
            torch.tensor(seq_lens_cpu, dtype=torch.int64)
            if seq_lens_cpu is not None
            else torch.tensor([8, 8], dtype=torch.int64)
        )
        return SimpleNamespace(
            forward_mode=_ForwardMode(decode=True),
            seq_lens=seq_lens,
            seq_lens_cpu=(
                torch.tensor(seq_lens_cpu, dtype=torch.int64)
                if seq_lens_cpu is not None
                else None
            ),
            req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
            runtime_sparse_page_capacity=None,
            spec_info=None,
        )

    @staticmethod
    def _make_coordinator(threshold=8):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.config = SimpleNamespace(
            min_sparse_prompt_len=0,
            sparse_extra_config={"dense_fallback_max_seq_len": threshold},
        )
        coordinator.device = torch.device("cpu")
        return coordinator

    def test_eager_whole_batch_gate_is_inclusive_and_requires_all_short(self):
        coordinator = self._make_coordinator()
        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ):
            self.assertTrue(
                coordinator._should_use_dense_fallback(
                    self._make_forward_batch([8, 7]), fixed_capacity=False
                )
            )
            self.assertFalse(
                coordinator._should_use_dense_fallback(
                    self._make_forward_batch([8, 9]), fixed_capacity=False
                )
            )

    def test_gate_falls_back_to_quest_when_runtime_identity_is_not_safe(self):
        coordinator = self._make_coordinator()
        cases = (
            ("disabled", self._make_forward_batch([8, 8]), False, 0),
            ("missing_host_lengths", self._make_forward_batch(None), False, 8),
            ("fixed_capacity_bool", self._make_forward_batch([8, 8]), True, 8),
            ("fixed_capacity_bucket", self._make_forward_batch([8, 8]), 64, 8),
        )
        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ):
            for name, forward_batch, fixed_capacity, threshold in cases:
                with self.subTest(name=name):
                    coordinator.config.sparse_extra_config[
                        "dense_fallback_max_seq_len"
                    ] = threshold
                    self.assertFalse(
                        coordinator._should_use_dense_fallback(
                            forward_batch, fixed_capacity=fixed_capacity
                        )
                    )

            coordinator.config.sparse_extra_config["dense_fallback_max_seq_len"] = 8
            graph_batch = self._make_forward_batch([8, 8])
            graph_batch.runtime_sparse_page_capacity = 64
            self.assertFalse(
                coordinator._should_use_dense_fallback(
                    graph_batch, fixed_capacity=False
                )
            )

    def test_cuda_graph_context_gates_keep_quest_retrieval(self):
        coordinator = self._make_coordinator()
        contexts = (
            ("breakable", True, False, None),
            ("tc_piecewise_capture", False, True, None),
            ("tc_piecewise_forward", False, False, object()),
        )
        for name, breakable, tc_piecewise, forward_context in contexts:
            with self.subTest(name=name), patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "breakable_cuda_graph.is_in_breakable_cuda_graph",
                return_value=breakable,
            ), patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph",
                return_value=tc_piecewise,
            ), patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
                return_value=forward_context,
            ):
                self.assertFalse(
                    coordinator._should_use_dense_fallback(
                        self._make_forward_batch([8, 8]), fixed_capacity=False
                    )
                )

    def test_dense_lifecycle_updates_representations_before_crossing_threshold(self):
        coordinator = self._make_coordinator()
        algorithm = SimpleNamespace(
            begin_dense_forward=Mock(),
            begin_forward=Mock(),
            should_update_representations=Mock(return_value=True),
            construct_representations=Mock(),
            update_representations=Mock(),
            finalize_forward=Mock(),
        )
        coordinator.algorithm = algorithm
        coordinator.backend_adaptor = SimpleNamespace(
            save_original_metadata=Mock(),
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int64).view(2, 16)
        )
        coordinator.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=Mock(return_value=torch.empty(1))
        )
        coordinator.states = SimpleNamespace(
            repr_constructed=torch.tensor([True, True]),
            prompt_lens=torch.tensor([8, 8], dtype=torch.int64),
        )
        coordinator.page_size = 4
        coordinator._forward_sparse_mask = None
        coordinator._forward_dense_fallback = False
        coordinator._last_sparse_layer_id = None
        coordinator._forward_started = False
        metadata = object()
        layer = SimpleNamespace(layer_id=0)
        short_batch = self._make_forward_batch([8, 8])

        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ), patch.object(
            coordinator, "_handle_sparse_retrieve", return_value="sparse"
        ) as sparse_retrieve:
            self.assertIs(
                coordinator.attention_begin(
                    torch.empty(2, 1),
                    torch.empty(2, 1),
                    torch.empty(2, 1),
                    layer,
                    short_batch,
                    metadata,
                ),
                metadata,
            )
            algorithm.begin_dense_forward.assert_called_once_with(short_batch)
            algorithm.begin_forward.assert_not_called()
            coordinator.backend_adaptor.save_original_metadata.assert_not_called()
            sparse_retrieve.assert_not_called()

            coordinator.attention_end(torch.empty(2, 1), layer, short_batch)
            algorithm.construct_representations.assert_not_called()
            algorithm.update_representations.assert_called_once()
            coordinator.token_to_kv_pool.get_key_buffer.assert_called_once_with(0)
            coordinator.finalize_forward(short_batch)
            algorithm.finalize_forward.assert_called_once_with(short_batch)

            long_batch = self._make_forward_batch([9, 8])
            self.assertEqual(
                coordinator.attention_begin(
                    torch.empty(2, 1),
                    torch.empty(2, 1),
                    torch.empty(2, 1),
                    layer,
                    long_batch,
                    metadata,
                ),
                "sparse",
            )
            algorithm.begin_forward.assert_called_once()
            coordinator.backend_adaptor.save_original_metadata.assert_called_once_with(
                metadata
            )
            sparse_retrieve.assert_called_once()

    def test_sparse_dense_sparse_transition_does_not_restore_stale_metadata(self):
        coordinator = self._make_coordinator()
        selected_pages = torch.tensor([[1]], dtype=torch.int32)
        valid_lengths = torch.tensor([1], dtype=torch.int32)
        algorithm = SimpleNamespace(
            begin_dense_forward=Mock(),
            begin_forward=Mock(),
            should_update_metadata_lengths=Mock(return_value=True),
            retrieve_topk=Mock(return_value=(selected_pages, valid_lengths)),
            get_selected_physical_pages=Mock(return_value=None),
            finalize_forward=Mock(),
        )
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        coordinator.algorithm = algorithm
        coordinator.backend_adaptor = adaptor
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16)
        )
        coordinator.states = SimpleNamespace(
            repr_constructed=torch.tensor([True]),
            prompt_lens=torch.tensor([9], dtype=torch.int64),
        )
        coordinator.page_size = 4
        coordinator._forward_sparse_mask = None
        coordinator._forward_dense_fallback = False
        coordinator._last_sparse_layer_id = None
        coordinator._forward_started = False
        layer = SimpleNamespace(layer_id=0)
        query = torch.empty(1, 1)
        metadata = SimpleNamespace(
            page_table=torch.zeros((1, 4), dtype=torch.int32),
            cache_seqlens_int32=torch.zeros(1, dtype=torch.int32),
            cu_seqlens_k=torch.zeros(2, dtype=torch.int32),
            max_seq_len_k=0,
            scheduler_metadata=None,
        )

        def reset_dense_metadata(seq_len):
            metadata.page_table.copy_(torch.tensor([[0, 1, 2, 3]], dtype=torch.int32))
            metadata.cache_seqlens_int32.fill_(seq_len)
            metadata.cu_seqlens_k.copy_(torch.tensor([0, seq_len], dtype=torch.int32))
            metadata.max_seq_len_k = seq_len
            metadata.scheduler_metadata = torch.ones(1, dtype=torch.int32)

        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ), patch.object(
            adaptor, "save_original_metadata", wraps=adaptor.save_original_metadata
        ) as save_metadata, patch.object(
            adaptor,
            "adapt_for_attn_metadata",
            wraps=adaptor.adapt_for_attn_metadata,
        ) as adapt_metadata:
            long_batch = self._make_forward_batch([9])
            reset_dense_metadata(9)
            coordinator.attention_begin(
                query, query, query, layer, long_batch, metadata
            )
            self.assertEqual(metadata.cache_seqlens_int32.tolist(), [1])
            coordinator.finalize_forward(long_batch)

            short_batch = self._make_forward_batch([8])
            reset_dense_metadata(8)
            dense_snapshot = (
                metadata.page_table.clone(),
                metadata.cache_seqlens_int32.clone(),
                metadata.cu_seqlens_k.clone(),
                metadata.scheduler_metadata,
            )
            coordinator.attention_begin(
                query, query, query, layer, short_batch, metadata
            )
            torch.testing.assert_close(metadata.page_table, dense_snapshot[0])
            torch.testing.assert_close(metadata.cache_seqlens_int32, dense_snapshot[1])
            torch.testing.assert_close(metadata.cu_seqlens_k, dense_snapshot[2])
            self.assertIs(metadata.scheduler_metadata, dense_snapshot[3])
            self.assertEqual(save_metadata.call_count, 1)
            self.assertEqual(adapt_metadata.call_count, 1)
            coordinator.finalize_forward(short_batch)

            reset_dense_metadata(9)
            coordinator.attention_begin(
                query, query, query, layer, long_batch, metadata
            )

        self.assertEqual(save_metadata.call_count, 2)
        self.assertEqual(adapt_metadata.call_count, 2)
        self.assertEqual(algorithm.retrieve_topk.call_count, 2)
        algorithm.begin_dense_forward.assert_called_once_with(short_batch)


class TestSparseRepresentationLifecycle(unittest.TestCase):
    def test_extend_resets_slot_and_decode_updates_only_at_page_boundary(self):
        algorithm = _RecordingAlgorithm()
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        k_buffer = torch.empty(1)

        algorithm.construct_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([10], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(extend=True),
                extend_prefix_lens=torch.tensor([0], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [0])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [2])
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2])

        algorithm.construct_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([14], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(extend=True),
                extend_prefix_lens=torch.tensor([10], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [2])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [3])

        num_calls = len(algorithm.calls)
        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([15], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([15], dtype=torch.int64),
            ),
        )
        self.assertEqual(len(algorithm.calls), num_calls)

        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([16], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [3])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [4])

    def test_short_prompt_constructs_first_page_during_decode(self):
        algorithm = _RecordingAlgorithm()
        algorithm.end_layer = 2
        algorithm.states.repr_constructed[0] = False
        algorithm.states.prompt_lens[0] = 3
        algorithm.states.last_constructed_page[0] = 0
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        k_buffer = torch.empty(1)

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([3], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
                ),
            )
        self.assertEqual(algorithm.calls, [])

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([4], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
                ),
            )
            if layer_id == 0:
                self.assertFalse(algorithm.states.repr_constructed[0].item())

        self.assertEqual([call[3].tolist() for call in algorithm.calls], [[0], [0]])
        self.assertEqual([call[4].tolist() for call in algorithm.calls], [[1], [1]])
        self.assertTrue(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [1])

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([8], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                ),
            )
        self.assertEqual(
            [call[3].tolist() for call in algorithm.calls[-2:]], [[1], [1]]
        )
        self.assertEqual(
            [call[4].tolist() for call in algorithm.calls[-2:]], [[2], [2]]
        )
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2])

    def test_decode_boundary_handles_constructed_and_unconstructed_requests(self):
        algorithm = _RecordingAlgorithm()
        algorithm.states = SimpleNamespace(
            repr_constructed=torch.tensor([True, False]),
            prompt_lens=torch.tensor([5, 3], dtype=torch.int64),
            last_constructed_page=torch.tensor([1, 0], dtype=torch.int64),
        )

        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([8, 4], dtype=torch.int64),
            k_buffer=torch.empty(1),
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([8, 4], dtype=torch.int64),
            ),
        )

        self.assertEqual(algorithm.calls[-1][1].tolist(), [0, 1])
        self.assertEqual(algorithm.calls[-1][3].tolist(), [1, 0])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [2, 1])
        self.assertEqual(algorithm.states.repr_constructed.tolist(), [True, True])
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
