import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.utils import enable_fused_set_kv_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _metadata():
    return SimpleNamespace(
        page_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        cache_seqlens_int32=torch.tensor([7, 6], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
        max_seq_len_k=7,
        scheduler_metadata=torch.ones(1, dtype=torch.int32),
    )


class TestFlashAttentionAdaptor(unittest.TestCase):
    def setUp(self):
        self.adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        self.metadata = _metadata()
        self.forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([7, 6], dtype=torch.int64),
        )
        self.req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [4, 5, 6, 7, 12, 13, 14, 15],
            ],
            dtype=torch.int64,
        )

    def test_rewrites_mixed_batch_in_place_and_lengths_once(self):
        pointers = (
            self.metadata.page_table.data_ptr(),
            self.metadata.cache_seqlens_int32.data_ptr(),
            self.metadata.cu_seqlens_k.data_ptr(),
        )
        self.adaptor.save_original_metadata(self.metadata)

        result = self.adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=self.metadata,
            forward_batch=self.forward_batch,
            req_to_token=self.req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertIs(result, self.metadata)
        self.assertEqual(self.metadata.page_table.tolist(), [[2, 2], [1, 3]])
        self.assertEqual(self.metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(self.metadata.cu_seqlens_k.tolist(), [0, 3, 9])
        self.assertIsNone(self.metadata.scheduler_metadata)
        self.assertEqual(
            (
                self.metadata.page_table.data_ptr(),
                self.metadata.cache_seqlens_int32.data_ptr(),
                self.metadata.cu_seqlens_k.data_ptr(),
            ),
            pointers,
        )

        length_versions = (
            self.metadata.cache_seqlens_int32._version,
            self.metadata.cu_seqlens_k._version,
        )
        self.adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=self.metadata,
            forward_batch=self.forward_batch,
            req_to_token=self.req_to_token,
            page_size=4,
            layer_id=1,
        )

        self.assertEqual(self.metadata.page_table.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(
            (
                self.metadata.cache_seqlens_int32._version,
                self.metadata.cu_seqlens_k._version,
            ),
            length_versions,
        )

    def test_precomputed_physical_pages_skip_remapping(self):
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        self.adaptor.save_original_metadata(metadata)

        with patch.object(
            self.adaptor,
            "_logical_to_physical_pages_batch",
            side_effect=AssertionError("unexpected remap"),
        ):
            self.adaptor.adapt_for_attn_metadata(
                selected_indices=torch.tensor([[0, 1]], dtype=torch.int32),
                valid_lengths=torch.tensor([2], dtype=torch.int32),
                sparse_mask=torch.tensor([True]),
                current_metadata=metadata,
                forward_batch=SimpleNamespace(
                    req_pool_indices=torch.tensor([0]),
                    seq_lens=torch.tensor([8]),
                ),
                req_to_token=torch.arange(8).view(1, 8),
                page_size=4,
                layer_id=0,
                selected_physical_indices=torch.tensor([[7, 3]], dtype=torch.int32),
            )

        self.assertEqual(metadata.page_table.tolist(), [[7, 3]])

    def test_new_forward_resets_layer_invariant_state(self):
        self.adaptor.save_original_metadata(self.metadata)
        self.adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=self.metadata,
            forward_batch=self.forward_batch,
            req_to_token=self.req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.metadata.page_table.copy_(torch.tensor([[0, 2], [1, 3]]))
        self.metadata.cache_seqlens_int32.copy_(torch.tensor([7, 6]))
        self.metadata.cu_seqlens_k.copy_(torch.tensor([0, 7, 13]))
        self.adaptor.save_original_metadata(self.metadata)
        self.adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0, 1], [0, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=self.metadata,
            forward_batch=self.forward_batch,
            req_to_token=self.req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertEqual(self.metadata.cache_seqlens_int32.tolist(), [7, 6])
        self.assertEqual(self.metadata.cu_seqlens_k.tolist(), [0, 7, 13])


class TestRuntimeSparseKvWritePolicy(unittest.TestCase):
    def test_runtime_coordinator_disables_fused_kv_prewrite(self):
        with forward_context(
            ForwardContext(
                attn_backend=None,
                runtime_sparse_coordinator=object(),
            )
        ):
            self.assertFalse(enable_fused_set_kv_buffer(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
