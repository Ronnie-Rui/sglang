import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import radix_attention
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardMode:
    def __init__(self, *, decode=False):
        self._decode = decode

    def is_decode(self):
        return self._decode


def _metadata():
    return SimpleNamespace(
        page_table=torch.zeros((1, 2), dtype=torch.int32),
        cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
        cu_seqlens_q=torch.arange(2, dtype=torch.int32),
        cu_seqlens_k=torch.arange(2, dtype=torch.int32),
        scheduler_metadata=None,
    )


class TestBreakableSparseAttention(unittest.TestCase):
    def test_decode_break_only_runs_retrieval_and_skips_per_layer_update(self):
        backend = SimpleNamespace(forward_metadata=_metadata())
        coordinator = SimpleNamespace(attention_end=Mock())
        output = torch.ones((1, 4))
        forward_batch = SimpleNamespace(forward_mode=_ForwardMode(decode=True))
        context = SimpleNamespace(
            attn_backend=backend, runtime_sparse_coordinator=coordinator
        )

        with (
            patch.object(
                radix_attention,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
            patch.object(
                radix_attention, "breakable_sparse_attention_begin", return_value=None
            ) as attention_begin,
            patch.object(
                radix_attention, "_dense_attention_forward", return_value=output
            ) as dense_forward,
        ):
            result = radix_attention.sparse_attention_forward(
                torch.empty((1, 4)),
                torch.empty((1, 4)),
                torch.empty((1, 4)),
                SimpleNamespace(),
                forward_batch,
                True,
            )

        self.assertIs(result, output)
        attention_begin.assert_called_once()
        dense_forward.assert_called_once()
        coordinator.attention_end.assert_not_called()

    def test_eager_decode_keeps_per_layer_retrieval_and_update(self):
        backend = SimpleNamespace(forward_metadata=_metadata())
        coordinator = SimpleNamespace(attention_end=Mock())
        output = torch.ones((1, 4))
        forward_batch = SimpleNamespace(forward_mode=_ForwardMode(decode=True))
        context = SimpleNamespace(
            attn_backend=backend, runtime_sparse_coordinator=coordinator
        )

        with (
            patch.object(
                radix_attention,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=False
            ),
            patch.object(
                radix_attention, "sparse_attention_begin", return_value=None
            ) as attention_begin,
            patch.object(
                radix_attention, "_dense_attention_forward", return_value=output
            ),
        ):
            result = radix_attention.sparse_attention_forward(
                torch.empty((1, 4)),
                torch.empty((1, 4)),
                torch.empty((1, 4)),
                SimpleNamespace(),
                forward_batch,
                True,
            )

        self.assertIs(result, output)
        attention_begin.assert_called_once()
        coordinator.attention_end.assert_called_once_with(
            output, unittest.mock.ANY, forward_batch
        )

    def test_breakable_retrieval_requires_same_metadata_object(self):
        backend = SimpleNamespace(forward_metadata=_metadata())
        coordinator = SimpleNamespace(attention_begin=Mock(return_value=_metadata()))
        context = SimpleNamespace(
            attn_backend=backend, runtime_sparse_coordinator=coordinator
        )

        with (
            patch.object(
                radix_attention,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "rewrite forward metadata in place"
            ):
                radix_attention.sparse_attention_begin(
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    SimpleNamespace(),
                    SimpleNamespace(),
                )

    def test_breakable_retrieval_requires_same_tensor_addresses(self):
        metadata = _metadata()
        backend = SimpleNamespace(forward_metadata=metadata)

        def replace_page_table(*args, **kwargs):
            metadata.page_table = metadata.page_table.clone()
            return metadata

        coordinator = SimpleNamespace(attention_begin=replace_page_table)
        context = SimpleNamespace(
            attn_backend=backend, runtime_sparse_coordinator=coordinator
        )
        with (
            patch.object(
                radix_attention,
                "get_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "tensor addresses must remain stable"
            ):
                radix_attention.sparse_attention_begin(
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    SimpleNamespace(),
                    SimpleNamespace(),
                )


class TestSparseCoordinatorForwardEnd(unittest.TestCase):
    def test_dispatches_decode_representation_updates_once_per_layer(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 2
        coordinator.end_layer = 5
        coordinator.algorithm = SimpleNamespace(update_representations=Mock())
        coordinator.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=Mock(side_effect=lambda layer_id: f"key-{layer_id}")
        )
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(decode=True),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([32]),
        )

        coordinator.forward_end(forward_batch)

        self.assertEqual(coordinator.algorithm.update_representations.call_count, 3)
        self.assertEqual(
            [
                call.kwargs["layer_id"]
                for call in coordinator.algorithm.update_representations.call_args_list
            ],
            [2, 3, 4],
        )
        for call in coordinator.algorithm.update_representations.call_args_list:
            self.assertIs(call.kwargs["forward_batch"], forward_batch)


if __name__ == "__main__":
    unittest.main()
