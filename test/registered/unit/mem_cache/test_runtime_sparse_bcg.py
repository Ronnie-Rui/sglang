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

    def test_graph_safe_decode_captures_retrieval_without_a_break(self):
        backend = SimpleNamespace(forward_metadata=_metadata())
        coordinator = SimpleNamespace(
            enable_cuda_graph_retrieval=True,
            attention_end=Mock(),
        )
        output = torch.ones((1, 4))
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(decode=True),
            runtime_sparse_page_capacity=640,
        )
        context = SimpleNamespace(
            attn_backend=backend, runtime_sparse_coordinator=coordinator
        )

        with (
            patch.object(radix_attention, "get_forward_context", return_value=context),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
            patch.object(
                radix_attention, "sparse_attention_begin", return_value=None
            ) as attention_begin,
            patch.object(
                radix_attention, "breakable_sparse_attention_begin", return_value=None
            ) as breakable_begin,
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
        self.assertEqual(attention_begin.call_args.kwargs["fixed_capacity"], 640)
        breakable_begin.assert_not_called()
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
                RuntimeError, "replaced one or more forward metadata tensors"
            ):
                radix_attention.sparse_attention_begin(
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    torch.empty((1, 4)),
                    SimpleNamespace(),
                    SimpleNamespace(),
                )


class TestSparseCoordinatorForwardLifecycle(unittest.TestCase):
    @staticmethod
    def _make_coordinator():
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 12
        coordinator.end_layer = 16
        coordinator._last_sparse_layer_id = None
        coordinator._forward_started = False
        coordinator.algorithm = SimpleNamespace(
            begin_forward=Mock(), finalize_forward=Mock()
        )
        coordinator.backend_adaptor = SimpleNamespace(save_original_metadata=Mock())
        coordinator._compute_sparse_mask = Mock(
            return_value=torch.tensor([True], dtype=torch.bool)
        )
        coordinator._handle_sparse_retrieve = Mock(return_value="adapted")
        return coordinator

    def test_first_actual_pp_layer_initializes_and_same_batch_wrap_reinitializes(self):
        coordinator = self._make_coordinator()
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([64]),
        )
        metadata = object()

        # Configured layer 12 may have key=None and never reach the coordinator.
        # The first actual local layer still owns forward initialization.
        for layer_id in (13, 14):
            result = coordinator.attention_begin(
                torch.empty((1, 1, 1)),
                torch.empty((1, 1, 1)),
                torch.empty((1, 1, 1)),
                SimpleNamespace(layer_id=layer_id),
                forward_batch,
                metadata,
                fixed_capacity=640,
            )
            self.assertEqual(result, "adapted")

        # Reusing and mutating the same ForwardBatch must not preserve the old
        # retrieval plan. Layer order wraps from 14 back to 13.
        forward_batch.seq_lens = torch.tensor([32000])
        coordinator.attention_begin(
            torch.empty((1, 1, 1)),
            torch.empty((1, 1, 1)),
            torch.empty((1, 1, 1)),
            SimpleNamespace(layer_id=13),
            forward_batch,
            metadata,
            fixed_capacity=2112,
        )

        self.assertEqual(coordinator.algorithm.begin_forward.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["fixed_capacity"]
                for call in coordinator.algorithm.begin_forward.call_args_list
            ],
            [640, 2112],
        )
        self.assertEqual(
            coordinator.backend_adaptor.save_original_metadata.call_count, 2
        )
        self.assertEqual(coordinator._compute_sparse_mask.call_count, 2)

    def test_all_skipped_eager_forward_does_not_finalize_stale_algorithm_state(self):
        coordinator = self._make_coordinator()
        forward_batch = SimpleNamespace()

        coordinator.finalize_forward(forward_batch)

        coordinator.algorithm.finalize_forward.assert_not_called()
        self.assertFalse(coordinator._forward_started)
        self.assertIsNone(coordinator._last_sparse_layer_id)

    def test_eager_begin_clears_capacity_published_by_previous_graph_replay(self):
        coordinator = self._make_coordinator()
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([64]),
            runtime_sparse_page_capacity=640,
        )

        coordinator.forward_begin(forward_batch, fixed_capacity=False)

        self.assertIsNone(forward_batch.runtime_sparse_page_capacity)
        coordinator.algorithm.begin_forward.assert_called_once()


class TestSparseCoordinatorForwardEnd(unittest.TestCase):
    def test_selects_smallest_graph_page_bucket_from_host_lengths(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.page_size = 16
        coordinator.cuda_graph_page_buckets = (640, 2112, 2560)

        self.assertEqual(
            coordinator.select_cuda_graph_page_capacity(torch.tensor([8192, 9000])),
            640,
        )
        self.assertEqual(
            coordinator.select_cuda_graph_page_capacity([32000, 32128]),
            2112,
        )
        self.assertEqual(
            coordinator.select_cuda_graph_page_capacity([40000]),
            2560,
        )
        self.assertIsNone(coordinator.select_cuda_graph_page_capacity([40961]))

    def test_dispatches_decode_representation_updates_once_per_layer(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 2
        coordinator.end_layer = 5
        coordinator.algorithm = SimpleNamespace(
            should_finalize_graph_forward=Mock(return_value=True),
            should_update_representations=Mock(return_value=True),
            update_representations=Mock(),
            finalize_forward=Mock(),
        )
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
        coordinator.algorithm.finalize_forward.assert_called_once_with(forward_batch)

    def test_skips_all_layer_buffers_away_from_page_boundary(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 2
        coordinator.end_layer = 5
        coordinator.algorithm = SimpleNamespace(
            should_finalize_graph_forward=Mock(return_value=True),
            should_update_representations=Mock(return_value=False),
            update_representations=Mock(),
            finalize_forward=Mock(),
        )
        coordinator.token_to_kv_pool = SimpleNamespace(get_key_buffer=Mock())
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(decode=True),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([31]),
        )

        coordinator.forward_end(forward_batch)

        coordinator.algorithm.should_update_representations.assert_called_once_with(
            forward_batch
        )
        coordinator.token_to_kv_pool.get_key_buffer.assert_not_called()
        coordinator.algorithm.update_representations.assert_not_called()
        coordinator.algorithm.finalize_forward.assert_called_once_with(forward_batch)

    def test_skips_graph_tail_when_sparse_path_was_not_captured(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 2
        coordinator.end_layer = 5
        coordinator.algorithm = SimpleNamespace(
            should_finalize_graph_forward=Mock(return_value=False),
            should_update_representations=Mock(),
            update_representations=Mock(),
            finalize_forward=Mock(),
        )
        coordinator.token_to_kv_pool = SimpleNamespace(get_key_buffer=Mock())
        forward_batch = SimpleNamespace(forward_mode=_ForwardMode(decode=True))

        coordinator.forward_end(forward_batch)

        coordinator.algorithm.should_finalize_graph_forward.assert_called_once_with(
            forward_batch
        )
        coordinator.algorithm.should_update_representations.assert_not_called()
        coordinator.algorithm.update_representations.assert_not_called()
        coordinator.algorithm.finalize_forward.assert_not_called()

    def test_bcg_tail_runs_after_eager_break_started_the_forward(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator.start_layer = 2
        coordinator.end_layer = 3
        coordinator._forward_started = True
        coordinator._last_sparse_layer_id = 2
        coordinator.algorithm = SimpleNamespace(
            should_finalize_graph_forward=Mock(return_value=False),
            should_update_representations=Mock(return_value=False),
            update_representations=Mock(),
            finalize_forward=Mock(),
        )
        coordinator.token_to_kv_pool = SimpleNamespace(get_key_buffer=Mock())
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(decode=True),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([31]),
        )

        coordinator.forward_end(forward_batch)

        coordinator.algorithm.finalize_forward.assert_called_once_with(forward_batch)
        self.assertFalse(coordinator._forward_started)
        self.assertIsNone(coordinator._last_sparse_layer_id)

    def test_prepare_graph_forward_discards_capture_time_lifecycle_state(self):
        coordinator = object.__new__(SparseCoordinator)
        coordinator._forward_started = True
        coordinator._last_sparse_layer_id = 27

        coordinator.prepare_graph_forward()

        self.assertFalse(coordinator._forward_started)
        self.assertIsNone(coordinator._last_sparse_layer_id)


if __name__ == "__main__":
    unittest.main()
