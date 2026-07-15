import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import radix_attention
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ForwardMode:
    def is_decode(self):
        return True


def _metadata():
    return SimpleNamespace(
        page_table=torch.zeros((1, 2), dtype=torch.int32),
        cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
        cu_seqlens_q=torch.arange(2, dtype=torch.int32),
        cu_seqlens_k=torch.arange(2, dtype=torch.int32),
        scheduler_metadata=None,
    )


class TestRuntimeSparseDispatch(unittest.TestCase):
    def _run_decode(self, *, in_breakable_graph, graph_retrieval, capacity=None):
        metadata = _metadata()
        backend = SimpleNamespace(forward_metadata=metadata)
        coordinator = SimpleNamespace(
            enable_cuda_graph_retrieval=graph_retrieval,
            attention_end=Mock(),
        )
        context = SimpleNamespace(
            attn_backend=backend,
            runtime_sparse_coordinator=coordinator,
        )
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(),
            runtime_sparse_page_capacity=capacity,
        )
        output = torch.ones((1, 4))

        with (
            patch.object(radix_attention, "get_forward_context", return_value=context),
            patch.object(
                radix_attention,
                "is_in_breakable_cuda_graph",
                return_value=in_breakable_graph,
            ),
            patch.object(radix_attention, "sparse_attention_begin") as captured_begin,
            patch.object(
                radix_attention, "breakable_sparse_attention_begin"
            ) as eager_break_begin,
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

        return (
            result,
            coordinator,
            captured_begin,
            eager_break_begin,
        )

    def test_eager_decode_retrieves_and_updates_per_layer(self):
        result, coordinator, captured_begin, eager_break_begin = self._run_decode(
            in_breakable_graph=False,
            graph_retrieval=False,
        )

        self.assertEqual(result.tolist(), [[1.0, 1.0, 1.0, 1.0]])
        captured_begin.assert_called_once()
        self.assertFalse(captured_begin.call_args.kwargs["fixed_capacity"])
        eager_break_begin.assert_not_called()
        coordinator.attention_end.assert_called_once()

    def test_breakable_decode_runs_uncaptured_retrieval_without_layer_update(self):
        _, coordinator, captured_begin, eager_break_begin = self._run_decode(
            in_breakable_graph=True,
            graph_retrieval=False,
        )

        captured_begin.assert_not_called()
        eager_break_begin.assert_called_once()
        self.assertFalse(eager_break_begin.call_args.kwargs["fixed_capacity"])
        coordinator.attention_end.assert_not_called()

    def test_bucketed_decode_captures_retrieval_with_fixed_capacity(self):
        _, coordinator, captured_begin, eager_break_begin = self._run_decode(
            in_breakable_graph=True,
            graph_retrieval=True,
            capacity=1024,
        )

        captured_begin.assert_called_once()
        self.assertEqual(captured_begin.call_args.kwargs["fixed_capacity"], 1024)
        eager_break_begin.assert_not_called()
        coordinator.attention_end.assert_not_called()


class TestBreakableMetadataInvariants(unittest.TestCase):
    def test_rejects_replaced_metadata_object(self):
        metadata = _metadata()
        context = SimpleNamespace(
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            runtime_sparse_coordinator=SimpleNamespace(
                attention_begin=Mock(return_value=_metadata())
            ),
        )

        with (
            patch.object(radix_attention, "get_forward_context", return_value=context),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
            self.assertRaisesRegex(RuntimeError, "rewrite metadata in place"),
        ):
            radix_attention.sparse_attention_begin(
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                SimpleNamespace(),
                SimpleNamespace(),
            )

    def test_rejects_replaced_metadata_tensor(self):
        metadata = _metadata()

        def replace_page_table(*args, **kwargs):
            metadata.page_table = metadata.page_table.clone()
            return metadata

        context = SimpleNamespace(
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            runtime_sparse_coordinator=SimpleNamespace(
                attention_begin=Mock(side_effect=replace_page_table)
            ),
        )

        with (
            patch.object(radix_attention, "get_forward_context", return_value=context),
            patch.object(
                radix_attention, "is_in_breakable_cuda_graph", return_value=True
            ),
            self.assertRaisesRegex(RuntimeError, "replaced metadata tensors"),
        ):
            radix_attention.sparse_attention_begin(
                torch.empty(1),
                torch.empty(1),
                torch.empty(1),
                SimpleNamespace(),
                SimpleNamespace(),
            )


class TestSparseCoordinatorGraphCapacity(unittest.TestCase):
    def setUp(self):
        self.coordinator = SparseCoordinator.__new__(SparseCoordinator)
        self.coordinator.page_size = 16
        self.coordinator.cuda_graph_page_buckets = (256, 1024, 2048)

    def test_selects_smallest_capacity_covering_host_lengths(self):
        cases = (
            ([1, 4096], 256),
            (torch.tensor([4097, 12000]), 1024),
            ([32768], 2048),
            ([32769], None),
            ([], 256),
        )
        for seq_lens, expected in cases:
            with self.subTest(seq_lens=seq_lens):
                self.assertEqual(
                    self.coordinator.select_cuda_graph_page_capacity(seq_lens),
                    expected,
                )

    def test_disabled_buckets_return_none(self):
        self.coordinator.cuda_graph_page_buckets = ()

        self.assertIsNone(self.coordinator.select_cuda_graph_page_capacity([1, 32768]))


class TestSparseCoordinatorForwardLifecycle(unittest.TestCase):
    def setUp(self):
        self.coordinator = SparseCoordinator.__new__(SparseCoordinator)
        self.coordinator.algorithm = Mock()
        self.coordinator._forward_sparse_mask = torch.tensor([True])

    def test_prepare_graph_forward_discards_capture_sparse_mask(self):
        self.coordinator.prepare_graph_forward()

        self.assertIsNone(self.coordinator._forward_sparse_mask)
        self.coordinator.algorithm.prepare_graph_forward.assert_called_once_with()

    def test_finalize_forward_delegates_to_algorithm(self):
        forward_batch = SimpleNamespace()

        self.coordinator.finalize_forward(forward_batch)

        self.coordinator.algorithm.finalize_forward.assert_called_once_with(
            forward_batch
        )

    def test_eager_forward_initializes_only_at_start_layer(self):
        self.coordinator.start_layer = 2
        self.coordinator.forward_begin = Mock()
        self.coordinator.backend_adaptor = Mock()
        metadata = object()
        self.coordinator._handle_sparse_retrieve = Mock(return_value=metadata)
        forward_batch = SimpleNamespace()

        for layer_id in (2, 3):
            result = self.coordinator.attention_begin(
                query=torch.empty(1),
                key=torch.empty(1),
                value=torch.empty(1),
                layer=SimpleNamespace(layer_id=layer_id),
                forward_batch=forward_batch,
                attn_metadata=metadata,
                fixed_capacity=256,
            )
            self.assertIs(result, metadata)

        self.coordinator.forward_begin.assert_called_once_with(
            forward_batch, fixed_capacity=256
        )
        self.coordinator.backend_adaptor.save_original_metadata.assert_called_once_with(
            metadata
        )
        self.assertEqual(self.coordinator._handle_sparse_retrieve.call_count, 2)

    def test_graph_tail_updates_each_local_layer_once(self):
        self.coordinator.start_layer = 2
        self.coordinator.end_layer = 4
        self.coordinator.algorithm.should_update_representations.return_value = True
        self.coordinator.token_to_kv_pool = Mock()
        self.coordinator.token_to_kv_pool.get_key_buffer.side_effect = lambda layer_id: (
            f"key-{layer_id}"
        )
        forward_batch = SimpleNamespace(
            forward_mode=_ForwardMode(),
            req_pool_indices=torch.tensor([3]),
            seq_lens=torch.tensor([17]),
        )

        self.coordinator.forward_end(forward_batch)

        self.assertEqual(
            [
                call.kwargs["layer_id"]
                for call in self.coordinator.algorithm.update_representations.call_args_list
            ],
            [2, 3],
        )
        self.assertEqual(
            [
                call.kwargs["k_buffer"]
                for call in self.coordinator.algorithm.update_representations.call_args_list
            ],
            ["key-2", "key-3"],
        )


class TestDenseFallbackDispatch(unittest.TestCase):
    @staticmethod
    def _batch(seq_lens_cpu, *, spec_info=None, capacity=None):
        seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int64)
        return SimpleNamespace(
            forward_mode=_ForwardMode(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=torch.arange(len(seq_lens_cpu)),
            runtime_sparse_page_capacity=capacity,
            spec_info=spec_info,
        )

    @staticmethod
    def _coordinator(threshold=8):
        coordinator = SparseCoordinator.__new__(SparseCoordinator)
        coordinator.config = SimpleNamespace(
            sparse_extra_config={"dense_fallback_max_seq_len": threshold}
        )
        coordinator.device = torch.device("cpu")
        return coordinator

    def test_gate_is_inclusive_for_an_entire_eager_batch(self):
        coordinator = self._coordinator()
        self.assertTrue(coordinator.should_use_dense_fallback(self._batch([8, 7])))
        self.assertFalse(coordinator.should_use_dense_fallback(self._batch([8, 9])))

        with (
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "breakable_cuda_graph.is_in_breakable_cuda_graph",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
                return_value=None,
            ),
        ):
            self.assertTrue(
                coordinator._should_use_dense_fallback(
                    self._batch([8, 7]), fixed_capacity=False
                )
            )
            self.assertFalse(
                coordinator._should_use_dense_fallback(
                    self._batch([8, 9]), fixed_capacity=False
                )
            )
            self.assertFalse(
                coordinator._should_use_dense_fallback(
                    self._batch([8, 8]), fixed_capacity=64
                )
            )
            self.assertFalse(
                coordinator._should_use_dense_fallback(
                    self._batch([8, 8], spec_info=object()), fixed_capacity=False
                )
            )

    def test_dense_dispatch_preserves_metadata_and_updates_representations(self):
        coordinator = self._coordinator()
        coordinator.start_layer = 0
        coordinator.algorithm = Mock()
        coordinator.algorithm.should_update_representations.return_value = True
        coordinator.backend_adaptor = Mock()
        coordinator.token_to_kv_pool = Mock()
        coordinator.token_to_kv_pool.get_key_buffer.return_value = "key-buffer"
        coordinator.forward_begin = Mock()
        coordinator._handle_sparse_retrieve = Mock(return_value="sparse")
        coordinator._forward_sparse_mask = None
        coordinator._forward_dense_fallback = False
        metadata = object()
        layer = SimpleNamespace(layer_id=0)
        short_batch = self._batch([8, 7])

        with (
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "breakable_cuda_graph.is_in_breakable_cuda_graph",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_backend_utils."
                "tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
                return_value=None,
            ),
        ):
            result = coordinator.attention_begin(
                torch.empty(2, 1),
                torch.empty(2, 1),
                torch.empty(2, 1),
                layer,
                short_batch,
                metadata,
            )

        self.assertIs(result, metadata)
        coordinator.algorithm.begin_dense_forward.assert_called_once_with(short_batch)
        coordinator.forward_begin.assert_not_called()
        coordinator.backend_adaptor.save_original_metadata.assert_not_called()
        coordinator._handle_sparse_retrieve.assert_not_called()

        coordinator.attention_end(torch.empty(2, 1), layer, short_batch)
        coordinator.algorithm.update_representations.assert_called_once_with(
            layer_id=0,
            req_pool_indices=short_batch.req_pool_indices,
            seq_lens=short_batch.seq_lens,
            k_buffer="key-buffer",
            forward_batch=short_batch,
        )


if __name__ == "__main__":
    unittest.main()
