import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeCudaGraphSparseCapacity(unittest.TestCase):
    def test_publishes_each_selected_capacity_to_scheduler_batch(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        coordinator = Mock()
        runner.model_runner = SimpleNamespace(runtime_sparse_coordinator=coordinator)
        runner._select_sparse_graph_page_capacity = Mock(side_effect=[4, 16, 4, None])
        forward_batch = SimpleNamespace()

        self.assertEqual(runner._publish_sparse_graph_page_capacity(forward_batch), 4)
        self.assertEqual(forward_batch.runtime_sparse_page_capacity, 4)
        self.assertEqual(runner._publish_sparse_graph_page_capacity(forward_batch), 16)
        self.assertEqual(forward_batch.runtime_sparse_page_capacity, 16)
        self.assertEqual(runner._publish_sparse_graph_page_capacity(forward_batch), 4)
        self.assertEqual(forward_batch.runtime_sparse_page_capacity, 4)
        self.assertIsNone(runner._publish_sparse_graph_page_capacity(forward_batch))
        self.assertIsNone(forward_batch.runtime_sparse_page_capacity)
        self.assertEqual(coordinator.prepare_graph_forward.call_count, 4)


if __name__ == "__main__":
    unittest.main()
