import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeCudaGraphSparseCapacity(unittest.TestCase):
    def test_capture_context_publishes_runtime_sparse_coordinator(self):
        source = textwrap.dedent(
            inspect.getsource(DecodeCudaGraphRunner.capture_one_shape)
        )
        tree = ast.parse(source)
        context_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ForwardContext"
        ]
        coordinator_bindings = [
            ast.unparse(keyword.value)
            for call in context_calls
            for keyword in call.keywords
            if keyword.arg == "runtime_sparse_coordinator"
        ]

        self.assertEqual(
            coordinator_bindings,
            ["self.model_runner.runtime_sparse_coordinator"],
        )

    def test_publishes_selected_capacity_and_resets_graph_forward(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        coordinator = Mock()
        runner.model_runner = SimpleNamespace(runtime_sparse_coordinator=coordinator)
        runner._select_sparse_graph_page_capacity = Mock(side_effect=[4, 16, None])
        forward_batch = SimpleNamespace()

        for expected in (4, 16, None):
            actual = runner._publish_sparse_graph_page_capacity(forward_batch)
            self.assertEqual(actual, expected)
            self.assertEqual(forward_batch.runtime_sparse_page_capacity, expected)

        self.assertEqual(coordinator.prepare_graph_forward.call_count, 3)

    def test_graph_key_isolated_by_sparse_capacity(self):
        runner = object.__new__(DecodeCudaGraphRunner)

        small = runner._make_graph_key(8, sparse_page_capacity=256)
        large = runner._make_graph_key(8, sparse_page_capacity=1024)
        dense = runner._make_graph_key(8, sparse_page_capacity=None)

        self.assertNotEqual(small, large)
        self.assertNotEqual(small, dense)
        self.assertEqual(small.sparse_page_capacity, 256)
        self.assertEqual(large.sparse_page_capacity, 1024)
        self.assertIsNone(dense.sparse_page_capacity)


if __name__ == "__main__":
    unittest.main()
