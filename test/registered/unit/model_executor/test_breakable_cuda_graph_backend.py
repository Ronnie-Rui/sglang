import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBreakableCudaGraphDataclassOutput(unittest.TestCase):
    def setUp(self):
        self.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)

    def test_dataclass_tensor_fields_are_buffered_copied_and_sliced(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(6, dtype=torch.float32).view(2, 3),
            hidden_states=torch.arange(8, dtype=torch.float32).view(2, 4),
            customized_info={"source": ["decode"]},
        )

        self.assertEqual(self.backend._output_rows(output, cap=4), 2)
        buffer = self.backend._alloc_full_buffer(output, size=4)
        self.assertIsInstance(buffer, LogitsProcessorOutput)
        self.assertEqual(buffer.next_token_logits.shape, (4, 3))
        self.assertEqual(buffer.hidden_states.shape, (4, 4))

        self.backend._copy_output_to_buffer(output, buffer, num_tokens=2)
        sliced = self.backend._slice_output(buffer, num_tokens=2)

        torch.testing.assert_close(
            sliced.next_token_logits, output.next_token_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sliced.hidden_states, output.hidden_states, rtol=0, atol=0
        )
        self.assertEqual(sliced.customized_info, {"source": ["decode"]})

    def test_optional_dataclass_fields_remain_none(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.ones((1, 3)),
            hidden_states=None,
            customized_info=None,
        )

        buffer = self.backend._alloc_full_buffer(output, size=2)
        self.backend._copy_output_to_buffer(output, buffer, num_tokens=1)
        sliced = self.backend._slice_output(buffer, num_tokens=1)

        self.assertIsNone(sliced.hidden_states)
        self.assertIsNone(sliced.customized_info)
        self.assertEqual(sliced.next_token_logits.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
