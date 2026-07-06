import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBreakableCudaGraphOutputBuffer(unittest.TestCase):
    def test_logits_processor_output_is_buffered_and_sliced(self):
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(6, dtype=torch.float32).view(2, 3),
            hidden_states=torch.arange(8, dtype=torch.float32).view(2, 4),
            customized_info={"source": ["decode"]},
        )

        self.assertEqual(backend._output_rows(output), 2)
        output_buffer = backend._alloc_full_buffer(output, size=4)
        self.assertIsInstance(output_buffer, LogitsProcessorOutput)
        self.assertEqual(output_buffer.next_token_logits.shape, (4, 3))
        self.assertEqual(output_buffer.hidden_states.shape, (4, 4))

        backend._copy_output_to_buffer(output, output_buffer, num_tokens=2)
        sliced = backend._slice_output(output_buffer, num_tokens=2)

        self.assertTrue(torch.equal(sliced.next_token_logits, output.next_token_logits))
        self.assertTrue(torch.equal(sliced.hidden_states, output.hidden_states))
        self.assertEqual(sliced.customized_info, {"source": ["decode"]})

    def test_expanded_decode_output_is_not_clamped_to_batch_size(self):
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(24, dtype=torch.float32).view(8, 3),
            hidden_states=torch.arange(32, dtype=torch.float32).view(8, 4),
        )

        batch_size = 2
        output_rows = backend._output_rows(output)
        output_capacity = max(batch_size, output_rows)
        output_buffer = backend._alloc_full_buffer(output, output_capacity)
        backend._copy_output_to_buffer(output, output_buffer, output_rows)
        sliced = backend._slice_output(output_buffer, output_rows)

        self.assertEqual(output_rows, 8)
        self.assertEqual(sliced.next_token_logits.shape, (8, 3))
        self.assertTrue(torch.equal(sliced.next_token_logits, output.next_token_logits))


if __name__ == "__main__":
    unittest.main()
