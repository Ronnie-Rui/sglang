import unittest

import torch

from sglang.jit_kernel.quest.topk import (
    quest_topk,
    quest_topk_to_flashattention_metadata_out,
)
from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
    quest_finalize_to_flashattention_metadata_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestFusedTopKMetadataKernel(unittest.TestCase):
    page_size = 4

    @staticmethod
    def _make_inputs(num_scores: int, topk_width: int, index_dtype: torch.dtype):
        device = torch.device("cuda")
        batch_size = 3
        torch.manual_seed(41 + num_scores)
        scores = torch.randn(batch_size, num_scores, dtype=torch.float32, device=device)
        # Exercise the existing select-then-filter semantics: non-finite values
        # occupy top-k slots and are discarded without backfilling.
        scores[0, 0] = float("nan")
        scores[0, 1] = float("inf")
        scores[1, -1] = float("-inf")
        scores[2].fill_(float("-inf"))

        k_per_req = torch.tensor(
            [topk_width, max(topk_width // 2, 1), 0],
            dtype=index_dtype,
            device=device,
        )
        recent_width = 4
        recent_start = torch.tensor(
            [num_scores - recent_width, num_scores - recent_width - 3, 0],
            dtype=index_dtype,
            device=device,
        )
        recent_indices = recent_start[:, None] + torch.arange(
            recent_width, dtype=index_dtype, device=device
        )
        recent_valid = torch.tensor(
            [[True, True, True, True], [True, True, False, False], [False] * 4],
            dtype=torch.bool,
            device=device,
        )
        sparse_mask = torch.tensor([True, True, False], dtype=torch.bool, device=device)
        seq_lens = torch.tensor(
            [num_scores * 4, num_scores * 4 - 11, 17],
            dtype=index_dtype,
            device=device,
        )
        req_pool_indices = torch.tensor([2, 0, 1], dtype=index_dtype, device=device)

        # Each logical page maps to a deterministic but non-identity physical
        # page so the fused kernel must perform the req_to_token lookup.
        req_to_token = torch.empty(
            (batch_size, num_scores * 4), dtype=index_dtype, device=device
        )
        logical_tokens = torch.arange(num_scores * 4, device=device)
        for req_idx in range(batch_size):
            req_to_token[req_idx] = logical_tokens + (req_idx + 1) * num_scores * 8

        return {
            "scores": scores,
            "k_per_req": k_per_req,
            "recent_indices": recent_indices,
            "recent_valid": recent_valid,
            "sparse_mask": sparse_mask,
            "seq_lens": seq_lens,
            "req_pool_indices": req_pool_indices,
            "req_to_token": req_to_token,
        }

    def _run_reference(self, inputs, *, update_lengths=True):
        topk_scores, topk_indices = quest_topk(
            inputs["scores"],
            inputs["k_per_req"].to(torch.int32),
            self._topk_width,
        )
        batch_size = inputs["scores"].shape[0]
        output_width = self._topk_width + inputs["recent_indices"].shape[1]
        page_table = torch.full(
            (batch_size, output_width), -99, dtype=torch.int32, device="cuda"
        )
        valid_lengths = torch.empty(batch_size, dtype=torch.int32, device="cuda")
        cache_seqlens = torch.empty(batch_size, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")
        quest_finalize_to_flashattention_metadata_(
            topk_scores=topk_scores,
            topk_indices=topk_indices,
            k_per_req=inputs["k_per_req"],
            recent_indices=inputs["recent_indices"],
            recent_valid=inputs["recent_valid"],
            valid_lengths=valid_lengths,
            sparse_mask=inputs["sparse_mask"],
            seq_lens=inputs["seq_lens"],
            req_pool_indices=inputs["req_pool_indices"],
            req_to_token=inputs["req_to_token"],
            page_table=page_table,
            cache_seqlens_int32=cache_seqlens,
            cu_seqlens_k=cu_seqlens,
            page_size=self.page_size,
            update_lengths=update_lengths,
        )
        return page_table, valid_lengths, cache_seqlens, cu_seqlens

    def _make_outputs(self, batch_size: int, output_width: int):
        return (
            torch.full(
                (batch_size, output_width), -99, dtype=torch.int32, device="cuda"
            ),
            torch.empty(batch_size, dtype=torch.int32, device="cuda"),
            torch.empty(batch_size, dtype=torch.int32, device="cuda"),
            torch.empty(batch_size + 1, dtype=torch.int32, device="cuda"),
        )

    def _run_fused(self, inputs, outputs, *, update_lengths=True):
        page_table, valid_lengths, cache_seqlens, cu_seqlens = outputs
        quest_topk_to_flashattention_metadata_out(
            **inputs,
            page_table=page_table,
            valid_lengths=valid_lengths,
            cache_seqlens_int32=cache_seqlens,
            cu_seqlens_k=cu_seqlens,
            topk_width=self._topk_width,
            page_size=self.page_size,
            update_lengths=update_lengths,
        )

    @staticmethod
    def _assert_outputs_equal(actual, expected):
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def _assert_case(self, num_scores, topk_width, index_dtype):
        self._topk_width = topk_width
        inputs = self._make_inputs(num_scores, topk_width, index_dtype)
        output_width = topk_width + inputs["recent_indices"].shape[1]
        outputs = self._make_outputs(inputs["scores"].shape[0], output_width)

        # Compile both paths before capture.
        expected = self._run_reference(inputs)
        self._run_fused(inputs, outputs)
        torch.cuda.synchronize()
        self._assert_outputs_equal(outputs, expected)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_fused(inputs, outputs)
        output_ptrs = tuple(tensor.data_ptr() for tensor in outputs)

        torch.manual_seed(117 + num_scores)
        inputs["scores"].copy_(torch.randn_like(inputs["scores"]))
        inputs["scores"][0, 4] = float("nan")
        inputs["scores"][1, 5] = float("inf")
        outputs[0].fill_(-99)
        outputs[1].zero_()
        outputs[2].zero_()
        outputs[3].zero_()
        expected = self._run_reference(inputs)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(tuple(tensor.data_ptr() for tensor in outputs), output_ptrs)
        self._assert_outputs_equal(outputs, expected)

    def test_matches_existing_pipeline_and_replays_graph(self):
        cases = (
            (640, 39, (torch.int32, torch.int64)),
            (2112, 131, (torch.int32, torch.int64)),
            (4096, 252, (torch.int32,)),
            (4096, 508, (torch.int32,)),
            (8192, 1020, (torch.int32,)),
        )
        for num_scores, topk_width, index_dtypes in cases:
            for index_dtype in index_dtypes:
                with self.subTest(
                    num_scores=num_scores,
                    topk_width=topk_width,
                    index_dtype=index_dtype,
                ):
                    self._assert_case(num_scores, topk_width, index_dtype)

    def test_batch_one_writes_cu_seqlens_in_main_kernel(self):
        self._topk_width = 39
        full_inputs = self._make_inputs(640, self._topk_width, torch.int32)
        inputs = {
            name: tensor if name == "req_to_token" else tensor[:1]
            for name, tensor in full_inputs.items()
        }
        output_width = self._topk_width + inputs["recent_indices"].shape[1]
        expected = self._run_reference(inputs)
        actual = self._make_outputs(1, output_width)
        self._run_fused(inputs, actual)
        torch.cuda.synchronize()
        self._assert_outputs_equal(actual, expected)

    def test_batch_one_inactive_row_returns_zero_valid_length(self):
        self._topk_width = 39
        full_inputs = self._make_inputs(640, self._topk_width, torch.int32)
        inputs = {
            name: tensor if name == "req_to_token" else tensor[:1]
            for name, tensor in full_inputs.items()
        }
        inputs["sparse_mask"].zero_()
        output_width = self._topk_width + inputs["recent_indices"].shape[1]
        expected = self._run_reference(inputs)
        actual = self._make_outputs(1, output_width)
        self._run_fused(inputs, actual)
        torch.cuda.synchronize()

        self._assert_outputs_equal(actual, expected)
        self.assertEqual(actual[1].item(), 0)

    def test_update_lengths_false_preserves_sequence_metadata(self):
        self._topk_width = 39
        inputs = self._make_inputs(640, self._topk_width, torch.int32)
        output_width = self._topk_width + inputs["recent_indices"].shape[1]
        expected = self._make_outputs(inputs["scores"].shape[0], output_width)
        actual = self._make_outputs(inputs["scores"].shape[0], output_width)
        for outputs in (expected, actual):
            outputs[2].fill_(123)
            outputs[3].fill_(456)

        topk_scores, topk_indices = quest_topk(
            inputs["scores"], inputs["k_per_req"], self._topk_width
        )
        quest_finalize_to_flashattention_metadata_(
            topk_scores=topk_scores,
            topk_indices=topk_indices,
            k_per_req=inputs["k_per_req"],
            recent_indices=inputs["recent_indices"],
            recent_valid=inputs["recent_valid"],
            valid_lengths=expected[1],
            sparse_mask=inputs["sparse_mask"],
            seq_lens=inputs["seq_lens"],
            req_pool_indices=inputs["req_pool_indices"],
            req_to_token=inputs["req_to_token"],
            page_table=expected[0],
            cache_seqlens_int32=expected[2],
            cu_seqlens_k=expected[3],
            page_size=self.page_size,
            update_lengths=False,
        )
        self._run_fused(inputs, actual, update_lengths=False)
        torch.cuda.synchronize()
        self._assert_outputs_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
