import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
    quest_finalize_to_flashattention_metadata_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _reference_direct_metadata(
    topk_scores,
    topk_indices,
    k_per_req,
    recent_indices,
    recent_valid,
    sparse_mask,
    seq_lens,
    req_pool_indices,
    req_to_token,
    page_size,
):
    batch_size = topk_scores.shape[0]
    output_width = topk_scores.shape[1] + recent_indices.shape[1]
    page_table = torch.full(
        (batch_size, output_width), -99, dtype=torch.int32, device=topk_scores.device
    )
    valid_lengths = torch.empty(
        batch_size, dtype=torch.int32, device=topk_scores.device
    )
    cache_seqlens = torch.empty(
        batch_size, dtype=torch.int32, device=topk_scores.device
    )
    cu_seqlens = torch.empty(
        batch_size + 1, dtype=torch.int32, device=topk_scores.device
    )

    cumulative = 0
    for row in range(batch_size):
        selected = []
        row_k = int(k_per_req[row].item())
        for col in range(row_k):
            score = float(topk_scores[row, col].item())
            if (
                score != float("-inf")
                and score != float("inf")
                and not torch.isnan(topk_scores[row, col])
            ):
                selected.append(int(topk_indices[row, col].item()))
        for col in range(recent_indices.shape[1]):
            if bool(recent_valid[row, col].item()):
                selected.append(int(recent_indices[row, col].item()))

        selected.sort()
        valid_lengths[row] = len(selected)
        if bool(sparse_mask[row].item()) and selected:
            req_idx = int(req_pool_indices[row].item())
            for out_col, logical_page in enumerate(selected):
                first_token = req_to_token[req_idx, logical_page * page_size]
                page_table[row, out_col] = first_token // page_size

        last_page_len = int((seq_lens[row].item() - 1) % page_size + 1)
        sparse_len = (len(selected) - 1) * page_size + last_page_len
        cache_len = (
            sparse_len
            if bool(sparse_mask[row].item()) and selected
            else int(seq_lens[row].item())
        )
        cache_seqlens[row] = cache_len
        cu_seqlens[row] = cumulative
        cumulative += cache_len
    cu_seqlens[batch_size] = cumulative
    return page_table, valid_lengths, cache_seqlens, cu_seqlens


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestDirectMetadataKernel(unittest.TestCase):
    page_size = 4

    @staticmethod
    def _make_inputs(index_dtype=torch.int32):
        device = torch.device("cuda")
        topk_scores = torch.tensor(
            [
                [5.0, float("nan"), 2.0, float("-inf")],
                [float("inf"), 9.0, 8.0, 7.0],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
            ],
            dtype=torch.float32,
            device=device,
        )
        topk_indices = torch.tensor(
            [[6, 1, 3, 0], [5, 4, 0, 2], [1, 2, 3, 4]],
            dtype=index_dtype,
            device=device,
        )
        return {
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "k_per_req": torch.tensor([3, 4, 0], dtype=index_dtype, device=device),
            "recent_indices": torch.tensor(
                [[7, 8], [6, 7], [4, 5]], dtype=index_dtype, device=device
            ),
            "recent_valid": torch.tensor(
                [[True, True], [True, False], [False, False]],
                dtype=torch.bool,
                device=device,
            ),
            "sparse_mask": torch.tensor(
                [True, True, False], dtype=torch.bool, device=device
            ),
            "seq_lens": torch.tensor([35, 31, 9], dtype=torch.int64, device=device),
            "req_pool_indices": torch.tensor(
                [2, 0, 1], dtype=torch.int64, device=device
            ),
            "req_to_token": torch.arange(
                3 * 64, dtype=torch.int32, device=device
            ).reshape(3, 64)
            + 16,
        }

    def _assert_matches_reference_and_replays_graph(self, index_dtype):
        inputs = self._make_inputs(index_dtype)
        batch_size = inputs["topk_scores"].shape[0]
        output_width = (
            inputs["topk_scores"].shape[1] + inputs["recent_indices"].shape[1]
        )
        page_table = torch.full(
            (batch_size, output_width), -99, dtype=torch.int32, device="cuda"
        )
        valid_lengths = torch.empty(batch_size, dtype=torch.int32, device="cuda")
        cache_seqlens = torch.empty(batch_size, dtype=torch.int32, device="cuda")
        cu_seqlens = torch.empty(batch_size + 1, dtype=torch.int32, device="cuda")

        quest_finalize_to_flashattention_metadata_(
            **inputs,
            valid_lengths=valid_lengths,
            page_table=page_table,
            cache_seqlens_int32=cache_seqlens,
            cu_seqlens_k=cu_seqlens,
            page_size=self.page_size,
            update_lengths=True,
        )
        torch.cuda.synchronize()

        expected = _reference_direct_metadata(**inputs, page_size=self.page_size)
        torch.testing.assert_close(page_table, expected[0])
        torch.testing.assert_close(valid_lengths, expected[1])
        torch.testing.assert_close(cache_seqlens, expected[2])
        torch.testing.assert_close(cu_seqlens, expected[3])

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quest_finalize_to_flashattention_metadata_(
                **inputs,
                valid_lengths=valid_lengths,
                page_table=page_table,
                cache_seqlens_int32=cache_seqlens,
                cu_seqlens_k=cu_seqlens,
                page_size=self.page_size,
                update_lengths=True,
            )

        page_table.fill_(-99)
        valid_lengths.zero_()
        cache_seqlens.zero_()
        cu_seqlens.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(page_table, expected[0])
        torch.testing.assert_close(valid_lengths, expected[1])
        torch.testing.assert_close(cache_seqlens, expected[2])
        torch.testing.assert_close(cu_seqlens, expected[3])

    def test_matches_reference_and_replays_graph(self):
        for index_dtype in (torch.int32, torch.int64):
            with self.subTest(index_dtype=index_dtype):
                self._assert_matches_reference_and_replays_graph(index_dtype)


if __name__ == "__main__":
    unittest.main()
