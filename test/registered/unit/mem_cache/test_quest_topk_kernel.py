import unittest

import torch

from sglang.jit_kernel.quest.topk import quest_topk, quest_topk_out
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestTopKKernel(unittest.TestCase):
    @staticmethod
    def _assert_matches_reference(
        scores: torch.Tensor,
        k_per_req: torch.Tensor,
        actual_scores: torch.Tensor,
        actual_indices: torch.Tensor,
        *,
        check_indices: bool = True,
    ) -> None:
        output_width = actual_indices.shape[1]
        for row, row_k in enumerate(k_per_req.cpu().tolist()):
            expected_scores, expected_indices = torch.topk(
                scores[row], k=row_k, sorted=True
            )
            selected_scores = actual_scores[row, :row_k]
            selected_indices = actual_indices[row, :row_k]
            torch.testing.assert_close(
                selected_scores.sort(descending=True).values,
                expected_scores,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                selected_scores,
                scores[row, selected_indices],
                rtol=0,
                atol=0,
            )
            if check_indices:
                torch.testing.assert_close(
                    selected_indices.sort().values,
                    expected_indices.to(selected_indices.dtype).sort().values,
                    rtol=0,
                    atol=0,
                )
            if row_k < output_width:
                self_suffix = actual_indices[row, row_k:]
                assert torch.equal(self_suffix, torch.full_like(self_suffix, -1))
                assert torch.isneginf(actual_scores[row, row_k:]).all()

    def test_random_ragged_rows_match_torch(self):
        torch.manual_seed(42)
        for num_pages, output_width in (
            (640, 39),
            (1536, 95),
            (2112, 131),
            (2560, 155),
        ):
            with self.subTest(num_pages=num_pages, output_width=output_width):
                scores = torch.randn(4, num_pages, dtype=torch.float32, device="cuda")
                k_per_req = torch.tensor(
                    [0, 1, output_width // 2, output_width],
                    dtype=torch.int32,
                    device="cuda",
                )
                actual_scores, actual_indices = quest_topk(
                    scores, k_per_req, output_width
                )
                self._assert_matches_reference(
                    scores, k_per_req, actual_scores, actual_indices
                )

    def test_ties_and_negative_infinity(self):
        scores = torch.tensor(
            [
                [4.0, 4.0, 3.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, float("-inf"), float("-inf"), -2.0, -3.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        scores = torch.nn.functional.pad(scores, (0, 2), value=-10.0)
        k_per_req = torch.tensor([3, 7], dtype=torch.int32, device="cuda")
        actual_scores, actual_indices = quest_topk(scores, k_per_req, 7)
        self._assert_matches_reference(
            scores,
            k_per_req,
            actual_scores,
            actual_indices,
            check_indices=False,
        )

    def test_more_than_2048_candidates_in_one_coarse_bin(self):
        # 1.0 and 1.001 map to adjacent fp16 values whose low four bits differ,
        # so the old 12-bit coarse histogram put all 8192 candidates in one bin
        # and truncated the exact-selection set to 2048 entries.
        scores = torch.cat(
            [
                torch.full((4096,), 1.0, dtype=torch.float32, device="cuda"),
                torch.full((4096,), 1.001, dtype=torch.float32, device="cuda"),
            ]
        ).unsqueeze(0)
        k_per_req = torch.tensor([2048], dtype=torch.int32, device="cuda")

        actual_scores, actual_indices = quest_topk(scores, k_per_req, 2048)
        self._assert_matches_reference(
            scores,
            k_per_req,
            actual_scores,
            actual_indices,
            check_indices=False,
        )
        self.assertTrue((actual_indices >= 4096).all().item())

    def test_nan_is_selected_with_original_score_for_finalize_filtering(self):
        scores = torch.zeros((1, 1024), dtype=torch.float32, device="cuda")
        scores[0, 0] = float("inf")
        scores[0, -1] = float("nan")
        k_per_req = torch.tensor([1], dtype=torch.int32, device="cuda")

        actual_scores, actual_indices = quest_topk(scores, k_per_req, 1)
        expected = torch.topk(scores, k=1, dim=1)

        torch.testing.assert_close(actual_indices, expected.indices.to(torch.int32))
        self.assertTrue(torch.isnan(actual_scores).all().item())
        self.assertFalse(torch.isfinite(actual_scores).any().item())

    def test_select_all_and_padding(self):
        scores = torch.tensor([[3.0, -1.0, 2.0]], dtype=torch.float32, device="cuda")
        k_per_req = torch.tensor([5], dtype=torch.int32, device="cuda")
        actual_scores, actual_indices = quest_topk(scores, k_per_req, 5)
        torch.testing.assert_close(actual_scores[0, :3], scores[0], rtol=0, atol=0)
        torch.testing.assert_close(
            actual_indices[0, :3],
            torch.arange(3, dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.isneginf(actual_scores[0, 3:]).all().item())
        self.assertTrue((actual_indices[0, 3:] == -1).all().item())

    def test_cuda_graph_replay_uses_stable_outputs(self):
        torch.manual_seed(7)
        scores = torch.randn(4, 640, dtype=torch.float32, device="cuda")
        k_per_req = torch.tensor([39, 20, 1, 0], dtype=torch.int32, device="cuda")
        output_scores = torch.empty(4, 39, dtype=torch.float32, device="cuda")
        output_indices = torch.empty(4, 39, dtype=torch.int32, device="cuda")

        quest_topk_out(scores, k_per_req, output_scores, output_indices)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quest_topk_out(scores, k_per_req, output_scores, output_indices)

        score_ptr = output_scores.data_ptr()
        index_ptr = output_indices.data_ptr()
        scores.copy_(torch.randn_like(scores))
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(output_scores.data_ptr(), score_ptr)
        self.assertEqual(output_indices.data_ptr(), index_ptr)
        self._assert_matches_reference(scores, k_per_req, output_scores, output_indices)


if __name__ == "__main__":
    unittest.main()
