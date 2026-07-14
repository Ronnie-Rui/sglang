import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
    quest_update_page_representations_,
)
from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
    quest_lazy_update_page_scores,
    quest_page_scores,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestLazyUpdateScoreKernel(unittest.TestCase):
    page_size = 4

    @staticmethod
    def _make_inputs(*, k_dtype=torch.float16):
        device = torch.device("cuda")
        batch_size, max_pages = 3, 4
        kv_heads, query_heads, head_dim = 2, 4, 16
        num_pool_pages = 32
        torch.manual_seed(29)

        req_pool_indices = torch.tensor([2, 0, 1], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([13, 10, 9], dtype=torch.int64, device=device)
        req_to_token = torch.empty(
            (batch_size, max_pages * 4), dtype=torch.int64, device=device
        )
        physical_page_rows = torch.tensor(
            [[2, 5, 8, 11], [13, 16, 19, 22], [24, 27, 29, 31]],
            dtype=torch.int64,
            device=device,
        )
        token_offsets = torch.arange(4, dtype=torch.int64, device=device)
        for req_idx in range(batch_size):
            req_to_token[req_idx] = (
                physical_page_rows[req_idx, :, None] * 4 + token_offsets
            ).reshape(-1)

        physical_pages = (req_to_token[req_pool_indices, ::4] // 4).contiguous()
        num_k_tokens = (num_pool_pages + 1) * 4
        inputs = {
            "queries": torch.randn(
                batch_size,
                query_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            "page_k_min": torch.randn(
                num_pool_pages,
                kv_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            "page_k_max": torch.randn(
                num_pool_pages,
                kv_heads,
                head_dim,
                dtype=torch.float32,
                device=device,
            ),
            "page_valid": torch.ones(num_pool_pages, dtype=torch.bool, device=device),
            "physical_pages": physical_pages,
            "req_pool_indices": req_pool_indices,
            "seq_lens": seq_lens,
            "req_to_token": req_to_token,
            "k_buffer": torch.randn(
                num_k_tokens,
                kv_heads,
                head_dim,
                dtype=k_dtype,
                device=device,
            ),
            "repr_constructed": torch.tensor(
                [True, False, True], dtype=torch.bool, device=device
            ),
            "last_constructed_page": torch.tensor(
                [1, 99, 1], dtype=torch.int64, device=device
            ),
            "active_mask": torch.tensor(
                [True, False, True], dtype=torch.bool, device=device
            ),
            "history_page_counts": torch.tensor(
                [3, 2, 1], dtype=torch.int64, device=device
            ),
        }
        # Force pages due on the next lazy update to start invalid. This proves
        # the update runs even for inactive requests and recent masked pages.
        due_pages = (
            physical_pages[0, 1:3].tolist()
            + physical_pages[1, 1:2].tolist()
            + physical_pages[2, 0:2].tolist()
        )
        inputs["page_valid"][due_pages] = False
        return inputs

    @staticmethod
    def _clone_inputs(inputs):
        return {name: tensor.clone() for name, tensor in inputs.items()}

    def _run_reference(self, inputs):
        ready_end_page = torch.clamp((inputs["seq_lens"] - 1) // self.page_size, min=0)
        ready_seq_lens = ready_end_page * self.page_size
        quest_update_page_representations_(
            req_pool_indices=inputs["req_pool_indices"],
            seq_lens=ready_seq_lens,
            req_to_token=inputs["req_to_token"],
            k_buffer=inputs["k_buffer"],
            repr_constructed=inputs["repr_constructed"],
            last_constructed_page=inputs["last_constructed_page"],
            page_k_min=inputs["page_k_min"],
            page_k_max=inputs["page_k_max"],
            page_valid=inputs["page_valid"],
            page_size=self.page_size,
            advance_trackers=True,
        )
        return quest_page_scores(
            inputs["queries"],
            inputs["page_k_min"],
            inputs["page_k_max"],
            inputs["page_valid"],
            inputs["physical_pages"],
            active_mask=inputs["active_mask"],
            history_page_counts=inputs["history_page_counts"],
        )

    def _run_lazy(self, inputs):
        return quest_lazy_update_page_scores(
            **inputs,
            page_size=self.page_size,
            advance_trackers=True,
        )

    def _assert_results(
        self, actual_scores, actual_inputs, expected_scores, expected_inputs
    ):
        torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
        for name in (
            "page_k_min",
            "page_k_max",
            "page_valid",
            "repr_constructed",
            "last_constructed_page",
        ):
            torch.testing.assert_close(
                actual_inputs[name], expected_inputs[name], rtol=0, atol=0
            )

    def test_matches_update_then_score_and_updates_masked_recent_pages(self):
        source = self._make_inputs()
        expected_inputs = self._clone_inputs(source)
        actual_inputs = self._clone_inputs(source)

        expected_scores = self._run_reference(expected_inputs)
        actual_scores = self._run_lazy(actual_inputs)
        torch.cuda.synchronize()
        self._assert_results(
            actual_scores, actual_inputs, expected_scores, expected_inputs
        )

        # Batch row 1 is inactive for retrieval, but its request slot 0 still
        # advances from page 1 to ready page 2 and updates that representation.
        self.assertEqual(actual_inputs["last_constructed_page"][0].item(), 2)
        updated_physical_page = int(actual_inputs["physical_pages"][1, 1].item())
        self.assertTrue(actual_inputs["page_valid"][updated_physical_page].item())

    def test_cuda_graph_replay_uses_new_ready_boundary(self):
        source = self._make_inputs()
        actual_inputs = self._clone_inputs(source)

        # Compile before capture, then restore the mutable pools and trackers.
        self._run_lazy(actual_inputs)
        torch.cuda.synchronize()
        for name in (
            "page_k_min",
            "page_k_max",
            "page_valid",
            "repr_constructed",
            "last_constructed_page",
        ):
            actual_inputs[name].copy_(source[name])

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_scores = self._run_lazy(actual_inputs)

        # Move every row across one more ready page and compare replay against
        # the existing update-then-score pipeline from identical state.
        for name in (
            "page_k_min",
            "page_k_max",
            "page_valid",
            "repr_constructed",
            "last_constructed_page",
        ):
            actual_inputs[name].copy_(source[name])
        actual_inputs["seq_lens"].copy_(torch.tensor([14, 13, 13], device="cuda"))
        expected_inputs = self._clone_inputs(actual_inputs)
        expected_scores = self._run_reference(expected_inputs)

        graph.replay()
        torch.cuda.synchronize()
        self._assert_results(
            captured_scores, actual_inputs, expected_scores, expected_inputs
        )

    def test_tracker_does_not_advance_past_physical_page_grid(self):
        inputs = self._clone_inputs(self._make_inputs())
        inputs["seq_lens"].fill_(100)
        self._run_lazy(inputs)
        torch.cuda.synchronize()

        max_pages = inputs["physical_pages"].shape[1]
        self.assertTrue(torch.all(inputs["last_constructed_page"] <= max_pages).item())


if __name__ == "__main__":
    unittest.main()
