import unittest

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
    quest_exact_superpage_page_scores,
    quest_page_scores,
    quest_superpage_scores,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _reference_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    history_page_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    k_min = page_k_min[physical_pages].to(torch.float32)
    k_max = page_k_max[physical_pages].to(torch.float32)
    valid = page_valid[physical_pages]
    batch_size, query_heads, head_dim = queries.shape
    kv_heads = k_min.shape[-2]
    group_size = query_heads // kv_heads
    query = queries.reshape(batch_size, kv_heads, group_size, head_dim)
    query = query.to(torch.float32).unsqueeze(1)
    bounds = torch.where(
        query >= 0,
        query * k_max.unsqueeze(3),
        query * k_min.unsqueeze(3),
    ).sum(dim=-1)
    scores = bounds.amax(dim=(2, 3))
    if active_mask is not None:
        page_idx = torch.arange(physical_pages.shape[1], device=physical_pages.device)
        valid = (
            valid
            & active_mask.unsqueeze(1)
            & (page_idx.unsqueeze(0) < history_page_counts.unsqueeze(1))
        )
    return torch.where(valid, scores, torch.full_like(scores, float("-inf")))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestScoreKernel(unittest.TestCase):
    def test_superpage_scores_are_member_page_upper_bounds(self):
        torch.manual_seed(123)
        device = torch.device("cuda")
        batch_size, pool_pages, logical_pages = 2, 13, 7
        kv_heads, query_heads, head_dim = 2, 4, 16
        page_k_min = torch.randn(pool_pages, kv_heads, head_dim, device=device)
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.ones(pool_pages, dtype=torch.bool, device=device)
        page_valid[5] = False
        physical_pages = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [12, 11, 10, 9, 8, 7, -1]],
            dtype=torch.int32,
            device=device,
        )
        queries = torch.randn(batch_size, query_heads, head_dim, device=device)
        active_mask = torch.tensor([True, True], device=device)
        history_counts = torch.tensor([7, 5], dtype=torch.int32, device=device)
        superpage_size = 4

        page_scores = quest_page_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask=active_mask,
            history_page_counts=history_counts,
        )
        superpage_scores = quest_superpage_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask,
            history_counts,
            superpage_size,
        )

        for batch_idx in range(batch_size):
            for superpage_idx in range(superpage_scores.shape[1]):
                start = superpage_idx * superpage_size
                end = min(start + superpage_size, logical_pages)
                member_scores = page_scores[batch_idx, start:end]
                finite_members = member_scores[torch.isfinite(member_scores)]
                if finite_members.numel() == 0:
                    self.assertTrue(
                        torch.isneginf(
                            superpage_scores[batch_idx, superpage_idx]
                        ).item()
                    )
                else:
                    self.assertGreaterEqual(
                        superpage_scores[batch_idx, superpage_idx].item(),
                        finite_members.max().item(),
                    )

    def test_exact_superpage_scores_repair_and_replay_graph(self):
        device = torch.device("cuda")
        page_k_min = torch.zeros((8, 1, 2), device=device)
        page_k_max = torch.zeros_like(page_k_min)
        page_valid = torch.ones(8, dtype=torch.bool, device=device)
        physical_pages = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=device
        )
        queries = torch.ones((2, 1, 2), device=device)
        active_mask = torch.ones(2, dtype=torch.bool, device=device)
        history_counts = torch.full((2,), 4, dtype=torch.int32, device=device)
        k_per_req = torch.ones(2, dtype=torch.int32, device=device)
        clear = torch.tensor(
            [[20.0, 20.0], [0.0, 0.0], [11.0, 1.0], [0.0, 0.0]],
            device=device,
        )
        loose = torch.tensor(
            [[10.0, 0.0], [0.0, 10.0], [11.0, 1.0], [0.0, 0.0]],
            device=device,
        )
        page_k_min[:4].copy_(clear.unsqueeze(1))
        page_k_min[4:].copy_(loose.unsqueeze(1))
        page_k_max.copy_(page_k_min)

        def hierarchical():
            return quest_exact_superpage_page_scores(
                queries,
                page_k_min,
                page_k_max,
                page_valid,
                physical_pages,
                active_mask,
                history_counts,
                k_per_req,
                max_k=1,
                superpage_size=2,
                oversample=1,
            )

        def full_topk():
            scores = quest_page_scores(
                queries,
                page_k_min,
                page_k_max,
                page_valid,
                physical_pages,
                active_mask=active_mask,
                history_page_counts=history_counts,
            )
            return torch.topk(scores, k=1, dim=1).indices

        eager_scores, eager_certified = hierarchical()
        self.assertEqual(eager_certified.tolist(), [True, False])
        torch.testing.assert_close(
            torch.topk(eager_scores, k=1, dim=1).indices, full_topk()
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                hierarchical()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_scores, captured_certified = hierarchical()

        page_k_min[:4].copy_(loose.unsqueeze(1))
        page_k_min[4:].copy_(clear.unsqueeze(1))
        page_k_max.copy_(page_k_min)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(captured_certified.tolist(), [False, True])
        torch.testing.assert_close(
            torch.topk(captured_scores, k=1, dim=1).indices, full_topk()
        )

    def test_exact_superpage_scores_match_full_scan_for_ragged_rows(self):
        torch.manual_seed(314)
        device = torch.device("cuda")
        batch_size, num_pages = 3, 17
        kv_heads, query_heads, head_dim = 2, 4, 16
        total_pages = batch_size * num_pages
        page_k_min = torch.randn(total_pages, kv_heads, head_dim, device=device)
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.ones(total_pages, dtype=torch.bool, device=device)
        physical_pages = torch.arange(
            total_pages, dtype=torch.int32, device=device
        ).reshape(batch_size, num_pages)
        physical_pages[1] = physical_pages[1].flip(0)
        queries = torch.randn(batch_size, query_heads, head_dim, device=device)
        active_mask = torch.tensor([True, True, False], device=device)
        history_counts = torch.tensor([17, 11, 0], dtype=torch.int32, device=device)

        full_scores = quest_page_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask=active_mask,
            history_page_counts=history_counts,
        )
        for superpage_size in (2, 4, 8, 16):
            for max_k in sorted({1, min(superpage_size, 5), 5}):
                with self.subTest(superpage_size=superpage_size, max_k=max_k):
                    k_per_req = torch.tensor(
                        [max_k, min(max_k, 11), 0],
                        dtype=torch.int32,
                        device=device,
                    )
                    hierarchical_scores, _ = quest_exact_superpage_page_scores(
                        queries,
                        page_k_min,
                        page_k_max,
                        page_valid,
                        physical_pages,
                        active_mask,
                        history_counts,
                        k_per_req,
                        max_k=max_k,
                        superpage_size=superpage_size,
                        oversample=1,
                    )
                    full_topk = torch.topk(
                        full_scores, k=max_k, dim=1, sorted=True
                    ).indices
                    hierarchical_topk = torch.topk(
                        hierarchical_scores, k=max_k, dim=1, sorted=True
                    ).indices
                    for row, row_k in enumerate(k_per_req.tolist()):
                        if row_k == 0:
                            continue
                        self.assertEqual(
                            set(hierarchical_topk[row, :row_k].tolist()),
                            set(full_topk[row, :row_k].tolist()),
                        )

    def test_equal_and_nonfinite_certificates_always_repair(self):
        device = torch.device("cuda")
        queries = torch.ones((3, 1, 2), device=device)
        physical_pages = torch.arange(12, dtype=torch.int32, device=device).reshape(
            3, 4
        )
        page_valid = torch.ones(12, dtype=torch.bool, device=device)
        active_mask = torch.ones(3, dtype=torch.bool, device=device)
        history_counts = torch.full((3,), 4, dtype=torch.int32, device=device)
        k_per_req = torch.ones(3, dtype=torch.int32, device=device)
        bounds = torch.tensor(
            [
                [5.0, 0.0],
                [0.0, 5.0],
                [3.0, 2.0],
                [0.0, 0.0],
                [float("nan"), 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
                [float("inf"), 0.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.0, 0.0],
            ],
            device=device,
        ).unsqueeze(1)

        max_bounds = bounds.clone()
        scores, certified = quest_exact_superpage_page_scores(
            queries,
            bounds,
            max_bounds,
            page_valid,
            physical_pages,
            active_mask,
            history_counts,
            k_per_req,
            max_k=1,
            superpage_size=2,
            oversample=1,
        )

        self.assertEqual(certified.tolist(), [False, False, False])
        full_scores = quest_page_scores(
            queries,
            bounds,
            max_bounds,
            page_valid,
            physical_pages,
            active_mask=active_mask,
            history_page_counts=history_counts,
        )
        torch.testing.assert_close(scores, full_scores, equal_nan=True)

    def test_matches_reference_for_gqa_and_noncontiguous_queries(self):
        torch.manual_seed(42)
        device = torch.device("cuda")
        batch_size = 3
        num_pool_pages = 19
        num_pages = 11
        query_heads = 16
        kv_heads = 8
        head_dim = 128

        page_k_min = torch.randn(
            num_pool_pages,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.ones(num_pool_pages, dtype=torch.bool, device=device)
        page_valid[[2, 13]] = False
        physical_page_storage = torch.empty(
            (batch_size, num_pages * 2), dtype=torch.int64, device=device
        )
        pool_indices = torch.arange(num_pool_pages, device=device)
        for batch_idx in range(batch_size):
            physical_page_storage[batch_idx, ::2] = torch.roll(
                pool_indices, shifts=batch_idx * 3
            )[:num_pages]
        physical_pages = physical_page_storage[:, ::2]

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                query_storage = torch.randn(
                    batch_size,
                    query_heads,
                    head_dim * 2,
                    dtype=dtype,
                    device=device,
                )
                queries = query_storage[:, :, ::2]
                expected = _reference_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                )

                actual_3d = quest_page_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                )
                flat_storage = torch.empty(
                    batch_size,
                    query_heads * head_dim * 2,
                    dtype=dtype,
                    device=device,
                )
                flat_storage[:, ::2] = queries.reshape(batch_size, -1)
                actual_2d = quest_page_scores(
                    flat_storage[:, ::2],
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                )

                torch.testing.assert_close(actual_3d, expected, rtol=2e-4, atol=2e-3)
                torch.testing.assert_close(actual_2d, expected, rtol=2e-4, atol=2e-3)
                topk = min(5, num_pages)
                self.assertTrue(
                    torch.equal(
                        actual_3d.topk(topk, dim=1).indices,
                        expected.topk(topk, dim=1).indices,
                    )
                )

    def test_native_bounds_are_scored_against_float32_reference(self):
        torch.manual_seed(9)
        device = torch.device("cuda")
        physical_pages = torch.tensor(
            [[0, 2, 4, 6], [1, 3, 5, 7]],
            dtype=torch.int64,
            device=device,
        )
        page_valid = torch.tensor(
            [True, True, False, True, True, True, True, False],
            dtype=torch.bool,
            device=device,
        )

        for bounds_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(bounds_dtype=bounds_dtype):
                page_k_min = torch.randn((8, 2, 64), dtype=bounds_dtype, device=device)
                page_k_max = (
                    page_k_min.to(torch.float32)
                    + torch.rand((8, 2, 64), dtype=torch.float32, device=device)
                ).to(bounds_dtype)
                queries = torch.randn((2, 4, 64), dtype=bounds_dtype, device=device)

                expected = _reference_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                )
                actual = quest_page_scores(
                    queries,
                    page_k_min,
                    page_k_max,
                    page_valid,
                    physical_pages,
                )

                self.assertEqual(expected.dtype, torch.float32)
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)

    def test_rejects_mismatched_page_bounds_dtype(self):
        device = torch.device("cuda")
        with self.assertRaisesRegex(ValueError, "must share.*dtype"):
            quest_page_scores(
                torch.empty((1, 2, 8), dtype=torch.float16, device=device),
                torch.empty((4, 1, 8), dtype=torch.float16, device=device),
                torch.empty((4, 1, 8), dtype=torch.bfloat16, device=device),
                torch.ones(4, dtype=torch.bool, device=device),
                torch.zeros((1, 1), dtype=torch.int64, device=device),
            )

    def test_empty_batch_and_page_dimensions(self):
        device = torch.device("cuda")
        page_k_min = torch.empty((4, 1, 8), device=device)
        page_k_max = torch.empty_like(page_k_min)
        page_valid = torch.ones(4, dtype=torch.bool, device=device)

        empty_batch = quest_page_scores(
            torch.empty((0, 1, 8), device=device),
            page_k_min,
            page_k_max,
            page_valid,
            torch.empty((0, 3), dtype=torch.int64, device=device),
        )
        empty_pages = quest_page_scores(
            torch.empty((2, 1, 8), device=device),
            page_k_min,
            page_k_max,
            page_valid,
            torch.empty((2, 0), dtype=torch.int64, device=device),
        )

        self.assertEqual(empty_batch.shape, (0, 3))
        self.assertEqual(empty_pages.shape, (2, 0))

    def test_fuses_retrieval_mask_and_replays_graph(self):
        torch.manual_seed(17)
        device = torch.device("cuda")
        batch_size, num_pool_pages = 3, 13
        queries = torch.randn((batch_size, 4, 16), device=device)
        page_k_min = torch.randn((num_pool_pages, 2, 16), device=device)
        page_k_max = page_k_min + torch.rand_like(page_k_min)
        page_valid = torch.ones(num_pool_pages, dtype=torch.bool, device=device)
        page_valid[5] = False
        physical_pages = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0], [7, 8, 9, 10, 11, 12, 0]],
            dtype=torch.int64,
            device=device,
        )
        active_mask = torch.tensor([True, False, True], device=device)
        history_page_counts = torch.tensor([4, 6, 0], device=device)

        def score():
            return quest_page_scores(
                queries,
                page_k_min,
                page_k_max,
                page_valid,
                physical_pages,
                active_mask=active_mask,
                history_page_counts=history_page_counts,
            )

        actual = score()
        expected = _reference_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask,
            history_page_counts,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = score()
        active_mask.copy_(torch.tensor([False, True, True], device=device))
        history_page_counts.copy_(torch.tensor([7, 3, 5], device=device))
        graph.replay()
        torch.cuda.synchronize()
        replay_expected = _reference_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask,
            history_page_counts,
        )
        torch.testing.assert_close(captured, replay_expected, rtol=2e-4, atol=2e-3)

    def test_rejects_noncontiguous_representation_pool(self):
        device = torch.device("cuda")
        contiguous_min = torch.empty((4, 2, 8), device=device)
        contiguous_max = torch.empty_like(contiguous_min)
        noncontiguous_min = torch.empty((2, 4, 8), device=device).transpose(0, 1)
        noncontiguous_max = torch.empty((2, 4, 8), device=device).transpose(0, 1)
        contiguous_valid = torch.ones(4, dtype=torch.bool, device=device)
        noncontiguous_valid = torch.ones(8, dtype=torch.bool, device=device)[::2]

        cases = (
            (noncontiguous_min, contiguous_max, contiguous_valid, "min/max"),
            (contiguous_min, noncontiguous_max, contiguous_valid, "min/max"),
            (contiguous_min, contiguous_max, noncontiguous_valid, "validity"),
        )
        for page_k_min, page_k_max, page_valid, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                quest_page_scores(
                    torch.empty((1, 2, 8), device=device),
                    page_k_min,
                    page_k_max,
                    page_valid,
                    torch.zeros((1, 1), dtype=torch.int64, device=device),
                )

    def test_out_of_range_pages_are_masked(self):
        device = torch.device("cuda")
        page_k_min = torch.zeros((2, 1, 8), device=device)
        page_k_max = torch.ones_like(page_k_min)
        scores = quest_page_scores(
            torch.ones((1, 1, 8), device=device),
            page_k_min,
            page_k_max,
            torch.ones(2, dtype=torch.bool, device=device),
            torch.tensor([[-1, 0, 2]], dtype=torch.int32, device=device),
        )

        self.assertEqual(scores[0, 1].item(), 8.0)
        self.assertTrue(torch.isneginf(scores[0, [0, 2]]).all().item())


if __name__ == "__main__":
    unittest.main()
