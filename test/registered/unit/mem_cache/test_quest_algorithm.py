"""Equivalence tests for the optimized Quest sparse attention algorithm.

These tests lock the behavior of the vectorized ``retrieve_topk`` path and the
full-page fast path in ``_compute_page_representations`` against the original
per-request reference implementations. They run on CPU and do not start a
server or call a real attention backend.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Config:
    def __init__(self, page_size, sparsity_ratio, num_recent_pages):
        self.page_size = page_size
        self.sparse_extra_config = {
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": num_recent_pages,
        }


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.max_context_len = req_to_token.shape[1]


class _FakeTokenToKVPool:
    def __init__(self, key_buffer):
        self.key_buffer = key_buffer

    def get_key_buffer(self, layer_id):
        return self.key_buffer


class _FakeStates:
    def __init__(self, size, device):
        self.repr_constructed = torch.zeros(size, dtype=torch.bool, device=device)
        self.prompt_lens = torch.zeros(size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(size, dtype=torch.int64, device=device)


class _FakeForwardBatch:
    def __init__(self, seq_lens, seq_lens_cpu=True):
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens.cpu() if seq_lens_cpu else None


def _build_req_to_token(batch_size, seq_lens, page_size, device):
    """Contiguous KV layout: request r owns a block of aligned token ids."""
    max_seq = int(seq_lens.max().item())
    tokens_per_req = ((max_seq + page_size - 1) // page_size) * page_size
    req_to_token = torch.zeros((batch_size, max_seq), dtype=torch.int32, device=device)
    positions = torch.arange(max_seq, dtype=torch.int32, device=device)
    for r in range(batch_size):
        req_to_token[r] = r * tokens_per_req + positions
    total_tokens = batch_size * tokens_per_req
    return req_to_token, total_tokens


def _make_algorithm(
    batch_size,
    seq_lens,
    page_size,
    sparsity_ratio,
    num_recent_pages,
    kv_heads,
    head_dim,
    device,
    seed,
):
    torch.manual_seed(seed)
    req_to_token, total_tokens = _build_req_to_token(
        batch_size, seq_lens, page_size, device
    )
    k_buffer = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=torch.float32, device=device
    )
    config = _Config(page_size, sparsity_ratio, num_recent_pages)
    algo = QuestAlgorithm(config, device)
    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=1,
        token_to_kv_pool=_FakeTokenToKVPool(k_buffer),
        req_to_token_pool=_FakeReqToTokenPool(req_to_token),
        states=_FakeStates(batch_size, device),
    )
    return algo, k_buffer


def _populate_page_reps(algo, batch_size, seq_lens, k_buffer, device):
    """Build page representations for all complete pages of every request."""
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    end_pages = seq_lens // algo.page_size
    algo._compute_page_representations(
        0,
        req_pool_indices,
        seq_lens,
        0,
        end_pages,
        k_buffer,
    )


def _reference_retrieve_topk(
    algo, queries, layer_id, req_pool_indices, sparse_mask, forward_batch
):
    """Original per-request loop implementation (pre-optimization)."""
    bs, device = queries.shape[0], queries.device
    seq_lens = forward_batch.seq_lens.to(device)
    req_to_token = algo.req_to_token_pool.req_to_token
    max_req_tokens = req_to_token.shape[1]

    per_request_indices = []
    per_request_lengths = []
    for i in range(bs):
        if not sparse_mask[i]:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue
        num_pages = int((seq_lens[i].item() + algo.page_size - 1) // algo.page_size)
        if num_pages <= algo.num_recent_pages:
            per_request_indices.append(torch.empty(0, device=device, dtype=torch.int32))
            per_request_lengths.append(0)
            continue

        page_idx = torch.arange(num_pages, device=device)
        page_start_token = req_to_token[
            req_pool_indices[i],
            (page_idx * algo.page_size).clamp(0, max_req_tokens - 1),
        ]
        phys_pages = (page_start_token // algo.page_size).unsqueeze(0)
        scores = algo._retrieve_page_scores(
            layer_id, phys_pages, req_pool_indices[i : i + 1], queries[i : i + 1]
        )
        recent_start = max(num_pages - algo.num_recent_pages, 0)
        scores = scores.clone()
        scores[:, recent_start:] = float("-inf")
        history_pages = max(recent_start, 1)
        k = max(int(history_pages * algo.sparsity_ratio), 1)
        k = min(k, history_pages)
        topk_idx = torch.topk(scores, k=k, dim=1, sorted=False)[1].squeeze(0)
        recent_idx = torch.arange(
            recent_start, recent_start + algo.num_recent_pages, device=device
        )
        recent_idx = recent_idx[recent_idx < num_pages]
        combined = torch.cat([topk_idx, recent_idx], dim=0).sort()[0].to(torch.int32)
        per_request_indices.append(combined)
        per_request_lengths.append(int(combined.numel()))

    return per_request_indices, per_request_lengths


def _sorted_rows(indices, lengths):
    """Per-request sorted page lists, truncated to each row's valid length.

    Works for both the optimized output (a padded ``[bs, max]`` tensor with a
    tensor of lengths) and the reference output (a list of variable-length
    tensors with a list of lengths).
    """
    rows = []
    for i in range(len(lengths)):
        length = int(lengths[i])
        rows.append(sorted(indices[i][:length].tolist()))
    return rows


class TestQuestRetrieveTopkEquivalence(CustomTestCase):
    device = torch.device("cpu")

    def _run_case(
        self,
        seq_lens_list,
        page_size=16,
        sparsity_ratio=0.5,
        num_recent_pages=4,
        kv_heads=2,
        q_heads=4,
        head_dim=8,
        sparse_mask_list=None,
        seed=0,
        use_seq_lens_cpu=True,
    ):
        device = self.device
        batch_size = len(seq_lens_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            sparsity_ratio,
            num_recent_pages,
            kv_heads,
            head_dim,
            device,
            seed,
        )
        _populate_page_reps(algo, batch_size, seq_lens, k_buffer, device)

        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        queries = torch.randn(
            batch_size, q_heads, head_dim, dtype=torch.float32, device=device
        )
        if sparse_mask_list is None:
            sparse_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            sparse_mask = torch.tensor(
                sparse_mask_list, dtype=torch.bool, device=device
            )
        forward_batch = _FakeForwardBatch(seq_lens, use_seq_lens_cpu)

        opt_indices, opt_lengths = algo.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        ref_indices, ref_lengths = _reference_retrieve_topk(
            algo, queries, 0, req_pool_indices, sparse_mask, forward_batch
        )

        opt_rows = _sorted_rows(opt_indices, opt_lengths)
        ref_rows = _sorted_rows(ref_indices, ref_lengths)
        self.assertEqual(
            opt_lengths.tolist(),
            [len(r) for r in ref_rows],
            msg=f"length mismatch for seq_lens={seq_lens_list}",
        )
        self.assertEqual(
            opt_rows,
            ref_rows,
            msg=f"selected pages mismatch for seq_lens={seq_lens_list}",
        )
        for row, length in zip(opt_indices.tolist(), opt_lengths.tolist()):
            self.assertEqual(row[:length], sorted(row[:length]))
            self.assertTrue(all(page >= 0 for page in row[:length]))
            self.assertTrue(all(page == -1 for page in row[length:]))

    def test_uniform_aligned(self):
        self._run_case([512, 512, 512, 512])

    def test_ragged_unaligned(self):
        self._run_case([511, 333, 257, 129])

    def test_ragged_unaligned_without_cpu_mirror(self):
        self._run_case([511, 333, 257, 129], use_seq_lens_cpu=False)

    def test_mixed_sparse_mask(self):
        self._run_case(
            [512, 480, 400, 320],
            sparse_mask_list=[True, False, True, False],
        )

    def test_short_context_no_sparsity(self):
        # All requests have num_pages <= num_recent_pages -> empty selection.
        self._run_case([32, 48, 16, 64], page_size=16, num_recent_pages=4)

    def test_mixed_short_and_long(self):
        self._run_case([48, 512, 32, 400], page_size=16, num_recent_pages=4)

    def test_gqa_grouped_heads(self):
        self._run_case([512, 448], kv_heads=2, q_heads=8, head_dim=8)

    def test_batch_size_one(self):
        self._run_case([777])

    def test_batch_size_one_without_cpu_mirror(self):
        self._run_case([777], use_seq_lens_cpu=False)

    def test_batch_size_one_inactive_has_only_padding(self):
        self._run_case([777], sparse_mask_list=[False])

    def test_different_page_size(self):
        self._run_case([1024, 800, 640], page_size=32)

    def test_ragged_k_selects_each_rows_actual_topk(self):
        seq_lens = torch.tensor([9, 5], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=2,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        scores = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
                [100.0, 1.0, 99.0, 98.0, 97.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=self.device,
        )

        def retrieve_scores(layer_id, phys_pages, req_pool_indices, queries):
            return scores[:, : phys_pages.shape[1]]

        algo._retrieve_page_scores = retrieve_scores
        indices, lengths = algo.retrieve_topk(
            queries=torch.zeros((2, 1, 1), device=self.device),
            layer_id=0,
            req_pool_indices=torch.arange(2, device=self.device),
            sparse_mask=torch.ones(2, dtype=torch.bool, device=self.device),
            forward_batch=_FakeForwardBatch(seq_lens),
        )

        self.assertEqual(lengths.tolist(), [5, 3])
        self.assertEqual(indices[0].tolist(), [4, 5, 6, 7, 8])
        self.assertEqual(indices[1].tolist(), [0, 2, 4, -1, -1])

    def test_invalid_history_pages_are_excluded(self):
        seq_lens = torch.tensor([8], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.75,
            num_recent_pages=2,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        algo.page_valid[0].zero_()
        algo.page_valid[0][torch.tensor([1, 4], device=self.device)] = True

        indices, lengths = algo.retrieve_topk(
            queries=torch.zeros((1, 1, 1), device=self.device),
            layer_id=0,
            req_pool_indices=torch.zeros(1, dtype=torch.long, device=self.device),
            sparse_mask=torch.ones(1, dtype=torch.bool, device=self.device),
            forward_batch=_FakeForwardBatch(seq_lens),
        )

        self.assertEqual(lengths.tolist(), [4])
        self.assertEqual(indices[0].tolist(), [1, 4, 6, 7, -1, -1])

    def test_forward_plan_reuses_ragged_layout_and_physical_mapping(self):
        seq_lens = torch.tensor([9, 5], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=2,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.arange(2, device=self.device)
        sparse_mask = torch.ones(2, dtype=torch.bool, device=self.device)

        with patch.object(
            algo,
            "_get_seq_lens_cpu",
            wraps=algo._get_seq_lens_cpu,
        ) as get_seq_lens_cpu:
            algo.begin_forward(
                forward_batch,
                req_pool_indices,
                sparse_mask,
                self.device,
            )
        get_seq_lens_cpu.assert_called_once_with(forward_batch, 2)
        plan = algo._retrieval_plan
        self.assertEqual(plan.seq_lens_cpu, [9, 5])
        self.assertEqual(plan.num_pages_cpu, [9, 5])
        self.assertEqual(plan.k_per_req.tolist(), [4, 2])
        self.assertEqual(plan.max_k, 4)

        queries = torch.zeros((2, 1, 1), device=self.device)
        with patch.object(
            algo,
            "_build_retrieval_plan",
            wraps=algo._build_retrieval_plan,
        ) as build_plan:
            selected_indices, _ = algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
            algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
        build_plan.assert_not_called()

        physical_pages = algo.get_selected_physical_pages(selected_indices)
        expected = torch.where(
            selected_indices >= 0,
            selected_indices + torch.tensor([[0], [9]], dtype=torch.int32),
            torch.zeros_like(selected_indices),
        )
        torch.testing.assert_close(physical_pages, expected)

    def test_fixed_capacity_plan_uses_pool_bound_without_host_lengths(self):
        capacity_lens = torch.tensor([32, 32], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=2,
            seq_lens=capacity_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        forward_batch = _FakeForwardBatch(
            torch.tensor([9, 5], dtype=torch.int64, device=self.device)
        )
        req_pool_indices = torch.arange(2, device=self.device)
        sparse_mask = torch.ones(2, dtype=torch.bool, device=self.device)

        with patch.object(
            algo,
            "_get_seq_lens_cpu",
            side_effect=AssertionError("fixed plans must not read host lengths"),
        ):
            algo.begin_forward(
                forward_batch,
                req_pool_indices,
                sparse_mask,
                self.device,
                fixed_capacity=16,
            )

        plan = algo._retrieval_plan
        self.assertTrue(plan.fixed_capacity)
        self.assertIsNone(plan.seq_lens_cpu)
        self.assertIsNone(plan.num_pages_cpu)
        self.assertEqual(plan.max_num_pages, 16)
        self.assertEqual(plan.max_k, 7)
        self.assertEqual(plan.page_idx.shape, (16,))
        self.assertEqual(plan.physical_pages.shape, (2, 16))
        self.assertEqual(plan.num_pages.tolist(), [9, 5])

    def test_fixed_capacity_batch_one_uses_masked_batched_retrieval(self):
        seq_lens = torch.tensor([16], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=self.device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=self.device)
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=True,
        )

        with (
            patch.object(
                algo,
                "_retrieve_topk_single",
                side_effect=AssertionError("single-request dynamic path was used"),
            ),
            patch.object(
                algo,
                "_retrieve_topk_batched",
                wraps=algo._retrieve_topk_batched,
            ) as batched,
        ):
            algo.retrieve_topk(
                torch.zeros((1, 1, 1), device=self.device),
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
        batched.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestQuestFixedCapacityCudaGraph(unittest.TestCase):
    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_triton_finalize_matches_torch_and_replays(self):
        from sglang.srt.mem_cache.sparsity.kernels.quest_finalize import (
            quest_finalize_selected_pages,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        topk_scores = torch.tensor(
            [
                [9.0, 8.0, float("-inf"), 6.0, 5.0],
                [4.0, float("nan"), 2.0, 1.0, float("inf")],
            ],
            dtype=torch.float32,
            device=device,
        )
        topk_indices = torch.tensor(
            [[7, 2, 9, 1, 4], [8, 3, 6, 0, 5]],
            dtype=torch.int64,
            device=device,
        )
        k_per_req = torch.tensor([4, 5], dtype=torch.int64, device=device)
        recent_indices = torch.tensor(
            [[10, 11], [9, 10]], dtype=torch.int64, device=device
        )
        recent_valid = torch.tensor(
            [[True, False], [True, True]], dtype=torch.bool, device=device
        )

        def run():
            return quest_finalize_selected_pages(
                topk_scores,
                topk_indices,
                k_per_req,
                recent_indices,
                recent_valid,
            )

        def reference():
            topk_rank = torch.arange(topk_scores.shape[1], device=device)
            topk_valid = (topk_rank < k_per_req.unsqueeze(1)) & torch.isfinite(
                topk_scores
            )
            combined_idx = torch.cat([topk_indices, recent_indices], dim=1)
            combined_valid = torch.cat([topk_valid, recent_valid], dim=1)
            return QuestAlgorithm._finalize_selected_pages(
                combined_idx, combined_valid, sentinel=32
            )

        expected_indices, expected_lengths = reference()
        actual_indices, actual_lengths = run()
        torch.testing.assert_close(actual_indices, expected_indices)
        torch.testing.assert_close(actual_lengths, expected_lengths)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_indices, captured_lengths = run()

        topk_scores.copy_(
            torch.tensor(
                [[5.0, 4.0, 3.0, 2.0, 1.0], [9.0, 8.0, 7.0, 6.0, 5.0]],
                device=device,
            )
        )
        k_per_req.copy_(torch.tensor([2, 3], device=device))
        recent_valid.copy_(torch.tensor([[True, True], [False, True]], device=device))
        graph.replay()
        torch.cuda.synchronize()

        expected_indices, expected_lengths = reference()
        torch.testing.assert_close(captured_indices, expected_indices)
        torch.testing.assert_close(captured_lengths, expected_lengths)

    def test_oversized_fixed_finalize_uses_torch_fallback(self):
        from sglang.srt.mem_cache.sparsity.kernels.quest_finalize import (
            QUEST_FINALIZE_MAX_WIDTH,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        capacity = QUEST_FINALIZE_MAX_WIDTH + 1
        capacity_lens = torch.tensor([capacity], dtype=torch.int64, device=device)
        algo, _ = _make_algorithm(
            batch_size=1,
            seq_lens=capacity_lens,
            page_size=1,
            sparsity_ratio=1.0,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=device,
            seed=0,
        )
        forward_batch = _FakeForwardBatch(capacity_lens)
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=device)
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
            fixed_capacity=True,
        )
        plan = algo._retrieval_plan
        topk_scores = torch.ones((1, plan.max_k), device=device)
        topk_idx = torch.arange(plan.max_k, device=device).unsqueeze(0)

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_finalize."
            "quest_finalize_selected_pages",
            side_effect=AssertionError("oversized input reached Triton finalize"),
        ):
            indices, lengths = algo._finalize_topk_with_recent(
                topk_scores, topk_idx, plan
            )

        self.assertEqual(indices.shape, (1, capacity))
        self.assertEqual(lengths.item(), capacity)
        torch.testing.assert_close(
            indices,
            torch.arange(capacity, dtype=torch.int32, device=device).unsqueeze(0),
        )

    def test_replay_grows_from_inactive_capture_to_sparse_long_context(self):
        device = torch.device("cuda", torch.cuda.current_device())
        capacity_lens = torch.tensor([32], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=capacity_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=7,
        )
        _populate_page_reps(algo, 1, capacity_lens, k_buffer, device)

        seq_lens = torch.tensor([1], dtype=torch.int64, device=device)
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        sparse_mask = torch.zeros(1, dtype=torch.bool, device=device)
        queries = torch.zeros((1, 1, 8), dtype=torch.float32, device=device)

        def run_fixed():
            algo.begin_forward(
                forward_batch,
                req_pool_indices,
                sparse_mask,
                device,
                fixed_capacity=True,
            )
            return algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run_fixed()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_indices, captured_lengths = run_fixed()

        seq_lens.copy_(torch.tensor([23], dtype=torch.int64, device=device))
        sparse_mask.fill_(True)
        queries.copy_(torch.randn_like(queries))
        graph.replay()
        torch.cuda.synchronize()

        expected_indices, expected_lengths = _reference_retrieve_topk(
            algo,
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch,
        )
        self.assertEqual(captured_lengths.cpu().tolist(), expected_lengths)
        self.assertEqual(
            _sorted_rows(captured_indices.cpu(), captured_lengths.cpu()),
            _sorted_rows(expected_indices, expected_lengths),
        )


def _reference_compute_page_reps_masked(
    algo, layer_id, reqs, seq_lens, end_page, k_buffer
):
    """Original masked min/max page representation (always-fallback path)."""
    device = k_buffer.device
    req_to_token = algo.req_to_token_pool.req_to_token
    n = reqs.shape[0]
    start_page = torch.zeros_like(end_page)
    max_pages = int((end_page - start_page).max().item())
    if max_pages <= 0:
        return

    pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
    pg_id = start_page.unsqueeze(1) + pg_off
    pg_mask = pg_id < end_page.unsqueeze(1)

    tok_start = pg_id * algo.page_size
    tok_off = torch.arange(algo.page_size, device=device).view(1, 1, -1)
    tok_pos = tok_start.unsqueeze(2) + tok_off
    tok_mask = (
        tok_pos
        < (tok_start + algo.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
    ) & pg_mask.unsqueeze(2)

    phys_tok = req_to_token[
        reqs.view(n, 1, 1).expand(n, max_pages, algo.page_size),
        tok_pos.clamp(0, req_to_token.shape[1] - 1),
    ].clamp(0, k_buffer.shape[0] - 1)
    keys = k_buffer[phys_tok].to(torch.float32)
    mask = tok_mask.unsqueeze(-1).unsqueeze(-1)
    page_min = torch.where(mask, keys, torch.full_like(keys, float("inf"))).amin(dim=2)
    page_max = torch.where(mask, keys, torch.full_like(keys, float("-inf"))).amax(dim=2)

    phys_pg = (
        req_to_token[
            reqs.unsqueeze(1).expand(n, max_pages),
            tok_start.clamp(0, req_to_token.shape[1] - 1),
        ]
        // algo.page_size
    )
    idx = pg_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return
    target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
        0, algo.page_k_min[layer_id].shape[0] - 1
    )
    algo.page_k_min[layer_id][target_pages] = page_min[idx[:, 0], idx[:, 1]]
    algo.page_k_max[layer_id][target_pages] = page_max[idx[:, 0], idx[:, 1]]
    algo.page_valid[layer_id][target_pages] = True


class TestQuestPageRepresentationEquivalence(CustomTestCase):
    device = torch.device("cpu")

    def _run_case(
        self,
        seq_lens_list,
        page_size=16,
        kv_heads=2,
        head_dim=8,
        seed=0,
        end_page_mode="floor",
    ):
        """Compare fast-path vs masked-fallback page representations.

        ``end_page_mode`` controls which branch of
        ``_compute_page_representations`` the new implementation takes:

        - ``"floor"``: ``end_page = seq_lens // page_size`` (only complete
          pages). This is what production callers
          (``construct_representations`` / ``update_representations``) always
          pass, so ``end_page * page_size <= seq_lens`` holds and the new code
          takes the full-page fast path.
        - ``"ceil"``: ``end_page = ceil(seq_lens / page_size)``. The last page
          of an unaligned request is partial, so the new code is forced down
          the masked fallback branch. Production never does this today, but the
          branch exists for safety and we lock its equivalence here.
        """
        device = self.device
        batch_size = len(seq_lens_list)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)

        algo_new, k_buffer = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            0.5,
            4,
            kv_heads,
            head_dim,
            device,
            seed,
        )
        algo_ref, _ = _make_algorithm(
            batch_size,
            seq_lens,
            page_size,
            0.5,
            4,
            kv_heads,
            head_dim,
            device,
            seed,
        )

        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        if end_page_mode == "floor":
            end_pages = seq_lens // page_size
        elif end_page_mode == "ceil":
            end_pages = (seq_lens + page_size - 1) // page_size
        else:
            raise ValueError(f"Unknown end_page_mode: {end_page_mode}")

        algo_new._compute_page_representations(
            0, req_pool_indices, seq_lens, 0, end_pages, k_buffer
        )
        _reference_compute_page_reps_masked(
            algo_ref, 0, req_pool_indices, seq_lens, end_pages, k_buffer
        )

        valid_new = algo_new.page_valid[0]
        valid_ref = algo_ref.page_valid[0]
        self.assertTrue(torch.equal(valid_new, valid_ref))
        torch.testing.assert_close(
            algo_new.page_k_min[0][valid_new],
            algo_ref.page_k_min[0][valid_ref],
        )
        torch.testing.assert_close(
            algo_new.page_k_max[0][valid_new],
            algo_ref.page_k_max[0][valid_ref],
        )

    def test_fast_path_aligned(self):
        # seq_len divisible by page_size -> all pages full -> fast path.
        self._run_case([512, 256, 384], end_page_mode="floor")

    def test_fast_path_production_floor(self):
        # Unaligned seq_lens but floor end_page (production semantics) still
        # only builds complete pages, so the fast path is taken.
        self._run_case([500, 300, 257], end_page_mode="floor")

    def test_fallback_unaligned(self):
        # Force a partial last page via ceil end_page -> masked fallback path.
        self._run_case([500, 300, 257], end_page_mode="ceil")

    def test_fallback_mixed_alignment(self):
        # Some requests aligned, some not, with ceil end_page -> fallback.
        self._run_case([512, 257, 384, 333], end_page_mode="ceil")


if __name__ == "__main__":
    unittest.main()
