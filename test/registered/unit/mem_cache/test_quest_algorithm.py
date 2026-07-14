"""Equivalence tests for the optimized Quest sparse attention algorithm.

These tests lock the behavior of the vectorized ``retrieve_topk`` path and the
full-page fast path in ``_compute_page_representations`` against the original
per-request reference implementations. They run on CPU and do not start a
server or call a real attention backend.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Config:
    def __init__(
        self, page_size, sparsity_ratio, num_recent_pages, sparse_extra_config=None
    ):
        self.page_size = page_size
        self.sparse_extra_config = {
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": num_recent_pages,
            **(sparse_extra_config or {}),
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
    sparse_extra_config=None,
    end_layer=1,
    k_dtype=torch.float32,
):
    torch.manual_seed(seed)
    req_to_token, total_tokens = _build_req_to_token(
        batch_size, seq_lens, page_size, device
    )
    k_buffer = torch.randn(
        total_tokens, kv_heads, head_dim, dtype=k_dtype, device=device
    )
    config = _Config(page_size, sparsity_ratio, num_recent_pages, sparse_extra_config)
    algo = QuestAlgorithm(config, device)
    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=end_layer,
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
        history_page_cap = algo.get_history_page_selection_cap()
        if history_page_cap is not None:
            k = min(k, history_page_cap)
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

    def test_jit_topk_plan_accepts_string_device_and_preserves_short_dtype(self):
        pool_seq_lens = torch.tensor([2048], dtype=torch.int64, device=self.device)
        algo, _ = _make_algorithm(
            batch_size=1,
            seq_lens=pool_seq_lens,
            page_size=1,
            sparsity_ratio=0.062124249,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=self.device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=self.device)

        # ModelRunner currently passes the device as the string "cuda". Keep
        # tensor construction on CPU here while exercising that representation.
        algo.device = "cuda"
        forward_batch = _FakeForwardBatch(
            torch.tensor([640], dtype=torch.int64, device=self.device)
        )
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=640,
        )
        self.assertEqual(algo._retrieval_plan.k_per_req.dtype, torch.int64)

        forward_batch = _FakeForwardBatch(
            torch.tensor([1024], dtype=torch.int64, device=self.device)
        )
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=1024,
        )
        self.assertEqual(algo._retrieval_plan.k_per_req.dtype, torch.int32)

        # This ratio rounds 1143 * ratio down in Python double but up to 279
        # in float32. max_k must remain a safe upper bound for the device k.
        algo.sparsity_ratio = 0.244094488
        forward_batch = _FakeForwardBatch(
            torch.tensor([1147], dtype=torch.int64, device=self.device)
        )
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=1147,
        )
        plan = algo._retrieval_plan
        self.assertEqual(plan.k_per_req.tolist(), [279])
        self.assertEqual(plan.max_k, 279)
        self.assertLessEqual(plan.k_per_req.max().item(), plan.max_k)


class TestQuestFixedSelectionBudget(CustomTestCase):
    device = torch.device("cpu")

    def _make_plan(
        self,
        seq_lens_list,
        *,
        cap_tokens=1024,
        page_size=16,
        ratio=0.5,
        recent_pages=4,
        sparse_mask=None,
        use_seq_lens_cpu=True,
        fixed_capacity=False,
        layer_page_budget=None,
    ):
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=self.device)
        algorithm, k_buffer = _make_algorithm(
            batch_size=len(seq_lens_list),
            seq_lens=seq_lens,
            page_size=page_size,
            sparsity_ratio=ratio,
            num_recent_pages=recent_pages,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=41,
            sparse_extra_config={
                "quest_max_selected_tokens": cap_tokens,
                "layer_page_budget": layer_page_budget or [],
            },
            end_layer=3 if layer_page_budget else 1,
        )
        req_pool_indices = torch.arange(
            len(seq_lens_list), dtype=torch.int64, device=self.device
        )
        if sparse_mask is None:
            sparse_mask = torch.ones(
                len(seq_lens_list), dtype=torch.bool, device=self.device
            )
        else:
            sparse_mask = torch.tensor(
                sparse_mask, dtype=torch.bool, device=self.device
            )
        forward_batch = _FakeForwardBatch(seq_lens, use_seq_lens_cpu)
        algorithm.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=fixed_capacity,
        )
        return (
            algorithm,
            k_buffer,
            forward_batch,
            req_pool_indices,
            sparse_mask,
        )

    def test_cap_is_default_off_and_generic_top_k_does_not_enable_it(self):
        config = _Config(page_size=16, sparsity_ratio=0.5, num_recent_pages=4)
        config.top_k = 80
        algorithm = QuestAlgorithm(config, self.device)

        self.assertIsNone(algorithm.quest_max_selected_pages)
        self.assertIsNone(algorithm.get_history_page_selection_cap())

    def test_ragged_budget_caps_total_width_and_preserves_inactive_rows(self):
        (
            algorithm,
            k_buffer,
            forward_batch,
            req_pool_indices,
            sparse_mask,
        ) = self._make_plan(
            [4096, 2048, 512],
            sparse_mask=[True, False, True],
        )
        plan = algorithm._retrieval_plan

        self.assertEqual(algorithm.quest_max_selected_pages, 64)
        self.assertEqual(algorithm.get_history_page_selection_cap(), 60)
        self.assertEqual(plan.k_per_req.tolist(), [60, 0, 14])
        self.assertEqual(plan.max_k, 60)

        _populate_page_reps(
            algorithm,
            batch_size=3,
            seq_lens=forward_batch.seq_lens,
            k_buffer=k_buffer,
            device=self.device,
        )
        selected, lengths = algorithm.retrieve_topk(
            torch.zeros((3, 1, 1), device=self.device),
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )

        self.assertEqual(selected.shape, (3, 64))
        self.assertEqual(lengths.tolist(), [64, 0, 18])
        self.assertTrue(torch.all(lengths <= 64).item())
        self.assertTrue(torch.all(selected[1] == -1).item())

    def test_nonbinding_budget_is_exactly_equivalent_to_default(self):
        seq_lens = torch.tensor([512, 384], dtype=torch.int64, device=self.device)
        common = dict(
            batch_size=2,
            seq_lens=seq_lens,
            page_size=16,
            sparsity_ratio=0.5,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=4,
            device=self.device,
            seed=57,
        )
        default, default_k = _make_algorithm(**common)
        capped, capped_k = _make_algorithm(
            **common,
            sparse_extra_config={"quest_max_selected_tokens": 1024},
        )
        self.assertTrue(torch.equal(default_k, capped_k))
        for algorithm, k_buffer in ((default, default_k), (capped, capped_k)):
            _populate_page_reps(algorithm, 2, seq_lens, k_buffer, self.device)

        req_pool_indices = torch.arange(2, dtype=torch.int64, device=self.device)
        sparse_mask = torch.ones(2, dtype=torch.bool, device=self.device)
        forward_batch = _FakeForwardBatch(seq_lens)
        torch.manual_seed(59)
        queries = torch.randn((2, 2, 4), device=self.device)
        default.begin_forward(forward_batch, req_pool_indices, sparse_mask, self.device)
        capped.begin_forward(forward_batch, req_pool_indices, sparse_mask, self.device)

        default_result = default.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        capped_result = capped.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )

        self.assertTrue(torch.equal(default_result[0], capped_result[0]))
        self.assertTrue(torch.equal(default_result[1], capped_result[1]))

    def test_device_only_plan_uses_the_same_cap(self):
        algorithm, _, _, _, _ = self._make_plan(
            [4096, 2048, 512],
            use_seq_lens_cpu=False,
        )
        plan = algorithm._retrieval_plan

        self.assertIsNone(plan.num_pages_cpu)
        self.assertEqual(plan.k_per_req.tolist(), [60, 60, 14])
        self.assertEqual(plan.max_k, 60)

    def test_extremely_large_cap_is_nonbinding_without_int64_overflow(self):
        cap_pages = (1 << 63) + 4
        algorithm, _, _, _, _ = self._make_plan(
            [512],
            cap_tokens=cap_pages * 16,
        )
        plan = algorithm._retrieval_plan

        self.assertEqual(algorithm.quest_max_selected_pages, cap_pages)
        self.assertEqual(plan.k_per_req.tolist(), [14])
        self.assertEqual(plan.max_k, 14)

    def test_partial_last_page_stays_below_the_token_cap(self):
        (
            algorithm,
            k_buffer,
            forward_batch,
            req_pool_indices,
            sparse_mask,
        ) = self._make_plan(
            [32001],
            cap_tokens=2048,
        )
        _populate_page_reps(
            algorithm,
            batch_size=1,
            seq_lens=forward_batch.seq_lens,
            k_buffer=k_buffer,
            device=self.device,
        )

        _, valid_lengths = algorithm.retrieve_topk(
            torch.zeros((1, 1, 1), device=self.device),
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        last_page_tokens = (forward_batch.seq_lens - 1) % algorithm.page_size + 1
        selected_tokens = (
            valid_lengths.to(torch.int64) - 1
        ) * algorithm.page_size + last_page_tokens

        self.assertEqual(valid_lengths.tolist(), [128])
        self.assertEqual(selected_tokens.tolist(), [2033])
        self.assertTrue(torch.all(selected_tokens <= 2048).item())

    def test_fixed_capacity_caps_device_k_and_static_output_width(self):
        algorithm, _, _, _, _ = self._make_plan(
            [8192],
            cap_tokens=2048,
            page_size=1,
            ratio=0.7,
            fixed_capacity=8192,
        )
        algorithm.device = "cuda"
        # Rebuild after using the production string device representation so
        # the JIT eligibility branch is covered without requiring a GPU.
        forward_batch = _FakeForwardBatch(torch.tensor([8192], dtype=torch.int64))
        req_pool_indices = torch.zeros(1, dtype=torch.int64)
        sparse_mask = torch.ones(1, dtype=torch.bool)
        algorithm.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=8192,
        )
        plan = algorithm._retrieval_plan

        self.assertEqual(plan.k_per_req.dtype, torch.int32)
        self.assertEqual(plan.k_per_req.tolist(), [2044])
        self.assertEqual(plan.max_k, 2044)
        self.assertEqual(plan.max_k + algorithm.num_recent_pages, 2048)

    def test_layer_ratio_is_applied_before_the_global_hard_cap(self):
        algorithm, _, _, _, _ = self._make_plan(
            [4096],
            layer_page_budget=[
                {"start_layer": 1, "end_layer": 2, "scale": 0.5},
                {"start_layer": 2, "end_layer": 3, "scale": 0.25},
            ],
        )
        base_plan = algorithm._retrieval_plan
        half_plan = algorithm._get_retrieval_plan_for_ratio(
            base_plan, algorithm.get_layer_sparsity_ratio(1)
        )
        quarter_plan = algorithm._get_retrieval_plan_for_ratio(
            base_plan, algorithm.get_layer_sparsity_ratio(2)
        )

        self.assertEqual(base_plan.k_per_req.tolist(), [60])
        self.assertEqual(half_plan.k_per_req.tolist(), [60])
        self.assertEqual(quarter_plan.k_per_req.tolist(), [31])
        self.assertEqual(
            [base_plan.max_k, half_plan.max_k, quarter_plan.max_k], [60, 60, 31]
        )


class TestQuestDecodeTokenSelectionReuse(unittest.TestCase):
    device = torch.device("cpu")

    @staticmethod
    def _make_reuse_algorithm(interval=None, *, pool_size=2):
        sparse_extra_config = {"layer_selection_reuse_interval": 1}
        if interval is not None:
            sparse_extra_config["decode_token_selection_reuse_interval"] = interval
        algorithm, _ = _make_algorithm(
            batch_size=pool_size,
            seq_lens=torch.full((pool_size,), 64, dtype=torch.int64),
            page_size=8,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=torch.device("cpu"),
            seed=0,
            sparse_extra_config=sparse_extra_config,
        )
        return algorithm

    @staticmethod
    def _make_decode_batch(seq_lens, *, rids=None, req_slots=None):
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
        batch_size = seq_lens.numel()
        if rids is None:
            rids = [f"request-{index}" for index in range(batch_size)]
        if req_slots is None:
            req_slots = list(range(batch_size))
        req_pool_indices = torch.tensor(req_slots, dtype=torch.int64)
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True,
                is_extend=lambda: False,
            ),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices.clone(),
            rids=list(rids),
        )

    @staticmethod
    def _fake_underlying_retrieve(queries, *args, **kwargs):
        batch_size = queries.shape[0]
        return (
            torch.zeros((batch_size, 1), dtype=torch.int32),
            torch.ones(batch_size, dtype=torch.int32),
        )

    def test_dense_forward_updates_bounds_before_sparse_transition(self):
        algorithm, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=torch.tensor([20], dtype=torch.int64),
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=2,
            device=self.device,
            seed=31,
            sparse_extra_config={"layer_selection_reuse_interval": 1},
            end_layer=2,
        )
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        algorithm.states.repr_constructed[0] = True
        algorithm.states.last_constructed_page[0] = 3
        k_buffer[12:16].fill_(100)
        for layer_id in range(2):
            algorithm._compute_page_representations(
                layer_id,
                req_pool_indices,
                torch.tensor([12], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([3], dtype=torch.int64),
                k_buffer,
            )

        dense_batch = self._make_decode_batch([16])
        algorithm.begin_dense_forward(dense_batch)
        for layer_id in range(2):
            algorithm.update_representations(
                layer_id,
                req_pool_indices,
                dense_batch.seq_lens,
                k_buffer,
                dense_batch,
            )
        algorithm.finalize_forward(dense_batch)

        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [4])
        physical_page = (
            int(algorithm.req_to_token_pool.req_to_token[0, 12].item())
            // algorithm.page_size
        )
        self.assertTrue(algorithm.page_valid[0][physical_page].item())
        self.assertTrue(algorithm.page_valid[1][physical_page].item())

        sparse_batch = self._make_decode_batch([17])
        algorithm.begin_forward(
            sparse_batch,
            sparse_batch.req_pool_indices,
            torch.ones(1, dtype=torch.bool),
            self.device,
        )
        selected_pages, valid_lengths = algorithm.retrieve_topk(
            torch.ones((1, 1, 2)),
            0,
            sparse_batch.req_pool_indices,
            torch.ones(1, dtype=torch.bool),
            forward_batch=sparse_batch,
        )

        self.assertGreater(valid_lengths.item(), 0)
        self.assertTrue((selected_pages[0, : valid_lengths.item()] == 3).any().item())

    def _run_decode_forward(
        self, algorithm, forward_batch, *, fixed_capacity=False, finalize=True
    ):
        batch_size = forward_batch.seq_lens.numel()
        sparse_mask = torch.ones(batch_size, dtype=torch.bool)
        algorithm.begin_forward(
            forward_batch,
            forward_batch.req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=fixed_capacity,
        )
        result = algorithm.retrieve_topk(
            torch.zeros((batch_size, 1, 1)),
            0,
            forward_batch.req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        if finalize:
            algorithm.finalize_forward(forward_batch)
        return result

    def test_default_and_explicit_one_refresh_every_decode_token(self):
        for interval in (None, 1):
            with self.subTest(interval=interval):
                algorithm = self._make_reuse_algorithm(interval)
                with (
                    patch.object(
                        BaseSparseAlgorithmImpl,
                        "retrieve_topk",
                        side_effect=self._fake_underlying_retrieve,
                    ) as underlying_retrieve,
                    patch.object(algorithm, "_finalize_representation_trackers"),
                ):
                    first = self._run_decode_forward(
                        algorithm, self._make_decode_batch([17])
                    )
                    second = self._run_decode_forward(
                        algorithm, self._make_decode_batch([18])
                    )

                self.assertEqual(underlying_retrieve.call_count, 2)
                self.assertEqual(len(first), 2)
                self.assertEqual(len(second), 2)
                self.assertIsNone(algorithm._decode_selection_cache_state)

    def test_interval_three_refreshes_then_reuses_twice(self):
        algorithm = self._make_reuse_algorithm(3)
        batches = [
            self._make_decode_batch([seq_len], req_slots=[1])
            for seq_len in (17, 18, 19, 20)
        ]

        with (
            patch.object(
                BaseSparseAlgorithmImpl,
                "retrieve_topk",
                side_effect=self._fake_underlying_retrieve,
            ) as underlying_retrieve,
            patch.object(algorithm, "_finalize_representation_trackers"),
        ):
            results = [self._run_decode_forward(algorithm, batch) for batch in batches]

        self.assertEqual(underlying_retrieve.call_count, 2)
        self.assertEqual([len(result) for result in results], [2, 4, 4, 2])
        # Request-pool slot 1 starts at physical page 8. Cross-token hits
        # return that owned physical page and identify it explicitly as such.
        for result in results[1:3]:
            self.assertEqual(result[0].tolist(), [[8]])
            self.assertIs(result[0], result[3])
            self.assertFalse(result[2])
        self.assertEqual(algorithm._decode_selection_cache_state.age, 0)

    def test_request_sequence_and_page_changes_force_refresh(self):
        cases = (
            (
                "request reorder",
                ([17, 17], ["a", "b"], [0, 1]),
                ([18, 18], ["b", "a"], [0, 1]),
            ),
            (
                "request slot reorder",
                ([17, 17], ["a", "b"], [0, 1]),
                ([18, 18], ["a", "b"], [1, 0]),
            ),
            (
                "batch churn",
                ([17, 17], ["a", "b"], [0, 1]),
                ([18], ["a"], [0]),
            ),
            (
                "sequence jump",
                ([17], ["a"], [0]),
                ([19], ["a"], [0]),
            ),
            (
                "sequence rollback",
                ([18], ["a"], [0]),
                ([17], ["a"], [0]),
            ),
            (
                "completed page",
                ([15], ["a"], [0]),
                ([16], ["a"], [0]),
            ),
            (
                "page count change",
                ([16], ["a"], [0]),
                ([17], ["a"], [0]),
            ),
        )

        for name, previous, current in cases:
            with self.subTest(name=name):
                algorithm = self._make_reuse_algorithm(8)
                previous_batch = self._make_decode_batch(
                    previous[0], rids=previous[1], req_slots=previous[2]
                )
                current_batch = self._make_decode_batch(
                    current[0], rids=current[1], req_slots=current[2]
                )
                with (
                    patch.object(
                        BaseSparseAlgorithmImpl,
                        "retrieve_topk",
                        side_effect=self._fake_underlying_retrieve,
                    ) as underlying_retrieve,
                    patch.object(algorithm, "_finalize_representation_trackers"),
                ):
                    self._run_decode_forward(algorithm, previous_batch)
                    result = self._run_decode_forward(algorithm, current_batch)

                self.assertEqual(underlying_retrieve.call_count, 2)
                self.assertEqual(len(result), 2)
                self.assertEqual(algorithm._decode_selection_cache_state.age, 0)

    def test_missing_host_identity_or_lengths_disables_reuse(self):
        for missing_attribute in ("req_pool_indices_cpu", "seq_lens_cpu"):
            with self.subTest(missing_attribute=missing_attribute):
                algorithm = self._make_reuse_algorithm(8)
                first_batch = self._make_decode_batch([17])
                second_batch = self._make_decode_batch([18])
                setattr(second_batch, missing_attribute, None)
                with (
                    patch.object(
                        BaseSparseAlgorithmImpl,
                        "retrieve_topk",
                        side_effect=self._fake_underlying_retrieve,
                    ) as underlying_retrieve,
                    patch.object(algorithm, "_finalize_representation_trackers"),
                ):
                    self._run_decode_forward(algorithm, first_batch)
                    result = self._run_decode_forward(algorithm, second_batch)

                self.assertEqual(underlying_retrieve.call_count, 2)
                self.assertEqual(len(result), 2)
                self.assertIsNone(algorithm._decode_selection_cache_state)

    def test_pending_cache_commits_only_after_successful_finalize(self):
        algorithm = self._make_reuse_algorithm(3)
        forward_batch = self._make_decode_batch([17])
        with (
            patch.object(
                BaseSparseAlgorithmImpl,
                "retrieve_topk",
                side_effect=self._fake_underlying_retrieve,
            ),
            patch.object(algorithm, "_finalize_representation_trackers"),
        ):
            self._run_decode_forward(algorithm, forward_batch, finalize=False)
            self.assertIsNone(algorithm._decode_selection_cache_state)
            self.assertIsNotNone(algorithm._pending_decode_selection_cache_state)
            algorithm.finalize_forward(forward_batch)

        self.assertIsNotNone(algorithm._decode_selection_cache_state)
        self.assertIsNone(algorithm._pending_decode_selection_cache_state)

        failed_algorithm = self._make_reuse_algorithm(3)
        failed_batch = self._make_decode_batch([17])
        with (
            patch.object(
                BaseSparseAlgorithmImpl,
                "retrieve_topk",
                side_effect=self._fake_underlying_retrieve,
            ),
            patch.object(
                failed_algorithm,
                "_finalize_representation_trackers",
                side_effect=RuntimeError("finalize failed"),
            ),
        ):
            self._run_decode_forward(failed_algorithm, failed_batch, finalize=False)
            with self.assertRaisesRegex(RuntimeError, "finalize failed"):
                failed_algorithm.finalize_forward(failed_batch)

        self.assertIsNone(failed_algorithm._decode_selection_cache_state)
        self.assertIsNone(failed_algorithm._pending_decode_selection_cache_state)

    def test_fixed_capacity_forwards_never_reuse_cross_token_selection(self):
        for fixed_capacity in (True, 4):
            with self.subTest(fixed_capacity=fixed_capacity):
                algorithm = self._make_reuse_algorithm(8)
                with (
                    patch.object(
                        BaseSparseAlgorithmImpl,
                        "retrieve_topk",
                        side_effect=self._fake_underlying_retrieve,
                    ) as underlying_retrieve,
                    patch.object(algorithm, "_finalize_representation_trackers"),
                ):
                    first = self._run_decode_forward(
                        algorithm,
                        self._make_decode_batch([17]),
                        fixed_capacity=fixed_capacity,
                    )
                    second = self._run_decode_forward(
                        algorithm,
                        self._make_decode_batch([18]),
                        fixed_capacity=fixed_capacity,
                    )

                self.assertEqual(underlying_retrieve.call_count, 2)
                self.assertEqual(len(first), 2)
                self.assertEqual(len(second), 2)
                self.assertIsNone(algorithm._decode_selection_cache_state)


class TestQuestLayerReuseAndBudget(unittest.TestCase):
    device = torch.device("cpu")

    @staticmethod
    def _make_wrapper_algorithm(*, interval=1, layer_page_budget=None):
        config = _Config(
            page_size=1,
            sparsity_ratio=0.062124249,
            num_recent_pages=1,
            sparse_extra_config={
                "layer_selection_reuse_interval": interval,
                "layer_page_budget": layer_page_budget or [],
            },
        )
        algorithm = QuestAlgorithm(config, torch.device("cpu"))
        algorithm.start_layer = 1
        return algorithm

    @staticmethod
    def _call_retrieve(algorithm, layer_id):
        return algorithm.retrieve_topk(
            torch.zeros((1, 1, 1)),
            layer_id,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.bool),
            forward_batch=object(),
        )

    def test_reuses_only_consecutive_layers_in_same_interval_and_budget(self):
        algorithm = self._make_wrapper_algorithm(
            interval=4,
            layer_page_budget=[{"start_layer": 3, "end_layer": 5, "scale": 0.5}],
        )
        algorithm.end_layer = 7
        selected = torch.tensor([[7]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)

        with patch.object(
            BaseSparseAlgorithmImpl,
            "retrieve_topk",
            return_value=(selected, lengths),
        ) as underlying_retrieve:
            results = [
                self._call_retrieve(algorithm, layer_id) for layer_id in range(1, 7)
            ]

            # Local PP start and budget/relative-interval boundaries are anchors.
            self.assertEqual(underlying_retrieve.call_count, 3)
            self.assertEqual(len(results[1]), 3)
            self.assertTrue(results[1][2])
            self.assertEqual(len(results[-1]), 3)
            self.assertTrue(results[-1][2])
            self.assertTrue(algorithm.should_update_metadata_lengths(1))
            self.assertTrue(algorithm.should_update_metadata_lengths(3))
            self.assertTrue(algorithm.should_update_metadata_lengths(5))
            self.assertFalse(algorithm.should_update_metadata_lengths(4))
            self.assertTrue(algorithm._is_selection_anchor(1))
            self.assertTrue(algorithm._is_selection_anchor(3))
            self.assertTrue(algorithm._is_selection_anchor(5))
            self.assertFalse(algorithm._is_selection_anchor(6))

            with patch.object(BaseSparseAlgorithmImpl, "begin_forward"):
                algorithm.begin_forward()
            self._call_retrieve(algorithm, 1)
            self.assertEqual(underlying_retrieve.call_count, 4)

    def test_default_interval_and_layer_discontinuity_force_retrieval(self):
        selected = torch.tensor([[0]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        with patch.object(
            BaseSparseAlgorithmImpl,
            "retrieve_topk",
            return_value=(selected, lengths),
        ) as underlying_retrieve:
            default_algorithm = self._make_wrapper_algorithm()
            self._call_retrieve(default_algorithm, 1)
            self._call_retrieve(default_algorithm, 2)
            self.assertEqual(underlying_retrieve.call_count, 2)

            reuse_algorithm = self._make_wrapper_algorithm(interval=4)
            reuse_algorithm.start_layer = 0
            self._call_retrieve(reuse_algorithm, 0)
            self._call_retrieve(reuse_algorithm, 2)
            self.assertEqual(underlying_retrieve.call_count, 4)

    def test_lazy_update_reanchors_on_noncontiguous_layer_order(self):
        algorithm = self._make_wrapper_algorithm(interval=4)
        algorithm.end_layer = 5
        algorithm._lazy_page_update_active = True
        algorithm._retrieval_plan = SimpleNamespace(max_num_pages=4)
        algorithm.states = _FakeStates(1, self.device)

        selected = torch.tensor([[3]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        lazy_updated_layers = []

        def fake_lazy_retrieve(*args, **kwargs):
            lazy_updated_layers.append(args[1])
            return selected, lengths

        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            seq_lens=torch.tensor([2], dtype=torch.long),
        )
        with patch.object(
            BaseSparseAlgorithmImpl,
            "retrieve_topk",
            side_effect=fake_lazy_retrieve,
        ) as underlying_retrieve:
            first_result = self._call_retrieve(algorithm, 1)
            algorithm.update_representations(
                1,
                torch.zeros(1, dtype=torch.long),
                torch.ones(1, dtype=torch.long),
                torch.zeros((1, 1, 1)),
                forward_batch,
            )
            result = self._call_retrieve(algorithm, 3)
            algorithm.update_representations(
                3,
                torch.zeros(1, dtype=torch.long),
                torch.ones(1, dtype=torch.long),
                torch.zeros((1, 1, 1)),
                forward_batch,
            )

        self.assertEqual(first_result, (selected, lengths))
        self.assertEqual(result, (selected, lengths))
        self.assertEqual(underlying_retrieve.call_count, 2)
        self.assertEqual(lazy_updated_layers, [1, 3])
        self.assertTrue(algorithm._lazy_page_update_active)
        self.assertEqual(algorithm._actual_selection_anchors, {1, 3})
        self.assertFalse(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 0)

        def fake_lazy_tracker(
            reqs, seq_lens, constructed, last_page, page_size, *, max_pages
        ):
            constructed[reqs] = True
            last_page[reqs] = torch.clamp(
                (seq_lens - 1) // page_size, min=0, max=max_pages
            )

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_advance_lazy_page_trackers_",
            side_effect=fake_lazy_tracker,
        ) as advance:
            algorithm.finalize_forward(forward_batch)

        advance.assert_called_once()
        self.assertEqual(advance.call_args.kwargs["max_pages"], 4)
        self.assertTrue(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 1)

    def test_noncontiguous_retrieval_refreshes_late_anchor_and_tail_finalizes(self):
        seq_lens = torch.tensor([2], dtype=torch.int64)
        algorithm, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
            sparse_extra_config={"layer_selection_reuse_interval": 4},
            end_layer=4,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long)
        sparse_mask = torch.ones(1, dtype=torch.bool)
        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        forward_batch.req_pool_indices = req_pool_indices
        algorithm.begin_forward(
            forward_batch, req_pool_indices, sparse_mask, self.device
        )

        selected = torch.tensor([[0]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)
        tracker_inputs = []

        def fake_page_update(*args, advance_trackers):
            tracker_inputs.append((bool(args[4][0].item()), int(args[5][0].item())))
            args[8].fill_(True)
            if advance_trackers:
                args[4][args[0]] = True
                args[5][args[0]] = args[1] // args[9]

        with (
            patch.object(
                BaseSparseAlgorithmImpl,
                "retrieve_topk",
                return_value=(selected, lengths),
            ) as underlying_retrieve,
            patch.object(algorithm, "_can_use_triton_page_update", return_value=True),
            patch(
                "sglang.srt.mem_cache.sparsity.kernels.quest_page_update."
                "quest_update_page_representations_",
                side_effect=fake_page_update,
            ) as page_update,
        ):
            algorithm.retrieve_topk(
                torch.zeros((1, 1, 1)),
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
            algorithm.update_representations(
                0, req_pool_indices, seq_lens, k_buffer, forward_batch
            )

            # Layer 1 is omitted (for example because its key input is None).
            # Layer 2 must become a fresh anchor even though it shares group 0.
            algorithm.retrieve_topk(
                torch.zeros((1, 1, 1)),
                2,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
            algorithm.update_representations(
                2, req_pool_indices, seq_lens, k_buffer, forward_batch
            )

            self.assertEqual(page_update.call_count, 2)
            self.assertTrue(algorithm.page_valid[0].any().item())
            self.assertTrue(algorithm.page_valid[2].any().item())
            self.assertFalse(algorithm.states.repr_constructed[0].item())
            self.assertEqual(algorithm.states.last_constructed_page[0].item(), 0)

            # Numerical tail layer 3 is absent. The runner-level finalizer must
            # still advance after every actual anchor has written its bounds.
            algorithm.finalize_forward(forward_batch)

        self.assertEqual(underlying_retrieve.call_count, 2)
        self.assertEqual(page_update.call_count, 2)
        self.assertEqual(tracker_inputs, [(False, 0), (False, 0)])
        self.assertTrue(algorithm.page_valid[0].any().item())
        self.assertTrue(algorithm.page_valid[2].any().item())
        self.assertFalse(algorithm.page_valid[1].any().item())
        self.assertFalse(algorithm.page_valid[3].any().item())
        self.assertTrue(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 2)

    def test_metadata_lengths_follow_previous_executed_layer(self):
        algorithm = self._make_wrapper_algorithm(
            interval=4,
            layer_page_budget=[{"start_layer": 3, "end_layer": 5, "scale": 0.5}],
        )
        algorithm.end_layer = 7
        selected = torch.tensor([[0]], dtype=torch.int32)
        lengths = torch.tensor([1], dtype=torch.int32)

        with patch.object(
            BaseSparseAlgorithmImpl,
            "retrieve_topk",
            return_value=(selected, lengths),
        ):
            # Layer 1 (the configured PP start) is absent. The first actual
            # sparse layer must still initialize sequence-length metadata.
            self.assertTrue(algorithm.should_update_metadata_lengths(2))
            self._call_retrieve(algorithm, 2)
            self.assertTrue(algorithm.should_update_metadata_lengths(2))

            # Numeric predecessors 3 and 5 have the same budget as layers 4
            # and 6 respectively, but neither predecessor actually executed.
            self.assertTrue(algorithm.should_update_metadata_lengths(4))
            self._call_retrieve(algorithm, 4)
            self.assertTrue(algorithm.should_update_metadata_lengths(4))
            self.assertTrue(algorithm.should_update_metadata_lengths(6))
            self._call_retrieve(algorithm, 6)
            self.assertTrue(algorithm.should_update_metadata_lengths(6))

            with patch.object(BaseSparseAlgorithmImpl, "begin_forward"):
                algorithm.begin_forward()
            self._call_retrieve(algorithm, 2)
            self._call_retrieve(algorithm, 6)
            self.assertFalse(algorithm.should_update_metadata_lengths(6))

    def test_graph_bucket_selects_matching_lazy_tracker_state(self):
        algorithm = self._make_wrapper_algorithm(interval=2)
        algorithm.end_layer = 3
        algorithm.states = _FakeStates(1, self.device)
        algorithm._lazy_page_update_active = True
        algorithm._lazy_page_update_graph_states = {
            4: (False, 4),
            16: (True, 16),
        }
        short_batch = SimpleNamespace(runtime_sparse_page_capacity=4)
        long_batch = SimpleNamespace(
            runtime_sparse_page_capacity=16,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )

        with patch.object(
            BaseSparseAlgorithmImpl,
            "should_update_representations",
            return_value=False,
        ):
            self.assertFalse(algorithm.should_update_representations(short_batch))
            self.assertTrue(algorithm.should_update_representations(long_batch))

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_advance_lazy_page_trackers_"
        ) as advance:
            long_batch.req_pool_indices = torch.zeros(1, dtype=torch.long)
            long_batch.seq_lens = torch.tensor([9], dtype=torch.long)
            algorithm.finalize_forward(long_batch)

        advance.assert_called_once()
        self.assertEqual(advance.call_args.kwargs["max_pages"], 16)

    def test_graph_finalize_requires_a_captured_capacity_marker(self):
        algorithm = self._make_wrapper_algorithm(interval=2)
        algorithm._lazy_page_update_graph_states = {4: (False, 4)}

        self.assertFalse(
            algorithm.should_finalize_graph_forward(
                SimpleNamespace(runtime_sparse_page_capacity=None)
            )
        )
        self.assertFalse(
            algorithm.should_finalize_graph_forward(
                SimpleNamespace(runtime_sparse_page_capacity=16)
            )
        )
        self.assertTrue(
            algorithm.should_finalize_graph_forward(
                SimpleNamespace(runtime_sparse_page_capacity=4)
            )
        )

    def test_same_batch_short_long_short_uses_current_lazy_graph_capacity(self):
        algorithm = self._make_wrapper_algorithm(interval=2)
        algorithm.end_layer = 3
        algorithm.states = _FakeStates(1, self.device)
        algorithm._lazy_page_update_active = True
        algorithm._lazy_page_update_graph_states = {
            4: (True, 4),
            16: (True, 16),
        }
        forward_batch = SimpleNamespace(
            runtime_sparse_page_capacity=4,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            req_pool_indices=torch.zeros(1, dtype=torch.long),
            seq_lens=torch.tensor([9], dtype=torch.long),
        )

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_advance_lazy_page_trackers_"
        ) as advance:
            for capacity, seq_len in ((4, 9), (16, 33), (4, 9)):
                forward_batch.runtime_sparse_page_capacity = capacity
                forward_batch.seq_lens.fill_(seq_len)
                algorithm.finalize_forward(forward_batch)

        self.assertEqual(
            [call.kwargs["max_pages"] for call in advance.call_args_list],
            [4, 16, 4],
        )

    def test_flags_on_cpu_fallback_finalizes_after_numeric_tail_is_skipped(self):
        seq_lens = torch.tensor([2], dtype=torch.int64)
        algorithm, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=7,
            sparse_extra_config={
                "layer_selection_reuse_interval": 2,
                "use_fused_topk_fa_metadata_kernel": True,
                "use_lazy_page_update_score_kernel": True,
                "use_triton_page_update_kernel": True,
            },
            end_layer=4,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long)
        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        forward_batch.req_pool_indices = req_pool_indices
        algorithm.begin_forward(
            forward_batch,
            req_pool_indices,
            torch.ones(1, dtype=torch.bool),
            self.device,
        )
        self.assertFalse(algorithm._lazy_page_update_active)

        for layer_id in (0, 2):
            algorithm.update_representations(
                layer_id,
                req_pool_indices,
                seq_lens,
                k_buffer,
                forward_batch,
            )

        self.assertFalse(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 0)
        algorithm.finalize_forward(forward_batch)
        self.assertTrue(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 2)
        self.assertTrue(algorithm.page_valid[0].any().item())
        self.assertTrue(algorithm.page_valid[2].any().item())
        self.assertFalse(algorithm.page_valid[1].any().item())
        self.assertFalse(algorithm.page_valid[3].any().item())

    def test_regular_finalize_respects_host_page_boundary_gate(self):
        seq_lens = torch.tensor([9], dtype=torch.int64)
        algorithm, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=11,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long)
        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        forward_batch.req_pool_indices = req_pool_indices
        algorithm.begin_forward(
            forward_batch,
            req_pool_indices,
            torch.ones(1, dtype=torch.bool),
            self.device,
        )

        algorithm.update_representations(
            0, req_pool_indices, seq_lens, k_buffer, forward_batch
        )
        algorithm.finalize_forward(forward_batch)

        self.assertFalse(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page[0].item(), 0)

    def test_layer_budget_scales_base_ratio_and_static_max_k(self):
        base_ratio = 0.062124249
        seq_lens = torch.tensor([1147], dtype=torch.int64)
        algorithm, _ = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=base_ratio,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=1,
            device=self.device,
            seed=0,
            sparse_extra_config={
                "layer_page_budget": [{"start_layer": 0, "end_layer": 1, "scale": 0.5}]
            },
        )
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.zeros(1, dtype=torch.long)
        sparse_mask = torch.ones(1, dtype=torch.bool)
        algorithm.device = "cuda"
        algorithm.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            self.device,
            fixed_capacity=1147,
        )

        result = (torch.tensor([[0]], dtype=torch.int32), torch.tensor([1]))
        with patch.object(
            algorithm, "_retrieve_topk_batched", return_value=result
        ) as retrieve:
            algorithm.retrieve_topk(
                torch.zeros((1, 1, 1)),
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )

        layer_plan = retrieve.call_args.args[2]
        effective_ratio = base_ratio * 0.5
        expected_k = int(
            (
                torch.tensor(1143, dtype=torch.float32)
                * torch.tensor(effective_ratio, dtype=torch.float32)
            ).item()
        )
        self.assertEqual(algorithm.get_layer_sparsity_ratio(0), effective_ratio)
        self.assertEqual(layer_plan.k_per_req.tolist(), [expected_k])
        self.assertEqual(layer_plan.max_k, expected_k)
        self.assertEqual(layer_plan.k_per_req.dtype, torch.int32)
        self.assertLess(layer_plan.max_k, algorithm._retrieval_plan.max_k)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestQuestFixedCapacityCudaGraph(unittest.TestCase):
    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_reuse_updates_only_selection_anchor_representations(self):
        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([8], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=17,
            sparse_extra_config={"layer_selection_reuse_interval": 2},
            end_layer=4,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=device)
        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        forward_batch.req_pool_indices = req_pool_indices
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
        )

        for layer_id in range(4):
            algo.update_representations(
                layer_id,
                req_pool_indices,
                seq_lens,
                k_buffer,
                forward_batch,
            )
        algo.finalize_forward(forward_batch)
        torch.cuda.synchronize()

        self.assertTrue(algo.page_valid[0].any().item())
        self.assertFalse(algo.page_valid[1].any().item())
        self.assertTrue(algo.page_valid[2].any().item())
        self.assertFalse(algo.page_valid[3].any().item())
        self.assertTrue(algo.states.repr_constructed[0].item())
        self.assertEqual(algo.states.last_constructed_page[0].item(), 2)

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_fused_topk_metadata_flag_reaches_production_retrieval(self):
        from sglang.jit_kernel.quest.topk import (
            quest_topk_to_flashattention_metadata_out,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([600, 511], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=2,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.062124249,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=23,
            sparse_extra_config={"use_fused_topk_fa_metadata_kernel": True},
        )
        _populate_page_reps(algo, 2, seq_lens, k_buffer, device)
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.arange(2, dtype=torch.long, device=device)
        sparse_mask = torch.ones(2, dtype=torch.bool, device=device)
        queries = torch.randn((2, 1, 8), device=device)
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
        )
        width = algo._retrieval_plan.max_k + algo.num_recent_pages

        def make_metadata():
            return SimpleNamespace(
                page_table=torch.full(
                    (2, width), -99, dtype=torch.int32, device=device
                ),
                cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
                cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
            )

        fused_metadata = make_metadata()
        with patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            wraps=quest_topk_to_flashattention_metadata_out,
        ) as fused_kernel:
            fused_pages, fused_lengths, fused_prepared = algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
                attn_metadata=fused_metadata,
            )
        fused_kernel.assert_called_once()
        self.assertTrue(fused_prepared)

        algo.use_fused_topk_fa_metadata_kernel = False
        reference_indices, reference_lengths = algo.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        reference_pages = algo.get_selected_physical_pages(reference_indices)
        torch.testing.assert_close(fused_lengths, reference_lengths, rtol=0, atol=0)
        for row, length in enumerate(reference_lengths.tolist()):
            torch.testing.assert_close(
                fused_pages[row, :length],
                reference_pages[row, :length],
                rtol=0,
                atol=0,
            )
        expected_cache_lengths = reference_lengths.to(torch.int32)
        expected_cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                expected_cache_lengths.cumsum(0, dtype=torch.int32),
            ]
        )
        torch.testing.assert_close(
            fused_metadata.cache_seqlens_int32,
            expected_cache_lengths,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fused_metadata.cu_seqlens_k,
            expected_cu_seqlens,
            rtol=0,
            atol=0,
        )

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_lazy_update_score_flag_updates_previous_completed_page(self):
        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([9], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=31,
            sparse_extra_config={"use_lazy_page_update_score_kernel": True},
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        algo.states.repr_constructed.fill_(True)
        algo.states.last_constructed_page.fill_(1)
        algo._compute_page_representations(
            0,
            req_pool_indices,
            seq_lens,
            0,
            torch.ones(1, dtype=torch.long, device=device),
            k_buffer,
        )
        physical_page = int(
            (
                algo.req_to_token_pool.req_to_token[0, algo.page_size] // algo.page_size
            ).item()
        )
        self.assertFalse(algo.page_valid[0][physical_page].item())

        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.req_pool_indices = req_pool_indices
        sparse_mask = torch.ones(1, dtype=torch.bool, device=device)
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
            fixed_capacity=4,
        )
        self.assertTrue(algo._lazy_page_update_active)
        algo.retrieve_topk(
            torch.randn((1, 1, 8), device=device),
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        self.assertEqual(algo.states.last_constructed_page.item(), 1)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        algo.finalize_forward(forward_batch)
        torch.cuda.synchronize()

        self.assertTrue(algo.page_valid[0][physical_page].item())
        self.assertEqual(algo.states.last_constructed_page.item(), 2)

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_lazy_noncontiguous_layers_refresh_both_actual_anchors(self):
        from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
            quest_lazy_update_page_scores,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([9], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=41,
            sparse_extra_config={
                "layer_selection_reuse_interval": 4,
                "use_lazy_page_update_score_kernel": True,
            },
            end_layer=4,
        )
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=device)
        algo.states.repr_constructed.fill_(True)
        algo.states.last_constructed_page.fill_(1)
        physical_page = int(
            (
                algo.req_to_token_pool.req_to_token[0, algo.page_size] // algo.page_size
            ).item()
        )
        self.assertFalse(algo.page_valid[0][physical_page].item())
        self.assertFalse(algo.page_valid[2][physical_page].item())

        forward_batch = _FakeForwardBatch(seq_lens)
        forward_batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        forward_batch.req_pool_indices = req_pool_indices
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
            fixed_capacity=4,
        )
        self.assertTrue(algo._lazy_page_update_active)

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_lazy_update_page_scores",
            wraps=quest_lazy_update_page_scores,
        ) as lazy_score:
            algo.retrieve_topk(
                torch.randn((1, 1, 8), device=device),
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )
            # Layer 1 is absent; layer 2 must run a fresh lazy score instead of
            # reusing layer 0 or falling back after scoring stale bounds.
            algo.retrieve_topk(
                torch.randn((1, 1, 8), device=device),
                2,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
            )

        torch.cuda.synchronize()
        self.assertEqual(lazy_score.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["advance_trackers"] is False
                for call in lazy_score.call_args_list
            )
        )
        self.assertEqual(algo._actual_selection_anchors, {0, 2})
        self.assertTrue(algo.page_valid[0][physical_page].item())
        self.assertTrue(algo.page_valid[2][physical_page].item())
        self.assertEqual(algo.states.last_constructed_page.item(), 1)

        algo.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        self.assertEqual(algo.states.last_constructed_page.item(), 2)

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_fused_topk_metadata_handles_single_request_eager_path(self):
        from sglang.jit_kernel.quest.topk import (
            quest_topk_to_flashattention_metadata_out,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([600], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.062124249,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=37,
            sparse_extra_config={"use_fused_topk_fa_metadata_kernel": True},
        )
        _populate_page_reps(algo, 1, seq_lens, k_buffer, device)
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.zeros(1, dtype=torch.long, device=device)
        sparse_mask = torch.ones(1, dtype=torch.bool, device=device)
        queries = torch.randn((1, 1, 8), device=device)
        algo.begin_forward(forward_batch, req_pool_indices, sparse_mask, device)
        width = algo._retrieval_plan.max_k + algo.num_recent_pages
        metadata = SimpleNamespace(
            page_table=torch.full((1, width), -99, dtype=torch.int32, device=device),
            cache_seqlens_int32=torch.empty(1, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.empty(2, dtype=torch.int32, device=device),
        )

        with patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            wraps=quest_topk_to_flashattention_metadata_out,
        ) as fused_kernel:
            fused_pages, fused_lengths, metadata_prepared = algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
                attn_metadata=metadata,
            )
        fused_kernel.assert_called_once()
        self.assertTrue(metadata_prepared)

        algo.use_fused_topk_fa_metadata_kernel = False
        reference_indices, reference_lengths = algo.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        reference_pages = algo.get_selected_physical_pages(reference_indices)
        length = reference_lengths.item()
        torch.testing.assert_close(fused_lengths, reference_lengths, rtol=0, atol=0)
        torch.testing.assert_close(
            fused_pages[0, :length], reference_pages[0, :length], rtol=0, atol=0
        )
        self.assertEqual(metadata.cache_seqlens_int32.item(), length)
        torch.testing.assert_close(
            metadata.cu_seqlens_k,
            torch.tensor([0, length], dtype=torch.int32, device=device),
            rtol=0,
            atol=0,
        )

    @unittest.skipIf(torch.version.hip is not None, "NVIDIA CUDA is required")
    def test_torch_topk_uses_direct_flashattention_metadata(self):
        device = torch.device("cuda", torch.cuda.current_device())
        seq_lens = torch.tensor([600, 511], dtype=torch.int64, device=device)
        algo, k_buffer = _make_algorithm(
            batch_size=2,
            seq_lens=seq_lens,
            page_size=1,
            sparsity_ratio=0.062124249,
            num_recent_pages=4,
            kv_heads=1,
            head_dim=8,
            device=device,
            seed=19,
        )
        _populate_page_reps(algo, 2, seq_lens, k_buffer, device)
        forward_batch = _FakeForwardBatch(seq_lens)
        req_pool_indices = torch.arange(2, dtype=torch.long, device=device)
        sparse_mask = torch.ones(2, dtype=torch.bool, device=device)
        queries = torch.randn((2, 1, 8), device=device)
        algo.begin_forward(
            forward_batch,
            req_pool_indices,
            sparse_mask,
            device,
            fixed_capacity=640,
        )
        plan = algo._retrieval_plan
        self.assertEqual(plan.k_per_req.dtype, torch.int64)
        self.assertLess(plan.max_num_pages, 1024)

        width = plan.max_k + algo.num_recent_pages
        attn_metadata = SimpleNamespace(
            page_table=torch.full((2, width), -99, dtype=torch.int32, device=device),
            cache_seqlens_int32=torch.empty(2, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.empty(3, dtype=torch.int32, device=device),
        )
        algo.use_fused_topk_fa_metadata_kernel = True
        with patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            side_effect=AssertionError("fixed graph reached eager fused kernel"),
        ):
            direct_pages, direct_lengths, metadata_prepared = algo.retrieve_topk(
                queries,
                0,
                req_pool_indices,
                sparse_mask,
                forward_batch=forward_batch,
                attn_metadata=attn_metadata,
            )
        self.assertTrue(metadata_prepared)

        algo.use_direct_fa_metadata_kernel = False
        fallback_indices, fallback_lengths = algo.retrieve_topk(
            queries,
            0,
            req_pool_indices,
            sparse_mask,
            forward_batch=forward_batch,
        )
        fallback_pages = algo.get_selected_physical_pages(fallback_indices)
        torch.testing.assert_close(direct_lengths, fallback_lengths)
        for row, length in enumerate(fallback_lengths.tolist()):
            torch.testing.assert_close(
                direct_pages[row, :length], fallback_pages[row, :length]
            )

        expected_cache_lengths = fallback_lengths.to(torch.int32)
        expected_cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                expected_cache_lengths.cumsum(0, dtype=torch.int32),
            ]
        )
        torch.testing.assert_close(
            attn_metadata.cache_seqlens_int32, expected_cache_lengths
        )
        torch.testing.assert_close(attn_metadata.cu_seqlens_k, expected_cu_seqlens)

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


class TestQuestNativePageBoundsDtype(CustomTestCase):
    device = torch.device("cpu")

    def _make(self, k_dtype, *, native=None, seed=31):
        sparse_extra_config = (
            {} if native is None else {"use_native_page_bounds_dtype": native}
        )
        seq_lens = torch.tensor([9], dtype=torch.int64, device=self.device)
        return _make_algorithm(
            batch_size=1,
            seq_lens=seq_lens,
            page_size=4,
            sparsity_ratio=0.5,
            num_recent_pages=1,
            kv_heads=2,
            head_dim=8,
            device=self.device,
            seed=seed,
            sparse_extra_config=sparse_extra_config,
            k_dtype=k_dtype,
        )

    def test_default_and_explicit_false_keep_fp32_bounds(self):
        for k_dtype in (torch.float16, torch.bfloat16):
            for native in (None, False):
                with self.subTest(k_dtype=k_dtype, native=native):
                    algorithm, _ = self._make(k_dtype, native=native)
                    self.assertEqual(algorithm.page_k_min[0].dtype, torch.float32)
                    self.assertEqual(algorithm.page_k_max[0].dtype, torch.float32)

    def test_native_bounds_follow_supported_k_dtype_and_fallback_otherwise(self):
        cases = (
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.float64, torch.float32),
        )
        for k_dtype, expected_dtype in cases:
            with self.subTest(k_dtype=k_dtype):
                algorithm, _ = self._make(k_dtype, native=True)
                self.assertEqual(algorithm.page_k_min[0].dtype, expected_dtype)
                self.assertEqual(algorithm.page_k_max[0].dtype, expected_dtype)

    def test_native_representations_and_scores_match_fp32_bounds(self):
        reqs = torch.zeros(1, dtype=torch.int64, device=self.device)
        seq_lens = torch.tensor([9], dtype=torch.int64, device=self.device)

        for k_dtype in (torch.float16, torch.bfloat16):
            for partial_page in (False, True):
                with self.subTest(k_dtype=k_dtype, partial_page=partial_page):
                    legacy, legacy_k = self._make(k_dtype, native=False)
                    native, native_k = self._make(k_dtype, native=True)
                    self.assertTrue(torch.equal(legacy_k, native_k))
                    end_pages = (
                        (seq_lens + legacy.page_size - 1) // legacy.page_size
                        if partial_page
                        else seq_lens // legacy.page_size
                    )

                    legacy._compute_page_representations(
                        0, reqs, seq_lens, 0, end_pages, legacy_k
                    )
                    native._compute_page_representations(
                        0, reqs, seq_lens, 0, end_pages, native_k
                    )

                    valid = legacy.page_valid[0]
                    self.assertTrue(torch.equal(valid, native.page_valid[0]))
                    for legacy_bounds, native_bounds in (
                        (legacy.page_k_min[0], native.page_k_min[0]),
                        (legacy.page_k_max[0], native.page_k_max[0]),
                    ):
                        expected = legacy_bounds[valid].to(k_dtype)
                        actual = native_bounds[valid]
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int16), expected.view(torch.int16)
                            )
                        )

                    torch.manual_seed(73)
                    queries = torch.randn(1, 4, 8, dtype=k_dtype, device=self.device)
                    physical_pages = torch.arange(
                        legacy.page_k_min[0].shape[0],
                        dtype=torch.int64,
                        device=self.device,
                    ).unsqueeze(0)
                    legacy_scores = legacy._retrieve_page_scores(
                        0, physical_pages, reqs, queries
                    )
                    native_scores = native._retrieve_page_scores(
                        0, physical_pages, reqs, queries
                    )

                    self.assertEqual(native_scores.dtype, torch.float32)
                    self.assertTrue(torch.equal(native_scores, legacy_scores))
                    self.assertTrue(
                        torch.equal(
                            torch.topk(native_scores, k=2, dim=1).indices,
                            torch.topk(legacy_scores, k=2, dim=1).indices,
                        )
                    )


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
