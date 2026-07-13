import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


_END_LAYER = 28
_BUDGET = [{"start_layer": 4, "end_layer": 24, "scale": 0.75}]


class _Config:
    def __init__(self, page_size, sparsity_ratio, num_recent_pages, extra_config):
        self.page_size = page_size
        self.sparse_extra_config = {
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": num_recent_pages,
            **extra_config,
        }


class _ReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.max_context_len = req_to_token.shape[1]


class _TokenToKVPool:
    def __init__(self, key_buffer):
        self.key_buffer = key_buffer

    def get_key_buffer(self, _layer_id):
        return self.key_buffer


class _States:
    def __init__(self, size, device):
        self.repr_constructed = torch.zeros(size, dtype=torch.bool, device=device)
        self.prompt_lens = torch.zeros(size, dtype=torch.int64, device=device)
        self.last_constructed_page = torch.zeros(size, dtype=torch.int64, device=device)


class _ForwardBatch:
    def __init__(self, seq_lens, req_pool_indices):
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens.cpu()
        self.req_pool_indices = req_pool_indices
        self.forward_mode = SimpleNamespace(is_decode=lambda: True)


def _build_storage(seq_lens, page_size, device, seed):
    batch_size = seq_lens.numel()
    max_seq_len = int(seq_lens.max().item())
    tokens_per_req = ((max_seq_len + page_size - 1) // page_size) * page_size
    positions = torch.arange(max_seq_len, dtype=torch.int32, device=device)
    req_to_token = torch.stack(
        [request * tokens_per_req + positions for request in range(batch_size)]
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    key_buffer = torch.randn(
        batch_size * tokens_per_req,
        1,
        8,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return req_to_token, key_buffer


def _make_algorithm(
    *,
    seq_lens,
    page_size,
    sparsity_ratio,
    num_recent_pages,
    req_to_token,
    key_buffer,
    extra_config,
):
    device = seq_lens.device
    algorithm = QuestAlgorithm(
        _Config(page_size, sparsity_ratio, num_recent_pages, extra_config), device
    )
    algorithm.initialize_representation_pool(
        start_layer=0,
        end_layer=_END_LAYER,
        token_to_kv_pool=_TokenToKVPool(key_buffer),
        req_to_token_pool=_ReqToTokenPool(req_to_token),
        states=_States(seq_lens.numel(), device),
    )
    return algorithm


def _make_metadata(batch_size, width, device):
    return SimpleNamespace(
        page_table=torch.full(
            (batch_size, width), -99, dtype=torch.int32, device=device
        ),
        cache_seqlens_int32=torch.full(
            (batch_size,), -99, dtype=torch.int32, device=device
        ),
        cu_seqlens_k=torch.full(
            (batch_size + 1,), -99, dtype=torch.int32, device=device
        ),
    )


def _selected_rows(selected_pages, valid_lengths):
    return [
        sorted(selected_pages[row, : int(length)].tolist())
        for row, length in enumerate(valid_lengths.tolist())
    ]


def _all_four_config():
    return {
        "layer_selection_reuse_interval": 2,
        "layer_page_budget": _BUDGET,
        "use_fused_topk_fa_metadata_kernel": True,
        "use_lazy_page_update_score_kernel": True,
    }


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestAllFourIntegration(unittest.TestCase):
    def test_eager_all_four_matches_exact_reference_across_budget_boundaries(self):
        from sglang.jit_kernel.quest.topk import (
            quest_topk_to_flashattention_metadata_out,
        )
        from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
            quest_lazy_update_page_scores,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        page_size = 4
        # The first row produces a full-budget top-k width of 255 and a
        # [4, 24) budget width of 191. This exercises realistic non-trivial
        # fused-kernel widths while retaining ragged rows and an inactive row.
        seq_lens = torch.tensor([2564, 2308, 2052], device=device)
        req_to_token, key_buffer = _build_storage(seq_lens, page_size, device, seed=317)
        common = dict(
            seq_lens=seq_lens,
            page_size=page_size,
            sparsity_ratio=0.4,
            num_recent_pages=2,
            req_to_token=req_to_token,
            key_buffer=key_buffer,
        )
        actual = _make_algorithm(**common, extra_config=_all_four_config())
        reference = _make_algorithm(
            **common,
            extra_config={
                "layer_selection_reuse_interval": 2,
                "layer_page_budget": _BUDGET,
            },
        )

        req_pool_indices = torch.arange(3, dtype=torch.int64, device=device)
        sparse_mask = torch.tensor([True, False, True], device=device)
        forward_batch = _ForwardBatch(seq_lens, req_pool_indices)
        ready_end_page = (seq_lens - 1) // page_size
        for layer_id in range(_END_LAYER):
            reference._compute_page_representations(
                layer_id,
                req_pool_indices,
                seq_lens,
                0,
                ready_end_page,
                key_buffer,
            )

        actual.begin_forward(forward_batch, req_pool_indices, sparse_mask, device)
        reference.begin_forward(forward_batch, req_pool_indices, sparse_mask, device)
        full_width = actual._retrieval_plan.max_k + actual.num_recent_pages
        metadata = _make_metadata(3, full_width, device)
        metadata_ptrs = (
            metadata.page_table.data_ptr(),
            metadata.cache_seqlens_int32.data_ptr(),
            metadata.cu_seqlens_k.data_ptr(),
        )
        generator = torch.Generator(device=device).manual_seed(911)
        queries = [
            torch.randn(
                3, 1, 8, device=device, dtype=torch.float32, generator=generator
            )
            for _ in range(_END_LAYER)
        ]

        widths = {}
        length_snapshots = {}
        previous_anchor_pages = None
        previous_anchor_lengths = None
        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_lazy_update_page_scores",
            wraps=quest_lazy_update_page_scores,
        ) as lazy_score, patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            wraps=quest_topk_to_flashattention_metadata_out,
        ) as fused_topk:
            for layer_id in range(_END_LAYER):
                actual_pages, actual_lengths, metadata_prepared = actual.retrieve_topk(
                    queries[layer_id],
                    layer_id,
                    req_pool_indices,
                    sparse_mask,
                    forward_batch=forward_batch,
                    attn_metadata=metadata,
                )
                reference_result = reference.retrieve_topk(
                    queries[layer_id],
                    layer_id,
                    req_pool_indices,
                    sparse_mask,
                    forward_batch=forward_batch,
                )
                reference_indices, reference_lengths = reference_result[:2]
                reference_pages = reference.get_selected_physical_pages(
                    reference_indices
                )

                self.assertTrue(metadata_prepared)
                torch.testing.assert_close(
                    actual_lengths, reference_lengths, rtol=0, atol=0
                )
                self.assertEqual(
                    _selected_rows(actual_pages, actual_lengths),
                    _selected_rows(reference_pages, reference_lengths),
                )
                self.assertEqual(actual_lengths[1].item(), 0)
                torch.testing.assert_close(
                    metadata.cache_seqlens_int32,
                    actual_lengths.to(torch.int32),
                    rtol=0,
                    atol=0,
                )
                expected_cu_seqlens = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        actual_lengths.cumsum(0, dtype=torch.int32),
                    )
                )
                torch.testing.assert_close(
                    metadata.cu_seqlens_k,
                    expected_cu_seqlens,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(
                    (
                        metadata.page_table.data_ptr(),
                        metadata.cache_seqlens_int32.data_ptr(),
                        metadata.cu_seqlens_k.data_ptr(),
                    ),
                    metadata_ptrs,
                )

                if layer_id % 2:
                    self.assertEqual(actual_pages.data_ptr(), previous_anchor_pages)
                    self.assertEqual(actual_lengths.data_ptr(), previous_anchor_lengths)
                else:
                    previous_anchor_pages = actual_pages.data_ptr()
                    previous_anchor_lengths = actual_lengths.data_ptr()

                if layer_id in (3, 4, 23, 24):
                    widths[layer_id] = actual_pages.shape[1]
                    length_snapshots[layer_id] = actual_lengths.clone()
                actual.update_representations(
                    layer_id,
                    req_pool_indices,
                    seq_lens,
                    key_buffer,
                    forward_batch,
                )

        torch.cuda.synchronize()
        self.assertFalse(actual.states.repr_constructed.any().item())
        self.assertFalse(actual.states.last_constructed_page.any().item())
        actual.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        actual_anchors = list(range(0, _END_LAYER, 2))
        self.assertEqual(actual._actual_selection_anchors, set(actual_anchors))
        self.assertEqual(lazy_score.call_count, len(actual_anchors))
        self.assertEqual(fused_topk.call_count, len(actual_anchors))
        self.assertEqual(
            [call.kwargs["update_lengths"] for call in fused_topk.call_args_list],
            [layer_id in (0, 4, 24) for layer_id in actual_anchors],
        )

        lazy_pool_ptrs = [
            call.kwargs["page_k_min"].data_ptr() for call in lazy_score.call_args_list
        ]
        self.assertEqual(
            lazy_pool_ptrs,
            [actual.page_k_min[layer_id].data_ptr() for layer_id in actual_anchors],
        )
        for layer_id in range(_END_LAYER):
            if layer_id in actual_anchors:
                self.assertTrue(actual.page_valid[layer_id].any().item())
            else:
                self.assertFalse(actual.page_valid[layer_id].any().item())

        self.assertEqual(widths[3], full_width)
        self.assertLess(widths[4], widths[3])
        self.assertEqual(widths[23], widths[4])
        self.assertEqual(widths[24], widths[3])
        self.assertTrue(torch.all(length_snapshots[4] <= length_snapshots[3]).item())
        self.assertTrue(
            torch.all(
                length_snapshots[4][sparse_mask] < length_snapshots[3][sparse_mask]
            ).item()
        )
        torch.testing.assert_close(
            length_snapshots[24], length_snapshots[3], rtol=0, atol=0
        )
        self.assertTrue(torch.all(actual.states.repr_constructed).item())
        torch.testing.assert_close(
            actual.states.last_constructed_page,
            ready_end_page,
            rtol=0,
            atol=0,
        )

    def test_fixed_capacity_all_four_captures_direct_metadata_fallback(self):
        from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
            quest_finalize_to_flashattention_metadata_,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        page_size = 4
        seq_lens = torch.tensor([252, 228, 196], device=device)
        req_to_token, key_buffer = _build_storage(seq_lens, page_size, device, seed=419)
        algorithm = _make_algorithm(
            seq_lens=seq_lens,
            page_size=page_size,
            sparsity_ratio=0.4,
            num_recent_pages=2,
            req_to_token=req_to_token,
            key_buffer=key_buffer,
            extra_config=_all_four_config(),
        )
        req_pool_indices = torch.arange(3, dtype=torch.int64, device=device)
        sparse_mask = torch.tensor([True, False, True], device=device)
        forward_batch = _ForwardBatch(seq_lens, req_pool_indices)
        generator = torch.Generator(device=device).manual_seed(1231)
        queries = [
            torch.randn(
                3, 1, 8, device=device, dtype=torch.float32, generator=generator
            )
            for _ in range(_END_LAYER)
        ]

        def begin_fixed_forward():
            algorithm.begin_forward(
                forward_batch,
                req_pool_indices,
                sparse_mask,
                device,
                fixed_capacity=64,
            )
            # Build the alternate budget plan outside graph capture, just as
            # production graph setup precomputes shape-specific state.
            algorithm._get_retrieval_plan_for_ratio(
                algorithm._retrieval_plan,
                algorithm.sparsity_ratio * 0.75,
            )

        begin_fixed_forward()
        plan = algorithm._retrieval_plan
        self.assertTrue(plan.fixed_capacity)
        full_width = plan.max_k + algorithm.num_recent_pages
        metadata = _make_metadata(3, full_width, device)

        # Warm both budget widths and prove the eager-only fused kernel cannot
        # be selected by a fixed-capacity CUDA Graph plan.
        with patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            side_effect=AssertionError("fixed plan entered eager fused top-k"),
        ), patch(
            "sglang.srt.mem_cache.sparsity.kernels."
            "quest_flashattention_metadata."
            "quest_finalize_to_flashattention_metadata_",
            wraps=quest_finalize_to_flashattention_metadata_,
        ) as direct_finalize:
            for layer_id in range(_END_LAYER):
                algorithm.retrieve_topk(
                    queries[layer_id],
                    layer_id,
                    req_pool_indices,
                    sparse_mask,
                    forward_batch=forward_batch,
                    attn_metadata=metadata,
                )
                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    seq_lens,
                    key_buffer,
                    forward_batch,
                )
        torch.cuda.synchronize()
        self.assertEqual(direct_finalize.call_count, _END_LAYER // 2)
        self.assertEqual(
            [call.kwargs["update_lengths"] for call in direct_finalize.call_args_list],
            [layer_id in (0, 4, 24) for layer_id in range(0, _END_LAYER, 2)],
        )
        self.assertFalse(algorithm.states.repr_constructed.any().item())
        self.assertFalse(algorithm.states.last_constructed_page.any().item())
        algorithm.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            algorithm.states.last_constructed_page,
            (seq_lens - 1) // page_size,
            rtol=0,
            atol=0,
        )

        for tensor_by_layer in (
            algorithm.page_k_min,
            algorithm.page_k_max,
            algorithm.page_valid,
        ):
            for tensor in tensor_by_layer.values():
                tensor.zero_()
        algorithm.states.repr_constructed.zero_()
        algorithm.states.last_constructed_page.zero_()
        metadata.page_table.fill_(-99)
        metadata.cache_seqlens_int32.fill_(-99)
        metadata.cu_seqlens_k.fill_(-99)

        begin_fixed_forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer_id in range(_END_LAYER):
                selected_pages, valid_lengths, metadata_prepared = (
                    algorithm.retrieve_topk(
                        queries[layer_id],
                        layer_id,
                        req_pool_indices,
                        sparse_mask,
                        forward_batch=forward_batch,
                        attn_metadata=metadata,
                    )
                )
                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    seq_lens,
                    key_buffer,
                    forward_batch,
                )

        torch.cuda.synchronize()
        self.assertFalse(algorithm.states.repr_constructed.any().item())
        self.assertFalse(algorithm.states.last_constructed_page.any().item())
        algorithm.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            algorithm.states.last_constructed_page,
            (seq_lens - 1) // page_size,
            rtol=0,
            atol=0,
        )
        output_ptrs = (
            selected_pages.data_ptr(),
            valid_lengths.data_ptr(),
            metadata.page_table.data_ptr(),
            metadata.cache_seqlens_int32.data_ptr(),
            metadata.cu_seqlens_k.data_ptr(),
        )
        for tensor_by_layer in (
            algorithm.page_k_min,
            algorithm.page_k_max,
            algorithm.page_valid,
        ):
            for tensor in tensor_by_layer.values():
                tensor.zero_()
        algorithm.states.repr_constructed.zero_()
        algorithm.states.last_constructed_page.zero_()
        for query in queries:
            query.normal_()

        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(algorithm.states.repr_constructed.any().item())
        self.assertFalse(algorithm.states.last_constructed_page.any().item())
        algorithm.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        self.assertTrue(metadata_prepared)
        self.assertEqual(
            (
                selected_pages.data_ptr(),
                valid_lengths.data_ptr(),
                metadata.page_table.data_ptr(),
                metadata.cache_seqlens_int32.data_ptr(),
                metadata.cu_seqlens_k.data_ptr(),
            ),
            output_ptrs,
        )
        self.assertEqual(valid_lengths[1].item(), 0)
        torch.testing.assert_close(
            metadata.cache_seqlens_int32,
            valid_lengths.to(torch.int32),
            rtol=0,
            atol=0,
        )
        expected_cu_seqlens = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                valid_lengths.cumsum(0, dtype=torch.int32),
            )
        )
        torch.testing.assert_close(
            metadata.cu_seqlens_k,
            expected_cu_seqlens,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            algorithm.states.last_constructed_page,
            (seq_lens - 1) // page_size,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
