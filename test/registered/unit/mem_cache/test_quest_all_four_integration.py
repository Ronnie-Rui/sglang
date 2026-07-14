import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseCoordinator
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
    def __init__(self, seq_lens, req_pool_indices, *, rids=None):
        self.seq_lens = seq_lens
        self.seq_lens_cpu = seq_lens.cpu()
        self.req_pool_indices = req_pool_indices
        self.req_pool_indices_cpu = req_pool_indices.cpu()
        self.rids = list(
            rids
            if rids is not None
            else [f"request-{slot}" for slot in self.req_pool_indices_cpu.tolist()]
        )
        self.spec_info = None
        self.runtime_sparse_page_capacity = None
        self.forward_mode = SimpleNamespace(is_decode=lambda: True)


def _build_storage(seq_lens, page_size, device, seed, dtype=torch.float32):
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
        dtype=dtype,
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
        max_seq_len_k=0,
        scheduler_metadata=None,
    )


def _selected_rows(selected_pages, valid_lengths):
    return [
        sorted(selected_pages[row, : int(length)].tolist())
        for row, length in enumerate(valid_lengths.tolist())
    ]


def _expected_fa_lengths(valid_lengths, sparse_mask, seq_lens, page_size):
    last_page_lengths = torch.where(
        seq_lens > 0,
        (seq_lens - 1) % page_size + 1,
        torch.zeros_like(seq_lens),
    )
    sparse_seq_lens = (
        valid_lengths.to(seq_lens.dtype) - 1
    ) * page_size + last_page_lengths
    cache_seqlens = torch.where(
        sparse_mask & (valid_lengths > 0),
        sparse_seq_lens,
        seq_lens,
    ).to(torch.int32)
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=seq_lens.device),
            cache_seqlens.cumsum(0, dtype=torch.int32),
        )
    )
    return cache_seqlens, cu_seqlens


def _all_four_config(*, use_native_page_bounds_dtype=False):
    config = {
        "layer_selection_reuse_interval": 2,
        "layer_page_budget": _BUDGET,
        "use_fused_topk_fa_metadata_kernel": True,
        "use_lazy_page_update_score_kernel": True,
    }
    if use_native_page_bounds_dtype:
        config["use_native_page_bounds_dtype"] = True
    return config


class TestExpectedFALengths(unittest.TestCase):
    def test_uses_token_lengths_and_dense_fallbacks(self):
        cache_seqlens, cu_seqlens = _expected_fa_lengths(
            valid_lengths=torch.tensor([3, 0, 2], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True, False]),
            seq_lens=torch.tensor([33, 29, 18], dtype=torch.int64),
            page_size=4,
        )

        torch.testing.assert_close(
            cache_seqlens,
            torch.tensor([9, 29, 18], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cu_seqlens,
            torch.tensor([0, 9, 38, 56], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


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
        max_selected_tokens = 800
        max_selected_pages = max_selected_tokens // page_size
        # The first row would produce a full-budget history width of 255, so
        # the fixed token budget caps it at 198 history plus 2 recent pages.
        # The [4, 24) layer budget remains narrower at 191 history pages.
        seq_lens = torch.tensor([2564, 2308, 2052], device=device)
        req_to_token, key_buffer = _build_storage(
            seq_lens, page_size, device, seed=317, dtype=torch.float16
        )
        common = dict(
            seq_lens=seq_lens,
            page_size=page_size,
            sparsity_ratio=0.4,
            num_recent_pages=2,
            req_to_token=req_to_token,
            key_buffer=key_buffer,
        )
        actual = _make_algorithm(
            **common,
            extra_config={
                **_all_four_config(use_native_page_bounds_dtype=True),
                "quest_max_selected_tokens": max_selected_tokens,
            },
        )
        reference = _make_algorithm(
            **common,
            extra_config={
                "layer_selection_reuse_interval": 2,
                "layer_page_budget": _BUDGET,
                "quest_max_selected_tokens": max_selected_tokens,
            },
        )
        self.assertEqual(actual.page_k_min[0].dtype, torch.float16)
        self.assertEqual(actual.page_k_max[0].dtype, torch.float16)
        self.assertEqual(reference.page_k_min[0].dtype, torch.float32)

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
        self.assertEqual(actual.quest_max_selected_pages, max_selected_pages)
        self.assertEqual(reference.quest_max_selected_pages, max_selected_pages)
        self.assertEqual(full_width, max_selected_pages)
        self.assertEqual(
            reference._retrieval_plan.max_k + reference.num_recent_pages,
            max_selected_pages,
        )
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
                self.assertLessEqual(actual_pages.shape[1], max_selected_pages)
                self.assertLessEqual(reference_pages.shape[1], max_selected_pages)
                self.assertTrue(torch.all(actual_lengths <= max_selected_pages).item())
                self.assertTrue(
                    torch.all(reference_lengths <= max_selected_pages).item()
                )
                self.assertEqual(actual_lengths[1].item(), 0)
                expected_cache_seqlens, expected_cu_seqlens = _expected_fa_lengths(
                    actual_lengths,
                    sparse_mask,
                    seq_lens,
                    page_size,
                )
                torch.testing.assert_close(
                    metadata.cache_seqlens_int32,
                    expected_cache_seqlens,
                    rtol=0,
                    atol=0,
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

    def test_eager_all_four_reuses_owned_physical_selection_on_next_token(self):
        from sglang.jit_kernel.quest.topk import (
            quest_topk_to_flashattention_metadata_out,
        )
        from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
            quest_update_flashattention_metadata_,
        )
        from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
            quest_lazy_update_page_scores,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        page_size = 4
        token0_seq_lens = torch.tensor([193, 169, 145], device=device)
        token1_seq_lens = token0_seq_lens + 1
        req_to_token, key_buffer = _build_storage(
            token0_seq_lens,
            page_size,
            device,
            seed=1429,
            dtype=torch.float16,
        )
        algorithm = _make_algorithm(
            seq_lens=token0_seq_lens,
            page_size=page_size,
            sparsity_ratio=0.4,
            num_recent_pages=2,
            req_to_token=req_to_token,
            key_buffer=key_buffer,
            extra_config={
                **_all_four_config(use_native_page_bounds_dtype=True),
                "decode_token_selection_reuse_interval": 2,
                "quest_max_selected_tokens": 96,
            },
        )
        req_pool_indices = torch.arange(3, dtype=torch.int64, device=device)
        sparse_mask = torch.tensor([True, False, True], device=device)
        rids = ["request-a", "request-b", "request-c"]
        token0_batch = _ForwardBatch(token0_seq_lens, req_pool_indices, rids=rids)
        token1_batch = _ForwardBatch(token1_seq_lens, req_pool_indices, rids=rids)

        max_pages = int(((token1_seq_lens.max() + page_size - 1) // page_size).item())
        metadata = _make_metadata(3, max_pages, device)
        metadata_ptrs = (
            metadata.page_table.data_ptr(),
            metadata.cache_seqlens_int32.data_ptr(),
            metadata.cu_seqlens_k.data_ptr(),
        )

        def reset_dense_metadata(forward_batch):
            page_starts = (
                torch.arange(max_pages, dtype=torch.int64, device=device) * page_size
            )
            dense_pages = torch.div(
                req_to_token[
                    req_pool_indices.unsqueeze(1),
                    page_starts.unsqueeze(0),
                ],
                page_size,
                rounding_mode="floor",
            ).to(torch.int32)
            metadata.page_table.copy_(dense_pages)
            metadata.cache_seqlens_int32.copy_(forward_batch.seq_lens.to(torch.int32))
            metadata.cu_seqlens_k[0].zero_()
            metadata.cu_seqlens_k[1:].copy_(
                forward_batch.seq_lens.cumsum(0, dtype=torch.int32)
            )
            metadata.max_seq_len_k = int(forward_batch.seq_lens_cpu.max().item())
            metadata.scheduler_metadata = None

        adaptor = FlashAttentionAdaptor(device)
        coordinator = object.__new__(SparseCoordinator)
        coordinator.algorithm = algorithm
        coordinator.backend_adaptor = adaptor
        coordinator.req_to_token_pool = algorithm.req_to_token_pool
        coordinator.page_size = page_size
        coordinator._forward_sparse_mask = sparse_mask

        generator = torch.Generator(device=device).manual_seed(1543)
        token0_queries = [
            torch.randn(
                3, 1, 8, device=device, dtype=torch.float32, generator=generator
            )
            for _ in range(_END_LAYER)
        ]
        token1_queries = [
            torch.randn(
                3, 1, 8, device=device, dtype=torch.float32, generator=generator
            )
            for _ in range(_END_LAYER)
        ]
        layers = [SimpleNamespace(layer_id=layer_id) for layer_id in range(_END_LAYER)]
        anchors = list(range(0, _END_LAYER, 2))

        with patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_lazy_update_page_scores",
            wraps=quest_lazy_update_page_scores,
        ) as lazy_score, patch(
            "sglang.jit_kernel.quest.topk." "quest_topk_to_flashattention_metadata_out",
            wraps=quest_topk_to_flashattention_metadata_out,
        ) as fused_topk, patch(
            "sglang.srt.mem_cache.sparsity.kernels."
            "quest_flashattention_metadata.quest_update_flashattention_metadata_",
            wraps=quest_update_flashattention_metadata_,
        ) as metadata_update, patch.object(
            adaptor,
            "adapt_for_attn_metadata",
            wraps=adaptor.adapt_for_attn_metadata,
        ) as adapt_metadata:
            reset_dense_metadata(token0_batch)
            algorithm.begin_forward(
                forward_batch=token0_batch,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                device=device,
            )
            self.assertEqual(algorithm._decode_selection_cache_mode, "refresh")
            adaptor.save_original_metadata(metadata)
            for layer_id, layer in enumerate(layers):
                adapted = coordinator._handle_sparse_retrieve(
                    token0_queries[layer_id], layer, token0_batch, metadata
                )
                self.assertIs(adapted, metadata)
                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    token0_seq_lens,
                    key_buffer,
                    token0_batch,
                )
            algorithm.finalize_forward(token0_batch)
            torch.cuda.synchronize()

            self.assertEqual(lazy_score.call_count, len(anchors))
            self.assertEqual(fused_topk.call_count, len(anchors))
            self.assertEqual(metadata_update.call_count, 0)
            committed = algorithm._decode_selection_cache_state
            self.assertIsNotNone(committed)
            self.assertEqual(committed.age, 0)
            self.assertEqual(len(committed.selections), len(anchors))

            cached_ptrs = {}
            cached_snapshots = {}
            for group, (physical_pages, valid_lengths) in committed.selections.items():
                cached_ptrs[group] = (
                    physical_pages.data_ptr(),
                    valid_lengths.data_ptr(),
                )
                cached_snapshots[group] = (
                    physical_pages.clone(),
                    valid_lengths.clone(),
                )
                self.assertNotEqual(
                    physical_pages.untyped_storage().data_ptr(),
                    metadata.page_table.untyped_storage().data_ptr(),
                )

            reset_dense_metadata(token1_batch)
            for group, (physical_pages, valid_lengths) in committed.selections.items():
                expected_pages, expected_lengths = cached_snapshots[group]
                torch.testing.assert_close(
                    physical_pages, expected_pages, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    valid_lengths, expected_lengths, rtol=0, atol=0
                )

            token1_adapt_start = adapt_metadata.call_count
            algorithm.begin_forward(
                forward_batch=token1_batch,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                device=device,
            )
            self.assertEqual(algorithm._decode_selection_cache_mode, "reuse")
            adaptor.save_original_metadata(metadata)
            for layer_id, layer in enumerate(layers):
                adapted = coordinator._handle_sparse_retrieve(
                    token1_queries[layer_id], layer, token1_batch, metadata
                )
                self.assertIs(adapted, metadata)
                call = adapt_metadata.call_args
                group = algorithm._selection_group(layer_id)
                selected_pages = call.kwargs["selected_indices"]
                valid_lengths = call.kwargs["valid_lengths"]
                self.assertEqual(selected_pages.data_ptr(), cached_ptrs[group][0])
                self.assertEqual(valid_lengths.data_ptr(), cached_ptrs[group][1])

                if layer_id in anchors:
                    self.assertIs(
                        call.kwargs["selected_physical_indices"], selected_pages
                    )
                else:
                    self.assertIsNone(call.kwargs["selected_physical_indices"])
                    self.assertTrue(call.kwargs["metadata_prepared"])

                if layer_id == 0:
                    torch.cuda.synchronize()
                    for row in sparse_mask.nonzero(as_tuple=False).flatten().tolist():
                        length = int(valid_lengths[row].item())
                        torch.testing.assert_close(
                            metadata.page_table[row, :length],
                            selected_pages[row, :length].to(torch.int32),
                            rtol=0,
                            atol=0,
                        )

                if layer_id in (0, 4, 24):
                    expected_cache_seqlens, expected_cu_seqlens = _expected_fa_lengths(
                        valid_lengths,
                        sparse_mask,
                        token1_seq_lens,
                        page_size,
                    )
                    torch.testing.assert_close(
                        metadata.cache_seqlens_int32,
                        expected_cache_seqlens,
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        metadata.cu_seqlens_k,
                        expected_cu_seqlens,
                        rtol=0,
                        atol=0,
                    )

                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    token1_seq_lens,
                    key_buffer,
                    token1_batch,
                )
            algorithm.finalize_forward(token1_batch)
            torch.cuda.synchronize()

        self.assertEqual(adapt_metadata.call_count - token1_adapt_start, _END_LAYER)
        self.assertEqual(lazy_score.call_count, len(anchors))
        self.assertEqual(fused_topk.call_count, len(anchors))
        self.assertEqual(metadata_update.call_count, len(anchors))
        self.assertTrue(
            all(
                call.kwargs["selected_indices_are_physical"]
                for call in metadata_update.call_args_list
            )
        )
        self.assertEqual(
            [call.kwargs["update_lengths"] for call in metadata_update.call_args_list],
            [layer_id in (0, 4, 24) for layer_id in anchors],
        )
        self.assertEqual(
            (
                metadata.page_table.data_ptr(),
                metadata.cache_seqlens_int32.data_ptr(),
                metadata.cu_seqlens_k.data_ptr(),
            ),
            metadata_ptrs,
        )
        committed = algorithm._decode_selection_cache_state
        self.assertIsNotNone(committed)
        self.assertEqual(committed.age, 1)
        for group, (physical_pages, valid_lengths) in committed.selections.items():
            self.assertEqual(physical_pages.data_ptr(), cached_ptrs[group][0])
            self.assertEqual(valid_lengths.data_ptr(), cached_ptrs[group][1])

    def test_fixed_capacity_all_four_captures_direct_metadata_fallback(self):
        from sglang.srt.mem_cache.sparsity.kernels.quest_flashattention_metadata import (
            quest_finalize_to_flashattention_metadata_,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        page_size = 4
        max_selected_tokens = 64
        max_selected_pages = max_selected_tokens // page_size
        graph_capacity_pages = 48
        seq_lens = torch.tensor([192, 168, 144], device=device)
        req_to_token, key_buffer = _build_storage(seq_lens, page_size, device, seed=419)
        algorithm = _make_algorithm(
            seq_lens=seq_lens,
            page_size=page_size,
            sparsity_ratio=0.4,
            num_recent_pages=2,
            req_to_token=req_to_token,
            key_buffer=key_buffer,
            extra_config={
                **_all_four_config(),
                "decode_token_selection_reuse_interval": 2,
                "quest_max_selected_tokens": max_selected_tokens,
            },
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
                fixed_capacity=graph_capacity_pages,
            )
            # Build the alternate budget plan outside graph capture, just as
            # production graph setup precomputes shape-specific state.
            algorithm._get_retrieval_plan_for_ratio(
                algorithm._retrieval_plan,
                algorithm.sparsity_ratio * 0.75,
            )

        begin_fixed_forward()
        self.assertIsNone(algorithm._decode_selection_cache_state)
        self.assertIsNone(algorithm._pending_decode_selection_cache_state)
        self.assertIsNone(algorithm._decode_selection_cache_mode)
        plan = algorithm._retrieval_plan
        self.assertTrue(plan.fixed_capacity)
        full_width = plan.max_k + algorithm.num_recent_pages
        budget_plan = algorithm._get_retrieval_plan_for_ratio(
            plan, algorithm.sparsity_ratio * 0.75
        )
        budget_width = budget_plan.max_k + algorithm.num_recent_pages
        self.assertEqual(algorithm.quest_max_selected_pages, max_selected_pages)
        self.assertEqual(full_width, max_selected_pages)
        self.assertLess(budget_width, full_width)
        self.assertLessEqual(budget_width, max_selected_pages)
        self.assertTrue(
            torch.all(
                plan.k_per_req <= max_selected_pages - algorithm.num_recent_pages
            ).item()
        )
        metadata = _make_metadata(3, full_width, device)

        # Warm both budget widths and prove the eager-only fused kernel cannot
        # be selected by a fixed-capacity CUDA Graph plan.
        warm_valid_lengths = []
        warm_widths = {}
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
                self.assertTrue(metadata_prepared)
                self.assertLessEqual(selected_pages.shape[1], max_selected_pages)
                warm_valid_lengths.append(valid_lengths.clone())
                if layer_id in (3, 4, 23, 24):
                    warm_widths[layer_id] = selected_pages.shape[1]
                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    seq_lens,
                    key_buffer,
                    forward_batch,
                )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.all(torch.stack(warm_valid_lengths) <= max_selected_pages).item()
        )
        self.assertEqual(warm_widths[3], full_width)
        self.assertEqual(warm_widths[4], budget_width)
        self.assertEqual(warm_widths[23], budget_width)
        self.assertEqual(warm_widths[24], full_width)
        self.assertEqual(direct_finalize.call_count, _END_LAYER // 2)
        self.assertEqual(
            [call.kwargs["update_lengths"] for call in direct_finalize.call_args_list],
            [layer_id in (0, 4, 24) for layer_id in range(0, _END_LAYER, 2)],
        )
        self.assertFalse(algorithm.states.repr_constructed.any().item())
        self.assertFalse(algorithm.states.last_constructed_page.any().item())
        algorithm.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        self.assertIsNone(algorithm._decode_selection_cache_state)
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
        self.assertIsNone(algorithm._decode_selection_cache_state)
        self.assertIsNone(algorithm._pending_decode_selection_cache_state)
        self.assertIsNone(algorithm._decode_selection_cache_mode)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured_widths = {}
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
                if layer_id in (3, 4, 23, 24):
                    captured_widths[layer_id] = selected_pages.shape[1]
                algorithm.update_representations(
                    layer_id,
                    req_pool_indices,
                    seq_lens,
                    key_buffer,
                    forward_batch,
                )

        torch.cuda.synchronize()
        self.assertEqual(captured_widths, warm_widths)
        self.assertFalse(algorithm.states.repr_constructed.any().item())
        self.assertFalse(algorithm.states.last_constructed_page.any().item())
        algorithm.finalize_forward(forward_batch)
        torch.cuda.synchronize()
        self.assertIsNone(algorithm._decode_selection_cache_state)
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
        self.assertIsNone(algorithm._decode_selection_cache_state)
        self.assertTrue(metadata_prepared)
        self.assertLessEqual(selected_pages.shape[1], max_selected_pages)
        self.assertTrue(torch.all(valid_lengths <= max_selected_pages).item())
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
        expected_cache_seqlens, expected_cu_seqlens = _expected_fa_lengths(
            valid_lengths,
            sparse_mask,
            seq_lens,
            page_size,
        )
        torch.testing.assert_close(
            metadata.cache_seqlens_int32,
            expected_cache_seqlens,
            rtol=0,
            atol=0,
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
