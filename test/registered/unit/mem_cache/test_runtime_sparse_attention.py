import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.utils import enable_fused_set_kv_buffer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionAdaptor(unittest.TestCase):
    def test_rewrites_flashattention_metadata_in_place(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([7, 6], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
            max_seq_len_k=7,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens = metadata.cu_seqlens_k
        adaptor.save_original_metadata(metadata)

        # Runtime page_size=4. Request 0's second logical page starts at
        # physical token 8, hence FlashAttention physical page 2. Request 1
        # remains dense because sparse_mask[1] is false.
        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 8, 9, 10, 11],
                [4, 5, 6, 7, 12, 13, 14, 15],
            ],
            dtype=torch.int64,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([7, 6], dtype=torch.int64),
        )

        result = adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertIs(result, metadata)
        self.assertIs(metadata.page_table, page_table)
        self.assertIs(metadata.cache_seqlens_int32, cache_seqlens)
        self.assertIs(metadata.cu_seqlens_k, cu_seqlens)
        self.assertEqual(metadata.page_table.tolist(), [[2, 2], [1, 3]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 3, 9])
        # Request 1 remains dense, so the batch max must stay a safe upper
        # bound for its six-token cache rather than shrinking to one page.
        self.assertEqual(metadata.max_seq_len_k, 7)
        self.assertIsNone(metadata.scheduler_metadata)

        cache_seqlens_version = cache_seqlens._version
        cu_seqlens_version = cu_seqlens._version
        page_table_version = page_table._version

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[0], [0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
        )
        self.assertEqual(metadata.page_table.tolist(), [[0, 2], [1, 3]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [3, 6])
        self.assertEqual(cache_seqlens._version, cache_seqlens_version)
        self.assertEqual(cu_seqlens._version, cu_seqlens_version)
        self.assertGreater(page_table._version, page_table_version)

    def test_ragged_mixed_batch_reuses_layer_invariant_metadata(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32
            ),
            cache_seqlens_int32=torch.tensor([10, 7, 3], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 10, 17, 20], dtype=torch.int32),
            max_seq_len_k=10,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        req_to_token = torch.tensor(
            [
                [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27],
                [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31],
                [8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens=torch.tensor([10, 7, 3], dtype=torch.int64),
        )
        adaptor.save_original_metadata(metadata)

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor(
                [[2, 0], [1, -1], [0, -1]], dtype=torch.int32
            ),
            valid_lengths=torch.tensor([2, 1, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True, True]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertEqual(
            metadata.page_table.tolist(), [[6, 0, 12], [4, 21, 22], [30, 31, 32]]
        )
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [6, 3, 3])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 6, 9, 12])
        self.assertEqual(metadata.max_seq_len_k, 10)

        cache_seqlens_version = metadata.cache_seqlens_int32._version
        cu_seqlens_version = metadata.cu_seqlens_k._version
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor(
                [[1, 2], [0, -1], [2, -1]], dtype=torch.int32
            ),
            valid_lengths=torch.tensor([2, 1, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True, True]),
            current_metadata=metadata,
            forward_batch=forward_batch,
            req_to_token=req_to_token,
            page_size=4,
            layer_id=1,
        )

        self.assertEqual(
            metadata.page_table.tolist(), [[3, 6, 12], [1, 21, 22], [30, 31, 32]]
        )
        self.assertEqual(metadata.cache_seqlens_int32._version, cache_seqlens_version)
        self.assertEqual(metadata.cu_seqlens_k._version, cu_seqlens_version)

    def test_no_selection_keeps_dense_metadata(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[9, 10]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([5], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
            max_seq_len_k=5,
            scheduler_metadata=torch.ones(1, dtype=torch.int32),
        )
        adaptor.save_original_metadata(metadata)

        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[-1]], dtype=torch.int32),
            valid_lengths=torch.tensor([0], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([5], dtype=torch.int64),
            ),
            req_to_token=torch.arange(8, dtype=torch.int64).view(1, 8),
            page_size=4,
            layer_id=0,
        )

        self.assertEqual(metadata.page_table.tolist(), [[9, 10]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [5])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 5])
        self.assertEqual(metadata.max_seq_len_k, 5)
        self.assertIsNone(metadata.scheduler_metadata)

    def test_save_original_metadata_resets_plan_for_next_replay(self):
        adaptor = FlashAttentionAdaptor(torch.device("cpu"))
        metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([9], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 9], dtype=torch.int32),
            max_seq_len_k=9,
            scheduler_metadata=None,
        )
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens = metadata.cu_seqlens_k
        req_to_token = torch.tensor(
            [[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]], dtype=torch.int64
        )

        adaptor.save_original_metadata(metadata)
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[2, 0]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([9], dtype=torch.int64),
            ),
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )
        self.assertEqual(metadata.page_table.tolist(), [[4, 0, 2]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [5])

        # BCG replay refills the same fixed-address metadata buffers with the
        # next dense batch before the start-layer hook runs again.
        metadata.page_table.copy_(torch.tensor([[0, 1, 2]], dtype=torch.int32))
        metadata.cache_seqlens_int32.copy_(torch.tensor([6], dtype=torch.int32))
        metadata.cu_seqlens_k.copy_(torch.tensor([0, 6], dtype=torch.int32))
        metadata.max_seq_len_k = 6
        metadata.scheduler_metadata = torch.ones(1, dtype=torch.int32)

        adaptor.save_original_metadata(metadata)
        adaptor.adapt_for_attn_metadata(
            selected_indices=torch.tensor([[1]], dtype=torch.int32),
            valid_lengths=torch.tensor([1], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
            current_metadata=metadata,
            forward_batch=SimpleNamespace(
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([6], dtype=torch.int64),
            ),
            req_to_token=req_to_token,
            page_size=4,
            layer_id=0,
        )

        self.assertIs(metadata.page_table, page_table)
        self.assertIs(metadata.cache_seqlens_int32, cache_seqlens)
        self.assertIs(metadata.cu_seqlens_k, cu_seqlens)
        self.assertEqual(metadata.page_table.tolist(), [[2, 1, 2]])
        self.assertEqual(metadata.cache_seqlens_int32.tolist(), [2])
        self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 2])
        self.assertEqual(metadata.max_seq_len_k, 6)
        self.assertIsNone(metadata.scheduler_metadata)


class TestQuestScoring(unittest.TestCase):
    def test_gqa_heads_are_scored_without_sign_cancellation(self):
        algorithm = QuestAlgorithm.__new__(QuestAlgorithm)
        algorithm.page_k_min = {
            0: torch.tensor([[[-10.0]], [[0.0]]], dtype=torch.float32)
        }
        algorithm.page_k_max = {
            0: torch.tensor([[[1.0]], [[5.0]]], dtype=torch.float32)
        }
        algorithm.page_valid = {0: torch.tensor([True, True])}

        scores = algorithm._retrieve_page_scores(
            layer_id=0,
            phys_pages=torch.tensor([[0, 1]], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            queries=torch.tensor([[[1.0], [-1.0]]], dtype=torch.float32),
        )

        self.assertEqual(scores.tolist(), [[10.0, 5.0]])


class TestRuntimeSparseKvWritePolicy(unittest.TestCase):
    def test_disables_fused_kv_prewrite(self):
        with forward_context(
            ForwardContext(
                attn_backend=None,
                runtime_sparse_coordinator=object(),
            )
        ):
            self.assertFalse(enable_fused_set_kv_buffer(SimpleNamespace()))


class _ForwardMode:
    def __init__(self, *, extend=False, decode=False):
        self._extend = extend
        self._decode = decode

    def is_extend(self):
        return self._extend

    def is_decode(self):
        return self._decode


class _RecordingAlgorithm(BaseSparseAlgorithmImpl):
    def __init__(self):
        config = SimpleNamespace(
            page_size=4,
            sparse_extra_config={"sparsity_ratio": 0.5, "num_recent_pages": 1},
        )
        super().__init__(config, torch.device("cpu"))
        self.calls = []
        self.end_layer = 1
        self.states = SimpleNamespace(
            repr_constructed=torch.tensor([True]),
            prompt_lens=torch.tensor([99], dtype=torch.int64),
            last_constructed_page=torch.tensor([9], dtype=torch.int64),
        )

    def _compute_page_representations(
        self, layer_id, reqs, seq_lens, start_page, end_page, k_buffer
    ):
        self.calls.append(
            (
                layer_id,
                reqs.clone(),
                seq_lens.clone(),
                start_page.clone(),
                end_page.clone(),
            )
        )

    def _initialize_representation_pools(self, start_layer, end_layer, total_num_pages):
        pass

    def _retrieve_page_scores(self, layer_id, phys_pages, req_pool_indices, queries):
        raise NotImplementedError


class TestSparseRepresentationLifecycle(unittest.TestCase):
    def test_extend_resets_slot_and_decode_updates_only_at_page_boundary(self):
        algorithm = _RecordingAlgorithm()
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        k_buffer = torch.empty(1)

        algorithm.construct_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([10], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(extend=True),
                extend_prefix_lens=torch.tensor([0], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [0])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [2])
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2])

        algorithm.construct_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([14], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(extend=True),
                extend_prefix_lens=torch.tensor([10], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [2])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [3])

        num_calls = len(algorithm.calls)
        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([15], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([15], dtype=torch.int64),
            ),
        )
        self.assertEqual(len(algorithm.calls), num_calls)

        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=torch.tensor([16], dtype=torch.int64),
            k_buffer=k_buffer,
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([16], dtype=torch.int64),
            ),
        )
        self.assertEqual(algorithm.calls[-1][3].tolist(), [3])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [4])

    def test_short_prompt_constructs_first_page_during_decode(self):
        algorithm = _RecordingAlgorithm()
        algorithm.end_layer = 2
        algorithm.states.repr_constructed[0] = False
        algorithm.states.prompt_lens[0] = 3
        algorithm.states.last_constructed_page[0] = 0
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        k_buffer = torch.empty(1)

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([3], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
                ),
            )
        self.assertEqual(algorithm.calls, [])

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([4], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([4], dtype=torch.int64),
                ),
            )
            if layer_id == 0:
                self.assertFalse(algorithm.states.repr_constructed[0].item())

        self.assertEqual([call[3].tolist() for call in algorithm.calls], [[0], [0]])
        self.assertEqual([call[4].tolist() for call in algorithm.calls], [[1], [1]])
        self.assertTrue(algorithm.states.repr_constructed[0].item())
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [1])

        for layer_id in range(2):
            algorithm.update_representations(
                layer_id=layer_id,
                req_pool_indices=req_pool_indices,
                seq_lens=torch.tensor([8], dtype=torch.int64),
                k_buffer=k_buffer,
                forward_batch=SimpleNamespace(
                    forward_mode=_ForwardMode(decode=True),
                    seq_lens_cpu=torch.tensor([8], dtype=torch.int64),
                ),
            )
        self.assertEqual(
            [call[3].tolist() for call in algorithm.calls[-2:]], [[1], [1]]
        )
        self.assertEqual(
            [call[4].tolist() for call in algorithm.calls[-2:]], [[2], [2]]
        )
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2])

    def test_decode_boundary_handles_constructed_and_unconstructed_requests(self):
        algorithm = _RecordingAlgorithm()
        algorithm.states = SimpleNamespace(
            repr_constructed=torch.tensor([True, False]),
            prompt_lens=torch.tensor([5, 3], dtype=torch.int64),
            last_constructed_page=torch.tensor([1, 0], dtype=torch.int64),
        )

        algorithm.update_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([8, 4], dtype=torch.int64),
            k_buffer=torch.empty(1),
            forward_batch=SimpleNamespace(
                forward_mode=_ForwardMode(decode=True),
                seq_lens_cpu=torch.tensor([8, 4], dtype=torch.int64),
            ),
        )

        self.assertEqual(algorithm.calls[-1][1].tolist(), [0, 1])
        self.assertEqual(algorithm.calls[-1][3].tolist(), [1, 0])
        self.assertEqual(algorithm.calls[-1][4].tolist(), [2, 1])
        self.assertEqual(algorithm.states.repr_constructed.tolist(), [True, True])
        self.assertEqual(algorithm.states.last_constructed_page.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
