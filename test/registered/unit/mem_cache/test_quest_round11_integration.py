import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


_ADAPTIVE_INTERVAL = "context_adaptive_layer_selection_reuse_interval"
_ADAPTIVE_MIN_PAGES = "context_adaptive_layer_selection_reuse_min_pages"
_SUPERPAGE_SIZE = "quest_superpage_size"
_SUPERPAGE_OVERSAMPLE = "quest_superpage_oversample"
_DENSE_THRESHOLD = "dense_fallback_max_seq_len"


class _Config:
    def __init__(
        self,
        extra_config=None,
        *,
        page_size=4,
        sparsity_ratio=0.25,
        num_recent_pages=1,
    ):
        self.page_size = page_size
        self.backend = "fa3"
        self.min_sparse_prompt_len = 0
        self.sparse_extra_config = {
            "sparsity_ratio": sparsity_ratio,
            "num_recent_pages": num_recent_pages,
            **(extra_config or {}),
        }


class _ForwardMode:
    @staticmethod
    def is_decode():
        return True

    @staticmethod
    def is_extend():
        return False


class _ForwardBatch:
    def __init__(self, seq_lens, *, host_lengths=True):
        self.seq_lens = torch.as_tensor(seq_lens, dtype=torch.int64)
        self.seq_lens_cpu = self.seq_lens.clone() if host_lengths else None
        self.req_pool_indices = torch.arange(self.seq_lens.numel(), dtype=torch.int64)
        self.req_pool_indices_cpu = self.req_pool_indices.clone()
        self.rids = [f"request-{index}" for index in self.req_pool_indices.tolist()]
        self.runtime_sparse_page_capacity = None
        self.spec_info = None
        self.forward_mode = _ForwardMode()


class _RequestStates:
    def __init__(self, batch_size):
        self.repr_constructed = torch.zeros(batch_size, dtype=torch.bool)
        self.prompt_lens = torch.zeros(batch_size, dtype=torch.int64)
        self.last_constructed_page = torch.zeros(batch_size, dtype=torch.int64)


class _ReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.max_context_len = req_to_token.shape[1]


class _TokenToKVPool:
    def __init__(self, key_buffer):
        self.key_buffer = key_buffer

    def get_key_buffer(self, _layer_id):
        return self.key_buffer


class _RecordingAdaptor:
    requires_selected_physical_indices = False

    def __init__(self):
        self.save_original_metadata = Mock()
        self.adapt_for_attn_metadata = Mock(
            side_effect=lambda **kwargs: kwargs["current_metadata"]
        )


def _make_algorithm(
    *,
    max_context_len,
    extra_config,
    page_size=4,
    end_layer=4,
    batch_size=1,
    sparsity_ratio=0.25,
):
    aligned_tokens = ((max_context_len + page_size - 1) // page_size) * page_size
    positions = torch.arange(max_context_len, dtype=torch.int32)
    req_to_token = torch.stack(
        [request * aligned_tokens + positions for request in range(batch_size)]
    )
    key_buffer = torch.arange(batch_size * aligned_tokens, dtype=torch.float32).view(
        -1, 1, 1
    )
    config = _Config(
        extra_config,
        page_size=page_size,
        sparsity_ratio=sparsity_ratio,
        num_recent_pages=1,
    )
    algorithm = QuestAlgorithm(config, torch.device("cpu"))
    states = _RequestStates(batch_size)
    req_pool = _ReqToTokenPool(req_to_token)
    kv_pool = _TokenToKVPool(key_buffer)
    algorithm.initialize_representation_pool(
        0,
        end_layer,
        kv_pool,
        req_pool,
        states,
    )
    return algorithm, states, req_pool, kv_pool


def _make_coordinator(algorithm, states, req_pool, kv_pool, *, end_layer=4):
    coordinator = object.__new__(SparseCoordinator)
    coordinator.config = algorithm.config
    coordinator.algorithm = algorithm
    coordinator.backend_adaptor = _RecordingAdaptor()
    coordinator.req_to_token_pool = req_pool
    coordinator.token_to_kv_pool = kv_pool
    coordinator.states = states
    coordinator.start_layer = 0
    coordinator.end_layer = end_layer
    coordinator.device = torch.device("cpu")
    coordinator.page_size = algorithm.page_size
    coordinator._forward_sparse_mask = None
    coordinator._forward_dense_fallback = False
    coordinator._last_sparse_layer_id = None
    coordinator._forward_started = False
    return coordinator


class TestQuestRound11ConfigurationMatrix(unittest.TestCase):
    def test_all_eight_feature_combinations_construct_with_safe_defaults(self):
        signatures = set()
        for adaptive, superpage, dense in itertools.product((False, True), repeat=3):
            with self.subTest(adaptive=adaptive, superpage=superpage, dense=dense):
                extra_config = {"layer_selection_reuse_interval": 2}
                if adaptive:
                    extra_config.update(
                        {
                            _ADAPTIVE_INTERVAL: 4,
                            _ADAPTIVE_MIN_PAGES: 512,
                            "use_lazy_page_update_score_kernel": True,
                        }
                    )
                if superpage:
                    extra_config.update(
                        {
                            _SUPERPAGE_SIZE: 8,
                            _SUPERPAGE_OVERSAMPLE: 1,
                        }
                    )
                if dense:
                    extra_config[_DENSE_THRESHOLD] = 8192

                config = _Config(extra_config, page_size=16)
                algorithm = QuestAlgorithm(config, torch.device("cpu"))
                signature = (
                    algorithm.context_adaptive_layer_selection_reuse_interval,
                    algorithm.context_adaptive_layer_selection_reuse_min_pages,
                    algorithm.quest_superpage_size,
                    algorithm.quest_superpage_oversample,
                    config.sparse_extra_config.get(_DENSE_THRESHOLD, 0),
                )
                signatures.add(signature)

                self.assertEqual(
                    algorithm.context_adaptive_layer_selection_reuse_interval,
                    4 if adaptive else None,
                )
                self.assertEqual(
                    algorithm.context_adaptive_layer_selection_reuse_min_pages,
                    512 if adaptive else None,
                )
                self.assertEqual(algorithm._use_layer_representation_trackers, adaptive)
                self.assertEqual(algorithm.quest_superpage_size, 8 if superpage else 1)
                self.assertEqual(
                    algorithm.quest_superpage_oversample, 1 if superpage else 2
                )
                self.assertEqual(
                    config.sparse_extra_config.get(_DENSE_THRESHOLD, 0),
                    8192 if dense else 0,
                )

        self.assertEqual(len(signatures), 8)


class TestQuestRound11Contracts(unittest.TestCase):
    def test_adaptive_superpage_helper_publishes_final_scores_and_certificate(self):
        algorithm = QuestAlgorithm(
            _Config(
                {
                    "layer_selection_reuse_interval": 2,
                    _ADAPTIVE_INTERVAL: 4,
                    _ADAPTIVE_MIN_PAGES: 8,
                    _SUPERPAGE_SIZE: 4,
                    _SUPERPAGE_OVERSAMPLE: 1,
                }
            ),
            torch.device("cpu"),
        )
        algorithm._active_layer_selection_reuse_interval = 4
        algorithm._context_adaptive_layer_selection_reuse_active = True
        algorithm._mark_actual_selection_anchor(0)
        algorithm.page_k_min[0] = torch.empty(1)
        algorithm.page_k_max[0] = torch.empty(1)
        algorithm.page_valid[0] = torch.empty(1, dtype=torch.bool)

        plan = SimpleNamespace(
            fixed_capacity=False,
            batch_size=2,
            max_num_pages=16,
            max_k=3,
            physical_pages=torch.arange(16).repeat(2, 1),
            active_mask=torch.tensor([True, True]),
            recent_start=torch.tensor([14, 12]),
            k_per_req=torch.tensor([3, 2], dtype=torch.int64),
        )
        queries = torch.zeros((2, 1, 1))
        repaired_scores = torch.arange(32, dtype=torch.float32).view(2, 16)
        certified = torch.tensor([True, False])

        with patch.object(
            algorithm, "_can_use_superpage_scoring", return_value=True
        ), patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_exact_superpage_page_scores",
            return_value=(repaired_scores, certified),
        ) as exact_superpage:
            actual = algorithm._try_superpage_page_scores(0, queries, plan)

        self.assertIs(actual, repaired_scores)
        self.assertIs(algorithm._last_superpage_certified, certified)
        self.assertEqual(algorithm._last_superpage_candidate_group_count, 1)
        self.assertEqual(algorithm._actual_selection_anchors, {0})
        call = exact_superpage.call_args.kwargs
        self.assertIs(call["queries"], queries)
        self.assertEqual(call["superpage_size"], 4)
        self.assertEqual(call["oversample"], 1)
        torch.testing.assert_close(actual, repaired_scores, rtol=0, atol=0)
        self.assertEqual(certified.tolist(), [True, False])

    def test_adaptive_superpage_advances_per_layer_tracker_between_forwards(self):
        algorithm, _, _, _ = _make_algorithm(
            max_context_len=16,
            page_size=4,
            end_layer=1,
            extra_config={
                "layer_selection_reuse_interval": 2,
                _ADAPTIVE_INTERVAL: 4,
                _ADAPTIVE_MIN_PAGES: 2,
                "use_lazy_page_update_score_kernel": True,
                _SUPERPAGE_SIZE: 2,
                _SUPERPAGE_OVERSAMPLE: 1,
            },
        )
        algorithm._lazy_page_update_active = True
        plan = SimpleNamespace(
            fixed_capacity=False,
            batch_size=1,
            max_num_pages=4,
            max_k=1,
            physical_pages=torch.arange(4).view(1, -1),
            active_mask=torch.tensor([True]),
            recent_start=torch.tensor([3]),
            k_per_req=torch.tensor([1], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([9], dtype=torch.int64),
        )
        observed_start_pages = []

        def advance_tracker(*args, **kwargs):
            req_pool_indices = args[0]
            safe_seq_lens = args[1]
            repr_constructed = args[4]
            last_constructed_page = args[5]
            observed_start_pages.append(last_constructed_page.clone())
            self.assertTrue(kwargs["advance_trackers"])
            repr_constructed[req_pool_indices] = True
            last_constructed_page[req_pool_indices] = (
                safe_seq_lens // algorithm.page_size
            )

        scores = torch.zeros((1, 4))
        certified = torch.tensor([True])
        with patch.object(
            algorithm, "_can_use_superpage_scoring", return_value=True
        ), patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_page_update."
            "quest_update_page_representations_",
            side_effect=advance_tracker,
        ), patch(
            "sglang.srt.mem_cache.sparsity.kernels.quest_score."
            "quest_exact_superpage_page_scores",
            return_value=(scores, certified),
        ):
            algorithm._try_superpage_page_scores(0, torch.zeros((1, 1, 1)), plan)
            plan.seq_lens = torch.tensor([13], dtype=torch.int64)
            algorithm._try_superpage_page_scores(0, torch.zeros((1, 1, 1)), plan)

        self.assertEqual([pages.tolist() for pages in observed_start_pages], [[0], [2]])
        _, last_page = algorithm.get_layer_representation_trackers(0)
        self.assertEqual(last_page.tolist(), [3])

    def test_dense_bypass_advances_adaptive_trackers_then_restores_quest(self):
        algorithm, states, req_pool, kv_pool = _make_algorithm(
            max_context_len=16,
            page_size=4,
            end_layer=4,
            extra_config={
                "layer_selection_reuse_interval": 2,
                _ADAPTIVE_INTERVAL: 4,
                _ADAPTIVE_MIN_PAGES: 3,
                "use_lazy_page_update_score_kernel": True,
                _DENSE_THRESHOLD: 8,
            },
        )
        coordinator = _make_coordinator(algorithm, states, req_pool, kv_pool)
        short_batch = _ForwardBatch([8])
        query = torch.zeros((1, 1, 1))
        metadata = object()

        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ), patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph",
            return_value=False,
        ), patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
            return_value=None,
        ), patch.object(
            coordinator,
            "_handle_sparse_retrieve",
            side_effect=AssertionError("all-short dense fallback ran Quest"),
        ):
            for layer_id in range(4):
                layer = SimpleNamespace(layer_id=layer_id)
                self.assertIs(
                    coordinator.attention_begin(
                        query, query, query, layer, short_batch, metadata
                    ),
                    metadata,
                )
                coordinator.attention_end(query, layer, short_batch)
            coordinator.finalize_forward(short_batch)

        for layer_id, expected_page in ((0, 2), (1, 0), (2, 2), (3, 0)):
            constructed, last_page = algorithm.get_layer_representation_trackers(
                layer_id
            )
            self.assertEqual(constructed[0].item(), expected_page > 0)
            self.assertEqual(last_page[0].item(), expected_page)
        self.assertTrue(states.repr_constructed[0].item())
        self.assertEqual(states.last_constructed_page[0].item(), 2)

        long_batch = _ForwardBatch([12])
        selected = torch.tensor([[0, 2]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)
        with patch.object(
            algorithm, "_can_enable_lazy_page_update", return_value=True
        ), patch.object(
            BaseSparseAlgorithmImpl,
            "retrieve_topk",
            return_value=(selected, lengths),
        ) as underlying_retrieve:
            for layer_id in range(4):
                coordinator.attention_begin(
                    query,
                    query,
                    query,
                    SimpleNamespace(layer_id=layer_id),
                    long_batch,
                    metadata,
                )

        self.assertFalse(coordinator._forward_dense_fallback)
        self.assertTrue(algorithm._context_adaptive_layer_selection_reuse_active)
        self.assertEqual(algorithm._active_layer_selection_reuse_interval, 4)
        self.assertEqual(algorithm._actual_selection_anchors, {0})
        underlying_retrieve.assert_called_once()

    def test_asd_uses_dense_at_8k_and_runs_adaptive_superpage_at_32k(self):
        algorithm, states, req_pool, kv_pool = _make_algorithm(
            max_context_len=32768,
            page_size=16,
            end_layer=4,
            sparsity_ratio=0.01,
            extra_config={
                "layer_selection_reuse_interval": 2,
                _ADAPTIVE_INTERVAL: 4,
                _ADAPTIVE_MIN_PAGES: 512,
                "use_lazy_page_update_score_kernel": True,
                _SUPERPAGE_SIZE: 8,
                _SUPERPAGE_OVERSAMPLE: 1,
                _DENSE_THRESHOLD: 8192,
            },
        )
        states.repr_constructed.fill_(True)
        states.prompt_lens.fill_(32768)
        coordinator = _make_coordinator(algorithm, states, req_pool, kv_pool)
        query = torch.zeros((1, 1, 1))
        metadata = object()
        long_scores = torch.arange(2048, dtype=torch.float32).view(1, -1)

        with patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ), patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "tc_piecewise_cuda_graph.is_in_tc_piecewise_cuda_graph",
            return_value=False,
        ), patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "tc_piecewise_cuda_graph.get_tc_piecewise_forward_context",
            return_value=None,
        ), patch.object(
            algorithm,
            "_select_active_layer_selection_reuse_interval",
            wraps=algorithm._select_active_layer_selection_reuse_interval,
        ) as select_adaptive, patch.object(
            algorithm,
            "_try_superpage_page_scores",
            return_value=long_scores,
        ) as superpage_scores, patch.object(
            algorithm, "_can_enable_lazy_page_update", return_value=True
        ):
            short_batch = _ForwardBatch([8192])
            self.assertIs(
                coordinator.attention_begin(
                    query,
                    query,
                    query,
                    SimpleNamespace(layer_id=0),
                    short_batch,
                    metadata,
                ),
                metadata,
            )
            select_adaptive.assert_not_called()
            superpage_scores.assert_not_called()
            coordinator.finalize_forward(short_batch)

            long_batch = _ForwardBatch([32768])
            for layer_id in range(4):
                coordinator.attention_begin(
                    query,
                    query,
                    query,
                    SimpleNamespace(layer_id=layer_id),
                    long_batch,
                    metadata,
                )

        self.assertFalse(coordinator._forward_dense_fallback)
        select_adaptive.assert_called_once()
        superpage_scores.assert_called_once()
        self.assertTrue(algorithm._context_adaptive_layer_selection_reuse_active)
        self.assertEqual(algorithm._actual_selection_anchors, {0})

    def test_fixed_capacity_and_graph_contexts_keep_quest_path(self):
        config = _Config(
            {
                "layer_selection_reuse_interval": 2,
                _ADAPTIVE_INTERVAL: 4,
                _ADAPTIVE_MIN_PAGES: 512,
                _SUPERPAGE_SIZE: 8,
                _SUPERPAGE_OVERSAMPLE: 1,
                _DENSE_THRESHOLD: 8192,
            },
            page_size=16,
        )
        algorithm = SimpleNamespace(
            config=config,
            page_size=16,
            begin_dense_forward=Mock(),
            begin_forward=Mock(),
        )
        states = SimpleNamespace(
            repr_constructed=torch.tensor([True]),
            prompt_lens=torch.tensor([8192], dtype=torch.int64),
        )
        req_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).view(1, 16)
        )
        kv_pool = SimpleNamespace(get_key_buffer=Mock(return_value=torch.empty(1)))
        coordinator = _make_coordinator(
            algorithm, states, req_pool, kv_pool, end_layer=1
        )
        batch = _ForwardBatch([8192])
        query = torch.zeros((1, 1, 1))
        metadata = object()

        with patch.object(
            coordinator, "_handle_sparse_retrieve", return_value="quest"
        ) as sparse_retrieve, patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=False,
        ):
            self.assertEqual(
                coordinator.attention_begin(
                    query,
                    query,
                    query,
                    SimpleNamespace(layer_id=0),
                    batch,
                    metadata,
                    fixed_capacity=2048,
                ),
                "quest",
            )
            algorithm.begin_dense_forward.assert_not_called()
            algorithm.begin_forward.assert_called_once()
            sparse_retrieve.assert_called_once()

        coordinator.prepare_graph_forward()
        algorithm.begin_forward.reset_mock()
        with patch.object(
            coordinator, "_handle_sparse_retrieve", return_value="quest"
        ) as sparse_retrieve, patch(
            "sglang.srt.model_executor.runner_backend_utils."
            "breakable_cuda_graph.is_in_breakable_cuda_graph",
            return_value=True,
        ):
            self.assertEqual(
                coordinator.attention_begin(
                    query,
                    query,
                    query,
                    SimpleNamespace(layer_id=0),
                    batch,
                    metadata,
                    fixed_capacity=False,
                ),
                "quest",
            )
            algorithm.begin_dense_forward.assert_not_called()
            algorithm.begin_forward.assert_called_once()
            sparse_retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
