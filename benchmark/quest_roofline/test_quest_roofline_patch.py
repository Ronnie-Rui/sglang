from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import launch
from quest_roofline_patch import (
    FIXED_SPARSE,
    RETRIEVAL_DENSE,
    RUNTIME_DENSE,
    _patch_adaptor_module,
    _patch_coordinator_module,
    _patch_quest_module,
    build_fixed_page_plan,
)


class _Algorithm:
    page_size = 4
    num_recent_pages = 2
    sparsity_ratio = 0.5

    @staticmethod
    def _get_seq_lens_cpu(forward_batch, batch_size):
        assert batch_size == len(forward_batch.seq_lens_cpu)
        return list(forward_batch.seq_lens_cpu)


class TestFixedPagePlan(unittest.TestCase):
    def test_matches_quest_page_counts_and_keeps_recent_suffix(self):
        selected, lengths = build_fixed_page_plan(
            _Algorithm(),
            queries=torch.empty((3, 2, 4)),
            sparse_mask=torch.tensor([True, True, False]),
            forward_batch=SimpleNamespace(seq_lens_cpu=[40, 17, 40]),
        )

        self.assertEqual(lengths.tolist(), [6, 3, 0])
        self.assertEqual(selected[0].tolist(), [0, 1, 2, 3, 8, 9])
        self.assertEqual(selected[1].tolist(), [0, 3, 4, -1, -1, -1])
        self.assertEqual(selected[2].tolist(), [-1, -1, -1, -1, -1, -1])

    def test_short_context_stays_dense(self):
        selected, lengths = build_fixed_page_plan(
            _Algorithm(),
            queries=torch.empty((1, 2, 4)),
            sparse_mask=torch.tensor([True]),
            forward_batch=SimpleNamespace(seq_lens_cpu=[8]),
        )

        self.assertEqual(selected.tolist(), [[-1]])
        self.assertEqual(lengths.tolist(), [0])

    def test_ragged_counts_match_batched_float32_selection(self):
        algorithm = _Algorithm()
        algorithm.sparsity_ratio = 0.244094488
        selected, lengths = build_fixed_page_plan(
            algorithm,
            queries=torch.empty((2, 2, 4)),
            sparse_mask=torch.tensor([True, True]),
            forward_batch=SimpleNamespace(seq_lens_cpu=[2040, 4008]),
        )

        self.assertEqual(lengths.tolist(), [126, 246])
        self.assertEqual(selected[0, :124].tolist(), list(range(124)))
        self.assertEqual(selected[0, 124:126].tolist(), [508, 509])

    def test_single_request_uses_production_float32_page_count(self):
        algorithm = _Algorithm()
        algorithm.page_size = 1
        algorithm.num_recent_pages = 0
        algorithm.sparsity_ratio = 0.244094488
        selected, lengths = build_fixed_page_plan(
            algorithm,
            queries=torch.empty((1, 2, 4)),
            sparse_mask=torch.tensor([True]),
            forward_batch=SimpleNamespace(seq_lens_cpu=[1143]),
        )

        self.assertEqual(lengths.tolist(), [279])
        self.assertEqual(selected[0, :279].tolist(), list(range(279)))


class TestPatches(unittest.TestCase):
    def test_fixed_quest_disables_forward_plan_and_representation_pool(self):
        class Quest:
            def __init__(self):
                self.page_k_min = {0: object()}
                self.page_k_max = {0: object()}
                self.page_valid = {0: object()}
                self._retrieval_plan = object()

        module = SimpleNamespace(QuestAlgorithm=Quest)
        _patch_quest_module(module)
        quest = Quest()

        quest._initialize_representation_pools(0, 2, 10)
        # SparseCoordinator forwards the selected CUDA graph page bucket as a
        # keyword argument in production. The roofline replacement must accept
        # that call while continuing to bypass the real retrieval plan.
        quest.begin_forward(None, None, None, None, fixed_capacity=640)

        self.assertEqual(quest.page_k_min, {})
        self.assertEqual(quest.page_k_max, {})
        self.assertEqual(quest.page_valid, {})
        self.assertIsNone(quest._retrieval_plan)
        self.assertIsNone(quest.get_selected_physical_pages(torch.tensor([[0]])))

    def test_retrieval_dense_adaptor_preserves_metadata(self):
        class Adaptor:
            def __init__(self):
                self._original_metadata = object()
                self.reset = False

            def _reset_forward_state(self):
                self.reset = True

        module = SimpleNamespace(FlashAttentionAdaptor=Adaptor)
        _patch_adaptor_module(module, RETRIEVAL_DENSE)
        adaptor = Adaptor()
        metadata = SimpleNamespace(scheduler_metadata=object())
        scheduler_metadata = metadata.scheduler_metadata

        adaptor.save_original_metadata(metadata)
        result = adaptor.adapt_for_attn_metadata(
            None, None, None, metadata, None, None, 4, 0
        )

        self.assertIs(result, metadata)
        self.assertIs(metadata.scheduler_metadata, scheduler_metadata)
        self.assertIsNone(adaptor._original_metadata)
        self.assertTrue(adaptor.reset)

    def test_fixed_coordinator_only_retrieves_at_start_layer(self):
        class Coordinator:
            def __init__(self):
                self.algorithm = SimpleNamespace()

            def _handle_sparse_retrieve(self, *args, **kwargs):
                return "adapted"

            def _compute_sparse_mask(self, req_pool_indices):
                return req_pool_indices

            def attention_end(self, *args, **kwargs):
                return "attention-end"

            def forward_end(self, *args, **kwargs):
                return "forward-end"

        # Use a dynamically named class so _is_quest sees the exact production
        # identity without importing SGLang in this lightweight test.
        quest_type = type("QuestAlgorithm", (), {})
        quest_type.__module__ = (
            "sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm"
        )
        calls = []

        def original_handle(self, *args, **kwargs):
            calls.append(args[1].layer_id)
            return "adapted"

        Coordinator._handle_sparse_retrieve = original_handle
        module = SimpleNamespace(SparseCoordinator=Coordinator)
        _patch_coordinator_module(module, FIXED_SPARSE)
        coordinator = Coordinator()
        coordinator.algorithm = quest_type()
        coordinator.start_layer = 2
        coordinator.config = SimpleNamespace(min_sparse_prompt_len=8)
        coordinator.states = SimpleNamespace(
            prompt_lens=torch.tensor([7, 8, 10], dtype=torch.int64)
        )

        start = coordinator._handle_sparse_retrieve(
            None, SimpleNamespace(layer_id=2), None, "dense"
        )
        later = coordinator._handle_sparse_retrieve(
            None, SimpleNamespace(layer_id=3), None, "fixed"
        )

        self.assertEqual(start, "adapted")
        self.assertEqual(later, "fixed")
        self.assertEqual(calls, [2])
        self.assertEqual(
            coordinator._compute_sparse_mask(torch.tensor([0, 1, 2])).tolist(),
            [False, True, True],
        )
        self.assertIsNone(coordinator.attention_end(None, None, None))
        self.assertIsNone(coordinator.forward_end(None))

    def test_runtime_dense_bypasses_quest_retrieval(self):
        class Coordinator:
            def _handle_sparse_retrieve(self, *args, **kwargs):
                raise AssertionError("retrieval should be bypassed")

            def _compute_sparse_mask(self, req_pool_indices):
                return req_pool_indices

            def attention_end(self, *args, **kwargs):
                raise AssertionError("representation update should be bypassed")

            def forward_end(self, *args, **kwargs):
                raise AssertionError("representation update should be bypassed")

        quest_type = type("QuestAlgorithm", (), {})
        quest_type.__module__ = (
            "sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm"
        )
        module = SimpleNamespace(SparseCoordinator=Coordinator)
        _patch_coordinator_module(module, RUNTIME_DENSE)
        coordinator = Coordinator()
        coordinator.algorithm = quest_type()

        metadata = object()
        result = coordinator._handle_sparse_retrieve(
            None, SimpleNamespace(layer_id=0), None, metadata
        )

        self.assertIs(result, metadata)
        self.assertIsNone(coordinator.attention_end(None, None, None))
        self.assertIsNone(coordinator.forward_end(None))


class TestLauncher(unittest.TestCase):
    def test_environment_prepends_patch_and_repo_python(self):
        fake_launch = Path("repo/benchmark/quest_roofline/launch.py").resolve()
        with patch.object(launch, "__file__", str(fake_launch)):
            env = launch.build_environment("fixed-sparse", {"PYTHONPATH": "existing"})
        entries = env["PYTHONPATH"].split(__import__("os").pathsep)

        self.assertEqual(Path(entries[0]).name, "quest_roofline")
        self.assertEqual(Path(entries[1]).name, "python")
        self.assertEqual(Path(entries[1]).parent.name, "repo")
        self.assertEqual(entries[2], "existing")
        self.assertEqual(env["SGLANG_QUEST_ROOFLINE_MODE"], "fixed-sparse")

    def test_validates_quest_fa3_server(self):
        launch.validate_server_args(
            [
                "--enable-hisparse",
                "--attention-backend",
                "fa3",
                "--hisparse-config",
                '{"algorithm":"quest","backend":"fa3"}',
            ]
        )

    def test_rejects_non_quest_server(self):
        with self.assertRaisesRegex(ValueError, "algorithm=quest"):
            launch.validate_server_args(
                [
                    "--enable-hisparse",
                    "--attention-backend",
                    "fa3",
                    "--hisparse-config",
                    '{"algorithm":"deepseek_dsa","backend":"fa3"}',
                ]
            )


if __name__ == "__main__":
    unittest.main()
