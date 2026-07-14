import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.kernels.quest_page_update import (
    quest_update_page_representations_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class _DecodeMode:
    @staticmethod
    def is_decode():
        return True


class TestQuestPageUpdateDispatch(unittest.TestCase):
    def test_small_batches_stay_on_the_portable_update_path(self):
        config = SimpleNamespace(page_size=4, sparse_extra_config={})
        algorithm = QuestAlgorithm(config, torch.device("cpu"))
        algorithm.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((4, 4), dtype=torch.int32)
        )
        algorithm.states = SimpleNamespace(
            repr_constructed=torch.ones(4, dtype=torch.bool),
            last_constructed_page=torch.zeros(4, dtype=torch.int64),
        )
        k_buffer = SimpleNamespace(
            is_cuda=True,
            ndim=3,
            dtype=torch.float16,
            shape=(16, 1, 64),
            device=torch.device("cpu"),
        )
        seq_lens = torch.full((4,), 4, dtype=torch.int64)

        self.assertFalse(
            algorithm._can_use_triton_page_update(
                torch.arange(3), seq_lens[:3], k_buffer
            )
        )
        self.assertTrue(
            algorithm._can_use_triton_page_update(torch.arange(4), seq_lens, k_buffer)
        )

    def test_algorithm_advances_trackers_only_on_last_layer(self):
        config = SimpleNamespace(page_size=4, sparse_extra_config={})
        algorithm = QuestAlgorithm(config, torch.device("cpu"))
        algorithm.end_layer = 2
        algorithm.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 4), dtype=torch.int32)
        )
        algorithm.states = SimpleNamespace(
            repr_constructed=torch.ones(1, dtype=torch.bool),
            last_constructed_page=torch.zeros(1, dtype=torch.int64),
        )
        algorithm.page_k_min = {
            layer_id: torch.zeros((1, 1, 1)) for layer_id in range(2)
        }
        algorithm.page_k_max = {
            layer_id: torch.zeros((1, 1, 1)) for layer_id in range(2)
        }
        algorithm.page_valid = {
            layer_id: torch.zeros(1, dtype=torch.bool) for layer_id in range(2)
        }
        forward_batch = SimpleNamespace(forward_mode=_DecodeMode())
        req_pool_indices = torch.zeros(1, dtype=torch.int64)
        seq_lens = torch.full((1,), 4, dtype=torch.int64)
        k_buffer = torch.zeros((4, 1, 1))

        with (
            patch.object(algorithm, "_can_use_triton_page_update", return_value=True),
            patch(
                "sglang.srt.mem_cache.sparsity.kernels.quest_page_update."
                "quest_update_page_representations_"
            ) as update_kernel,
        ):
            algorithm.update_representations(
                0, req_pool_indices, seq_lens, k_buffer, forward_batch
            )
            algorithm.update_representations(
                1, req_pool_indices, seq_lens, k_buffer, forward_batch
            )

        self.assertEqual(update_kernel.call_count, 2)
        self.assertFalse(update_kernel.call_args_list[0].kwargs["advance_trackers"])
        self.assertTrue(update_kernel.call_args_list[1].kwargs["advance_trackers"])


def _reference_update(
    req_pool_indices,
    seq_lens,
    req_to_token,
    k_buffer,
    repr_constructed,
    last_constructed_page,
    page_k_min,
    page_k_max,
    page_valid,
    page_size,
    *,
    advance_trackers,
):
    for batch_idx, req_idx_tensor in enumerate(req_pool_indices):
        req_idx = int(req_idx_tensor.item())
        end_page = int(seq_lens[batch_idx].item()) // page_size
        start_page = (
            int(last_constructed_page[req_idx].item())
            if bool(repr_constructed[req_idx].item())
            else 0
        )
        for logical_page in range(start_page, end_page):
            logical_start = logical_page * page_size
            physical_tokens = req_to_token[
                req_idx, logical_start : logical_start + page_size
            ].clamp(0, k_buffer.shape[0] - 1)
            keys = k_buffer[physical_tokens.to(torch.long)].to(torch.float32)
            target_page = int(
                (req_to_token[req_idx, logical_start] // page_size)
                .clamp(0, page_k_min.shape[0] - 1)
                .item()
            )
            page_k_min[target_page] = keys.amin(dim=0)
            page_k_max[target_page] = keys.amax(dim=0)
            page_valid[target_page] = True

        if advance_trackers and start_page < end_page:
            repr_constructed[req_idx] = True
            last_constructed_page[req_idx] = end_page


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required",
)
class TestQuestPageUpdateKernel(unittest.TestCase):
    page_size = 4

    @staticmethod
    def _make_inputs(
        *,
        k_dtype: torch.dtype = torch.float16,
    ):
        device = torch.device("cuda")
        req_to_token = torch.tensor(
            [
                [8, 13, 10, 15, 24, 31, 26, 29, 40, 47, 42, 45],
                [48, 55, 50, 53, 60, 67, 62, 65, 72, 79, 74, 77],
                [56, 63, 58, 61, 80, 87, 82, 85, 96, 103, 98, 101],
                [88, 95, 90, 93, 104, 111, 106, 109, 112, 127, 114, 125],
            ],
            dtype=torch.int64,
            device=device,
        )
        torch.manual_seed(17)
        k_buffer = torch.randn(128, 3, 33, dtype=k_dtype, device=device)
        return {
            "req_pool_indices": torch.tensor(
                [3, 0, 2], dtype=torch.int64, device=device
            ),
            "seq_lens": torch.tensor([12, 8, 7], dtype=torch.int64, device=device),
            "req_to_token": req_to_token,
            "k_buffer": k_buffer,
            "repr_constructed": torch.tensor(
                [False, False, True, True], dtype=torch.bool, device=device
            ),
            # Slot 0's stale value must be ignored because it is unconstructed.
            "last_constructed_page": torch.tensor(
                [99, 0, 1, 2], dtype=torch.int64, device=device
            ),
            "page_k_min": torch.zeros(32, 3, 33, dtype=torch.float32, device=device),
            "page_k_max": torch.zeros(32, 3, 33, dtype=torch.float32, device=device),
            "page_valid": torch.zeros(32, dtype=torch.bool, device=device),
        }

    def _assert_matches_reference(self, *, advance_trackers):
        actual = self._make_inputs()
        expected = {key: value.clone() for key, value in actual.items()}

        _reference_update(
            **expected,
            page_size=self.page_size,
            advance_trackers=advance_trackers,
        )
        quest_update_page_representations_(
            **actual,
            page_size=self.page_size,
            advance_trackers=advance_trackers,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(actual["page_valid"], expected["page_valid"]))
        torch.testing.assert_close(actual["page_k_min"], expected["page_k_min"])
        torch.testing.assert_close(actual["page_k_max"], expected["page_k_max"])
        self.assertTrue(
            torch.equal(actual["repr_constructed"], expected["repr_constructed"])
        )
        self.assertTrue(
            torch.equal(
                actual["last_constructed_page"],
                expected["last_constructed_page"],
            )
        )

    def test_mixed_batch_uses_each_mapped_physical_token(self):
        self._assert_matches_reference(advance_trackers=False)

    def test_trackers_advance_only_when_requested(self):
        self._assert_matches_reference(advance_trackers=True)

    def test_cuda_graph_replay_reads_new_boundaries(self):
        inputs = self._make_inputs()
        inputs["req_pool_indices"] = inputs["req_pool_indices"][:1]
        inputs["seq_lens"] = inputs["seq_lens"][:1]
        inputs["seq_lens"].fill_(8)
        inputs["last_constructed_page"][3] = 1

        # Compile both launches before capture, then restore persistent state.
        quest_update_page_representations_(
            **inputs,
            page_size=self.page_size,
            advance_trackers=True,
        )
        torch.cuda.synchronize()
        inputs["repr_constructed"][3] = True
        inputs["last_constructed_page"][3] = 1
        inputs["page_k_min"].zero_()
        inputs["page_k_max"].zero_()
        inputs["page_valid"].zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quest_update_page_representations_(
                **inputs,
                page_size=self.page_size,
                advance_trackers=True,
            )

        inputs["repr_constructed"][3] = True
        inputs["last_constructed_page"][3] = 1
        inputs["page_k_min"].zero_()
        inputs["page_k_max"].zero_()
        inputs["page_valid"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(inputs["last_constructed_page"][3].item(), 2)
        self.assertTrue(inputs["page_valid"][26].item())

        inputs["seq_lens"].fill_(12)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(inputs["last_constructed_page"][3].item(), 3)
        self.assertTrue(inputs["page_valid"][28].item())

        inputs["seq_lens"].fill_(8)
        graph.replay()
        torch.cuda.synchronize()
        # A shorter/stale replay must not move the tracker backwards.
        self.assertEqual(inputs["last_constructed_page"][3].item(), 3)


if __name__ == "__main__":
    unittest.main()
