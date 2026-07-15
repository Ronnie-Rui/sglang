from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ScoreAlgorithm(BaseSparseAlgorithmImpl):
    def _retrieve_page_scores(self, layer_id, phys_pages, req_pool_indices, queries):
        del layer_id, req_pool_indices, queries
        self.score_calls = getattr(self, "score_calls", 0) + 1
        return phys_pages.to(torch.float32)


def _config(**extra):
    return SimpleNamespace(
        page_size=4,
        sparse_extra_config={
            "sparsity_ratio": 0.5,
            "num_recent_pages": 2,
            **extra,
        },
    )


def _reference_selected(seq_len: int, sparse: bool) -> list[int]:
    num_pages = (seq_len + 3) // 4
    if not sparse or num_pages <= 2:
        return []
    history_pages = num_pages - 2
    k = min(max(int(history_pages * 0.5), 1), history_pages)
    history = list(range(history_pages - k, history_pages))
    return history + list(range(history_pages, num_pages))


def _build_plan(page_counts, sparse_mask, *, sparsity_ratio=0.25):
    algorithm = _ScoreAlgorithm(
        _config(sparsity_ratio=sparsity_ratio), torch.device("cpu")
    )
    max_tokens = max(max(page_counts, default=0) * algorithm.page_size, 1)
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(max_tokens, dtype=torch.int64).repeat(
            len(page_counts), 1
        )
    )
    seq_lens = torch.tensor(
        [count * algorithm.page_size for count in page_counts], dtype=torch.int64
    )
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=seq_lens.tolist())
    plan = algorithm._build_retrieval_plan(
        forward_batch,
        torch.arange(len(page_counts)),
        torch.tensor(sparse_mask, dtype=torch.bool),
        torch.device("cpu"),
    )
    return algorithm, plan


def _layer_policy_algorithm(*, interval=1, layer_page_budget=None):
    algorithm = QuestAlgorithm(
        _config(
            layer_selection_reuse_interval=interval,
            layer_page_budget=layer_page_budget or [],
        ),
        torch.device("cpu"),
    )
    algorithm.start_layer = 0
    algorithm.end_layer = 5
    return algorithm


def test_batched_retrieval_matches_per_request_reference():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    req_to_token = torch.arange(64, dtype=torch.int64).repeat(4, 1)
    algorithm.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    seq_lens = torch.tensor([40, 12, 28, 36], dtype=torch.int64)
    sparse_mask = torch.tensor([True, True, False, True])
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=seq_lens.tolist())

    selected, lengths = algorithm.retrieve_topk(
        queries=torch.zeros((4, 1)),
        layer_id=0,
        req_pool_indices=torch.arange(4),
        sparse_mask=sparse_mask,
        forward_batch=forward_batch,
    )

    for row, (seq_len, sparse) in enumerate(zip(seq_lens.tolist(), sparse_mask)):
        expected = _reference_selected(seq_len, bool(sparse))
        length = int(lengths[row])
        assert length == len(expected)
        assert selected[row, :length].tolist() == expected
        assert (selected[row, length:] == -1).all()


def test_single_request_uses_low_overhead_retrieval_path():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int64).unsqueeze(0)
    )
    seq_lens = torch.tensor([40], dtype=torch.int64)
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=[40])

    with patch.object(
        algorithm,
        "_build_retrieval_plan",
        side_effect=AssertionError("single-request path built a batched plan"),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=forward_batch,
        )

    expected = _reference_selected(40, True)
    assert lengths.tolist() == [len(expected)]
    assert selected.tolist() == [expected]


def test_single_request_requires_host_seq_lens_for_fast_path():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int64).unsqueeze(0)
    )
    seq_lens = torch.tensor([40], dtype=torch.int64)

    with patch.object(
        algorithm,
        "_retrieve_topk_single",
        side_effect=AssertionError("single-request path read device seq_lens"),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=SimpleNamespace(seq_lens=seq_lens),
        )

    expected = _reference_selected(40, True)
    assert lengths.tolist() == [len(expected)]
    assert selected.tolist() == [expected]


def test_single_request_without_host_mask_does_not_read_scalar_and_gates_output():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int64).unsqueeze(0)
    )
    seq_lens = torch.tensor([40], dtype=torch.int64)
    forward_batch = SimpleNamespace(seq_lens=seq_lens, seq_lens_cpu=[40])

    with (
        patch.object(algorithm, "_get_bool_mask_cpu", return_value=None),
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("device sparse_mask scalar was read"),
        ),
    ):
        selected, lengths = algorithm.retrieve_topk(
            queries=torch.zeros((1, 1)),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([False]),
            forward_batch=forward_batch,
        )

    assert algorithm.score_calls == 1
    assert lengths.tolist() == [0]
    assert selected.tolist() == [[-1] * selected.shape[1]]


def test_single_request_exact_host_false_skips_score_calculation():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64, dtype=torch.int64).unsqueeze(0)
    )
    seq_lens = torch.tensor([40], dtype=torch.int64)
    forward_batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=[40],
        sparse_mask_cpu=[False],
    )

    selected, lengths = algorithm.retrieve_topk(
        queries=torch.zeros((1, 1)),
        layer_id=0,
        req_pool_indices=torch.tensor([0]),
        sparse_mask=torch.tensor([False]),
        forward_batch=forward_batch,
    )

    assert getattr(algorithm, "score_calls", 0) == 0
    assert lengths.tolist() == [0]
    assert selected.tolist() == [[-1]]


def test_inactive_requests_do_not_inflate_host_topk_width():
    algorithm, all_inactive = _build_plan([2048, 1024], [False, False])
    assert all_inactive.max_k == 0
    assert all_inactive.k_per_req.tolist() == [0, 0]

    selected, lengths = algorithm.retrieve_topk(
        queries=torch.zeros((2, 1)),
        layer_id=0,
        req_pool_indices=torch.arange(2),
        sparse_mask=torch.tensor([False, False]),
        forward_batch=SimpleNamespace(
            seq_lens=all_inactive.seq_lens,
            seq_lens_cpu=all_inactive.seq_lens.tolist(),
        ),
    )
    assert getattr(algorithm, "score_calls", 0) == 0
    assert lengths.tolist() == [0, 0]
    assert selected.tolist() == [[-1], [-1]]

    _, mixed = _build_plan([2048, 1024], [False, True])
    assert mixed.max_num_pages == 2048
    assert mixed.k_per_req.tolist() == [0, 255]
    assert mixed.max_k == int(mixed.k_per_req.max()) == 255


def test_non_cpu_sparse_mask_never_forces_a_host_copy():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))
    non_cpu_mask = torch.ones(2, dtype=torch.bool, device="meta")

    assert algorithm._get_bool_mask_cpu(non_cpu_mask, 2) is None
    assert algorithm._get_bool_mask_cpu(
        non_cpu_mask,
        2,
        forward_batch=SimpleNamespace(sparse_mask_cpu=[False, True]),
    ) == [False, True]


def test_layer_policy_reuses_only_consecutive_layers_with_matching_budget():
    algorithm = _layer_policy_algorithm(
        interval=4,
        layer_page_budget=[{"start_layer": 2, "end_layer": 4, "scale": 0.5}],
    )
    selected = torch.tensor([[7]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)

    with patch.object(
        BaseSparseAlgorithmImpl,
        "retrieve_topk",
        return_value=(selected, lengths),
    ) as underlying:
        results = [
            algorithm.retrieve_topk(
                torch.zeros((1, 1, 1)),
                layer_id,
                torch.zeros(1, dtype=torch.long),
                torch.ones(1, dtype=torch.bool),
                forward_batch=object(),
            )
            for layer_id in range(5)
        ]

    assert underlying.call_count == 3
    assert [len(result) for result in results] == [2, 3, 2, 3, 2]
    assert [algorithm.should_update_metadata_lengths(i) for i in range(5)] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_layer_budget_builds_and_caches_ratio_specific_plan():
    algorithm = _layer_policy_algorithm(
        layer_page_budget=[{"start_layer": 1, "end_layer": 2, "scale": 0.5}],
    )
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(400, dtype=torch.int64).unsqueeze(0),
        max_context_len=400,
    )
    forward_batch = SimpleNamespace(seq_lens=torch.tensor([400]), seq_lens_cpu=[400])
    algorithm.begin_forward(
        forward_batch,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.device("cpu"),
    )

    base_plan = algorithm._retrieval_plan
    budget_plan = algorithm._get_retrieval_plan_for_ratio(
        base_plan, algorithm.get_layer_sparsity_ratio(1)
    )

    assert base_plan.k_per_req.tolist() == [49]
    assert budget_plan.k_per_req.tolist() == [24]
    assert budget_plan.max_k == 24
    assert budget_plan.physical_pages.data_ptr() == base_plan.physical_pages.data_ptr()
    assert (
        algorithm._get_retrieval_plan_for_ratio(
            base_plan, algorithm.get_layer_sparsity_ratio(1)
        )
        is budget_plan
    )


def test_layer_reuse_updates_only_actual_anchors_and_advances_final_group():
    algorithm = _layer_policy_algorithm(interval=2)
    algorithm.end_layer = 4
    algorithm.states = SimpleNamespace(
        repr_constructed=torch.tensor([True]),
        last_constructed_page=torch.tensor([1]),
    )
    algorithm._actual_selection_anchors.update({0, 2})
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    req_pool_indices = torch.tensor([0])
    seq_lens = torch.tensor([8])
    k_buffer = torch.zeros((8, 1, 1))

    with (
        patch.object(algorithm, "should_update_representations", return_value=True),
        patch.object(algorithm, "_compute_page_representations") as compute,
    ):
        for layer_id in range(4):
            algorithm.update_representations(
                layer_id,
                req_pool_indices,
                seq_lens,
                k_buffer,
                forward_batch,
            )

    assert [call.args[0] for call in compute.call_args_list] == [0, 2]
    assert algorithm.states.repr_constructed.tolist() == [True]
    assert algorithm.states.last_constructed_page.tolist() == [2]


def _capped_plan(*, seq_lens_cpu=True, fixed_capacity=False, layer_budget=None):
    algorithm = QuestAlgorithm(
        _config(
            quest_max_selected_tokens=48,
            layer_page_budget=layer_budget or [],
        ),
        torch.device("cpu"),
    )
    algorithm.start_layer = 0
    algorithm.end_layer = 2
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(400, dtype=torch.int64).unsqueeze(0),
        max_context_len=400,
    )
    forward_batch = SimpleNamespace(seq_lens=torch.tensor([400]))
    if seq_lens_cpu:
        forward_batch.seq_lens_cpu = [400]
    algorithm.begin_forward(
        forward_batch,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.device("cpu"),
        fixed_capacity=fixed_capacity,
    )
    return algorithm


def test_selected_token_cap_is_default_off_and_caps_host_device_and_graph_plans():
    assert (
        QuestAlgorithm(_config(), torch.device("cpu")).quest_max_selected_pages is None
    )

    host = _capped_plan()
    device_only = _capped_plan(seq_lens_cpu=False)
    fixed = _capped_plan(fixed_capacity=100)

    assert host.quest_max_selected_pages == 12
    assert host.get_history_page_selection_cap() == 10
    for algorithm in (host, device_only, fixed):
        assert algorithm._retrieval_plan.k_per_req.tolist() == [10]
        assert algorithm._retrieval_plan.max_k == 10
        assert algorithm._retrieval_plan.max_k + algorithm.num_recent_pages == 12


def test_layer_ratio_is_applied_before_global_selected_token_cap():
    algorithm = _capped_plan(
        layer_budget=[{"start_layer": 1, "end_layer": 2, "scale": 0.1}]
    )
    base_plan = algorithm._retrieval_plan
    layer_plan = algorithm._get_retrieval_plan_for_ratio(
        base_plan, algorithm.get_layer_sparsity_ratio(1)
    )

    assert base_plan.k_per_req.tolist() == [10]
    assert layer_plan.k_per_req.tolist() == [4]
    assert [base_plan.max_k, layer_plan.max_k] == [10, 4]


def test_full_page_fast_path_and_partial_fallback_match_reference():
    algorithm = QuestAlgorithm(_config(), torch.device("cpu"))
    algorithm.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(12, dtype=torch.int64).unsqueeze(0)
    )
    algorithm.page_k_min[0] = torch.zeros((3, 1, 2), dtype=torch.float32)
    algorithm.page_k_max[0] = torch.zeros((3, 1, 2), dtype=torch.float32)
    algorithm.page_valid[0] = torch.zeros(3, dtype=torch.bool)
    keys = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)

    algorithm._compute_page_representations(
        0,
        torch.tensor([0]),
        torch.tensor([12]),
        0,
        torch.tensor([3]),
        keys,
    )
    expected = keys.reshape(3, 4, 1, 2)
    torch.testing.assert_close(algorithm.page_k_min[0], expected.amin(dim=1))
    torch.testing.assert_close(algorithm.page_k_max[0], expected.amax(dim=1))

    algorithm.page_k_min[0].zero_()
    algorithm.page_k_max[0].zero_()
    algorithm.page_valid[0].zero_()
    algorithm._compute_page_representations(
        0,
        torch.tensor([0]),
        torch.tensor([7]),
        0,
        torch.tensor([2]),
        keys,
    )
    torch.testing.assert_close(algorithm.page_k_min[0][1], keys[4:7].amin(dim=0))
    torch.testing.assert_close(algorithm.page_k_max[0][1], keys[4:7].amax(dim=0))


def test_gqa_correctness_change_replaces_mean_sum_with_conservative_max():
    algorithm = QuestAlgorithm(_config(), torch.device("cpu"))
    algorithm.page_k_min[0] = torch.tensor(
        [
            [[-4.0, -2.0], [-6.0, -3.0]],
            [[-1.0, -8.0], [-2.0, -5.0]],
        ]
    )
    algorithm.page_k_max[0] = torch.tensor(
        [
            [[5.0, 3.0], [7.0, 4.0]],
            [[9.0, 2.0], [3.0, 6.0]],
        ]
    )
    algorithm.page_valid[0] = torch.ones(2, dtype=torch.bool)
    physical_pages = torch.tensor([[0, 1]])
    queries = torch.tensor([[[2.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [0.0, -2.0]]])

    actual = algorithm._retrieve_page_scores(
        0, physical_pages, torch.tensor([0]), queries
    )
    k_min = algorithm.page_k_min[0][physical_pages].unsqueeze(3)
    k_max = algorithm.page_k_max[0][physical_pages].unsqueeze(3)
    grouped_query = queries.reshape(1, 2, 2, 2).unsqueeze(1)
    per_head = torch.where(
        grouped_query >= 0,
        grouped_query * k_max,
        grouped_query * k_min,
    ).sum(dim=-1)
    expected = per_head.amax(dim=(2, 3))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([[12.0, 18.0]]))

    averaged_query = queries.reshape(1, 2, 2, 2).mean(dim=2).unsqueeze(1)
    old_approximation = torch.where(
        averaged_query >= 0,
        averaged_query * k_max.squeeze(3),
        averaged_query * k_min.squeeze(3),
    ).sum(dim=(2, 3))
    torch.testing.assert_close(old_approximation, torch.tensor([[4.5, 7.5]]))
    assert not torch.equal(actual, old_approximation)


def test_multi_token_decode_falls_back_to_device_page_check():
    algorithm = _ScoreAlgorithm(_config(), torch.device("cpu"))

    single_token = SimpleNamespace(
        seq_lens_cpu=torch.tensor([17]),
        num_token_non_padded_cpu=1,
        positions=torch.tensor([16]),
        spec_info=None,
    )
    assert not algorithm.should_update_representations(single_token)

    page_boundary = SimpleNamespace(
        seq_lens_cpu=torch.tensor([16]),
        num_token_non_padded_cpu=1,
        positions=torch.tensor([15]),
        spec_info=None,
    )
    assert algorithm.should_update_representations(page_boundary)

    # A three-token jump from length 15 to 18 crosses a page boundary even
    # though the final length is not divisible by page_size.
    multi_token = SimpleNamespace(
        seq_lens_cpu=torch.tensor([18]),
        num_token_non_padded_cpu=3,
        positions=torch.tensor([15, 16, 17]),
        spec_info=object(),
    )
    assert algorithm.should_update_representations(multi_token)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
