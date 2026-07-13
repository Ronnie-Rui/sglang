"""Compare Quest page scoring with a KV-head-parallel atomic-max prototype.

This is intentionally a benchmark-only experiment. The production implementation
continues to come from ``quest_page_scores``; the Triton kernel below exists only to
measure whether exposing KV-head parallelism is worth its output-initialization and
atomic-reduction costs.
"""

import argparse
import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.sparsity.kernels.quest_score import quest_page_scores

Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
GROUP_SIZE = Q_HEADS // KV_HEADS


@triton.jit
def _quest_page_score_kv_head_atomic_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    kv_head = tl.program_id(2)

    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
    history_page_count = tl.load(history_page_counts_ptr + batch_idx)
    page_is_selected = (
        page_in_bounds & request_is_active & (page_idx < history_page_count)
    )
    physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    page_is_valid = tl.load(
        page_valid_ptr + physical_page, mask=page_is_selected, other=0
    )

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = (dim_offsets < HEAD_DIM) & page_is_selected & page_is_valid
    key_offsets = physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dim_offsets
    key_min = tl.load(page_k_min_ptr + key_offsets, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    key_max = tl.load(page_k_max_ptr + key_offsets, mask=dim_mask, other=0.0).to(
        tl.float32
    )

    best_bound = -float("inf")
    for group_idx in range(GROUP_SIZE):
        query_head = kv_head * GROUP_SIZE + group_idx
        query_offsets = (
            batch_idx * Q_HEADS * HEAD_DIM + query_head * HEAD_DIM + dim_offsets
        )
        query = tl.load(queries_ptr + query_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        bound_keys = tl.where(query >= 0, key_max, key_min)
        head_bound = tl.sum(query * bound_keys, axis=0)
        best_bound = tl.maximum(best_bound, head_bound)

    tl.atomic_max(
        output_ptr + batch_idx * num_pages + page_idx,
        best_bound,
        mask=page_is_selected & page_is_valid,
    )


def quest_page_scores_kv_head_atomic(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    active_mask: torch.Tensor,
    history_page_counts: torch.Tensor,
) -> torch.Tensor:
    """Launch the benchmark-only KV-head-parallel implementation."""
    batch_size, num_pages = physical_pages.shape
    num_pool_pages = page_k_min.shape[0]
    output = torch.full(
        (batch_size, num_pages),
        -float("inf"),
        dtype=torch.float32,
        device=queries.device,
    )
    if batch_size == 0 or num_pages == 0:
        return output

    _quest_page_score_kv_head_atomic_kernel[(num_pages, batch_size, KV_HEADS)](
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
        output,
        num_pages,
        num_pool_pages,
        Q_HEADS=Q_HEADS,
        KV_HEADS=KV_HEADS,
        GROUP_SIZE=GROUP_SIZE,
        HEAD_DIM=HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        num_warps=4,
    )
    return output


@dataclass
class Result:
    batch_size: int
    num_pages: int
    baseline_median_ms: float
    atomic_median_ms: float
    speedup: float
    baseline_min_ms: float
    baseline_max_ms: float
    atomic_min_ms: float
    atomic_max_ms: float
    baseline_round_ms: str
    atomic_round_ms: str
    max_abs_error: float
    finite_scores: int
    masked_scores: int


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _measure_rounds(
    fn: Callable[[], torch.Tensor],
    *,
    rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> list[float]:
    return [
        float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms))
        for _ in range(rounds)
    ]


def _make_inputs(
    batch_size: int,
    num_pages: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + batch_size * 10_000 + num_pages)

    # Leave spare pool entries so logical pages can be shuffled independently.
    num_pool_pages = batch_size * num_pages + 37
    queries = torch.randn(
        (batch_size, Q_HEADS, HEAD_DIM),
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    page_k_min = torch.randn(
        (num_pool_pages, KV_HEADS, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    page_k_max = page_k_min + torch.rand(
        page_k_min.shape,
        dtype=page_k_min.dtype,
        device=page_k_min.device,
        generator=generator,
    )
    page_valid = torch.ones(num_pool_pages, dtype=torch.bool, device="cuda")
    page_valid[5::97] = False

    physical_pages = torch.arange(
        batch_size * num_pages, dtype=torch.int64, device="cuda"
    ).reshape(batch_size, num_pages)
    shifts = torch.arange(batch_size, device="cuda").unsqueeze(1) * 13
    physical_pages = (physical_pages + shifts) % num_pool_pages

    # Exercise both lower/upper out-of-range inputs before the history cutoff.
    physical_pages[:, 7] = -1
    physical_pages[:, 13] = num_pool_pages
    active_mask = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    if batch_size > 1:
        active_mask[1::3] = False
    history_page_counts = (
        num_pages - 11 - torch.arange(batch_size, dtype=torch.int64, device="cuda")
    )
    return (
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask,
        history_page_counts,
    )


def _format_rounds(values: list[float]) -> str:
    return ";".join(f"{value:.6f}" for value in values)


def run_case(
    batch_size: int,
    num_pages: int,
    *,
    dtype: torch.dtype,
    seed: int,
    rounds: int,
    warmup_ms: int,
    rep_ms: int,
) -> Result:
    inputs = _make_inputs(batch_size, num_pages, dtype, seed)

    def baseline() -> torch.Tensor:
        return quest_page_scores(
            *inputs[:5],
            active_mask=inputs[5],
            history_page_counts=inputs[6],
        )

    def kv_head_atomic() -> torch.Tensor:
        return quest_page_scores_kv_head_atomic(*inputs)

    expected = baseline()
    actual = kv_head_atomic()
    torch.cuda.synchronize()
    expected_finite = torch.isfinite(expected)
    actual_finite = torch.isfinite(actual)
    if not torch.equal(actual_finite, expected_finite):
        mismatch_count = int((actual_finite != expected_finite).sum().item())
        raise AssertionError(f"finite/masked score mismatch: {mismatch_count}")
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)
    finite_error = (actual[expected_finite] - expected[expected_finite]).abs()
    max_abs_error = float(finite_error.max().item())

    baseline_round_ms = _measure_rounds(
        baseline,
        rounds=rounds,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )
    atomic_round_ms = _measure_rounds(
        kv_head_atomic,
        rounds=rounds,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )
    baseline_median_ms = statistics.median(baseline_round_ms)
    atomic_median_ms = statistics.median(atomic_round_ms)
    finite_scores = int(expected_finite.sum().item())
    return Result(
        batch_size=batch_size,
        num_pages=num_pages,
        baseline_median_ms=baseline_median_ms,
        atomic_median_ms=atomic_median_ms,
        speedup=baseline_median_ms / atomic_median_ms,
        baseline_min_ms=min(baseline_round_ms),
        baseline_max_ms=max(baseline_round_ms),
        atomic_min_ms=min(atomic_round_ms),
        atomic_max_ms=max(atomic_round_ms),
        baseline_round_ms=_format_rounds(baseline_round_ms),
        atomic_round_ms=_format_rounds(atomic_round_ms),
        max_abs_error=max_abs_error,
        finite_scores=finite_scores,
        masked_scores=batch_size * num_pages - finite_scores,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 8])
    parser.add_argument("--num-pages", type=_parse_int_list, default=[640, 2112])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=61718073)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU")
    if args.rounds <= 0 or args.warmup_ms <= 0 or args.rep_ms <= 0:
        raise ValueError("rounds, warmup-ms, and rep-ms must be positive")

    dtype = getattr(torch, args.dtype)
    results = []
    for num_pages in args.num_pages:
        for batch_size in args.batch_sizes:
            result = run_case(
                batch_size,
                num_pages,
                dtype=dtype,
                seed=args.seed,
                rounds=args.rounds,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            results.append(result)
            print(
                f"B={batch_size} N={num_pages} "
                f"baseline={result.baseline_median_ms:.6f} ms "
                f"kv_atomic={result.atomic_median_ms:.6f} ms "
                f"speedup={result.speedup:.3f}x "
                f"range_baseline=[{result.baseline_min_ms:.6f}, "
                f"{result.baseline_max_ms:.6f}] ms "
                f"range_atomic=[{result.atomic_min_ms:.6f}, "
                f"{result.atomic_max_ms:.6f}] ms "
                f"max_abs={result.max_abs_error:.6g} "
                f"finite/masked={result.finite_scores}/{result.masked_scores}"
            )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
        print(f"saved={args.output_csv}")


if __name__ == "__main__":
    main()
