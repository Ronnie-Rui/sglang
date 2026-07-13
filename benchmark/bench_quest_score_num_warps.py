#!/usr/bin/env python3
"""Sweep ``num_warps`` for Quest's current serial page-score kernel.

This benchmark imports the production Triton kernel and changes only its launch
configuration. The output buffer is preallocated so the timing isolates kernel
execution from ``torch.empty`` and Python-side validation in the public wrapper.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import triton

from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
    _quest_page_score_kernel,
    quest_page_scores,
)

DEFAULT_CASES = (
    (1, 640),
    (4, 640),
    (8, 640),
    (1, 2112),
    (4, 2112),
    (8, 2112),
)
WARP_COUNTS = (1, 2, 4, 8)
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


@dataclass(frozen=True)
class Inputs:
    queries: torch.Tensor
    page_k_min: torch.Tensor
    page_k_max: torch.Tensor
    page_valid: torch.Tensor
    physical_pages: torch.Tensor
    active_mask: torch.Tensor
    history_page_counts: torch.Tensor


@dataclass(frozen=True)
class Result:
    batch_size: int
    num_pages: int
    num_warps: int
    mean_ms: float
    std_ms: float
    speedup_vs_warp4: float
    exact_match: bool
    max_abs_error: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def _make_inputs(
    batch_size: int,
    num_pages: int,
    dtype: torch.dtype,
    seed: int,
) -> Inputs:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    num_pool_pages = batch_size * num_pages

    queries = torch.randn((batch_size, Q_HEADS, HEAD_DIM), dtype=dtype, device=device)
    page_k_min = torch.randn(
        (num_pool_pages, KV_HEADS, HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    page_k_max = page_k_min + torch.rand_like(page_k_min)
    page_valid = torch.ones(num_pool_pages, dtype=torch.bool, device=device)
    physical_pages = torch.arange(
        num_pool_pages, dtype=torch.int64, device=device
    ).reshape(batch_size, num_pages)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    history_page_counts = torch.full(
        (batch_size,), num_pages, dtype=torch.int32, device=device
    )
    return Inputs(
        queries=queries,
        page_k_min=page_k_min,
        page_k_max=page_k_max,
        page_valid=page_valid,
        physical_pages=physical_pages,
        active_mask=active_mask,
        history_page_counts=history_page_counts,
    )


def _launch_serial_score(
    inputs: Inputs,
    output: torch.Tensor,
    num_warps: int,
) -> None:
    batch_size, query_heads, head_dim = inputs.queries.shape
    num_pages = inputs.physical_pages.shape[1]
    num_pool_pages, kv_heads, _ = inputs.page_k_min.shape

    _quest_page_score_kernel[(num_pages, batch_size)](
        inputs.queries,
        inputs.page_k_min,
        inputs.page_k_max,
        inputs.page_valid,
        inputs.physical_pages,
        inputs.active_mask,
        inputs.history_page_counts,
        output,
        num_pages,
        num_pool_pages,
        APPLY_RETRIEVAL_MASK=True,
        Q_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
    )


def _measure_ms(
    fn: Callable[[], None], warmup: int, iters: int, repeats: int
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / iters)
    return statistics.mean(samples), statistics.pstdev(samples)


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    finite = torch.isfinite(expected)
    if not finite.any():
        return 0.0
    return float((actual[finite] - expected[finite]).abs().max().item())


def _run_case(
    batch_size: int,
    num_pages: int,
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> list[Result]:
    inputs = _make_inputs(batch_size, num_pages, dtype, seed)
    production = quest_page_scores(
        inputs.queries,
        inputs.page_k_min,
        inputs.page_k_max,
        inputs.page_valid,
        inputs.physical_pages,
        active_mask=inputs.active_mask,
        history_page_counts=inputs.history_page_counts,
    )

    timings: dict[int, tuple[float, float]] = {}
    correctness: dict[int, tuple[bool, float]] = {}
    outputs = {num_warps: torch.empty_like(production) for num_warps in WARP_COUNTS}
    for num_warps in WARP_COUNTS:
        output = outputs[num_warps]

        def launch() -> None:
            _launch_serial_score(inputs, output, num_warps)

        launch()
        torch.cuda.synchronize()
        exact_match = torch.equal(output, production)
        max_abs_error = _max_abs_error(output, production)
        torch.testing.assert_close(output, production, rtol=2e-4, atol=2e-3)
        correctness[num_warps] = (exact_match, max_abs_error)
        timings[num_warps] = _measure_ms(launch, warmup, iters, repeats)

    warp4_ms = timings[4][0]
    return [
        Result(
            batch_size=batch_size,
            num_pages=num_pages,
            num_warps=num_warps,
            mean_ms=timings[num_warps][0],
            std_ms=timings[num_warps][1],
            speedup_vs_warp4=warp4_ms / timings[num_warps][0],
            exact_match=correctness[num_warps][0],
            max_abs_error=correctness[num_warps][1],
        )
        for num_warps in WARP_COUNTS
    ]


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This benchmark requires NVIDIA CUDA")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError(
            "warmup must be non-negative; iters and repeats must be positive"
        )

    dtype = getattr(torch, args.dtype)
    results = []
    for case_idx, (batch_size, num_pages) in enumerate(DEFAULT_CASES):
        case_results = _run_case(
            batch_size,
            num_pages,
            dtype,
            args.seed + case_idx,
            args.warmup,
            args.iters,
            args.repeats,
        )
        results.extend(case_results)
        best = min(case_results, key=lambda result: result.mean_ms)
        for result in case_results:
            print(
                f"B={result.batch_size} N={result.num_pages} "
                f"warps={result.num_warps} {result.mean_ms:.5f} "
                f"+/- {result.std_ms:.5f} ms "
                f"vs_warp4={result.speedup_vs_warp4:.3f}x "
                f"exact={result.exact_match} "
                f"max_abs={result.max_abs_error:.6g}"
            )
        print(
            f"B={batch_size} N={num_pages} best_num_warps={best.num_warps} "
            f"({best.mean_ms:.5f} ms)"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
