#!/usr/bin/env python3
"""Benchmark Quest's row-wise selection against torch.topk."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

DEFAULT_CASES = (
    (1, 640, 39),
    (4, 640, 39),
    (8, 640, 39),
    (1, 2112, 131),
    (4, 2112, 131),
    (8, 2112, 131),
    (1, 2560, 155),
    (4, 2560, 155),
    (8, 2560, 155),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def _measure(fn, warmup: int, iters: int, repeats: int) -> tuple[float, float]:
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
        samples.append(start.elapsed_time(end) / iters)
    return statistics.mean(samples), statistics.pstdev(samples)


def _assert_same_set(
    scores: torch.Tensor,
    k_per_req: torch.Tensor,
    actual_scores: torch.Tensor,
    actual_indices: torch.Tensor,
) -> None:
    for row, row_k in enumerate(k_per_req.cpu().tolist()):
        expected_indices = torch.topk(scores[row], k=row_k, dim=0, sorted=False).indices
        actual = actual_indices[row, :row_k].sort().values
        expected = expected_indices.to(actual.dtype).sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_scores[row, :row_k],
            scores[row, actual_indices[row, :row_k]],
            rtol=0,
            atol=0,
        )
        if row_k < actual_indices.shape[1]:
            assert torch.equal(
                actual_indices[row, row_k:],
                torch.full_like(actual_indices[row, row_k:], -1),
            )
            assert torch.isneginf(actual_scores[row, row_k:]).all()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This benchmark requires NVIDIA CUDA")

    from sglang.jit_kernel.quest.topk import quest_topk

    torch.manual_seed(args.seed)
    rows = []
    for batch_size, num_pages, max_k in DEFAULT_CASES:
        scores = torch.randn(batch_size, num_pages, dtype=torch.float32, device="cuda")
        k_per_req = torch.full((batch_size,), max_k, dtype=torch.int32, device="cuda")

        actual_scores, actual_indices = quest_topk(scores, k_per_req, max_k)
        _assert_same_set(scores, k_per_req, actual_scores, actual_indices)

        torch_mean, torch_std = _measure(
            lambda: torch.topk(scores, k=max_k, dim=1, sorted=True),
            args.warmup,
            args.iters,
            args.repeats,
        )
        quest_mean, quest_std = _measure(
            lambda: quest_topk(scores, k_per_req, max_k),
            args.warmup,
            args.iters,
            args.repeats,
        )
        row = {
            "batch_size": batch_size,
            "num_pages": num_pages,
            "max_k": max_k,
            "torch_ms": torch_mean,
            "torch_std_ms": torch_std,
            "quest_ms": quest_mean,
            "quest_std_ms": quest_std,
            "speedup": torch_mean / quest_mean,
        }
        rows.append(row)
        print(
            f"B={batch_size:2d} N={num_pages:4d} K={max_k:3d} "
            f"torch={torch_mean:.4f} ms quest={quest_mean:.4f} ms "
            f"speedup={row['speedup']:.2f}x"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
