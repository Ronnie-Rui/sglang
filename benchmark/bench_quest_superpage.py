"""Microbenchmark exact Quest superpage pruning against full page scoring."""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_score import (
    quest_exact_superpage_page_scores,
    quest_page_scores,
)


@dataclass
class Result:
    batch_size: int
    num_pages: int
    topk: int
    superpage_size: int
    oversample: int
    candidate_page_pct: float
    certified_pct: float
    baseline_ms: float
    superpage_ms: float
    speedup: float
    exact: bool


def measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iters


def make_bounds(args, batch_size: int):
    dtype = getattr(torch, args.dtype)
    total_pages = batch_size * args.num_pages
    keys = torch.randn(
        total_pages,
        args.page_size,
        args.kv_heads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    page_k_min = keys.amin(dim=1)
    page_k_max = keys.amax(dim=1)
    del keys
    return page_k_min, page_k_max


def run_case(args, batch_size: int, superpage_size: int, oversample: int) -> Result:
    torch.manual_seed(args.seed + batch_size)
    torch.cuda.manual_seed_all(args.seed + batch_size)
    dtype = getattr(torch, args.dtype)
    total_pages = batch_size * args.num_pages
    page_k_min, page_k_max = make_bounds(args, batch_size)
    page_valid = torch.ones(total_pages, dtype=torch.bool, device="cuda")
    physical_pages = torch.arange(
        total_pages, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, args.num_pages)
    queries = torch.randn(
        batch_size,
        args.q_heads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    active_mask = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    history_counts = torch.full(
        (batch_size,), args.num_pages, dtype=torch.int32, device="cuda"
    )
    k_per_req = torch.full((batch_size,), args.topk, dtype=torch.int32, device="cuda")

    def baseline():
        return quest_page_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask=active_mask,
            history_page_counts=history_counts,
        )

    def hierarchical():
        return quest_exact_superpage_page_scores(
            queries,
            page_k_min,
            page_k_max,
            page_valid,
            physical_pages,
            active_mask,
            history_counts,
            k_per_req,
            max_k=args.topk,
            superpage_size=superpage_size,
            oversample=oversample,
        )

    full_scores = baseline()
    superpage_scores, certified = hierarchical()
    full_topk = torch.topk(full_scores, k=args.topk, dim=1).indices.sort(dim=1).values
    candidate_topk = (
        torch.topk(superpage_scores, k=args.topk, dim=1).indices.sort(dim=1).values
    )
    exact = bool(torch.equal(full_topk, candidate_topk))
    if not exact:
        raise AssertionError("exact superpage pruning changed the selected page set")

    baseline_ms = measure_ms(baseline, args.warmup, args.iters)
    superpage_ms = measure_ms(hierarchical, args.warmup, args.iters)
    candidate_groups = min(
        (args.num_pages + superpage_size - 1) // superpage_size,
        max(1, ((args.topk + superpage_size - 1) // superpage_size) * oversample),
    )
    candidate_pages = min(args.num_pages, candidate_groups * superpage_size)
    return Result(
        batch_size=batch_size,
        num_pages=args.num_pages,
        topk=args.topk,
        superpage_size=superpage_size,
        oversample=oversample,
        candidate_page_pct=100.0 * candidate_pages / args.num_pages,
        certified_pct=100.0 * certified.float().mean().item(),
        baseline_ms=baseline_ms,
        superpage_ms=superpage_ms,
        speedup=baseline_ms / superpage_ms,
        exact=exact,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--num-pages", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--superpage-sizes", default="4,8,16")
    parser.add_argument("--oversamples", default="1,2,4")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA is required")
    results = []
    for batch_size in map(int, args.batch_sizes.split(",")):
        for superpage_size in map(int, args.superpage_sizes.split(",")):
            for oversample in map(int, args.oversamples.split(",")):
                result = run_case(args, batch_size, superpage_size, oversample)
                results.append(result)
                print(
                    f"bs={batch_size} S={superpage_size} over={oversample} "
                    f"candidate={result.candidate_page_pct:.1f}% "
                    f"certified={result.certified_pct:.1f}% "
                    f"ms={result.baseline_ms:.4f}->{result.superpage_ms:.4f} "
                    f"speedup={result.speedup:.3f}x exact={result.exact}"
                )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)


if __name__ == "__main__":
    main()
