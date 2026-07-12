"""Benchmark and validate the fused Quest page-score kernel."""

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang.srt.mem_cache.sparsity.kernels.quest_score import quest_page_scores


@dataclass
class Result:
    batch_size: int
    num_pages: int
    q_heads: int
    kv_heads: int
    head_dim: int
    torch_ms: float
    triton_ms: float
    speedup: float
    torch_peak_mb: float
    triton_peak_mb: float
    max_abs_error: float
    topk_overlap_pct: float


def reference_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
) -> torch.Tensor:
    k_min = page_k_min[physical_pages]
    k_max = page_k_max[physical_pages]
    valid_mask = page_valid[physical_pages]
    batch_size, q_heads, head_dim = queries.shape
    kv_heads = k_min.shape[-2]
    group_size = q_heads // kv_heads
    query = queries.view(batch_size, kv_heads, group_size, head_dim)
    query = query.to(k_min.dtype).unsqueeze(1)
    per_head_bound = torch.where(
        query >= 0,
        query * k_max.unsqueeze(3),
        query * k_min.unsqueeze(3),
    ).sum(dim=-1)
    criticality = per_head_bound.amax(dim=(2, 3))
    return torch.where(
        valid_mask,
        criticality,
        torch.full_like(criticality, float("-inf")),
    )


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


def measure_peak_mb(fn) -> float:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    torch.cuda.empty_cache()
    return max(peak - base, 0) / (1024 * 1024)


def run_case(args, batch_size: int) -> Result:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed + batch_size)
    torch.cuda.manual_seed_all(args.seed + batch_size)

    total_pages = batch_size * args.num_pages
    queries = torch.randn(
        batch_size,
        args.q_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    page_k_min = torch.randn(
        total_pages,
        args.kv_heads,
        args.head_dim,
        dtype=torch.float32,
        device=device,
    )
    widths = torch.rand_like(page_k_min).abs()
    page_k_max = page_k_min + widths
    physical_pages = torch.arange(total_pages, dtype=torch.int64, device=device).view(
        batch_size, args.num_pages
    )
    page_valid = torch.rand(total_pages, device=device) > args.invalid_ratio

    def torch_fn():
        return reference_scores(
            queries, page_k_min, page_k_max, page_valid, physical_pages
        )

    def triton_fn():
        return quest_page_scores(
            queries, page_k_min, page_k_max, page_valid, physical_pages
        )

    reference = torch_fn()
    actual = triton_fn()
    torch.cuda.synchronize()
    finite = torch.isfinite(reference)
    max_abs_error = float((actual[finite] - reference[finite]).abs().max().item())
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-3)

    topk = min(args.topk, args.num_pages)
    reference_topk = torch.topk(reference, k=topk, dim=1, sorted=False).indices
    actual_topk = torch.topk(actual, k=topk, dim=1, sorted=False).indices
    overlap = (reference_topk.unsqueeze(2) == actual_topk.unsqueeze(1)).any(
        dim=2
    ).float().mean().item() * 100

    torch_peak_mb = measure_peak_mb(torch_fn)
    triton_peak_mb = measure_peak_mb(triton_fn)
    torch_ms = measure_ms(torch_fn, args.warmup, args.iters)
    triton_ms = measure_ms(triton_fn, args.warmup, args.iters)
    return Result(
        batch_size=batch_size,
        num_pages=args.num_pages,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        torch_ms=torch_ms,
        triton_ms=triton_ms,
        speedup=torch_ms / triton_ms,
        torch_peak_mb=torch_peak_mb,
        triton_peak_mb=triton_peak_mb,
        max_abs_error=max_abs_error,
        topk_overlap_pct=overlap,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--num-pages", type=int, default=2000)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--topk", type=int, default=124)
    parser.add_argument("--invalid-ratio", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    results = [
        run_case(args, int(batch_size))
        for batch_size in args.batch_sizes.split(",")
        if batch_size
    ]
    for result in results:
        print(
            f"bs={result.batch_size} torch={result.torch_ms:.4f} ms "
            f"triton={result.triton_ms:.4f} ms speedup={result.speedup:.2f}x "
            f"peak={result.torch_peak_mb:.1f}->{result.triton_peak_mb:.1f} MB "
            f"max_abs={result.max_abs_error:.6f} "
            f"topk_overlap={result.topk_overlap_pct:.2f}%"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"elapsed={time.perf_counter() - started:.2f}s")
