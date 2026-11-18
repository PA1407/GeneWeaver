
import random
import statistics
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dask import delayed
from distributed import Client, LocalCluster

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.alignment.gpu_engine import gpu_alignment
from src.scheduler.distributed import split_genome
from src.scheduler.gpu_distributed import process_gpu_chunk


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

    genome = genome[:100] + guide + genome[100 + len(guide):]
    genome = genome[:500] + guide + genome[500 + len(guide):]

    return genome


def process_original_chunk(
    guide_sequence,
    chunk,
    start_position,
    max_mismatches
):
    """Original approach: transfer guide for every chunk."""

    matches = gpu_alignment(
        guide_sequence,
        chunk,
        max_mismatches
    )

    return [
        {
            "position": match["position"] + start_position,
            "sequence": match["sequence"],
            "mismatches": match["mismatches"]
        }
        for match in matches
    ]


def run_distributed(
    client,
    processor,
    guide,
    genome,
    chunk_size,
    max_mismatches
):
    chunks = split_genome(
        genome,
        chunk_size,
        len(guide)
    )

    tasks = [
        delayed(processor)(
            guide,
            chunk,
            start,
            max_mismatches
        )
        for chunk, start in chunks
    ]

    futures = client.compute(tasks)
    chunk_results = client.gather(futures)

    results = []

    for matches in chunk_results:
        results.extend(matches)

    results.sort(key=lambda item: item["position"])

    return results


def measure_time(function, runs=5):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function()
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = generate_genome(1_000_000, guide)

    chunk_size = 250_000
    max_mismatches = 2

    expected = find_matches_with_mismatches(
        guide,
        genome,
        max_mismatches
    )

    print("ORIGINAL vs REUSABLE GPU GUIDE CACHE")
    print("=" * 65)

    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as client:

            # Warm up CUDA and Dask.
            run_distributed(
                client,
                process_original_chunk,
                guide,
                genome[:chunk_size],
                chunk_size,
                max_mismatches
            )

            # Initialize the reusable guide cache before measuring
            # steady-state performance.
            run_distributed(
                client,
                process_gpu_chunk,
                guide,
                genome[:chunk_size],
                chunk_size,
                max_mismatches
            )

            original_results, original_time = measure_time(
                lambda: run_distributed(
                    client,
                    process_original_chunk,
                    guide,
                    genome,
                    chunk_size,
                    max_mismatches
                )
            )

            optimized_results, optimized_time = measure_time(
                lambda: run_distributed(
                    client,
                    process_gpu_chunk,
                    guide,
                    genome,
                    chunk_size,
                    max_mismatches
                )
            )

    print(f"\nGenome size: {len(genome):,}")
    print(f"Chunk size: {chunk_size:,}")
    print("Dask workers: 1")

    print(f"\nOriginal GPU+Dask: {original_time:.6f} seconds")
    print(f"Cached GPU+Dask:   {optimized_time:.6f} seconds")

    if optimized_time > 0:
        improvement = (
            (original_time - optimized_time)
            / original_time
        ) * 100

        print(f"Performance improvement: {improvement:.2f}%")

    print("\nCorrectness checks:")
    print("Original matches CPU:", original_results == expected)
    print("Cached matches CPU:", optimized_results == expected)
    print(
        "Original and cached results identical:",
        original_results == optimized_results
    )

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
