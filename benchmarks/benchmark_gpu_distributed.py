

import random
import statistics
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from distributed import Client, LocalCluster

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.alignment.gpu_engine import gpu_alignment
from src.scheduler.gpu_distributed import gpu_distributed_alignment


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

    genome = genome[:100] + guide + genome[100 + len(guide):]
    genome = genome[:500] + guide + genome[500 + len(guide):]

    return genome


def measure_time(function, *args, runs=3):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function(*args)
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    sizes = [10_000, 100_000, 500_000]

    # Warm up CUDA before measurements.
    gpu_alignment(guide, "ATGC" * 1000, 2)

    print("CPU vs GPU vs GPU+DASK (REUSED CLUSTER)")
    print("=" * 65)

    # Create one cluster and reuse its client.
    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as client:

            # Warm up the Dask worker and GPU.
            gpu_distributed_alignment(
                guide,
                "ATGC" * 1000,
                chunk_size=1000,
                max_mismatches=2,
                client=client
            )

            for size in sizes:
                genome = generate_genome(size, guide)

                cpu_results, cpu_time = measure_time(
                    find_matches_with_mismatches,
                    guide, genome, 2
                )

                gpu_results, gpu_time = measure_time(
                    gpu_alignment,
                    guide, genome, 2
                )

                distributed_results, distributed_time = measure_time(
                    gpu_distributed_alignment,
                    guide,
                    genome,
                    100_000,
                    2,
                    1,
                    client
                )

                print(f"\nGenome size: {size:,}")
                print(f"CPU: {cpu_time:.6f} seconds")
                print(f"Direct GPU: {gpu_time:.6f} seconds")
                print(f"GPU + Dask: {distributed_time:.6f} seconds")

                print("GPU matches CPU:", gpu_results == cpu_results)
                print(
                    "GPU+Dask matches CPU:",
                    distributed_results == cpu_results
                )

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
