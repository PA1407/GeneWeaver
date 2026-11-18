
import random
import statistics
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from distributed import Client, LocalCluster

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.scheduler.gpu_distributed import gpu_distributed_alignment


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

    genome = genome[:100] + guide + genome[100 + len(guide):]
    genome = genome[:500] + guide + genome[500 + len(guide):]

    return genome


def measure_time(function, runs=3):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function()
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = generate_genome(500_000, guide)

    chunk_sizes = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]

    expected_results = find_matches_with_mismatches(
        guide, genome, 2
    )

    print("GPU + DASK CHUNK SIZE EXPERIMENT")
    print("=" * 65)

    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as client:

            print("\nChunk Size | Median Time | Matches Correct")

            for chunk_size in chunk_sizes:

                results, elapsed = measure_time(
                    lambda: gpu_distributed_alignment(
                        guide,
                        genome,
                        chunk_size=chunk_size,
                        max_mismatches=2,
                        num_workers=1,
                        client=client
                    )
                )

                correct = results == expected_results

                print(
                    f"{chunk_size:>10,} | "
                    f"{elapsed:>10.6f}s | "
                    f"{correct}"
                )

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
