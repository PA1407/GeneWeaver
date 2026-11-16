import random
import statistics
import time
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.scheduler.distributed import distributed_alignment

import random
import statistics
import time

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.scheduler.distributed import distributed_alignment


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

    genome = genome[:100] + guide + genome[100 + len(guide):]
    genome = genome[:500] + guide + genome[500 + len(guide):]

    return genome


def measure_time(function, *args, runs=3):
    timings = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)

    return result, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome_sizes = [10_000, 100_000, 500_000, 1_000_000]

    print("GENEWEAVER: CPU vs DASK BENCHMARK")
    print("=" * 60)

    for size in genome_sizes:
        print(f"\nGenome size: {size:,} bases")

        genome = generate_genome(size, guide)

        cpu_results, cpu_time = measure_time(
            find_matches_with_mismatches,
            guide,
            genome,
            2
        )

        dask_results, dask_time = measure_time(
            distributed_alignment,
            guide,
            genome,
            100_000,
            2,
            2
        )

        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"Dask time: {dask_time:.6f} seconds")
        print(f"Results identical: {cpu_results == dask_results}")

        if dask_time > 0:
            print(f"Speedup: {cpu_time / dask_time:.2f}x")

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
