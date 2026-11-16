
import random
import statistics
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    genome = generate_genome(1_000_000, guide)

    cpu_results, cpu_time = measure_time(
        find_matches_with_mismatches,
        guide, genome, 2
    )

    print("GENEWEAVER: DASK WORKER BENCHMARK")
    print(f"CPU baseline: {cpu_time:.6f} seconds")
    print("-" * 55)

    for workers in [1, 2, 4]:
        results, elapsed = measure_time(
            distributed_alignment,
            guide,
            genome,
            100_000,
            2,
            workers
        )

        print(f"Workers: {workers}")
        print(f"Execution time: {elapsed:.6f} seconds")
        print(f"Results identical: {results == cpu_results}")

        if elapsed > 0:
            print(f"CPU/Dask ratio: {cpu_time / elapsed:.2f}x")

        print("-" * 55)


if __name__ == "__main__":
    main()
