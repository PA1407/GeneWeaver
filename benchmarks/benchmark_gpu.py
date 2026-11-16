import random
import statistics
import time
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.alignment.gpu_engine import gpu_alignment
import random
import statistics
import time

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.alignment.gpu_engine import gpu_alignment


def generate_genome(size, guide):
    """Generate a reproducible DNA sequence with two known matches."""
    random.seed(42)

    genome = "".join(random.choices("ATGC", k=size))

    genome = (
        genome[:100]
        + guide
        + genome[100 + len(guide):]
    )

    genome = (
        genome[:500]
        + guide
        + genome[500 + len(guide):]
    )

    return genome


def measure_time(function, *args, runs=3):
    """Measure execution time and return the median."""
    timings = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = function(*args)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    return result, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"

    genome_sizes = [10_000, 100_000, 500_000, 1_000_000]

    print("GENEWEAVER: CPU vs GPU BENCHMARK")
    print("=" * 60)

    # Warm up GPU to exclude first-time CUDA compilation.
    print("\nWarming up GPU...")
    gpu_alignment(guide, "ATGCGTACGTAGCTAGCTAA", max_mismatches=2)
    print("GPU warm-up completed.")

    for size in genome_sizes:
        print("\n" + "-" * 60)
        print(f"Genome size: {size:,} bases")

        genome = generate_genome(size, guide)

        # Measure CPU execution.
        cpu_results, cpu_time = measure_time(
            find_matches_with_mismatches,
            guide,
            genome,
            2
        )

        # Measure GPU end-to-end execution, including transfers.
        gpu_results, gpu_time = measure_time(
            gpu_alignment,
            guide,
            genome,
            2
        )

        # Verify correctness.
        results_match = cpu_results == gpu_results

        print(f"CPU time: {cpu_time:.6f} seconds")
        print(f"GPU end-to-end time: {gpu_time:.6f} seconds")
        print(f"Results identical: {results_match}")

        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x")

        if not results_match:
            print("WARNING: CPU and GPU results differ!")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
