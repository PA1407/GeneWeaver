
import random
import statistics
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.gpu_engine import gpu_alignment
from src.alignment.gpu_shared_memory import gpu_alignment_shared


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

    genome = genome[:100] + guide + genome[100 + len(guide):]
    genome = genome[:500] + guide + genome[500 + len(guide):]

    return genome


def measure_time(function, guide, genome, runs=5):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function(guide, genome, 2)
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome_sizes = [10_000, 100_000, 500_000, 1_000_000]

    print("GENEWEAVER: GPU GLOBAL vs SHARED MEMORY")
    print("=" * 65)

    # Warm up both kernels before timing.
    warmup_genome = "ATGC" * 1000
    gpu_alignment(guide, warmup_genome, 2)
    gpu_alignment_shared(guide, warmup_genome, 2)

    print("GPU warm-up completed.")

    for size in genome_sizes:
        print("\n" + "-" * 65)
        print(f"Genome size: {size:,} bases")

        genome = generate_genome(size, guide)

        original_results, original_time = measure_time(
            gpu_alignment, guide, genome
        )

        shared_results, shared_time = measure_time(
            gpu_alignment_shared, guide, genome
        )

        print(f"Original GPU time: {original_time:.6f} seconds")
        print(f"Shared-memory GPU time: {shared_time:.6f} seconds")
        print(f"Results identical: {original_results == shared_results}")

        if shared_time > 0:
            print(
                f"Shared-memory speedup: "
                f"{original_time / shared_time:.2f}x"
            )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
