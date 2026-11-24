

import statistics
import time

from src.alignment.gpu_engine import gpu_alignment

from benchmarks.benchmark_gpu_compaction import (
    gpu_alignment_compact,
    generate_genome,
)


def controlled_measure(
    guide,
    genome,
    threshold,
    runs=15
):
    standard_times = []
    compact_times = []

    standard_results = None
    compact_results = None

    # Warm up both implementations.
    gpu_alignment(guide, genome, threshold)
    gpu_alignment_compact(guide, genome, threshold)

    for _ in range(runs):
        start = time.perf_counter()

        standard_results = gpu_alignment(
            guide,
            genome,
            threshold
        )

        standard_times.append(
            time.perf_counter() - start
        )

        start = time.perf_counter()

        compact_results = gpu_alignment_compact(
            guide,
            genome,
            threshold
        )

        compact_times.append(
            time.perf_counter() - start
        )

    return (
        standard_results,
        compact_results,
        statistics.median(standard_times),
        statistics.median(compact_times),
    )


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = generate_genome(500_000, guide)

    thresholds = [5, 6, 7, 8, 9, 10]

    print("CONTROLLED GPU COMPACTION BENCHMARK")
    print("=" * 75)

    for threshold in thresholds:
        (
            standard_results,
            compact_output,
            standard_time,
            compact_time,
        ) = controlled_measure(
            guide,
            genome,
            threshold
        )

        compact_results, windows, compact_bytes = compact_output

        standard_bytes = windows * 4

        speedup = standard_time / compact_time

        print(f"\nThreshold: {threshold}")
        print(f"Matches: {len(compact_results):,}")

        print(f"Standard median: {standard_time:.6f} s")
        print(f"Compact median:  {compact_time:.6f} s")

        print(f"Speedup: {speedup:.3f}x")

        print(
            f"Transfer reduction: "
            f"{(1 - compact_bytes / standard_bytes) * 100:.2f}%"
        )

        print(
            "Results correct:",
            standard_results == compact_results
        )

    print("\nCONTROLLED BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
