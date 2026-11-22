import time
import numpy as np
import statistics

from benchmarks.benchmark_result_processing import (
    original_processing,
    vectorized_processing
)


def measure(function, *args, repeats=7):
    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)

    return statistics.median(timings), result


def main():
    rng = np.random.default_rng(42)

    genome_length = 500_000
    guide_length = 20
    total_windows = genome_length - guide_length + 1
    genome = "A" * genome_length

    print("Result Processing: Match Density Comparison")
    print("=" * 65)

    for density in [0.0001, 0.01, 0.10, 0.60]:
        mismatch_counts = np.full(
            total_windows, 3, dtype=np.int32
        )

        match_count = int(total_windows * density)

        selected = rng.choice(
            total_windows,
            size=match_count,
            replace=False
        )

        mismatch_counts[selected] = 0

        old_time, old_result = measure(
            original_processing,
            mismatch_counts,
            genome,
            guide_length,
            2
        )

        new_time, new_result = measure(
            vectorized_processing,
            mismatch_counts,
            genome,
            guide_length,
            2
        )

        assert old_result == new_result

        speedup = old_time / new_time if new_time else 0

        print(f"\nMatch density: {density * 100:.2f}%")
        print(f"Matches: {len(old_result):,}")
        print(f"Original: {old_time:.6f} seconds")
        print(f"Vectorized: {new_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print("Output verification: PASSED")


if __name__ == "__main__":
    main()
