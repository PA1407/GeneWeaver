import time
import numpy as np
import statistics

from benchmarks.benchmark_result_processing import (
    original_processing,
    vectorized_processing
)


def timed_call(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return time.perf_counter() - start, result


def main():
    rng = np.random.default_rng(42)

    genome_length = 200_000
    guide_length = 20
    total_windows = genome_length - guide_length + 1
    genome = "A" * genome_length

    print("Controlled Result Processing Benchmark")
    print("=" * 65)

    for density in [0.10, 0.30, 0.50, 0.60]:
        counts = np.full(total_windows, 3, dtype=np.int32)
        match_count = int(total_windows * density)

        selected = rng.choice(
            total_windows,
            size=match_count,
            replace=False
        )
        counts[selected] = 0

        args = (counts, genome, guide_length, 2)

        # Warm-up both implementations.
        _, original_result = timed_call(original_processing, *args)
        _, vectorized_result = timed_call(vectorized_processing, *args)

        assert original_result == vectorized_result

        original_times = []
        vectorized_times = []

        for repeat in range(15):
            if repeat % 2 == 0:
                old_time, old_result = timed_call(
                    original_processing, *args
                )
                new_time, new_result = timed_call(
                    vectorized_processing, *args
                )
            else:
                new_time, new_result = timed_call(
                    vectorized_processing, *args
                )
                old_time, old_result = timed_call(
                    original_processing, *args
                )

            assert old_result == new_result

            original_times.append(old_time)
            vectorized_times.append(new_time)

        old_median = statistics.median(original_times)
        new_median = statistics.median(vectorized_times)

        print(f"\nMatch density: {density * 100:.0f}%")
        print(f"Original median:  {old_median:.6f}s")
        print(f"NumPy median:     {new_median:.6f}s")
        print(f"Speedup:          {old_median / new_median:.2f}x")
        print("Output verification: PASSED")


if __name__ == "__main__":
    main()
