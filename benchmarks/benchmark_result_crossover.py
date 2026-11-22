import numpy as np
from benchmarks.benchmark_result_processing import (
    original_processing,
    vectorized_processing
)
from benchmarks.benchmark_result_density import measure


def main():
    rng = np.random.default_rng(42)

    genome_length = 200_000
    guide_length = 20
    total_windows = genome_length - guide_length + 1
    genome = "A" * genome_length

    print("Hybrid Filtering Crossover Benchmark")
    print("=" * 65)

    for density in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
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

        print(f"\nDensity: {density * 100:.0f}%")
        print(f"Original: {old_time:.6f}s")
        print(f"NumPy:    {new_time:.6f}s")
        print(f"Speedup:  {old_time / new_time:.2f}x")


if __name__ == "__main__":
    main()
