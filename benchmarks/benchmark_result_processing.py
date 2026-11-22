import time
import numpy as np
import statistics


def original_processing(mismatch_counts, genome, guide_length, max_mismatches):
    matches = []

    for position, count in enumerate(mismatch_counts):
        if count <= max_mismatches:
            matches.append({
                "position": position,
                "sequence": genome[position:position + guide_length],
                "mismatches": int(count)
            })

    return matches


def vectorized_processing(mismatch_counts, genome, guide_length, max_mismatches):
    matching_positions = np.flatnonzero(
        mismatch_counts <= max_mismatches
    )

    return [
        {
            "position": int(position),
            "sequence": genome[position:position + guide_length],
            "mismatches": int(mismatch_counts[position])
        }
        for position in matching_positions
    ]


def measure(function, *args, repeats=7):
    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)

    return statistics.median(timings), result


def main():
    guide_length = 20
    rng = np.random.default_rng(42)

    print("GPU Result Processing Benchmark")
    print("=" * 65)

    for genome_length in [100_000, 500_000, 1_000_000]:
        total_windows = genome_length - guide_length + 1
        genome = "A" * genome_length

        # Simulate mismatch counts from a GPU result array.
        mismatch_counts = rng.integers(
            0, 5, size=total_windows, dtype=np.int32
        )

        threshold = 2

        old_time, old_result = measure(
            original_processing,
            mismatch_counts,
            genome,
            guide_length,
            threshold
        )

        new_time, new_result = measure(
            vectorized_processing,
            mismatch_counts,
            genome,
            guide_length,
            threshold
        )

        assert old_result == new_result

        speedup = old_time / new_time if new_time > 0 else 0

        print(f"\nGenome size: {genome_length:,}")
        print(f"Windows: {total_windows:,}")
        print(f"Matches: {len(old_result):,}")
        print(f"Original processing: {old_time:.6f} seconds")
        print(f"Vectorized processing: {new_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print("Output verification: PASSED")


if __name__ == "__main__":
    main()
