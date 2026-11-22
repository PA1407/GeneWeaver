import time
import numpy as np
import statistics

from src.alignment.gpu_engine import process_alignment_results


def original_processing(counts, genome, guide_length, threshold):
    matches = []

    for position, count in enumerate(counts):
        if count <= threshold:
            matches.append({
                "position": position,
                "sequence": genome[position:position + guide_length],
                "mismatches": int(count)
            })

    return matches


def measure(function, *args, repeats=15):
    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        timings.append(time.perf_counter() - start)

    return statistics.median(timings), result


def main():
    rng = np.random.default_rng(42)
    guide_length = 20

    print("Integrated Result Processing Benchmark")
    print("=" * 65)

    for genome_length in [100_000, 500_000, 1_000_000]:
        genome = "A" * genome_length
        total_windows = genome_length - guide_length + 1

        counts = rng.integers(
            0, 5, size=total_windows, dtype=np.int32
        )

        args = (counts, genome, guide_length, 2)

        old_time, old_result = measure(original_processing, *args)
        new_time, new_result = measure(process_alignment_results, *args)

        assert old_result == new_result

        print(f"\nGenome size: {genome_length:,}")
        print(f"Matches: {len(new_result):,}")
        print(f"Original: {old_time:.6f}s")
        print(f"Optimized: {new_time:.6f}s")
        print(f"Speedup: {old_time / new_time:.2f}x")
        print("Output verification: PASSED")


if __name__ == "__main__":
    main()
