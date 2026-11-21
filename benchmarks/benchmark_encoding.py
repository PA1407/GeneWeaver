
import random
import time
import statistics

from src.alignment.gpu_engine import (
    encode_sequence,
    encode_sequence_vectorized
)


def generate_dna_sequence(size):
    random.seed(42)

    return "".join(
        random.choices("ATGC", k=size)
    )


def benchmark(function, sequence, runs=7):

    timings = []

    for _ in range(runs):
        start = time.perf_counter()

        result = function(sequence)

        elapsed = time.perf_counter() - start

        timings.append(elapsed)

    return result, statistics.median(timings)


def main():

    genome_sizes = [
        100_000,
        500_000,
        1_000_000
    ]

    print("\nDNA ENCODING PERFORMANCE BENCHMARK")
    print("=" * 65)

    for size in genome_sizes:

        sequence = generate_dna_sequence(size)

        # Warm-up both implementations.
        original_warmup = encode_sequence(sequence)
        vectorized_warmup = encode_sequence_vectorized(sequence)

        if not (original_warmup == vectorized_warmup).all():
            raise RuntimeError("Encoding results differ!")

        # Measure original implementation.
        original_result, original_time = benchmark(
            encode_sequence,
            sequence
        )

        # Measure vectorized implementation.
        vectorized_result, vectorized_time = benchmark(
            encode_sequence_vectorized,
            sequence
        )

        if not (original_result == vectorized_result).all():
            raise RuntimeError("Benchmark results differ!")

        speedup = original_time / vectorized_time

        improvement = (
            (original_time - vectorized_time)
            / original_time
        ) * 100

        print(f"\nGenome size: {size:,} bases")
        print("-" * 50)

        print(
            f"Original encoding:   {original_time:.6f} seconds"
        )

        print(
            f"Vectorized encoding: {vectorized_time:.6f} seconds"
        )

        print(f"Speedup:             {speedup:.2f}x")
        print(f"Time reduction:      {improvement:.2f}%")
        print("Results:             IDENTICAL")

    print("\nENCODING BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
