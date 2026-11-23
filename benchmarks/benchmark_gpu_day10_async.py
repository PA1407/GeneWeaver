import time
import statistics
import numpy as np

from src.alignment.gpu_engine import gpu_alignment
from src.alignment.gpu_async import GPUAsyncWorkspace


def measure(function, repeats=7):
    timings = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)

    return statistics.median(timings), result


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"

    print("STANDARD GPU vs REUSABLE ASYNC GPU")
    print("=" * 65)

    for size in [100_000, 500_000, 1_000_000]:

        genome = ("ATGC" * ((size + 3) // 4))[:size]

        workspace = GPUAsyncWorkspace(
            len(guide),
            len(genome)
        )

        # Warm up both implementations.
        standard_result = gpu_alignment(guide, genome, 2)
        async_result = workspace.align(guide, genome, 2)

        assert standard_result == async_result

        standard_time, standard_result = measure(
            lambda: gpu_alignment(guide, genome, 2)
        )

        async_time, async_result = measure(
            lambda: workspace.align(guide, genome, 2)
        )

        assert standard_result == async_result

        speedup = standard_time / async_time

        print(f"\nGenome size: {size:,}")
        print(f"Matches: {len(standard_result):,}")
        print(f"Standard GPU: {standard_time:.6f} seconds")
        print(f"Reusable async GPU: {async_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print("Output verification: PASSED")

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
