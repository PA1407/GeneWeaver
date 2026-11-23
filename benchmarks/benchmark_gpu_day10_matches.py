import time
import statistics

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


def create_genome(size, guide):
    genome = list(("ATGC" * ((size + 3) // 4))[:size])

    # Insert known guide matches at regular intervals.
    for position in range(1000, size - len(guide), 10000):
        genome[position:position + len(guide)] = list(guide)

    return "".join(genome)


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"

    print("GPU WORKSPACE REUSE: MATCH-POSITIVE BENCHMARK")
    print("=" * 65)

    for size in [100_000, 500_000, 1_000_000]:
        genome = create_genome(size, guide)

        workspace = GPUAsyncWorkspace(
            len(guide),
            len(genome)
        )

        # Warm-up.
        standard_result = gpu_alignment(guide, genome, 0)
        async_result = workspace.align(guide, genome, 0)

        assert standard_result == async_result
        assert len(standard_result) > 0

        standard_time, standard_result = measure(
            lambda: gpu_alignment(guide, genome, 0)
        )

        async_time, async_result = measure(
            lambda: workspace.align(guide, genome, 0)
        )

        assert standard_result == async_result

        print(f"\nGenome size: {size:,}")
        print(f"Matches: {len(standard_result):,}")
        print(f"Standard GPU: {standard_time:.6f} seconds")
        print(f"Reusable async: {async_time:.6f} seconds")
        print(f"Speedup: {standard_time / async_time:.2f}x")
        print("Output verification: PASSED")

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
