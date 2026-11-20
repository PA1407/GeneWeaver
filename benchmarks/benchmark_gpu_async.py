

import random
import statistics
import time

from src.alignment.gpu_engine import gpu_alignment
from src.alignment.gpu_async import (
    gpu_alignment_async,
    GPUAsyncWorkspace
)


def generate_genome(size, guide):
    random.seed(42)

    genome = list(
        random.choices("ATGC", k=size)
    )

    for position in [100, 500]:
        if position + len(guide) <= size:
            genome[position:position + len(guide)] = guide

    return "".join(genome)


def measure(function, runs=7):
    timings = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)

    return result, statistics.median(timings)


def main():
    guide = "ATGCGATCGATCGATCGATC"

    genome_sizes = [
        10_000,
        100_000,
        500_000,
        1_000_000
    ]

    print("\nTHREE-WAY GPU ASYNC BENCHMARK")
    print("=" * 75)

    for size in genome_sizes:
        genome = generate_genome(size, guide)

        # Prepare reusable workspace and measure its setup cost.
        setup_start = time.perf_counter()

        workspace = GPUAsyncWorkspace(
            len(guide),
            len(genome)
        )

        setup_time = time.perf_counter() - setup_start

        # Warm up all implementations.
        original_warmup = gpu_alignment(guide, genome, 2)

        fresh_warmup = gpu_alignment_async(
            guide, genome, 2
        )

        reused_warmup = gpu_alignment_async(
            guide, genome, 2, workspace
        )

        if not (
            original_warmup == fresh_warmup == reused_warmup
        ):
            raise RuntimeError(
                f"Warm-up results differ at size {size}"
            )

        # Original GPU implementation.
        original_results, original_time = measure(
            lambda: gpu_alignment(guide, genome, 2)
        )

        # Async implementation with fresh allocations.
        fresh_results, fresh_time = measure(
            lambda: gpu_alignment_async(guide, genome, 2)
        )

        # Async implementation reusing the same workspace.
        reused_results, reused_time = measure(
            lambda: gpu_alignment_async(
                guide, genome, 2, workspace
            )
        )

        if not (
            original_results == fresh_results == reused_results
        ):
            raise RuntimeError(
                f"Results differ at genome size {size}"
            )

        print(f"\nGenome size: {size:,} bases")
        print("-" * 55)
        print(f"Original GPU:       {original_time:.6f} seconds")
        print(f"Async fresh:        {fresh_time:.6f} seconds")
        print(f"Async reused:       {reused_time:.6f} seconds")
        print(f"Workspace setup:    {setup_time:.6f} seconds")

        print("\nSpeedup compared with original:")
        print(f"Fresh workspace:    {original_time / fresh_time:.2f}x")
        print(f"Reused workspace:   {original_time / reused_time:.2f}x")

        print("\nCorrectness: IDENTICAL")

        # Release workspace resources before next genome size.
        workspace.stream.synchronize()
        del workspace

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
