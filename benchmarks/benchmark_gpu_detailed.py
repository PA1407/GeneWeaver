
import time
import statistics

import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence_vectorized,
    alignment_kernel
)


def profile_gpu_alignment(guide_sequence, genome_sequence):

    cuda.synchronize()
    total_start = time.perf_counter()

    # 1. Measure DNA encoding.
    start = time.perf_counter()

    guide = encode_sequence_vectorized(guide_sequence)
    genome = encode_sequence_vectorized(genome_sequence)

    encoding_time = time.perf_counter() - start

    total_windows = len(genome) - len(guide) + 1

    if total_windows <= 0:
        raise ValueError(
            "Genome must be at least as long as guide"
        )

    threads_per_block = 128

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    # 2. Measure guide transfer.
    start = time.perf_counter()

    d_guide = cuda.to_device(guide)
    cuda.synchronize()

    guide_transfer = time.perf_counter() - start

    # 3. Measure genome transfer.
    start = time.perf_counter()

    d_genome = cuda.to_device(genome)
    cuda.synchronize()

    genome_transfer = time.perf_counter() - start

    # 4. Measure GPU output memory allocation.
    start = time.perf_counter()

    d_results = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    cuda.synchronize()

    allocation_time = time.perf_counter() - start

    # 5. Measure CUDA kernel execution.
    start = time.perf_counter()

    alignment_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_guide,
        d_genome,
        d_results
    )

    cuda.synchronize()

    kernel_time = time.perf_counter() - start

    # 6. Measure GPU-to-CPU result transfer.
    start = time.perf_counter()

    mismatch_counts = d_results.copy_to_host()

    result_transfer = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    return {
        "encoding": encoding_time,
        "guide_transfer": guide_transfer,
        "genome_transfer": genome_transfer,
        "allocation": allocation_time,
        "kernel": kernel_time,
        "result_transfer": result_transfer,
        "total": total_time,
        "windows": total_windows,
        "result_count": len(mismatch_counts)
    }


def main():

    guide = "ATGCGTACGTAGCTAGCTAA"

    genome_sizes = [
        100_000,
        500_000,
        1_000_000
    ]

    runs = 7

    print("\nDETAILED GPU PERFORMANCE PROFILING")
    print("=" * 65)

    for size in genome_sizes:

        genome = "ATGC" * (size // 4)

        # Warm up CUDA compilation and execution.
        profile_gpu_alignment(guide, genome)

        measurements = []

        for _ in range(runs):
            measurements.append(
                profile_gpu_alignment(guide, genome)
            )

        stages = [
            "encoding",
            "guide_transfer",
            "genome_transfer",
            "allocation",
            "kernel",
            "result_transfer",
            "total"
        ]

        print(f"\nGenome size: {len(genome):,} bases")
        print("-" * 55)

        for stage in stages:

            values = [
                result[stage]
                for result in measurements
            ]

            median_time = statistics.median(values)

            print(
                f"{stage.replace('_', ' ').title():22}"
                f"{median_time:.6f} seconds"
            )

        print(
            f"Windows processed: "
            f"{measurements[-1]['windows']:,}"
        )

        print(
            f"Result elements: "
            f"{measurements[-1]['result_count']:,}"
        )

    print("\nDETAILED PROFILING COMPLETED")


if __name__ == "__main__":
    main()
