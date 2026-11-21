
import time
import statistics

import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence,
    encode_sequence_vectorized,
    alignment_kernel
)


def profile_pipeline(guide_sequence, genome_sequence, encoder):

    cuda.synchronize()
    total_start = time.perf_counter()

    # 1. DNA encoding
    start = time.perf_counter()

    guide = encoder(guide_sequence)
    genome = encoder(genome_sequence)

    encoding_time = time.perf_counter() - start

    total_windows = len(genome) - len(guide) + 1

    threads_per_block = 128

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    # 2. Guide transfer
    start = time.perf_counter()

    d_guide = cuda.to_device(guide)
    cuda.synchronize()

    guide_transfer = time.perf_counter() - start

    # 3. Genome transfer
    start = time.perf_counter()

    d_genome = cuda.to_device(genome)
    cuda.synchronize()

    genome_transfer = time.perf_counter() - start

    # 4. GPU output allocation
    start = time.perf_counter()

    d_results = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    cuda.synchronize()

    allocation_time = time.perf_counter() - start

    # 5. CUDA kernel execution
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

    # 6. Result transfer
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
        "mismatch_counts": mismatch_counts
    }


def benchmark(guide, genome, encoder, runs=7):

    measurements = []

    for _ in range(runs):
        measurements.append(
            profile_pipeline(guide, genome, encoder)
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

    return {
        stage: statistics.median(
            [measurement[stage] for measurement in measurements]
        )
        for stage in stages
    }, measurements[-1]["mismatch_counts"]


def main():

    guide = "ATGCGTACGTAGCTAGCTAA"

    genome_sizes = [
        100_000,
        500_000,
        1_000_000
    ]

    print("\nGPU PIPELINE: ORIGINAL VS VECTORIZED ENCODING")
    print("=" * 70)

    for size in genome_sizes:

        genome = "ATGC" * (size // 4)

        # Warm-up both implementations.
        profile_pipeline(guide, genome, encode_sequence)
        profile_pipeline(guide, genome, encode_sequence_vectorized)

        original, original_results = benchmark(
            guide,
            genome,
            encode_sequence
        )

        optimized, optimized_results = benchmark(
            guide,
            genome,
            encode_sequence_vectorized
        )

        if not np.array_equal(
            original_results,
            optimized_results
        ):
            raise RuntimeError(
                f"GPU results differ for genome size {size}"
            )

        print(f"\nGenome size: {size:,} bases")
        print("-" * 60)

        print(
            f"Original total:    {original['total']:.6f} seconds"
        )

        print(
            f"Optimized total:   {optimized['total']:.6f} seconds"
        )

        print(
            f"Original encoding: {original['encoding']:.6f} seconds"
        )

        print(
            f"Optimized encoding:{optimized['encoding']:.6f} seconds"
        )

        print(
            f"Pipeline speedup:  "
            f"{original['total'] / optimized['total']:.2f}x"
        )

        print(
            f"Encoding speedup:  "
            f"{original['encoding'] / optimized['encoding']:.2f}x"
        )

        print("GPU mismatch counts: IDENTICAL")

    print("\nPIPELINE COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
