import time
import sys
from pathlib import Path

import numpy as np
from numba import cuda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.gpu_engine import (
    encode_sequence,
    alignment_kernel
)

import time
import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence,
    alignment_kernel
)


def profile_gpu_alignment(guide_sequence, genome_sequence):
    guide = encode_sequence(guide_sequence)
    genome = encode_sequence(genome_sequence)

    total_windows = len(genome) - len(guide) + 1

    if total_windows <= 0:
        raise ValueError("Genome must be at least as long as guide")

    threads_per_block = 128
    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    # 1. Measure guide transfer
    start = time.perf_counter()
    d_guide = cuda.to_device(guide)
    cuda.synchronize()
    guide_transfer = time.perf_counter() - start

    # 2. Measure genome transfer
    start = time.perf_counter()
    d_genome = cuda.to_device(genome)
    cuda.synchronize()
    genome_transfer = time.perf_counter() - start

    # 3. Allocate output memory
    d_mismatch_counts = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    # 4. Measure kernel execution
    start = time.perf_counter()

    alignment_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_guide,
        d_genome,
        d_mismatch_counts
    )

    cuda.synchronize()
    kernel_time = time.perf_counter() - start

    # 5. Measure GPU-to-CPU result transfer
    start = time.perf_counter()
    mismatch_counts = d_mismatch_counts.copy_to_host()
    result_transfer = time.perf_counter() - start

    return {
        "guide_transfer": guide_transfer,
        "genome_transfer": genome_transfer,
        "kernel_execution": kernel_time,
        "result_transfer": result_transfer,
        "total": (
            guide_transfer
            + genome_transfer
            + kernel_time
            + result_transfer
        ),
        "windows": total_windows
    }


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = "ATGC" * 125_000

    # Warm up CUDA compilation
    profile_gpu_alignment(guide, genome)

    results = profile_gpu_alignment(guide, genome)

    print("\nGPU MEMORY TRANSFER PROFILING")
    print("=" * 50)

    print(f"Guide transfer:  {results['guide_transfer']:.6f} s")
    print(f"Genome transfer: {results['genome_transfer']:.6f} s")
    print(f"Kernel execution: {results['kernel_execution']:.6f} s")
    print(f"Result transfer: {results['result_transfer']:.6f} s")

    print("-" * 50)
    print(f"Total measured time: {results['total']:.6f} s")
    print(f"Windows processed: {results['windows']:,}")

    print("\nPROFILING COMPLETED")


if __name__ == "__main__":
    main()
