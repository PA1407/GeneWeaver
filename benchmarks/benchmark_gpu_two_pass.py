
import statistics
import time

import numpy as np
from numba import cuda, int32

from src.alignment.gpu_engine import (
    encode_sequence_vectorized,
    gpu_alignment,
)

from benchmarks.benchmark_gpu_compaction import (
    gpu_alignment_compact,
    generate_genome,
)


THREADS_PER_BLOCK = 128


@cuda.jit
def count_matches_kernel(
    guide,
    genome,
    max_mismatches,
    mismatch_counts,
    block_counts,
):
    local_id = cuda.threadIdx.x
    position = cuda.grid(1)

    total_windows = genome.size - guide.size + 1

    shared_flags = cuda.shared.array(
        THREADS_PER_BLOCK,
        dtype=int32,
    )

    mismatch_count = 0

    if position < total_windows:
        for j in range(guide.size):
            if guide[j] != genome[position + j]:
                mismatch_count += 1

        mismatch_counts[position] = mismatch_count

        if mismatch_count <= max_mismatches:
            shared_flags[local_id] = 1
        else:
            shared_flags[local_id] = 0
    else:
        shared_flags[local_id] = 0

    cuda.syncthreads()

    if local_id == 0:
        block_total = 0

        for i in range(THREADS_PER_BLOCK):
            block_total += shared_flags[i]

        block_counts[cuda.blockIdx.x] = block_total


@cuda.jit
def write_matches_kernel(
    guide,
    genome,
    max_mismatches,
    mismatch_counts,
    block_offsets,
    output_positions,
    output_mismatches,
):
    local_id = cuda.threadIdx.x
    position = cuda.grid(1)

    total_windows = genome.size - guide.size + 1

    shared_flags = cuda.shared.array(
        THREADS_PER_BLOCK,
        dtype=int32,
    )

    matched = 0
    mismatch_count = 0

    if position < total_windows:
        mismatch_count = mismatch_counts[position]

        if mismatch_count <= max_mismatches:
            matched = 1

    shared_flags[local_id] = matched

    cuda.syncthreads()

    local_offset = 0

    if matched == 1:
        for i in range(local_id):
            local_offset += shared_flags[i]

        block_start = block_offsets[cuda.blockIdx.x]
        output_index = block_start + local_offset

        output_positions[output_index] = position
        output_mismatches[output_index] = mismatch_count


def gpu_alignment_two_pass(
    guide_sequence,
    genomic_sequence,
    max_mismatches=2,
):
    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    guide = encode_sequence_vectorized(guide_sequence)
    genome = encode_sequence_vectorized(genomic_sequence)

    if len(guide) > len(genome):
        return [], 0, 0

    total_windows = len(genome) - len(guide) + 1

    blocks_per_grid = (
        total_windows + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    d_guide = cuda.to_device(guide)
    d_genome = cuda.to_device(genome)

    d_mismatch_counts = cuda.device_array(
        total_windows,
        dtype=np.int32,
    )

    d_block_counts = cuda.device_array(
        blocks_per_grid,
        dtype=np.int32,
    )

    # Pass 1: calculate mismatch counts and block match totals.
    count_matches_kernel[
        blocks_per_grid,
        THREADS_PER_BLOCK,
    ](
        d_guide,
        d_genome,
        max_mismatches,
        d_mismatch_counts,
        d_block_counts,
    )

    # Copy block totals to CPU and calculate exclusive offsets.
    block_counts = d_block_counts.copy_to_host()

    block_offsets = np.zeros(
        blocks_per_grid,
        dtype=np.int32,
    )

    if blocks_per_grid > 1:
        block_offsets[1:] = np.cumsum(
            block_counts[:-1],
            dtype=np.int64,
        ).astype(np.int32)

    match_count = int(block_counts.sum())

    if match_count == 0:
        return [], total_windows, (
            blocks_per_grid * 4
        )

    d_block_offsets = cuda.to_device(block_offsets)

    # Allocate output arrays using the actual match count.
    d_positions = cuda.device_array(
        match_count,
        dtype=np.int32,
    )

    d_mismatches = cuda.device_array(
        match_count,
        dtype=np.int32,
    )

    # Pass 2: write matches using precomputed block offsets.
    write_matches_kernel[
        blocks_per_grid,
        THREADS_PER_BLOCK,
    ](
        d_guide,
        d_genome,
        max_mismatches,
        d_mismatch_counts,
        d_block_offsets,
        d_positions,
        d_mismatches,
    )

    positions = d_positions.copy_to_host()
    mismatches = d_mismatches.copy_to_host()

    results = []

    for i in range(match_count):
        position = int(positions[i])

        results.append({
            "position": position,
            "sequence": genomic_sequence[
                position:position + len(guide)
            ],
            "mismatches": int(mismatches[i]),
        })

    results.sort(key=lambda item: item["position"])

    # Includes block-count D2H, offset H2D, and output D2H.
    transferred_bytes = (
        blocks_per_grid * 4
        + blocks_per_grid * 4
        + match_count * 8
    )

    return results, total_windows, transferred_bytes


def controlled_measure(
    guide,
    genome,
    threshold,
    runs=15,
):
    functions = {
        "standard": gpu_alignment,
        "atomic": gpu_alignment_compact,
        "two_pass": gpu_alignment_two_pass,
    }

    timings = {
        name: [] for name in functions
    }

    results = {}

    # Warm up every implementation.
    for name, function in functions.items():
        results[name] = function(
            guide,
            genome,
            threshold,
        )

    for run in range(runs):
        order = list(functions.items())

        if run % 2:
            order.reverse()

        for name, function in order:
            start = time.perf_counter()

            results[name] = function(
                guide,
                genome,
                threshold,
            )

            timings[name].append(
                time.perf_counter() - start
            )

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    return results, medians


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = generate_genome(500_000, guide)

    thresholds = [5, 7, 9, 10, 20]

    print("GPU TWO-PASS COMPACTION BENCHMARK")
    print("=" * 75)

    for threshold in thresholds:
        results, medians = controlled_measure(
            guide,
            genome,
            threshold,
        )

        standard_results = results["standard"]
        atomic_results, _, atomic_bytes = results["atomic"]
        two_pass_results, windows, two_pass_bytes = (
            results["two_pass"]
        )

        print(f"\nThreshold: {threshold}")
        print(f"Matches: {len(two_pass_results):,}")

        print(f"Standard GPU: {medians['standard']:.6f} s")
        print(f"Global atomic: {medians['atomic']:.6f} s")
        print(f"Two-pass: {medians['two_pass']:.6f} s")

        print(
            f"Two-pass vs atomic: "
            f"{medians['atomic'] / medians['two_pass']:.3f}x"
        )

        print(
            f"Two-pass vs standard: "
            f"{medians['standard'] / medians['two_pass']:.3f}x"
        )

        print(f"Atomic transfer: {atomic_bytes:,} bytes")
        print(f"Two-pass transfer: {two_pass_bytes:,} bytes")

        print(
            "All results correct:",
            standard_results == atomic_results == two_pass_results,
        )

    print("\nTWO-PASS BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
