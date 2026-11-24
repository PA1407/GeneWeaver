
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



@cuda.jit
def block_compact_alignment_kernel(
    guide,
    genome,
    max_mismatches,
    output_positions,
    output_mismatches,
    match_count,
    block_counts,
    block_offsets,
):
    shared_flags = cuda.shared.array(128, dtype=int32)
    shared_positions = cuda.shared.array(128, dtype=int32)
    shared_mismatches = cuda.shared.array(128, dtype=int32)

    local_id = cuda.threadIdx.x
    position = cuda.grid(1)

    total_windows = genome.size - guide.size + 1

    matched = 0
    mismatch_count = 0

    if position < total_windows:
        for j in range(guide.size):
            if guide[j] != genome[position + j]:
                mismatch_count += 1

        if mismatch_count <= max_mismatches:
            matched = 1

    shared_flags[local_id] = matched
    shared_positions[local_id] = position
    shared_mismatches[local_id] = mismatch_count

    cuda.syncthreads()

    # Thread 0 counts all matches in this block.
    if local_id == 0:
        block_total = 0

        for i in range(128):
            block_total += shared_flags[i]

        block_counts[cuda.blockIdx.x] = block_total

    cuda.syncthreads()

    # Every matching thread calculates its local output offset.
    local_offset = 0

    if matched == 1:
        for i in range(local_id):
            local_offset += shared_flags[i]

    # IMPORTANT: Thread 0 reserves space even when it is not a match.
    if local_id == 0:
        block_total = block_counts[cuda.blockIdx.x]

        if block_total > 0:
            block_start = cuda.atomic.add(
                match_count,
                0,
                block_total,
            )
            block_offsets[cuda.blockIdx.x] = block_start

    cuda.syncthreads()

    # Matching threads write into their reserved block range.
    if matched == 1:
        block_start = block_offsets[cuda.blockIdx.x]
        output_index = block_start + local_offset

        output_positions[output_index] = shared_positions[local_id]
        output_mismatches[output_index] = shared_mismatches[local_id]



def gpu_alignment_block_compact(
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

    threads_per_block = 128
    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    d_guide = cuda.to_device(guide)
    d_genome = cuda.to_device(genome)

    d_positions = cuda.device_array(
        total_windows,
        dtype=np.int32,
    )

    d_mismatches = cuda.device_array(
        total_windows,
        dtype=np.int32,
    )

    d_match_count = cuda.to_device(
        np.zeros(1, dtype=np.int32)
    )

    d_block_counts = cuda.device_array(
        blocks_per_grid,
        dtype=np.int32,
    )

    d_block_offsets = cuda.device_array(
        blocks_per_grid,
        dtype=np.int32,
    )

    block_compact_alignment_kernel[
        blocks_per_grid,
        threads_per_block,
    ](
        d_guide,
        d_genome,
        max_mismatches,
        d_positions,
        d_mismatches,
        d_match_count,
        d_block_counts,
        d_block_offsets,
    )

    match_count = int(d_match_count.copy_to_host()[0])

    if match_count == 0:
        return [], total_windows, 4

    positions = d_positions[:match_count].copy_to_host()
    mismatches = d_mismatches[:match_count].copy_to_host()

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

    transferred_bytes = (
        match_count * 2 * np.dtype(np.int32).itemsize
        + np.dtype(np.int32).itemsize
    )

    return results, total_windows, transferred_bytes


def controlled_measure(
    guide,
    genome,
    threshold,
    runs=15,
):
    standard_times = []
    atomic_times = []
    block_times = []

    standard_results = None
    atomic_results = None
    block_results = None

    # Warm up all three implementations.
    gpu_alignment(guide, genome, threshold)
    gpu_alignment_compact(guide, genome, threshold)
    gpu_alignment_block_compact(guide, genome, threshold)

    for run in range(runs):
        implementations = [
            ("standard", gpu_alignment),
            ("atomic", gpu_alignment_compact),
            ("block", gpu_alignment_block_compact),
        ]

        # Alternate execution order to reduce systematic timing bias.
        if run % 2:
            implementations.reverse()

        for name, function in implementations:
            start = time.perf_counter()
            result = function(guide, genome, threshold)
            elapsed = time.perf_counter() - start

            if name == "standard":
                standard_results = result
                standard_times.append(elapsed)
            elif name == "atomic":
                atomic_results = result
                atomic_times.append(elapsed)
            else:
                block_results = result
                block_times.append(elapsed)

    return (
        standard_results,
        atomic_results,
        block_results,
        statistics.median(standard_times),
        statistics.median(atomic_times),
        statistics.median(block_times),
    )


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    genome = generate_genome(500_000, guide)

    thresholds = [5, 7, 9, 10, 20]

    print("GPU BLOCK-LEVEL COMPACTION BENCHMARK")
    print("=" * 75)

    for threshold in thresholds:
        (
            standard_results,
            atomic_output,
            block_output,
            standard_time,
            atomic_time,
            block_time,
        ) = controlled_measure(
            guide,
            genome,
            threshold,
        )

        atomic_results, windows, atomic_bytes = atomic_output
        block_results, _, block_bytes = block_output

        print(f"\nThreshold: {threshold}")
        print(f"Matches: {len(block_results):,}")

        print(f"Standard GPU: {standard_time:.6f} s")
        print(f"Global atomic: {atomic_time:.6f} s")
        print(f"Block aggregation: {block_time:.6f} s")

        print(f"Atomic speedup: {atomic_time / block_time:.3f}x")
        print(f"Block vs standard: {standard_time / block_time:.3f}x")

        print(f"Global atomic transfer: {atomic_bytes:,} bytes")
        print(f"Block aggregation transfer: {block_bytes:,} bytes")

        print(
            "All results correct:",
            standard_results == atomic_results == block_results,
        )

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()

