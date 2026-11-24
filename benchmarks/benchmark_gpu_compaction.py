
import statistics
import time

import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence_vectorized,
    alignment_kernel,
    gpu_alignment,
)


@cuda.jit
def compact_alignment_kernel(
    guide,
    genome,
    max_mismatches,
    output_positions,
    output_mismatches,
    match_count,
):
    position = cuda.grid(1)

    total_windows = genome.size - guide.size + 1

    if position < total_windows:
        mismatches = 0

        for j in range(guide.size):
            if guide[j] != genome[position + j]:
                mismatches += 1

        if mismatches <= max_mismatches:
            output_index = cuda.atomic.add(
                match_count,
                0,
                1
            )

            output_positions[output_index] = position
            output_mismatches[output_index] = mismatches


def gpu_alignment_compact(
    guide_sequence,
    genomic_sequence,
    max_mismatches=2
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

    d_guide = cuda.to_device(guide)
    d_genome = cuda.to_device(genome)

    # Worst-case output capacity: every window can match.
    d_positions = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    d_mismatches = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    d_match_count = cuda.to_device(
        np.zeros(1, dtype=np.int32)
    )

    threads_per_block = 128

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    compact_alignment_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_guide,
        d_genome,
        max_mismatches,
        d_positions,
        d_mismatches,
        d_match_count
    )

    # Transfer the number of matches first.
    match_count = int(d_match_count.copy_to_host()[0])

    # Transfer only the populated output entries.
    if match_count == 0:
        return [], total_windows, 1

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
            "mismatches": int(mismatches[i])
        })

    results.sort(key=lambda item: item["position"])

    # Two output arrays plus one count scalar.
    transferred_bytes = (
        match_count * 2 * np.dtype(np.int32).itemsize
        + np.dtype(np.int32).itemsize
    )

    return results, total_windows, transferred_bytes


def measure(function, *args, runs=7):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function(*args)
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


def generate_genome(size, guide):
    rng = np.random.default_rng(42)

    genome = "".join(
        rng.choice(
            np.array(list("ATGC")),
            size=size
        )
    )

    # Insert predictable exact matches.
    genome = (
        genome[:100]
        + guide
        + genome[100 + len(guide):]
    )

    genome = (
        genome[:500]
        + guide
        + genome[500 + len(guide):]
    )

    return genome


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"
    sizes = [100_000, 500_000, 1_000_000]
    max_mismatches = 0

    print("GPU RESULT COMPACTION EXPERIMENT")
    print("=" * 70)

    for size in sizes:
        genome = generate_genome(size, guide)

        # Warm up CUDA compilation.
        expected = gpu_alignment(
            guide,
            genome,
            max_mismatches
        )

        compact_result = gpu_alignment_compact(
            guide,
            genome,
            max_mismatches
        )

        # Warm up before timing.
        gpu_alignment_compact(
            guide,
            genome,
            max_mismatches
        )

        baseline_results, baseline_time = measure(
            gpu_alignment,
            guide,
            genome,
            max_mismatches
        )

        compact_output, compact_time = measure(
            gpu_alignment_compact,
            guide,
            genome,
            max_mismatches
        )

        compact_results, windows, transferred_bytes = (
            compact_output
        )

        baseline_transfer_bytes = windows * 4

        print(f"\nGenome size: {size:,}")
        print(f"Windows processed: {windows:,}")
        print(f"Matches found: {len(compact_results)}")

        print(
            f"Standard GPU time: {baseline_time:.6f} seconds"
        )

        print(
            f"Compaction GPU time: {compact_time:.6f} seconds"
        )

        print(
            f"Standard result transfer: "
            f"{baseline_transfer_bytes:,} bytes"
        )

        print(
            f"Compacted result transfer: "
            f"{transferred_bytes:,} bytes"
        )

        print(
            "Results correct:",
            baseline_results == compact_results == expected
        )

        if transferred_bytes > 0:
            print(
                f"Transfer reduction: "
                f"{(1 - transferred_bytes / baseline_transfer_bytes) * 100:.2f}%"
            )

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
