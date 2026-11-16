
import numpy as np
from numba import cuda, int32

from src.alignment.gpu_engine import encode_sequence


@cuda.jit
def shared_alignment_kernel(guide, genome, mismatch_counts):

    position = cuda.grid(1)

    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    guide_length = guide.size
    total_windows = mismatch_counts.size

    # Shared memory tile:
    # One block loads its genome region once.
    shared_genome = cuda.shared.array(
        shape=0,
        dtype=int32
    )

    tile_start = cuda.blockIdx.x * block_size

    tile_length = block_size + guide_length - 1

    # Cooperatively load genome bases into shared memory.
    for i in range(thread_id, tile_length, block_size):

        genome_index = tile_start + i

        if genome_index < genome.size:
            shared_genome[i] = genome[genome_index]

    cuda.syncthreads()

    if position < total_windows:

        mismatches = 0

        local_position = thread_id

        for j in range(guide_length):

            if shared_genome[local_position + j] != guide[j]:
                mismatches += 1

        mismatch_counts[position] = mismatches


def gpu_alignment_shared(
    guide_sequence,
    genomic_sequence,
    max_mismatches=2
):

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    guide = encode_sequence(guide_sequence)
    genome = encode_sequence(genomic_sequence)

    if len(guide) > len(genome):
        return []

    total_windows = len(genome) - len(guide) + 1

    threads_per_block = 128

    tile_length = threads_per_block + len(guide) - 1

    d_guide = cuda.to_device(guide)
    d_genome = cuda.to_device(genome)

    d_mismatch_counts = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    shared_alignment_kernel[
        blocks_per_grid,
        threads_per_block,
        0,
        tile_length * np.dtype(np.int32).itemsize
    ](
        d_guide,
        d_genome,
        d_mismatch_counts
    )

    mismatch_counts = d_mismatch_counts.copy_to_host()

    matches = []

    for position, count in enumerate(mismatch_counts):

        if count <= max_mismatches:

            window = genomic_sequence[
                position:position + len(guide_sequence)
            ]

            matches.append({
                "position": position,
                "sequence": window,
                "mismatches": int(count)
            })

    return matches
