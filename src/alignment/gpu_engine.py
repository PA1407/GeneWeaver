
import numpy as np
from numba import cuda


# Encode DNA bases as integers:
# A = 0, C = 1, G = 2, T = 3

BASE_ENCODING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def encode_sequence(sequence):
    """Convert a DNA sequence into integer values."""

    sequence = sequence.upper()

    try:
        return np.array(
            [BASE_ENCODING[base] for base in sequence],
            dtype=np.int32
        )
    except KeyError as error:
        raise ValueError(
            f"Invalid DNA base: {error.args[0]}"
        )


# Vectorized DNA encoding lookup table.
# Invalid characters are represented by 255.

DNA_LOOKUP = np.full(256, 255, dtype=np.uint8)

for base, value in BASE_ENCODING.items():
    DNA_LOOKUP[ord(base)] = value


def encode_sequence_vectorized(sequence):
    """
    Encode DNA using NumPy vectorized lookup.

    A = 0, C = 1, G = 2, T = 3
    """

    sequence = sequence.upper()

    try:
        sequence_bytes = sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"Invalid DNA base: {sequence[error.start]}"
        ) from error

    # Convert ASCII characters into NumPy integers.
    ascii_values = np.frombuffer(
        sequence_bytes,
        dtype=np.uint8
    )

    # Perform vectorized lookup.
    encoded = DNA_LOOKUP[ascii_values]

    # Check for invalid DNA characters.
    invalid_positions = np.flatnonzero(encoded == 255)

    if invalid_positions.size > 0:
        invalid_base = sequence[invalid_positions[0]]

        raise ValueError(
            f"Invalid DNA base: {invalid_base}"
        )

    return encoded.astype(np.int32)


@cuda.jit
def alignment_kernel(guide, genome, mismatch_counts):
    """Compare one genomic window per CUDA thread."""

    position = cuda.grid(1)

    if position < mismatch_counts.size:

        mismatches = 0

        for j in range(guide.size):
            if guide[j] != genome[position + j]:
                mismatches += 1

        mismatch_counts[position] = mismatches


def gpu_alignment(guide_sequence, genomic_sequence,
                  max_mismatches=2):

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError(
            "Maximum mismatches cannot be negative"
        )

    guide = encode_sequence_vectorized(guide_sequence)
    genome = encode_sequence_vectorized(genomic_sequence)

    if len(guide) > len(genome):
        return []

    total_windows = len(genome) - len(guide) + 1

    # Transfer input arrays from CPU RAM to GPU VRAM
    d_guide = cuda.to_device(guide)
    d_genome = cuda.to_device(genome)

    d_mismatch_counts = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    threads_per_block = 128

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    # Execute CUDA kernel
    alignment_kernel[
        blocks_per_grid,
        threads_per_block
    ](
        d_guide,
        d_genome,
        d_mismatch_counts
    )

    # Transfer results back to CPU RAM
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


def gpu_alignment_with_device_guide(
    d_guide,
    guide_sequence,
    genomic_sequence,
    max_mismatches=2
):
    """Align a genome chunk using an already-transferred GPU guide."""

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    genome = encode_sequence_vectorized(genomic_sequence)

    if len(guide_sequence) > len(genome):
        return []

    total_windows = len(genome) - len(guide_sequence) + 1

    d_genome = cuda.to_device(genome)

    d_mismatch_counts = cuda.device_array(
        total_windows,
        dtype=np.int32
    )

    threads_per_block = 128

    blocks_per_grid = (
        total_windows + threads_per_block - 1
    ) // threads_per_block

    alignment_kernel[
        blocks_per_grid,
        threads_per_block
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

if __name__ == "__main__":

    guide = "ATGC"
    genome = "GGATGCTTATGC"

    results = gpu_alignment(
        guide,
        genome,
        max_mismatches=1
    )

    print("Guide sequence:", guide)
    print("Genomic sequence:", genome)
    print("GPU alignment results:")

    for result in results:
        print(result)

    print("GPU alignment completed successfully!")
