
from time import perf_counter


def find_exact_matches(guide_sequence, genomic_sequence):
    """
    Find exact occurrences of a guide sequence
    inside a genomic sequence.

    Returns zero-based starting positions.
    """

    guide_sequence = guide_sequence.upper()
    genomic_sequence = genomic_sequence.upper()

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if len(guide_sequence) > len(genomic_sequence):
        return []

    matches = []

    for position in range(
        len(genomic_sequence) - len(guide_sequence) + 1
    ):
        window = genomic_sequence[
            position:position + len(guide_sequence)
        ]

        if window == guide_sequence:
            matches.append(position)

    return matches


def find_matches_with_mismatches(
    guide_sequence,
    genomic_sequence,
    max_mismatches=2
):
    """
    Find genomic windows with mismatches
    up to the specified maximum.
    """

    guide_sequence = guide_sequence.upper()
    genomic_sequence = genomic_sequence.upper()

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    if len(guide_sequence) > len(genomic_sequence):
        return []

    matches = []

    for position in range(
        len(genomic_sequence) - len(guide_sequence) + 1
    ):
        window = genomic_sequence[
            position:position + len(guide_sequence)
        ]

        mismatch_count = sum(
            guide_base != genome_base
            for guide_base, genome_base
            in zip(guide_sequence, window)
        )

        if mismatch_count <= max_mismatches:
            matches.append({
                "position": position,
                "sequence": window,
                "mismatches": mismatch_count
            })

    return matches

def benchmark_alignment(guide_sequence, genomic_sequence):
    """Measure CPU alignment execution time."""

    start_time = perf_counter()

    matches = find_exact_matches(
        guide_sequence,
        genomic_sequence
    )

    end_time = perf_counter()

    execution_time = end_time - start_time

    return matches, execution_time


if __name__ == "__main__":

    guide = "ATGC"
    genome = "GGATGCTTATGC"

    matches, execution_time = benchmark_alignment(
        guide,
        genome
    )

    print("Guide sequence:", guide)
    print("Genomic sequence:", genome)
    print("Matching positions:", matches)
    print(f"Execution time: {execution_time:.8f} seconds")
