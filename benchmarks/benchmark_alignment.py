import random

random.seed(42)


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import random
import time

from src.alignment.cpu_baseline import (
    find_matches_with_mismatches
)


def generate_dna_sequence(length):
    """Generate a random DNA sequence."""
    return "".join(
        random.choices("ATGC", k=length)
    )


def run_benchmark():

    guide = "ATGCGTACGTAGCTAGCTAG"

    genome_sizes = [
        10_000,
        100_000,
        500_000,
        1_000_000
    ]

    print("\nGENEWEAVER CPU ALIGNMENT BENCHMARK")
    print("-" * 55)

    for size in genome_sizes:

        genome = generate_dna_sequence(size)

# Insert known guide matches for correctness validation
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

        start = time.perf_counter()

        results = find_matches_with_mismatches(
            guide,
            genome,
            max_mismatches=2
        )

        end = time.perf_counter()

        execution_time = end - start

        print(f"Genome size: {size:,} bases")
        print(f"Matches found: {len(results)}")
        print(f"Execution time: {execution_time:.6f} seconds")
        print("Matching positions:", [
        item["position"] for item in results
   ])
        print("-" * 55)


if __name__ == "__main__":
    run_benchmark()
