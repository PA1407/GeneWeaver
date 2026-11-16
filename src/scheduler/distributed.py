

from dask import delayed
from distributed import Client, LocalCluster

from src.alignment.cpu_baseline import find_matches_with_mismatches


def process_chunk(
    guide_sequence,
    chunk,
    start_position,
    max_mismatches=2
):
    """Process a genomic chunk and adjust match positions."""

    matches = find_matches_with_mismatches(
        guide_sequence,
        chunk,
        max_mismatches
    )

    adjusted_matches = []

    for match in matches:
        adjusted_matches.append({
            "position": match["position"] + start_position,
            "sequence": match["sequence"],
            "mismatches": match["mismatches"]
        })

    return adjusted_matches


def split_genome(genome_sequence, chunk_size, guide_length):
    """Split the genome into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    overlap = guide_length - 1
    chunks = []

    for start in range(0, len(genome_sequence), chunk_size):
        end = min(
            start + chunk_size + overlap,
            len(genome_sequence)
        )

        chunk = genome_sequence[start:end]
        chunks.append((chunk, start))

    return chunks


def distributed_alignment(
    guide_sequence,
    genome_sequence,
    chunk_size=100_000,
    max_mismatches=2,
    num_workers=2
):
    """
    Execute genomic alignment using Dask Distributed.

    LocalCluster manages workers and Client submits
    and gathers distributed tasks.
    """

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty.")

    if num_workers <= 0:
        raise ValueError("Number of workers must be positive.")

    if len(guide_sequence) > len(genome_sequence):
        return []

    chunks = split_genome(
        genome_sequence,
        chunk_size,
        len(guide_sequence)
    )

    tasks = []

    for chunk, start_position in chunks:
        task = delayed(process_chunk)(
            guide_sequence,
            chunk,
            start_position,
            max_mismatches
        )
        tasks.append(task)

    combined_results = []

    # Start a local Dask distributed cluster.
    with LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as client:

            # Submit tasks to distributed workers.
            futures = client.compute(tasks)

            # Collect results from workers.
            results = client.gather(futures)

            for chunk_results in results:
                combined_results.extend(chunk_results)

    combined_results.sort(
        key=lambda item: item["position"]
    )

    return combined_results

