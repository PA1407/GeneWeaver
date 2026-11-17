

from dask import delayed
from distributed import Client, LocalCluster

from src.alignment.gpu_engine import gpu_alignment
from src.scheduler.distributed import split_genome


def process_gpu_chunk(
    guide_sequence,
    chunk,
    start_position,
    max_mismatches=2
):
    """Process one genomic chunk using CUDA GPU alignment."""

    matches = gpu_alignment(
        guide_sequence,
        chunk,
        max_mismatches
    )

    return [
        {
            "position": match["position"] + start_position,
            "sequence": match["sequence"],
            "mismatches": match["mismatches"]
        }
        for match in matches
    ]


def gpu_distributed_alignment(
    guide_sequence,
    genome_sequence,
    chunk_size=100_000,
    max_mismatches=2,
    num_workers=1,
    client=None
):
    """
    Execute GPU alignment through Dask Distributed.

    An existing Client can be supplied to reuse a cluster.
    Otherwise, this function creates and closes its own cluster.
    """

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty.")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative.")

    if num_workers <= 0:
        raise ValueError("Number of workers must be positive.")

    if len(guide_sequence) > len(genome_sequence):
        return []

    chunks = split_genome(
        genome_sequence,
        chunk_size,
        len(guide_sequence)
    )

    tasks = [
        delayed(process_gpu_chunk)(
            guide_sequence,
            chunk,
            start_position,
            max_mismatches
        )
        for chunk, start_position in chunks
    ]

    def execute(client_instance):
        futures = client_instance.compute(tasks)
        results = client_instance.gather(futures)

        combined_results = []

        for chunk_results in results:
            combined_results.extend(chunk_results)

        combined_results.sort(
            key=lambda item: item["position"]
        )

        return combined_results

    # Reuse an existing cluster when a Client is supplied.
    if client is not None:
        return execute(client)

    # Otherwise, manage a cluster within this function.
    with LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as local_client:
            return execute(local_client)
