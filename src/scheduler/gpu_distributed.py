
from dask import delayed
from distributed import Client, LocalCluster, get_worker

from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence,
    gpu_alignment_with_device_guide
)
from src.scheduler.distributed import split_genome


# Worker-local cache for GPU guide arrays.
_GPU_GUIDE_CACHE = {}


def get_device_guide(guide_sequence):
    """Transfer a guide to GPU memory once per worker."""

    if guide_sequence not in _GPU_GUIDE_CACHE:
        encoded_guide = encode_sequence(guide_sequence)
        _GPU_GUIDE_CACHE[guide_sequence] = cuda.to_device(
            encoded_guide
        )

    return _GPU_GUIDE_CACHE[guide_sequence]


def process_gpu_chunk(
    guide_sequence,
    chunk,
    start_position,
    max_mismatches=2
):
    """Process a genomic chunk using a reusable GPU guide."""

    # This function is expected to execute inside a Dask worker.
    get_worker()

    d_guide = get_device_guide(guide_sequence)

    matches = gpu_alignment_with_device_guide(
        d_guide,
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
    chunk_size=250_000,
    max_mismatches=2,
    num_workers=1,
    client=None
):
    """Execute GPU alignment through Dask Distributed."""

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

    if client is not None:
        return execute(client)

    with LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None
    ) as cluster:

        with Client(cluster) as local_client:
            return execute(local_client)
