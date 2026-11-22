

import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence_vectorized,
    alignment_kernel,
    process_alignment_results
)


class GPUAsyncWorkspace:
    """
    Reusable host and GPU buffers for asynchronous alignment.

    A workspace is created for specific guide and genome lengths.
    It can be reused for different sequences of those same lengths.
    """

    def __init__(self, guide_length, genome_length):

        self.guide_length = guide_length
        self.genome_length = genome_length

        self.total_windows = (
            genome_length - guide_length + 1
        )

        self.threads_per_block = 128

        self.blocks_per_grid = (
            self.total_windows + self.threads_per_block - 1
        ) // self.threads_per_block

        # Create one reusable CUDA stream.
        self.stream = cuda.stream()

        # Allocate pinned host memory once.
        self.pinned_guide = cuda.pinned_array(
            guide_length,
            dtype=np.int32
        )

        self.pinned_genome = cuda.pinned_array(
            genome_length,
            dtype=np.int32
        )

        self.pinned_results = cuda.pinned_array(
            self.total_windows,
            dtype=np.int32
        )

        # Allocate GPU memory once.
        self.d_guide = cuda.device_array(
            guide_length,
            dtype=np.int32
        )

        self.d_genome = cuda.device_array(
            genome_length,
            dtype=np.int32
        )

        self.d_results = cuda.device_array(
            self.total_windows,
            dtype=np.int32
        )

    def align(self, guide_sequence, genomic_sequence,
              max_mismatches=2):

        if len(guide_sequence) != self.guide_length:
            raise ValueError("Guide length does not match workspace.")

        if len(genomic_sequence) != self.genome_length:
            raise ValueError("Genome length does not match workspace.")

        if not guide_sequence:
            raise ValueError("Guide sequence cannot be empty.")

        if max_mismatches < 0:
            raise ValueError(
                "Maximum mismatches cannot be negative"
            )

        # Encode input sequences.
        guide = encode_sequence_vectorized(guide_sequence)
        genome = encode_sequence_vectorized(genomic_sequence)

        # Copy encoded sequences into reusable pinned buffers.
        np.copyto(self.pinned_guide, guide)
        np.copyto(self.pinned_genome, genome)

        # Enqueue asynchronous host-to-device transfers.
        self.d_guide.copy_to_device(
            self.pinned_guide,
            stream=self.stream
        )

        self.d_genome.copy_to_device(
            self.pinned_genome,
            stream=self.stream
        )

        # Launch the alignment kernel on the same stream.
        alignment_kernel[
            self.blocks_per_grid,
            self.threads_per_block,
            self.stream
        ](
            self.d_guide,
            self.d_genome,
            self.d_results
        )

        # Enqueue asynchronous device-to-host transfer.
        self.d_results.copy_to_host(
            ary=self.pinned_results,
            stream=self.stream
        )

        # Ensure transfers and kernel execution are complete.
        self.stream.synchronize()

        return process_alignment_results(
            self.pinned_results,
            genomic_sequence,
            self.guide_length,
            max_mismatches
        )



def gpu_alignment_async(
    guide_sequence,
    genomic_sequence,
    max_mismatches=2,
    workspace=None
):
    """
    Perform asynchronous GPU alignment.

    Pass a reusable workspace to avoid repeated memory allocations.
    """

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError(
            "Maximum mismatches cannot be negative"
        )

    if len(guide_sequence) > len(genomic_sequence):
        return []

    if workspace is None:
        workspace = GPUAsyncWorkspace(
            len(guide_sequence),
            len(genomic_sequence)
        )

    return workspace.align(
        guide_sequence,
        genomic_sequence,
        max_mismatches
    )
