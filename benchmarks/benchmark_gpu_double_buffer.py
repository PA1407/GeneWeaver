

import random
import statistics
import time

import numpy as np
from numba import cuda

from src.alignment.gpu_engine import (
    encode_sequence_vectorized,
    alignment_kernel,
    gpu_alignment,
    process_alignment_results,
)
from src.alignment.gpu_async import GPUAsyncWorkspace


THREADS_PER_BLOCK = 128


def generate_genome(size, guide):
    random.seed(42)
    genome = "".join(random.choices("ATGC", k=size))

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


def split_into_chunks(genome, chunk_size, guide_length):
    overlap = guide_length - 1

    for start in range(0, len(genome), chunk_size):
        end = min(
            start + chunk_size + overlap,
            len(genome)
        )

        yield genome[start:end], start


def measure(function, *args, runs=7):
    timings = []
    results = None

    for _ in range(runs):
        start = time.perf_counter()
        results = function(*args)
        timings.append(time.perf_counter() - start)

    return results, statistics.median(timings)


class DoubleBufferSlot:

    def __init__(
        self,
        guide_encoded,
        max_genome_length
    ):
        self.guide_length = len(guide_encoded)
        self.max_genome_length = max_genome_length

        self.max_windows = (
            max_genome_length - self.guide_length + 1
        )

        self.stream = cuda.stream()

        # Reusable pinned host memory.
        self.pinned_guide = cuda.pinned_array(
            self.guide_length,
            dtype=np.int32
        )

        self.pinned_genome = cuda.pinned_array(
            max_genome_length,
            dtype=np.int32
        )

        self.pinned_results = cuda.pinned_array(
            self.max_windows,
            dtype=np.int32
        )

        np.copyto(self.pinned_guide, guide_encoded)

        # Reusable GPU memory.
        self.d_guide = cuda.device_array(
            self.guide_length,
            dtype=np.int32
        )

        self.d_genome = cuda.device_array(
            max_genome_length,
            dtype=np.int32
        )

        self.d_results = cuda.device_array(
            self.max_windows,
            dtype=np.int32
        )

        # Transfer the guide once for this buffer.
        self.d_guide.copy_to_device(
            self.pinned_guide,
            stream=self.stream
        )

        self.stream.synchronize()

        self.pending = False
        self.chunk = None
        self.offset = 0
        self.windows = 0
        self.keep_before = None
        self.max_mismatches = 0


class DoubleBufferAligner:

    def __init__(
        self,
        guide,
        chunk_size,
        max_mismatches=0
    ):
        self.guide = guide
        self.chunk_size = chunk_size
        self.max_mismatches = max_mismatches

        guide_encoded = encode_sequence_vectorized(guide)

        max_genome_length = (
            chunk_size + len(guide) - 1
        )

        # Two independent reusable CUDA buffer slots.
        self.slots = [
            DoubleBufferSlot(
                guide_encoded,
                max_genome_length
            ),
            DoubleBufferSlot(
                guide_encoded,
                max_genome_length
            )
        ]

    def submit(self, slot, chunk, offset, is_last):
        genome_encoded = encode_sequence_vectorized(chunk)

        windows = (
            len(chunk) - len(self.guide) + 1
        )

        if windows <= 0:
            slot.pending = False
            return

        np.copyto(
            slot.pinned_genome[:len(chunk)],
            genome_encoded
        )

        # Queue host-to-device transfer.
        slot.d_genome[:len(chunk)].copy_to_device(
            slot.pinned_genome[:len(chunk)],
            stream=slot.stream
        )

        blocks = (
            windows + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK

        # Queue GPU alignment.
        alignment_kernel[
            blocks,
            THREADS_PER_BLOCK,
            slot.stream
        ](
            slot.d_guide,
            slot.d_genome,
            slot.d_results
        )

        # Queue device-to-host transfer.
        slot.d_results[:windows].copy_to_host(
            ary=slot.pinned_results[:windows],
            stream=slot.stream
        )

        slot.chunk = chunk
        slot.offset = offset
        slot.windows = windows

        # Non-final chunks own only positions before
        # the next chunk's starting position.
        slot.keep_before = (
            None if is_last else self.chunk_size
        )

        slot.max_mismatches = self.max_mismatches
        slot.pending = True

    def collect(self, slot, output):
        if not slot.pending:
            return

        # Wait only when this slot needs to be collected.
        slot.stream.synchronize()

        chunk_results = process_alignment_results(
            slot.pinned_results[:slot.windows],
            slot.chunk,
            len(self.guide),
            slot.max_mismatches
        )

        for result in chunk_results:
            local_position = result["position"]

            if (
                slot.keep_before is not None
                and local_position >= slot.keep_before
            ):
                continue

            result = result.copy()
            result["position"] += slot.offset
            output.append(result)

        slot.pending = False

    def align(self, genome):
        output = []

        for index, (chunk, offset) in enumerate(
            split_into_chunks(
                genome,
                self.chunk_size,
                len(self.guide)
            )
        ):
            slot = self.slots[index % 2]

            # Collect old work before reusing this slot.
            self.collect(slot, output)

            is_last = (
                offset + self.chunk_size >= len(genome)
            )

            self.submit(
                slot,
                chunk,
                offset,
                is_last
            )

        # Collect any remaining work.
        for slot in self.slots:
            self.collect(slot, output)

        output.sort(key=lambda item: item["position"])

        return output


def chunked_sequential_alignment(
    guide,
    genome,
    chunk_size,
    max_mismatches
):
    output = []

    for index, (chunk, offset) in enumerate(
        split_into_chunks(
            genome,
            chunk_size,
            len(guide)
        )
    ):
        workspace = GPUAsyncWorkspace(
            len(guide),
            len(chunk)
        )

        chunk_results = workspace.align(
            guide,
            chunk,
            max_mismatches
        )

        is_last = (
            offset + chunk_size >= len(genome)
        )

        for result in chunk_results:
            if not is_last and result["position"] >= chunk_size:
                continue

            result = result.copy()
            result["position"] += offset
            output.append(result)

    output.sort(key=lambda item: item["position"])

    return output



def chunked_reusable_alignment(
    guide,
    genome,
    chunk_size,
    max_mismatches
):
    guide_encoded = encode_sequence_vectorized(guide)

    max_genome_length = (
        chunk_size + len(guide) - 1
    )

    slot = DoubleBufferSlot(
        guide_encoded,
        max_genome_length
    )

    # Initialize one aligner with exactly one buffer slot.
    aligner = DoubleBufferAligner.__new__(
        DoubleBufferAligner
    )

    aligner.guide = guide
    aligner.chunk_size = chunk_size
    aligner.max_mismatches = max_mismatches
    aligner.slots = [slot]

    output = []

    for chunk, offset in split_into_chunks(
        genome,
        chunk_size,
        len(guide)
    ):
        is_last = (
            offset + chunk_size >= len(genome)
        )

        # Collect previous work before reusing the buffer.
        aligner.collect(slot, output)

        aligner.submit(
            slot,
            chunk,
            offset,
            is_last
        )

    # Collect the final pending operation.
    aligner.collect(slot, output)

    output.sort(key=lambda item: item["position"])

    return output


def main():
    guide = "ATGCGTACGTAGCTAGCTAA"

    sizes = [100_000, 500_000, 1_000_000]
    chunk_size = 100_000
    max_mismatches = 0

    print("GPU DOUBLE-BUFFER EXPERIMENT")
    print("=" * 70)

    for size in sizes:
        genome = generate_genome(size, guide)

        # Compile and warm up CUDA.
        expected = gpu_alignment(
            guide,
            genome,
            max_mismatches
        )


        sequential_results, sequential_time = measure(
            chunked_sequential_alignment,
            guide,
            genome,
            chunk_size,
            max_mismatches
        )

        reusable_results, reusable_time = measure(
            chunked_reusable_alignment,
            guide,
            genome,
            chunk_size,
            max_mismatches
        )

        double_buffer = DoubleBufferAligner(
            guide,
            chunk_size,
            max_mismatches
        )

        double_buffer.align(genome)

        double_results, double_time = measure(
            double_buffer.align,
            genome
        )

        print(f"\nGenome size: {size:,}")
        print(
            f"Number of chunks: "
            f"{(size + chunk_size - 1) // chunk_size}"
        )

        print(f"Fresh workspace:       {sequential_time:.6f} seconds")
        print(f"One reusable buffer:   {reusable_time:.6f} seconds")
        print(f"Two reusable buffers:  {double_time:.6f} seconds")

        print(
            "Fresh workspace correct:",
            sequential_results == expected
        )
        print(
            "One buffer correct:",
            reusable_results == expected
        )
        print(
            "Two buffers correct:",
            double_results == expected
        )


    print("\nDOUBLE-BUFFER EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
