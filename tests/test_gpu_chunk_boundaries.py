

import unittest

from src.alignment.gpu_engine import gpu_alignment

from benchmarks.benchmark_gpu_double_buffer import (
    DoubleBufferAligner,
    chunked_sequential_alignment,
    chunked_reusable_alignment,
)


class TestGPUChunkBoundaries(unittest.TestCase):

    def setUp(self):
        self.guide = "ATGCGTACGTAGCTAGCTAA"
        self.chunk_size = 100_000
        self.genome_size = 200_100

        genome = list("A" * self.genome_size)

        self.expected_positions = [
            99_970,
            99_990,
            100_010,
            100_030,
        ]

        for position in self.expected_positions:
            genome[position:position + len(self.guide)] = self.guide

        self.genome = "".join(genome)

        self.expected = gpu_alignment(
            self.guide,
            self.genome,
            0
        )

    def test_direct_gpu_expected_positions(self):
        positions = [
            item["position"] for item in self.expected
        ]

        self.assertEqual(
            positions,
            self.expected_positions
        )

    def test_sequential_chunked_alignment(self):
        results = chunked_sequential_alignment(
            self.guide,
            self.genome,
            self.chunk_size,
            0
        )

        self.assertEqual(results, self.expected)

    def test_single_reusable_buffer_alignment(self):
        results = chunked_reusable_alignment(
            self.guide,
            self.genome,
            self.chunk_size,
            0
        )

        self.assertEqual(results, self.expected)

    def test_double_buffer_alignment(self):
        aligner = DoubleBufferAligner(
            self.guide,
            self.chunk_size,
            0
        )

        results = aligner.align(self.genome)

        self.assertEqual(results, self.expected)


if __name__ == "__main__":
    unittest.main()
