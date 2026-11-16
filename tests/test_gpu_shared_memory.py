
import unittest

from src.alignment.gpu_engine import gpu_alignment
from src.alignment.gpu_shared_memory import gpu_alignment_shared


class TestSharedMemoryAlignment(unittest.TestCase):

    def test_shared_matches_original_gpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        original = gpu_alignment(guide, genome, 2)
        optimized = gpu_alignment_shared(guide, genome, 2)

        self.assertEqual(optimized, original)

    def test_match_across_block_boundary(self):
        guide = "ATGC"
        genome = "A" * 126 + guide + "C" * 20

        original = gpu_alignment(guide, genome, 0)
        optimized = gpu_alignment_shared(guide, genome, 0)

        self.assertEqual(optimized, original)

    def test_large_genome(self):
        guide = "ATGC"
        genome = "A" * 1000 + guide + "C" * 2000

        original = gpu_alignment(guide, genome, 0)
        optimized = gpu_alignment_shared(guide, genome, 0)

        self.assertEqual(optimized, original)

    def test_no_matches(self):
        self.assertEqual(
            gpu_alignment_shared("AAAA", "CCCCCCCC", 0),
            []
        )

    def test_empty_guide(self):
        with self.assertRaises(ValueError):
            gpu_alignment_shared("", "ATGC")

    def test_invalid_dna(self):
        with self.assertRaises(ValueError):
            gpu_alignment_shared("ATGX", "ATGC")


if __name__ == "__main__":
    unittest.main()
