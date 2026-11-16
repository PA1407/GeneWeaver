
import unittest

from src.alignment.cpu_baseline import (
    find_matches_with_mismatches
)

from src.alignment.gpu_engine import gpu_alignment


class TestGPUAlignment(unittest.TestCase):

    def test_gpu_matches_cpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        cpu_results = find_matches_with_mismatches(
            guide, genome, max_mismatches=1
        )

        gpu_results = gpu_alignment(
            guide, genome, max_mismatches=1
        )

        self.assertEqual(cpu_results, gpu_results)

    def test_gpu_mismatch_detection(self):
        guide = "ATGC"
        genome = "ATGCATGA"

        results = gpu_alignment(
            guide, genome, max_mismatches=1
        )

        self.assertEqual(
            [item["mismatches"] for item in results],
            [0, 1]
        )

    def test_gpu_no_match(self):
        results = gpu_alignment(
            "AAAA", "TTTTGGGG", max_mismatches=0
        )

        self.assertEqual(results, [])

    def test_gpu_invalid_guide(self):
        with self.assertRaises(ValueError):
            gpu_alignment("", "ATGC")

    def test_gpu_invalid_dna(self):
        with self.assertRaises(ValueError):
            gpu_alignment("ATGX", "ATGC")


if __name__ == "__main__":
    unittest.main()
