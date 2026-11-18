

import unittest
from numba import cuda

from src.alignment.cpu_baseline import (
    find_matches_with_mismatches
)

from src.alignment.gpu_engine import (
    gpu_alignment,
    gpu_alignment_with_device_guide,
    encode_sequence
)


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

    def test_reusable_guide_matches_original_gpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        d_guide = cuda.to_device(encode_sequence(guide))

        original_results = gpu_alignment(
            guide, genome, max_mismatches=1
        )

        optimized_results = gpu_alignment_with_device_guide(
            d_guide, guide, genome, max_mismatches=1
        )

        self.assertEqual(original_results, optimized_results)

    def test_reusable_guide_matches_cpu(self):
        guide = "ATGC"
        genome = "ATGCATGA"

        d_guide = cuda.to_device(encode_sequence(guide))

        cpu_results = find_matches_with_mismatches(
            guide, genome, max_mismatches=1
        )

        optimized_results = gpu_alignment_with_device_guide(
            d_guide, guide, genome, max_mismatches=1
        )

        self.assertEqual(cpu_results, optimized_results)


if __name__ == "__main__":
    unittest.main()
