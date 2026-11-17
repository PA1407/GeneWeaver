
import unittest

from src.alignment.cpu_baseline import find_matches_with_mismatches
from src.scheduler.gpu_distributed import gpu_distributed_alignment


class TestGPUDistributedAlignment(unittest.TestCase):

    def test_gpu_distributed_matches_cpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        expected = find_matches_with_mismatches(
            guide, genome, 0
        )

        actual = gpu_distributed_alignment(
            guide,
            genome,
            chunk_size=5,
            max_mismatches=0,
            num_workers=1
        )

        self.assertEqual(actual, expected)

    def test_match_across_chunk_boundary(self):
        guide = "ATGC"
        genome = "A" * 4 + guide + "C" * 20

        results = gpu_distributed_alignment(
            guide,
            genome,
            chunk_size=6,
            max_mismatches=0,
            num_workers=1
        )

        self.assertEqual(
            [item["position"] for item in results],
            [4]
        )

    def test_no_matches(self):
        results = gpu_distributed_alignment(
            "AAAA",
            "CCCCCCCC",
            max_mismatches=0,
            num_workers=1
        )

        self.assertEqual(results, [])

    def test_empty_guide(self):
        with self.assertRaises(ValueError):
            gpu_distributed_alignment("", "ATGC")

    def test_negative_mismatch_threshold(self):
        with self.assertRaises(ValueError):
            gpu_distributed_alignment(
                "ATGC", "ATGC", max_mismatches=-1
            )

    def test_invalid_worker_count(self):
        with self.assertRaises(ValueError):
            gpu_distributed_alignment(
                "ATGC", "ATGC", num_workers=0
            )


if __name__ == "__main__":
    unittest.main()
