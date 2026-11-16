
import unittest

from src.scheduler.distributed import distributed_alignment
from src.alignment.cpu_baseline import find_matches_with_mismatches


class TestDistributedAlignment(unittest.TestCase):

    def test_distributed_matches_cpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        expected = find_matches_with_mismatches(
            guide, genome, 2
        )
        actual = distributed_alignment(
            guide, genome, chunk_size=5
        )

        self.assertEqual(actual, expected)

    def test_match_at_chunk_boundary(self):
        guide = "ATGC"
        genome = "AAAAATGCAAAA"

        results = distributed_alignment(
            guide, genome, chunk_size=6
        )

        self.assertEqual(
            [item["position"] for item in results],
            [4]
        )

    def test_no_matches(self):
        results = distributed_alignment(
            "AAAA", "CCCCCCCC", chunk_size=4
        )
        self.assertEqual(results, [])

    def test_empty_guide(self):
        with self.assertRaises(ValueError):
            distributed_alignment("", "ATGC")

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            distributed_alignment(
                "ATGC", "ATGCATGC", chunk_size=0
            )

    def test_invalid_worker_count(self):
        with self.assertRaises(ValueError):
            distributed_alignment(
                "ATGC", "ATGCATGC", num_workers=0
            )


if __name__ == "__main__":
    unittest.main()
