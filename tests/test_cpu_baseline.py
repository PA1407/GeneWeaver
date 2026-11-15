
import unittest

from src.alignment.cpu_baseline import find_exact_matches


class TestCPUAlignment(unittest.TestCase):

    def test_exact_match(self):
        self.assertEqual(
            find_exact_matches("ATGC", "GGATGCTTATGC"),
            [2, 8]
        )

    def test_no_match(self):
        self.assertEqual(
            find_exact_matches("AAAA", "TTTTGGGG"),
            []
        )

    def test_match_at_start(self):
        self.assertEqual(
            find_exact_matches("ATGC", "ATGCTTTT"),
            [0]
        )

    def test_match_at_end(self):
        self.assertEqual(
            find_exact_matches("ATGC", "TTTTATGC"),
            [4]
        )

    def test_multiple_matches(self):
        self.assertEqual(
            find_exact_matches("AA", "AAAAA"),
            [0, 1, 2, 3]
        )

    def test_case_insensitive(self):
        self.assertEqual(
            find_exact_matches("atgc", "GGATGCTT"),
            [2]
        )

    def test_empty_guide(self):
        with self.assertRaises(ValueError):
            find_exact_matches("", "ATGC")
            
    
    def test_mismatch_alignment(self):
        from src.alignment.cpu_baseline import (
            find_matches_with_mismatches
        )

        results = find_matches_with_mismatches(
            "ATGC",
            "ATGCATGA",
            max_mismatches=1
        )

        self.assertEqual(
            [item["mismatches"] for item in results],
            [0, 1]
        )

    def test_mismatch_threshold(self):
        from src.alignment.cpu_baseline import (
            find_matches_with_mismatches
        )

        results = find_matches_with_mismatches(
            "ATGC",
            "ATGATTTT",
            max_mismatches=1
        )

        self.assertEqual(len(results), 1)

    def test_negative_mismatch_threshold(self):
        from src.alignment.cpu_baseline import (
            find_matches_with_mismatches
        )

        with self.assertRaises(ValueError):
            find_matches_with_mismatches(
                "ATGC",
                "ATGC",
                max_mismatches=-1
            )



if __name__ == "__main__":
    unittest.main()
