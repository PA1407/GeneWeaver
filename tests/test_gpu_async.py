
import unittest

from src.alignment.gpu_engine import gpu_alignment
from src.alignment.gpu_async import gpu_alignment_async


class TestGPUAsync(unittest.TestCase):

    def test_async_matches_original_gpu(self):
        guide = "ATGC"
        genome = "GGATGCTTATGC"

        original = gpu_alignment(guide, genome, 1)
        asynchronous = gpu_alignment_async(guide, genome, 1)

        self.assertEqual(original, asynchronous)

    def test_async_exact_matches(self):
        results = gpu_alignment_async("ATGC", "GGATGCTTATGC", 0)

        self.assertEqual(
            [item["position"] for item in results],
            [2, 8]
        )

    def test_async_mismatch_detection(self):
        results = gpu_alignment_async("ATGC", "ATGT", 1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["mismatches"], 1)

    def test_async_no_matches(self):
        results = gpu_alignment_async("AAAA", "CCCC", 0)

        self.assertEqual(results, [])

    def test_empty_guide_raises_error(self):
        with self.assertRaises(ValueError):
            gpu_alignment_async("", "ATGC", 1)

    def test_negative_mismatch_threshold_raises_error(self):
        with self.assertRaises(ValueError):
            gpu_alignment_async("ATGC", "ATGC", -1)

    def test_invalid_dna_raises_error(self):
        with self.assertRaises(ValueError):
            gpu_alignment_async("ATGX", "ATGC", 1)

    def test_guide_longer_than_genome(self):
        results = gpu_alignment_async("ATGCTT", "ATGC", 2)

        self.assertEqual(results, [])
        
    
    def test_reusable_workspace_matches_original(self):
        from src.alignment.gpu_async import GPUAsyncWorkspace

        guide = "ATGC"
        genome = "GGATGCTTATGC"

        workspace = GPUAsyncWorkspace(
            len(guide),
            len(genome)
        )

        original = gpu_alignment(guide, genome, 1)

        first = gpu_alignment_async(
            guide, genome, 1, workspace
        )

        second = gpu_alignment_async(
            guide, genome, 1, workspace
        )

        self.assertEqual(original, first)
        self.assertEqual(first, second)

    def test_workspace_rejects_wrong_guide_length(self):
        from src.alignment.gpu_async import GPUAsyncWorkspace

        workspace = GPUAsyncWorkspace(4, 12)

        with self.assertRaises(ValueError):
            workspace.align("ATG", "GGATGCTTATGC", 1)

    def test_workspace_rejects_wrong_genome_length(self):
        from src.alignment.gpu_async import GPUAsyncWorkspace

        workspace = GPUAsyncWorkspace(4, 12)

        with self.assertRaises(ValueError):
            workspace.align("ATGC", "GGATGCT", 1)



if __name__ == "__main__":
    unittest.main()
