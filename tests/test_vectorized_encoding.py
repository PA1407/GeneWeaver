
import unittest
import numpy as np

from src.alignment.gpu_engine import (
    encode_sequence,
    encode_sequence_vectorized
)


class TestVectorizedEncoding(unittest.TestCase):

    def test_matches_original_encoder(self):
        sequences = [
            "ATGC",
            "GGATGCTTATGC",
            "AAAA",
            "CCCC",
            "GGGG",
            "TTTT"
        ]

        for sequence in sequences:
            with self.subTest(sequence=sequence):
                np.testing.assert_array_equal(
                    encode_sequence(sequence),
                    encode_sequence_vectorized(sequence)
                )

    def test_lowercase_sequence(self):
        result = encode_sequence_vectorized("atgc")

        np.testing.assert_array_equal(
            result,
            np.array([0, 3, 2, 1], dtype=np.int32)
        )

    def test_invalid_dna_raises_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid DNA base: X"
        ):
            encode_sequence_vectorized("ATGX")

    def test_non_ascii_input_raises_error(self):
        with self.assertRaises(ValueError):
            encode_sequence_vectorized("ATGÇ")

    def test_output_dtype(self):
        result = encode_sequence_vectorized("ATGC")

        self.assertEqual(result.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
