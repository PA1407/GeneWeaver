import pytest

from src.parser.genomic_parser import (
    parse_fasta,
    validate_dna_sequence
)


def test_valid_dna_sequence():
    result = validate_dna_sequence("ATGC")

    assert result == (True, set())


def test_invalid_dna_sequence():
    result = validate_dna_sequence("ATGX")

    assert result[0] is False
    assert result[1] == {"X"}


def test_parse_valid_fasta(tmp_path):
    fasta_file = tmp_path / "sample.fasta"

    fasta_file.write_text(
        ">gene_001 chromosome_region_1\n"
        "ATGCGTAC\n"
    )

    records = parse_fasta(fasta_file)

    assert len(records) == 1
    assert records[0]["record_id"] == "gene_001"
    assert records[0]["sequence"] == "ATGCGTAC"
    assert records[0]["length"] == 8


def test_missing_fasta_file(tmp_path):
    fasta_file = tmp_path / "missing.fasta"

    with pytest.raises(FileNotFoundError):
        parse_fasta(fasta_file)


def test_empty_fasta_file(tmp_path):
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    with pytest.raises(ValueError):
        parse_fasta(fasta_file)


def test_invalid_fasta_sequence(tmp_path):
    fasta_file = tmp_path / "invalid.fasta"

    fasta_file.write_text(
        ">gene_invalid\n"
        "ATGCXYZ\n"
    )

    with pytest.raises(ValueError):
        parse_fasta(fasta_file)