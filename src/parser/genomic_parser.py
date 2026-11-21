from pathlib import Path
from Bio import SeqIO


def validate_dna_sequence(sequence):
    valid_bases = {"A", "T", "G", "C"}
    sequence = sequence.upper()

    invalid_bases = set(sequence) - valid_bases

    if invalid_bases:
        return False, invalid_bases

    return True, set()


def parse_fasta(file_path):
    fasta_path = Path(file_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {file_path}"
        )

    try:
        records = []

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            sequence = str(record.seq).upper()

            is_valid, invalid_bases = validate_dna_sequence(sequence)

            if not is_valid:
                raise ValueError(
                    f"Invalid DNA sequence in {record.id}: {invalid_bases}"
                )

            record_data = {
                "record_id": record.id,
                "description": record.description,
                "sequence": sequence,
                "length": len(sequence)
            }

            records.append(record_data)

        if not records:
            raise ValueError("No FASTA records found.")

        return records

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to read FASTA file: {error}"
        ) from error

    if not records:
        raise ValueError("No FASTA records found.")

    return records


if __name__ == "__main__":
    fasta_file = "data/input/sample_genome.fasta"

    parsed_records = parse_fasta(fasta_file)

    print("GENEWEAVER - FASTA PARSER RESULTS")
    print("=" * 45)

    for record in parsed_records:
        print(f"Record ID: {record['record_id']}")
        print(f"Description: {record['description']}")
        print(f"DNA Sequence: {record['sequence']}")
        print(f"Sequence Length: {record['length']}")
        print("-" * 45)

    print(f"Total records: {len(parsed_records)}")