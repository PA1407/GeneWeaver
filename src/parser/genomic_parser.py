
from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path):
    """
    Parse a FASTA file and extract metadata and DNA sequences.

    Returns:
        A list of dictionaries containing:
        - record_id
        - description
        - sequence
        - length
    """

    fasta_path = Path(file_path)

    # Check whether the file exists
    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {file_path}"
        )

    records = []

    # Read multiple FASTA records
    for record in SeqIO.parse(str(fasta_path), "fasta"):

        sequence = str(record.seq).upper()

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


if __name__ == "__main__":

    fasta_file = "data/input/sample_genome.fasta"

    parsed_records = parse_fasta(fasta_file)

    print("\nGENEWEAVER - FASTA PARSER RESULTS")
    print("=" * 45)

    for record in parsed_records:
        print("Record ID:", record["record_id"])
        print("Description:", record["description"])
        print("DNA Sequence:", record["sequence"])
        print("Sequence Length:", record["length"])
        print("-" * 45)

    print("Total records:", len(parsed_records))
