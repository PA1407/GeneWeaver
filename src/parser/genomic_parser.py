
from pathlib import Path
from Bio import SeqIO


def parse_fasta(file_path):
    """
    Read a FASTA file and extract sequence information.

    Returns:
        A list of dictionaries containing:
        record_id, sequence and length.
    """

    fasta_path = Path(file_path)

    # Check whether the file exists
    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {file_path}"
        )

    records = []

    # Read FASTA records using BioPython
    for record in SeqIO.parse(str(fasta_path), "fasta"):

        sequence = str(record.seq).upper()

        record_data = {
            "record_id": record.id,
            "sequence": sequence,
            "length": len(sequence)
        }

        records.append(record_data)

    # Check whether the FASTA file contains records
    if not records:
        raise ValueError("No FASTA records found in the file.")

    return records


if __name__ == "__main__":

    fasta_file = "data/input/sample_genome.fasta"

    result = parse_fasta(fasta_file)

    for record in result:
        print("Record ID:", record["record_id"])
        print("Sequence:", record["sequence"])
        print("Length:", record["length"])
        print("-" * 40)
