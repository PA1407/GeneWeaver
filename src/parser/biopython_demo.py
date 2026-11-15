
from Bio import SeqIO

# Location of the FASTA file
fasta_file = "data/input/sample_genome.fasta"

# Read the FASTA records
records = SeqIO.parse(fasta_file, "fasta")

# Display each record
for record in records:
    print("Record ID:", record.id)
    print("Sequence:", record.seq)
    print("Sequence Length:", len(record.seq))
    print("-------------------------")
