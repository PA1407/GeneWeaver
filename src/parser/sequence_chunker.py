
def chunk_sequence(sequence, record_id, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    chunks = []

    for i in range(0, len(sequence), chunk_size):
        chunk = sequence[i:i + chunk_size]

        chunk_data = {
            "chunk_id": f"{record_id}_chunk_{len(chunks) + 1}",
            "record_id": record_id,
            "start": i,
            "end": i + len(chunk),
            "sequence": chunk
        }

        chunks.append(chunk_data)

    return chunks


if __name__ == "__main__":
    dna_sequence = "ATGCGTACGTAGCTAG"

    result = chunk_sequence(
        dna_sequence,
        "gene_001",
        4
    )

    print("GENEWEAVER - COORDINATE TRACKING")
    print("=" * 50)

    for chunk in result:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Sequence: {chunk['sequence']}")
        print(f"Start: {chunk['start']}")
        print(f"End: {chunk['end']}")
        print(f"Length: {chunk['end'] - chunk['start']}")
        print("-" * 50)

    print(f"Total chunks: {len(result)}")
