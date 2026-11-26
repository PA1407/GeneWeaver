
def chunk_sequence(sequence, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    chunks = []

    for i in range(0, len(sequence), chunk_size):
        chunk = sequence[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    dna_sequence = "ATGCGTACGTAGCTAG"

    result = chunk_sequence(dna_sequence, 4)

    print("GENEWEAVER - SEQUENCE CHUNKING")
    print("=" * 40)

    for index, chunk in enumerate(result, start=1):
        print(f"Chunk {index}: {chunk}")

    print(f"Total chunks: {len(result)}")
