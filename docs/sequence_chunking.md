
# Sequence Chunking

## What is Sequence Chunking?
Sequence Chunking is the process of dividing a large DNA sequence into smaller parts called chunks.

## Why is it needed?
1. Memory Management
2. Parallel Processing
3. GPU Processing
4. Performance
5. Scalability

## Important Terms
- Original Sequence: Complete DNA sequence.
- Chunk: Smaller part of the original sequence.
- Chunk Size: Number of bases in each chunk.
- Start Position: Position where a chunk begins.
- End Position: Position where a chunk ends.
- Overlap: Shared bases between consecutive chunks.

## Example

Original Sequence: ATGCGTACGTAGCTAG

Chunk Size: 4

- Chunk 1: ATGC
- Chunk 2: GTAC
- Chunk 3: GTAG
- Chunk 4: CTAG

## GeneWeaver Implementation Requirements
- Divide genomic sequences into smaller chunks.
- Generate a unique ID for every chunk.
- Maintain the original record ID.
- Store start and end positions.
- Handle overlapping regions.
