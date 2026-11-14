# GeneWeaver – System Architecture

## 1. Project Overview

**Project Name:** GeneWeaver – GPU-Accelerated CRISPR Alignment Engine

**Domain:** Bioinformatics and High-Performance Computing (HPC)

GeneWeaver is designed to accelerate genomic sequence alignment and identify potential CRISPR off-target locations. Traditional Python-based sequence comparisons can become computationally expensive when processing large genomic datasets.

The project uses CPU-based alignment as a baseline and NVIDIA CUDA with Numba to perform parallel sequence comparisons on the GPU. Dask is planned to support distributed workload execution across multiple GPUs or machines when available.

## 2. System Objectives

- Develop a CPU-based genomic alignment baseline.
- Implement GPU-accelerated alignment using Numba CUDA.
- Transfer genomic data between CPU memory and GPU memory.
- Distribute computational workloads using Dask.
- Benchmark CPU and GPU execution performance.
- Optimize GPU execution using CUDA techniques, including shared memory.

## 3. High-Level Architecture

The system is organized into the following major components:

### 3.1 Genomic Parser

**Technology:** Python and BioPython

Responsibilities:
- Read genomic sequences from FASTA files.
- Parse and clean genomic sequence data.
- Divide large genomic datasets into manageable chunks.
- Prepare sequence data for alignment processing.

**Ownership:** Member 2

### 3.2 CPU Alignment Engine

**Technology:** Python

Responsibilities:
- Implement the initial sequence alignment logic.
- Perform sequence comparisons using CPU processing.
- Establish a baseline for execution-time measurement.
- Provide reference results for validating GPU output.

**Ownership:** Member 1

### 3.3 GPU Alignment Engine

**Technology:** Python, Numba and CUDA

Responsibilities:
- Convert the CPU alignment logic into GPU-compatible CUDA kernels.
- Execute parallel sequence comparisons on the NVIDIA GPU.
- Manage data transfers between CPU RAM and GPU VRAM.
- Validate GPU results against CPU results.
- Optimize GPU execution performance.

**Ownership:** Member 1

### 3.4 Distributed Scheduler

**Technology:** Dask

Responsibilities:
- Divide computational workloads into tasks.
- Coordinate the execution of alignment workloads.
- Support distributed execution across multiple GPUs or machines when available.
- Collect and organize computation results.

**Ownership:** Member 1

### 3.5 Biological Scoring Module

**Technology:** Python

Responsibilities:
- Evaluate potential alignment results.
- Apply biological scoring criteria.
- Consider PAM sequence proximity as specified in the project requirements.

**Ownership:** Member 2

### 3.6 TUI Dashboard

**Technology:** Textual

Responsibilities:
- Display alignment progress.
- Present GPU memory and utilization information.
- Display processing status and alignment results.
- Highlight mismatched base pairs.

**Ownership:** Member 2

## 4. Execution Workflow

The planned system workflow is:

1. The user provides genomic data in FASTA format.
2. The genomic parser reads and prepares the sequence data.
3. The prepared sequences are passed to the alignment engine.
4. The CPU engine performs baseline alignment when required.
5. The GPU engine transfers the required data to GPU memory and executes CUDA kernels.
6. Dask coordinates distributed workloads when distributed execution is enabled.
7. Alignment results are passed to the biological scoring module.
8. The dashboard displays progress and results.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| BioPython | FASTA parsing and genomic data processing |
| NumPy | Numerical arrays and data handling |
| Numba | GPU kernel development |
| NVIDIA CUDA | Parallel GPU computation |
| Dask | Distributed workload scheduling |
| Textual | Terminal-based user interface |
| Git and GitHub | Version control and collaboration |

## 6. Member Responsibilities

### Member 1 – HPC and GPU Alignment

- Develop the CPU alignment baseline.
- Implement the CUDA GPU alignment engine.
- Manage CPU-to-GPU and GPU-to-CPU memory transfers.
- Develop the Dask distributed scheduler.
- Perform CPU and GPU benchmarking.
- Optimize CUDA execution using shared memory.

### Member 2 – Genomic Processing and Interface

- Implement FASTA parsing and sequence cleaning.
- Develop genomic data chunking.
- Implement biological scoring, including PAM proximity.
- Develop the Textual dashboard.
- Display alignment results and GPU information.

Both members will coordinate during integration and testing.

## 7. Performance Evaluation

The project will evaluate performance through:

- CPU execution time.
- GPU execution time.
- CPU-versus-GPU speedup.
- GPU memory usage.
- Correctness of CPU and GPU alignment results.
- Workload distribution and execution behavior.

The project brief specifies a target of at least 100x speedup for alignment calculations during the mid-review. This is a performance target to evaluate, not a guaranteed result.

## 8. Development and Version Control

Git and GitHub are used to manage project development.

- The `main` branch maintains the primary project version.
- The `parth` branch is used for Member 1 development.
- Team members work on their respective modules.
- Changes are committed and pushed regularly.
- Integration and testing are performed collaboratively.

## 9. Conclusion

GeneWeaver combines CPU-based sequence alignment, GPU acceleration and distributed workload scheduling to improve the computational performance of genomic sequence comparisons. The architecture separates genomic parsing, alignment, scheduling, scoring and visualization into independent components, allowing team members to develop and test their responsibilities before integration.