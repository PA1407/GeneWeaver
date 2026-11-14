# GeneWeaver – Team Responsibilities and Task Allocation

## 1. Project Information

**Project Name:** GeneWeaver – GPU-Accelerated CRISPR Alignment Engine

**Domain:** Bioinformatics and High-Performance Computing (HPC)

**Project Duration:** 4 Weeks

**Team Structure:** 2 Active Members

## 2. Team Overview

The GeneWeaver project is divided into two major development areas:

1. HPC, CPU/GPU alignment and distributed workload processing.
2. Genomic data processing, biological scoring and user interface development.

Each member is responsible for implementing their assigned modules. Both members will collaborate during integration, testing and final delivery.

## 3. Member 1 – HPC and GPU Alignment

**Primary Responsibility:** CPU/GPU computation, performance optimization and distributed scheduling.

### Assigned Tasks

| No. | Task | Technology |
|---|---|---|
| 1 | Develop the CPU alignment baseline | Python |
| 2 | Implement genomic sequence comparison logic for the alignment engine | Python |
| 3 | Develop GPU-based alignment kernels | Numba CUDA |
| 4 | Handle CPU RAM and GPU VRAM transfers | CUDA |
| 5 | Validate GPU results against CPU results | Python, NumPy |
| 6 | Implement distributed workload scheduling | Dask |
| 7 | Benchmark CPU and GPU execution time | Python |
| 8 | Optimize GPU execution using shared memory | CUDA |

### Expected Deliverables

- CPU alignment implementation.
- CUDA GPU alignment implementation.
- CPU/GPU memory transfer functionality.
- Dask distributed scheduler.
- CPU-versus-GPU benchmark results.
- Performance optimization results.
- Unit tests for assigned modules.

### Main Files

```text
src/
├── alignment/
│   ├── cpu_baseline.py
│   └── gpu_engine.py
│
└── scheduler/
    └── distributed.py

benchmarks/
└── benchmark_alignment.py

tests/
```

## 4. Member 2 – Genomic Processing and Interface

**Primary Responsibility:** Genomic data preparation, biological scoring and dashboard development.

### Assigned Tasks

| No. | Task | Technology |
|---|---|---|
| 1 | Read and parse FASTA genomic files | BioPython |
| 2 | Clean and validate genomic sequences | Python |
| 3 | Divide large genomic datasets into chunks | Python |
| 4 | Implement biological scoring logic | Python |
| 5 | Evaluate PAM sequence proximity | Python |
| 6 | Develop the terminal user interface | Textual |
| 7 | Display GPU utilization and memory information | Textual |
| 8 | Display alignment results and mismatched base pairs | Textual |

### Expected Deliverables

- FASTA genomic parser.
- Sequence cleaning and chunking functionality.
- Biological scoring module.
- PAM-based scoring functionality.
- Textual dashboard.
- Alignment result visualization.
- Unit tests for assigned modules.

### Main Files

```text
src/
├── parser/
│   └── genomic_parser.py
│
├── scoring/
│   └── mutation_scoring.py
│
└── dashboard/
    └── app.py
```

## 5. Shared Responsibilities

Both members will work together on:

- Defining the interface between genomic processing and alignment modules.
- Integrating the parser with the CPU/GPU alignment engine.
- Testing the complete execution workflow.
- Identifying and resolving integration issues.
- Maintaining GitHub repository quality.
- Updating project documentation.
- Preparing the final report and presentation.
- Demonstrating the completed project.

## 6. Task Execution Timeline

| Week | Member 1 | Member 2 |
|---|---|---|
| Week 1 | CPU baseline planning and implementation; environment setup | FASTA parser and sequence preparation |
| Week 2 | CUDA implementation and memory transfers | Textual dashboard scaffold |
| Week 3 | Dask scheduling and GPU performance optimization | Biological scoring and PAM proximity |
| Week 4 | Shared-memory optimization and benchmarking | Dashboard polishing and result visualization |

Both members will participate in integration and testing throughout the project.

## 7. Git and Collaboration Workflow

The team will use GitHub for source-code management.

- Each member will work on their assigned development branch.
- Member 1 will maintain development work on the `parth` branch.
- Changes will be committed with descriptive commit messages.
- Team members will push completed changes regularly.
- Branch integration will be performed collaboratively.
- The `main` branch will contain the integrated project version.

## 8. Completion Criteria

### Member 1

- CPU alignment baseline works correctly.
- CUDA kernels execute successfully on the available NVIDIA GPU.
- GPU results are validated against CPU results.
- CPU/GPU performance benchmarks are available.
- Dask scheduling functionality is implemented and tested.
- GPU optimization work is documented.

### Member 2

- FASTA files can be parsed and prepared.
- Genomic sequences can be divided into chunks.
- Biological scoring and PAM proximity are implemented.
- The Textual dashboard displays relevant progress and results.

### Overall Project

- Assigned modules are integrated.
- End-to-end execution is tested.
- Documentation and benchmark results are available.
- Final presentation and demonstration are prepared.

## 9. Conclusion

The responsibilities are divided to support parallel development and clear ownership. Member 1 focuses on computational performance, GPU acceleration and distributed scheduling, while Member 2 focuses on genomic data processing, biological scoring and the terminal dashboard. Both members collaborate to integrate, test and deliver the complete GeneWeaver application.