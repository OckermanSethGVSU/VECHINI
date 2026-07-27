# VECHINI

VECHINI (**V**ector Database **E**valuation and **C**haracterization for **H**PC **I**nsertion, Indexing, and **N**earest-neighbor Search) is the open-source research artifact accompanying our [report on vector database performance on HPC systems](https://arxiv.org/abs/2606.08950). This work-in-progress tool is designed to automate vector database (VDB) experiments on PBS HPC platforms that support Apptainer and MPI. Using `pbs_submit_manager.sh` and configuration files, users can launch customized single- and multi-node experiments to evaluate key aspects of VDB performance. The goal is to help users answer questions about performance on their target HPC platform. We focus on a few of the key stages of the VDB lifecycle, including data ingestion, index construction, and query serving. Currently, we support [Qdrant](https://qdrant.tech/), [Milvus](https://milvus.io/), and [Weaviate](https://weaviate.io/). 


<br> <br>

<p align="center">
  <img src="docs/figures/GithubHorizontalVectorDBLifecycle.png" alt="Benchmark architecture" width="600">
</p>





## Start Here
Using simple configurations files or command line parameters, (e.g., [Weaviate query experiment file](weaviate/sampleConfigs/query_core_testing.env)), users can specify key parameters that the submit manager will use to generate an experimental directory. If the submit manager is called with `--generate-only` it will only generate the directory, otherwise it will submit the job to the PBS queue specified in the configuration file. To view the supported experimental parameters, consult the [configuration page](config-reference.md) in the documentation. 


```bash
./pbs_submit_manager.sh --help
./pbs_submit_manager.sh --help --engine qdrant
./pbs_submit_manager.sh --engine milvus --config path/to/run.env
./pbs_submit_manager.sh --generate-only --engine weaviate --config path/to/run.env
./pbs_submit_manager.sh --generate-only --engine weaviate --set QUERY_BATCH_SIZE=32
```

## Task-Driven Benchmarking  
We designed our benchmark to allow us to test different common VDB tasks. Each task isolates a different part of vector database behavior so performance can be evaluated more cleanly across engines. 

* `INSERT`: measures data ingestion throughput, allowing users to test different storage mediums, batch sizes, clients per worker, and total workers

* `INDEX`: isolates index construction cost after the data has been inserted, enabling users to test different storage mediums, virtual-core availability, index parameters, and numbers of workers.

* `QUERY`: measures search latency and retrieval throughput on an already indexed collection, allowing users to test a variety of query parameters, performance with different numbers of cores, and scaling across multiple nodes.

* `MIXED`: evaluates concurrent read/write behavior under combined insert and query load. This task is in the process of being expanded upon. 


## A Focus on Modular Design
This repository is designed to be extensible to allow for new vector databases to be added. The core workflow is organized around a unified submit manager and a shared schema-driven configuration layer, so new engines can plug into the existing generation, staging, and submission flow. To add a new vector database, create a new folder and mirror the existing engine layout with the following files defined:

  - `engine.sh`
    * Implements the engine contract used by `pbs_submit_manager.sh`. This is where the engine registers itself, validates config, stages files into generated run directories, and tells the unified submit layer how to run the workflow.

  - `schema.sh`
    * Defines the engine-specific configuration variables and defaults. This is the source of truth for parameters exposed through `--help --engine <name>` and for any required engine-specific runtime settings.

  - `main.sh`
    * The PBS runtime entrypoint copied into generated run directories. It is responsible for launching the database on the target cluster, waiting for readiness, and invoking the client workload for the selected task.

  - `local_main.sh`
    * The local runtime entrypoint, if local mode is supported. This usually launches a local container or standalone process and then runs the same staged client workflow without PBS.

  - `clients/`
    * Contains the client source and build helpers used to drive insert, index, query, or mixed workloads. The engine should stage the built binaries from here into generated run directories.

  - `scripts/`
    * Holds engine-specific helpers such as collection creation, readiness checks, status inspection, result summarization, or setup utilities that are needed by the runtime scripts.

## Built in PBS Utilties
The unified submit manager also includes queue-watching functionality. It can monitor a set of specified PBS queues and automatically submit a job when an opening becomes available. This lets users stay within machine-specific queue limits while still defining all experimental configurations upfront, rather than repeatedly checking when experiments finish to then submit new jobs. This is separate from the experiment configuration schema and does not change how parameter combinations are generated. It only affects which PBS queue each already-generated run is submitted to.

Use these flags together on `pbs_submit_manager.sh`:

```bash
./pbs_submit_manager.sh \
  --engine qdrant \
  --config path/to/run.env \
  --queue-candidates debug,debug-scaling,capacity \
  --queue-limits 1,1,5 \
  --submit-username myuser
```

- `--queue-candidates`: ordered comma-separated list of PBS queues to try
- `--queue-limits`: ordered comma-separated list of per-queue job caps for the specified user
- `--submit-username`: required username used for `qstat -u ...` queue counting

Notes:

- Queue order is preference order. The launcher checks candidates from left to right and submits to the first queue where the user is below the configured limit.
- If no candidate queue is open, the launcher waits 60 seconds and checks again.
- With `--generate-only`, the manager does not poll PBS. It uses the first queue candidate when writing `submit.sh`.


## Scientfic Datasets
While vector search has been extensively studied in industry applications, less is known about its use in scientific domains (e.g, molecular search, meteorological trajectory detection, literature-driven hypothesis generation). This repository aims to address that gap by curating scientific embedding/query datasets for public use.

| Dataset | Source data | Original use case | Number of embeddings | Dimensionality | Intended distance metric | Embedding model |
|---|---|---|---:|---:|---|---|
| [Pes2o-VE](https://acdc.alcf.anl.gov/mdf/detail/e31e3225-7f75-4313-9983-f8b75811405f-1.0/) | [PeS2o text corpus](https://github.com/allenai/pes2o) | Scientific document retrieval | ~88 million | 2560 | Cosine | Qwen3-Embedding-4B |



## Docs

- [Repo overview](docs/overview.md)
- [Unified submit workflow](docs/unified-submit.md)
- [Config reference](docs/config-reference.md)
- [Qdrant engine docs](docs/engines/qdrant.md)
- [Milvus engine docs](docs/engines/milvus.md)
- [Weaviate engine docs](docs/engines/weaviate.md)


## Citation
```bibtex
@misc{ockerman2026coreshurtsvectordatabase,
  title         = {When More Cores Hurts: The Vector Database Scaling Paradox in HPC},
  author        = {Seth Ockerman and Song Young Oh and Amal Gueroudji and Rochana Chaturvedi and Philip Carns and Nicholas Chia and Matthieu Dorier and Robert Latham and Tanwi Mallick and Swann Perarnau and Robert Underwood and Kyle Chard and Ian Foster and Robert Ross and Shivaram Venkataraman},
  year          = {2026},
  eprint        = {2606.08950},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2606.08950}
}
```
