# Milvus workflow

This directory contains the Milvus workflows used in this repo: standalone and distributed cluster launch paths, Go client binaries for insert/query, and a Go mixed insert/query runner for event-timeline experiments.

Main HPC entrypoint:

- `pbs_submit_manager.sh`

## Current workflow capabilities

- Task modes: `INSERT`, `INDEX`, `QUERY`, `IMPORT`, `MIXED` (`TASK`)
- Deployment modes: `STANDALONE`, `DISTRIBUTED` (`MODE`)
- Platforms: `POLARIS`, `AURORA` (`PLATFORM`)
- Storage media: `memory`, `DAOS`, `lustre`, `SSD` (`STORAGE_MEDIUM`)
- Perf modes: `NONE`, `STAT`, `RECORD` (`PERF`)
- Optional tracing via `TRACING=True`
- Insert/query balancing used by the Go clients:
  - `INSERT_BALANCE_STRATEGY`: `NO_BALANCE`, `WORKER_BALANCE`
  - `QUERY_BALANCE_STRATEGY`: `NO_BALANCE`, `WORKER_BALANCE`
- Insert path selection for `TASK=INSERT` and the preload phase of `TASK=QUERY`:
  - `INSERT_METHOD`: `traditional`, `bulk`
- Bulk transport selection when `INSERT_METHOD=bulk` or `TASK=IMPORT`:
  - `BULK_UPLOAD_TRANSPORT`: `writer`, `mc`
- Staging-medium selection for the `mc` bulk path:
  - `BULK_UPLOAD_STAGING_MEDIUM`: `memory`, `lustre`, `SSD`
- Distributed controls:
  - `MINIO_MODE`: `single`, `stripped`
  - `MINIO_MEDIUM`: `DAOS`, `lustre`
  - `ETCD_MODE`: `single`, `replicated`
  - `ETCD_MEDIUM`
  - `STREAMING_NODES`, `STREAMING_NODES_PER_CN`
  - `NUM_PROXIES`, `NUM_PROXIES_PER_CN`
  - `DML_CHANNELS`

## Important files

- `pbs_submit_manager.sh`: sweep configuration and PBS script generation
- `main.sh`: runtime orchestration for standalone and distributed runs
- `check_dependencies.sh`: dependency validation helper
- `runtime/cluster/`: launch scripts for Milvus, etcd, and MinIO
- `runtime/configs/`: Milvus config templates used at runtime
- `scripts/`: collection setup, profiling, polling, indexing, summaries, and helper scripts
- `clients/batch_client/`: main Go client used by `main.sh`
- `clients/mixed/`: mixed insert/query Go runner with JSONL event logs
- `clients/run_mixed.sh`: example launcher for the mixed runner
- `utils/`: tracing helpers, status scripts, local standalone helpers, and timeline analysis

## HPC runtime flow (`main.sh`)

1. Export run variables and activate the configured platform-specific Python environment.
2. Optionally mount DAOS and start OpenTelemetry when tracing is enabled.
3. Launch either:
   - standalone Milvus on one worker node, or
   - distributed etcd/MinIO plus Milvus service roles across worker nodes.
4. Wait for readiness and determine the reachable Milvus address.
5. For insert/index paths, configure the collection with `scripts/setup_collection.py`.
6. For `TASK=INSERT` and the preload insert step of `TASK=QUERY`, choose the write path with `INSERT_METHOD`:
   - `traditional`: run the Go multi-client insert workload
   - `bulk`: run `scripts/bulk_upload_import.py`
   - alternative pipelined helper: `scripts/bulk_upload_import_mc.py` writes local bulk files and uploads them to MinIO with `mc cp` as each batch is committed, deleting staged files after successful upload by default
   - when `INSERT_METHOD=bulk`, choose the bulk transport with `BULK_UPLOAD_TRANSPORT`:
     - `writer`: current `RemoteBulkWriter` path
     - `mc`: local writer plus pipelined `mc cp`
   - when `BULK_UPLOAD_TRANSPORT=mc`, choose the temporary staging location with `BULK_UPLOAD_STAGING_MEDIUM`:
     - `memory`: `/dev/shm/<run>/bulk-import-stage`
     - `lustre`: `<run>/bulk-import-stage`
     - `SSD`: `/local/scratch/<run>/bulk-import-stage`
7. If `TASK=INDEX`, run `scripts/index.py`.
8. If `TASK=MIXED`, run the Go mixed client and then generate a merged timeline with `qdrant/scripts/mixed_timeline.py`.
9. Write timing outputs, summaries, optional tracing data, and worker logs.

## Important submit variables (`pbs_submit_manager.sh`)

### Sweep variables

- `NODES=(...)`
- `CORES=(...)`
- `INSERT_BATCH_SIZE=(...)`
- `QUERY_BATCH_SIZE=(...)`

### Runtime variables

- `TASK`
- `MODE`
- `RUN_MODE`
- `STORAGE_MEDIUM`
- `PERF`
- `TRACING`
- `WAL`
- `GPU_INDEX`
- `DEBUG`
- `AUTO_CLEANUP`
- `BASE_DIR`
- `ENV_PATH`
- `ALLOW_SYSTEM_PYTHON`
- `MILVUS_BUILD_DIR`
- `MILVUS_CONFIG_DIR`
- `MINIO_MODE`
- `MINIO_MEDIUM`
- `ETCD_MODE`
- `ETCD_MEDIUM`
- Insert path:
  - `INSERT_DATA_FILEPATH`
  - `INSERT_CORPUS_SIZE`
  - `INSERT_CLIENTS_PER_PROXY`
  - `INSERT_METHOD`
  - `BULK_UPLOAD_TRANSPORT`
  - `BULK_UPLOAD_STAGING_MEDIUM`
  - `INSERT_BALANCE_STRATEGY`
  - `INSERT_BATCH_SIZE`
- Query path:
  - `QUERY_DATA_FILEPATH`
  - `QUERY_CORPUS_SIZE`
  - `QUERY_CLIENTS_PER_PROXY`
  - `QUERY_BALANCE_STRATEGY`
  - `QUERY_BATCH_SIZE`
- Collection/index path:
  - `VECTOR_DIM`
  - `DISTANCE_METRIC`
  - `RESTORE_DIR`
  - `EXPECTED_CORPUS_SIZE`
- Mixed path:
  - `MIXED_DATA_FILEPATH`
  - `MIXED_CORPUS_SIZE`
  - `MIXED_RESULT_PATH`
  - `MIXED_INSERT_CLIENTS_PER_PROXY`
  - `MIXED_QUERY_CLIENTS_PER_PROXY`
  - `MIXED_INSERT_BATCH_SIZE`
  - `MIXED_QUERY_BATCH_SIZE`
  - `INSERT_MODE`, `QUERY_MODE`
  - `INSERT_OPS_PER_SEC`, `QUERY_OPS_PER_SEC`
  - `INSERT_START_ID`
- Distributed-only controls listed above

### Scheduler variables

- `WALLTIME`
- `QUEUE`

## Go clients

Build from `milvus/clients`:

```bash
cd clients
./build.sh batch_client
./build.sh mixed
```

Important binaries/scripts:

- `clients/batch_client/batch_client`: insert/query client used by `main.sh`
- `clients/mixed/mixed`: mixed insert/query runner
- `clients/run_mixed.sh`: example mixed-runner invocation

## Local mode

`local_main.sh` provides a container-backed local harness for:

- `TASK=INSERT`
- `TASK=INDEX`
- `TASK=QUERY`
- `TASK=IMPORT`
- `TASK=MIXED`

Local mode uses Docker or Podman plus local config generation under the run directory. `TASK=IMPORT` requires `MINIO_MODE=single` locally.

## Mixed runner notes

The Go mixed runner is not the main HPC entrypoint, but it is present in the repo and currently supports:

- dedicated insert and query worker counts
- `max` and `rate` modes per role
- deterministic row partitioning
- per-worker JSONL event logs
- fixed or bounded-random batch sizing
- optional dry-run backend

It is useful for reconstructing a mixed insert/query timeline outside the larger PBS workflow.

## Dependency check

Run manually:

```bash
./check_dependencies.sh
./check_dependencies.sh --missing-only
```

## Expected outputs

Per run directory you will typically see:

- `workflow.out`, `workflow.log`, and component logs
- timing/summary CSV or NPY files
- insert/query outputs under task-specific directories
- tracing outputs when `TRACING=True`
- worker status files under `workerOut/` or related runtime directories

## Required local artifacts (site-specific)

Examples of required non-committed artifacts include:

- Milvus/etcd/MinIO Apptainer images
- `milvus/sifs/` payloads expected by PBS staging
- built Go clients
- Python environments referenced by `ENV_PATH`
- dataset `.npy` files and optional restore directories
