# Qdrant

This directory contains the Qdrant engine implementation used by the unified submit interface.

## Main entrypoint

From the repo root:

```bash
./pbs_submit_manager.sh --help --engine qdrant
./pbs_submit_manager.sh --engine qdrant --config qdrant_run.env
./pbs_submit_manager.sh --generate-only --engine qdrant --config qdrant_run.env
```

## Directory structure

- `engine.sh`: Qdrant engine wiring for the unified submit manager
- `schema.sh`: Qdrant variable registry and defaults
- `main.sh`: PBS/HPC runtime flow
- `local_main.sh`: local container-backed runtime flow
- `runtime/cluster/`: Qdrant node launch scripts for PBS runs
- `scripts/`: collection setup, indexing, profiling, summaries, and mixed timeline tools
- `utils/`: dependency checks, SIF download helpers, and input inspection tools
- `sifs/`: local Qdrant SIF cache used by PBS run staging
- `sampleConfigs/`: example config files for the unified submit manager
- `clients/batch_client/`: insert/query Rust client
- `clients/mixed/`: mixed insert/query Rust client
- `runtime_state/`: optional seed runtime-state payload copied into generated runs

## Important run artifacts

Generated Qdrant runs now typically contain:

- `submit.sh`
- `run_config.env`
- `qdrant.sif`
- `runtime_state/`
- `uploadNPY/`
- `queryNPY/`
- `clientTiming/`
- `systemStats/`

`run_config.env` is the canonical resolved run config for the generated run directory.

## Common Qdrant variables

Required for runs:

- `TASK`
- `PLATFORM`
- `WALLTIME`
- `QUEUE`
- `ACCOUNT`

Required for PBS runs:

- `QDRANT_SIF` or `QDRANT_VERSION`
- `ENV_PATH`, unless `ALLOW_SYSTEM_PYTHON=True`

Common runtime knobs:

- `RUN_MODE`: `PBS` or `local`
- `BASE_DIR`: optional base Qdrant directory; auto-filled by the submit manager when empty
- `NODES`
- `WORKERS_PER_NODE`
- `CORES`
- `STORAGE_MEDIUM`
- `ENV_PATH`: Python environment root activated by PBS runs
- `ALLOW_SYSTEM_PYTHON`: set `True` to use the already-loaded Python environment instead of `ENV_PATH`
- `QDRANT_VERSION`: optional version such as `1.16.1`; derives `qdrant_v1.16.1.sif` for PBS and `qdrant/qdrant:v1.16.1` locally
- `QDRANT_SIF`: explicit source SIF filename copied from `qdrant/sifs/` into generated runs as `qdrant.sif`
- `QDRANT_LOCAL_IMAGE`: explicit Docker or Podman image for local runs
- `QDRANT_EXECUTABLE`: optional local executable override copied from `qdrant/qdrantBuilds/`; leave empty to use the executable inside the SIF
- `LOG_LEVEL`
- `VECTOR_DIM`
- `DISTANCE_METRIC`
- `QUANTIZATION_TYPE`: `NONE`, `SCALAR`, `BINARY`, `PRODUCT`, or `TURBO`
- `QUANTIZATION_ALWAYS_RAM`
- `QUANTIZATION_SCALAR_QUANTILE`
- `QUANTIZATION_BINARY_ENCODING`
- `QUANTIZATION_PRODUCT_COMPRESSION`
- `QUANTIZATION_TURBO_BITS`: TurboQuant requires Qdrant 1.18 or newer
- `GPU_INDEX`
- `REBALANCE_TOPOLOGY`

For example, enable scalar quantization when the collection is created with:

```bash
QUANTIZATION_TYPE=SCALAR
QUANTIZATION_ALWAYS_RAM=True
QUANTIZATION_SCALAR_QUANTILE=0.99
```

See [QUANTIZATION.md](QUANTIZATION.md) for a comparison of the available
methods, parameter details, and experiment guidance.

Insert/query file inputs:

- `INSERT_DATA_FILEPATH`
- `QUERY_DATA_FILEPATH`
- `INSERT_CORPUS_SIZE`
- `QUERY_CORPUS_SIZE`

Index/search tuning:

- `HNSW_M`
- `HNSW_EF_CONSTRUCTION`
- `HNSW_EF_SEARCH`

Mixed-workload knobs:

- `MIXED_DATA_FILEPATH`
- `MIXED_CORPUS_SIZE`
- `MIXED_INSERT_CLIENTS_PER_WORKER`
- `MIXED_QUERY_CLIENTS_PER_WORKER`
- `RESULT_PATH`
- `INSERT_MODE`
- `QUERY_MODE`
- `INSERT_OPS_PER_SEC`
- `QUERY_OPS_PER_SEC`
- `INSERT_START_ID`

For `TASK=MIXED`, `INSERT_START_ID` is used as the ID offset for mixed inserts. If it is empty, `main.sh` derives it in this order:

1. `RESTORE_DIR` set: use `EXPECTED_CORPUS_SIZE`
2. `INSERT_CORPUS_SIZE` set: use `INSERT_CORPUS_SIZE`
3. `INSERT_DATA_FILEPATH` set: run the staged `npy_inspect.py` helper to read the `.npy` row count
4. otherwise fail

Use:

```bash
./pbs_submit_manager.sh --help --engine qdrant
```

to see the full variable list, defaults, and requirement rules.

## PBS setup

Download or provide a Qdrant SIF before generating PBS runs. The helper defaults to the latest upstream image and saves a versioned filename:

```bash
qdrant/utils/download_sif.sh
# writes qdrant/sifs/qdrant_latest.sif

qdrant/utils/download_sif.sh 1.16.1
# writes qdrant/sifs/qdrant_v1.16.1.sif
```

Set `QDRANT_SIF` to the filename, not a path:

```bash
QDRANT_SIF=qdrant_v1.16.1.sif
```

Alternatively, select the matching version for both PBS and local runs:

```bash
QDRANT_VERSION=1.16.1
```

For PBS, the matching `qdrant_v1.16.1.sif` must already exist under
`qdrant/sifs/`. For local runs, the resolved image is
`qdrant/qdrant:v1.16.1`.

Explicit values take precedence independently:

```bash
QDRANT_VERSION=1.16.1
QDRANT_SIF=my_custom_qdrant.sif
QDRANT_LOCAL_IMAGE=my-registry/qdrant:custom
```

Generated PBS runs always receive the selected file as `qdrant.sif`, which is what `runtime/cluster/launchQdrantNode.sh` executes.

PBS runs require an explicit Python environment by default:

```bash
ENV_PATH=/path/to/python/env
```

If the loaded modules/current environment already provide all Python dependencies, opt out explicitly:

```bash
ALLOW_SYSTEM_PYTHON=True
```

## Example configs

Sample configs live under `qdrant/sampleConfigs/`. For example:

```bash
./pbs_submit_manager.sh --generate-only --config qdrant/sampleConfigs/aurora_yandex_query.env
```

Remove `--generate-only` when the config is ready to submit.

## Utilities

Use `utils/npy_inspect.py` to inspect `.npy` workload files:

```bash
qdrant/utils/npy_inspect.py /path/to/file.npy
qdrant/utils/npy_inspect.py /path/to/file.npy --field shape
qdrant/utils/npy_inspect.py /path/to/file.npy --field dtype
```

The default output is row count, which is useful for `INSERT_CORPUS_SIZE`, `QUERY_CORPUS_SIZE`, or `INSERT_START_ID`.

## Local mode

Local runs use:

- `local_main.sh`
- a local Qdrant container
- `clients/batch_client`
- optional `clients/mixed`

Supported local tasks are `INSERT`, `QUERY`, `MIXED`, and `LAUNCH`.

Typical workflow:

```bash
./pbs_submit_manager.sh --generate-only --engine qdrant --config qdrant_local.env
cd qdrant/<generated-run-dir>
bash submit.sh
```

## Notes on current behavior

- Empty `INSERT_CORPUS_SIZE` / `QUERY_CORPUS_SIZE` means use all rows in the `.npy` file.
- Empty `CORES` means no explicit CPU binding.
- `REBALANCE_TOPOLOGY=False` keeps `configure_collection.py` in simple setup mode.
- `clientTiming/` holds timing CSVs and summary outputs.
- `systemStats/` holds periodic and final profiler CSVs; profilers remove their non-final CSV after writing the final one.
- `runtime_state/` receives registry files, nodefiles, and top-level `config.yaml` during cleanup.
- `runtime_state/flag.txt` is the shared stop signal for worker-side consumers; the top-level `flag.txt` is cleaned up at the end.
