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

### Local recall sweeps

`utils/run_local_recall_sweep.py` runs a resumable local parameter sweep and
writes one aggregate CSV containing the dataset, Qdrant settings, timings, and
recall:

```bash
python3 qdrant/utils/run_local_recall_sweep.py \
  qdrant/sampleConfigs/local_recall_sweep.toml \
  --dry-run

python3 qdrant/utils/run_local_recall_sweep.py \
  qdrant/sampleConfigs/local_recall_sweep.toml
```

The TOML file explicitly pairs each data matrix with its query and ground-truth
ID matrices. The runner reads `distance_metric.txt` from each dataset directory.
Set `enabled=false` on a `[[datasets]]` entry to omit it from a particular run.
It recreates and inserts for each dataset, segment-count, and quantization
combination; rebuilds for each `HNSW_M` and `HNSW_EF_CONSTRUCTION` combination;
and reuses that index across all `HNSW_EF_SEARCH` values.
For each requested segment count, it calculates
`MAX_SEGMENT_SIZE=ceil(vector_payload_bytes / segments / 1024)` and records that
value as `segment_size_kb` in the aggregate CSV.

Successful rows already present in the output CSV are skipped when
`run.resume=true`. Use `--results-csv` to resume locally from any compatible
local or PBS aggregate CSV and append new local rows to that same file:

```bash
python3 qdrant/utils/run_local_recall_sweep.py \
  qdrant/sampleConfigs/local_recall_sweep.toml \
  --results-csv qdrant/pbs_sweep_queue/results.csv
```

Compatibility is determined by the exact `run_key`, which includes the dataset,
distance metric, segment count, quantization settings, HNSW build settings,
`ef_search`, and `top_k`. Imported rows retain their original timing, image, and
result-directory columns for provenance.
When `qdrant_version="latest"`, the local sweep always pulls the current
`qdrant/qdrant:latest` image and recreates its dedicated container. Persisted
Qdrant data remains in the mounted sweep output directory.

Quantization exploration is defined with named `[[quantization_variants]]`
tables. Variants may independently set `type`, `always_ram`,
`scalar_quantile`, `binary_encoding`, `product_compression`, and `turbo_bits`.
The variant name and all effective settings are written to the aggregate CSV.
`always_ram` defaults to `true`; set it to `false` for an explicit disk-backed
comparison.
Set `enabled=false` on expensive variants you want to omit.
The older `[sweep].quantization` plus `[quantization]` format remains supported.

`[sweep].top_k` accepts multiple values. Every `(ef_search, top_k)` pair is a
separate Qdrant query execution because the requested result limit can affect
effective HNSW search behavior. Query outputs are isolated under
`ef_search_<n>/top_k_<k>/`.
When a requested `ef_search` is below `top_k`, it is normalized to
`ef_search=top_k`. Duplicate normalized pairs are collapsed into one query.

Every collection `build/`, index-build, and `top_k_*` query directory
contains:

- `run_config.env`: the exact shell-sourceable configuration environment
  explicitly passed by the sweep runner
- `sweep_params.csv`: a compact one-row summary of the dataset, operation,
  image, segment, quantization, HNSW, and query settings

### PBS recall sweep workers

`utils/run_pbs_recall_sweep.py` divides the same TOML sweep into independent
collection-level units:

```text
dataset + number_of_segments + quantization_variant
```

Each unit inserts its dataset once, rebuilds each requested HNSW index, and runs
the full normalized `(ef_search, top_k)` query matrix. Workers do not communicate
with each other or share a running Qdrant instance.

Prepare the queue on the parallel filesystem:

```bash
qdrant/utils/run_pbs_recall_sweep.py \
  qdrant/sampleConfigs/local_recall_sweep.toml prepare
```

This creates `pending/`, `claimed/`, `completed/`, `failed/`, `heartbeats/`,
and `units/` directories plus `worker.pbs.sh`.

The recommended submission path is the continuous watcher:

```bash
qdrant/utils/run_pbs_recall_sweep.py \
  qdrant/sampleConfigs/local_recall_sweep.toml watch
```

Configure `queue_candidates`, `queue_limits`, `queue_queued_limits`,
`submit_username`, `walltime`, and `watch_poll_seconds` under `[pbs]`. The
watcher uses `qstat -u submit_username` and counts all of that user's jobs,
matching the existing queue-watch behavior. It submits workers with
`qsub -q <selected-queue>`, replenishes open slots until all units finish,
automatically requeues stale claims, stops if a unit enters `failed/`, and
aggregates results on completion when `aggregate_on_complete=true`.

Only one watcher may control a sweep queue at a time; an atomic `watch.lock`
prevents duplicate submitters. The watcher should run from a login or service
node for the duration of the sweep.

Workers can still be submitted manually:

```bash
cd qdrant/pbs_sweep_queue
qsub -q capacity worker.pbs.sh
```

Each worker atomically renames one pending JSON descriptor into `claimed/`,
writes a periodic heartbeat, launches Qdrant from the configured SIF with
Apptainer, executes the unit, and atomically moves the descriptor to
`completed/` or `failed/`. Qdrant storage is placed under the configured
node-local `scratch_root`; logs, audit files, recall, and timing results remain
under the shared `units/` directory.

Set `worker_max_units=0` to keep claiming units until the queue is empty or the
PBS allocation approaches its walltime. `stop_before_seconds=300` reserves five
minutes for shutdown. The worker reads allocated and used walltime from
`qstat -f $PBS_JOBID` when available. At the reserve boundary it terminates the
active insert/index/query process group, stops Apptainer, deletes node-local
scratch, and atomically returns the current unit to `pending/`. Completed query
rows in that unit remain recorded and are skipped when another worker resumes
the unit.

Queue operations:

```bash
# Inspect counts.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml status

# Continuously fill configured queue slots until completion.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml watch

# Recover claims whose heartbeat has expired.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml requeue-stale

# Retry units that exited with an error.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml requeue-failed

# Combine per-unit CSVs without concurrent shared-file appends.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml aggregate

# Import successful local or PBS rows into an existing queue.
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml import-results RESULTS.csv
```

Results can also be imported while preparing a new queue:

```bash
qdrant/utils/run_pbs_recall_sweep.py CONFIG.toml prepare \
  --results-csv RESULTS.csv
```

The importer copies only successful rows whose `run_key` exactly matches the
current TOML. A fully covered unit is moved to `completed/`. A partially covered
unit remains pending and its worker skips the imported query configurations.
Rows outside the current sweep and failed rows are ignored. Running/claimed
units are never moved by the importer.

The TOML file is hashed during `prepare`; workers refuse to run if it changes.
Run `prepare` again to add missing descriptors, or `prepare --reset` to discard
all existing queue state and rebuild it.

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
