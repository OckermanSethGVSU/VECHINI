# Unified Submit Workflow

The repo-wide entrypoint is `pbs_submit_manager.sh`.

## Common Commands

```bash
./pbs_submit_manager.sh --help
./pbs_submit_manager.sh --help --engine qdrant
./pbs_submit_manager.sh --engine milvus --config path/to/run.env
./pbs_submit_manager.sh --generate-only --engine weaviate --config path/to/run.env
./pbs_submit_manager.sh --engine qdrant --set TASK=QUERY --set RUN_MODE=LOCAL
```

## Inputs

Run config can come from:

- defaults defined in schema files
- `--config path/to/file.env`
- `--set KEY=value`

Config files are plain `KEY=value` env files.

## Behavior

- Normal execution generates a run directory and submits automatically when
  `RUN_MODE=PBS`.
- `--generate-only` generates the run directory without submitting it.
- `--help --engine <engine>` shows the engine variables, defaults, and
  requirement rules.

## Key Files

- [pbs_submit_manager.sh](../pbs_submit_manager.sh)
- [common/submit_lib.sh](../common/submit_lib.sh)
- [common/schema.sh](../common/schema.sh)
- [common/engine_schema_lib.sh](../common/engine_schema_lib.sh)

## Validation

Useful low-cost checks for submit-layer changes:

```bash
./pbs_submit_manager.sh --help
./pbs_submit_manager.sh --help --engine qdrant
./pbs_submit_manager.sh --help --engine milvus
./pbs_submit_manager.sh --help --engine weaviate
```

When cluster-only assets are required, validate generation and static config
inspection locally, then call out the runtime gap explicitly.
