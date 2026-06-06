# Runtime State Seed

Files placed in this directory are copied into each generated Milvus run's
`runtime_state/` directory.

For PBS runs, that directory is mounted read-write inside Milvus containers at
`/runtime_state`.

Runtime scripts also write coordination files and experiment output into this
directory, so avoid names used by the workflow, including:

- `worker.ip`
- `PROXY_registry.txt`
- `minio_registry.txt`
- `workflow_start.txt`
- `workflow_end.txt`
- `flag.txt`
