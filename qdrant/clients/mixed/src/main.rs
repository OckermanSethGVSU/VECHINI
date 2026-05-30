use anyhow::{anyhow, bail, Context, Result};
use ndarray::{s, Array2, Axis};
use ndarray_npy::read_npy;
use qdrant_client::qdrant::{
    point_id::PointIdOptions, PointStruct, Query, QueryBatchPointsBuilder, QueryPoints,
    QueryPointsBuilder, ScoredPoint, SearchParamsBuilder, UpsertPointsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Barrier;
use tokio::task::JoinSet;
use tokio::time::sleep_until;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Insert,
    Query,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Query => "query",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Max,
    Rate,
}

#[derive(Clone, Debug)]
struct BatchConfig {
    fixed: usize,
    min: Option<usize>,
    max: Option<usize>,
}

#[derive(Clone, Debug)]
struct RoleConfig {
    clients: usize,
    clients_per_worker: Option<usize>,
    corpus_size: Option<usize>,
    vectors_path: PathBuf,
    batch: BatchConfig,
    mode: RunMode,
    ops_per_sec: f64,
    balance_strategy: String,
}

#[derive(Clone, Debug)]
struct Config {
    collection_name: String,
    insert_start_id: u64,
    output_dir: PathBuf,
    insert: RoleConfig,
    query: RoleConfig,
    top_k: u64,
    ef_search: u64,
    rpc_timeout: Duration,
    qdrant_url: Option<String>,
    registry_path: PathBuf,
    n_workers: Option<usize>,
}

#[derive(Clone, Debug)]
struct MatrixData {
    rows: usize,
    dim: usize,
    data: Arc<Array2<f32>>,
}

#[derive(Clone, Copy, Debug)]
struct Assignment {
    client_id: usize,
    start_row: usize,
    end_row: usize,
}

#[derive(Debug)]
struct Endpoint {
    ip: String,
    port: u16,
}

#[derive(Serialize)]
struct InsertLogRecord {
    client_role: Role,
    client_id: usize,
    op_index: usize,
    issued_at_ns: i64,
    completed_at_ns: i64,
    duration_ns: i64,
    batch_start_row: usize,
    batch_end_row: usize,
    inserted_id_start: u64,
    inserted_id_end_exclusive: u64,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct QueryLogRecord {
    client_role: Role,
    client_id: usize,
    op_index: usize,
    issued_at_ns: i64,
    completed_at_ns: i64,
    duration_ns: i64,
    query_start_row: usize,
    query_end_row: usize,
    query_row_indices: Vec<usize>,
    result_ids: Vec<Vec<i64>>,
    result_scores: Vec<Vec<f32>>,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tokio::main]
// Load configuration and both vector corpora once, then launch the mixed workload.
async fn main() -> Result<()> {
    let cfg = Arc::new(parse_config()?);
    fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("create output dir {}", cfg.output_dir.display()))?;

    let insert_data =
        load_matrix(&cfg.insert.vectors_path, cfg.insert.corpus_size).with_context(|| {
            format!(
                "load insert vectors from {}",
                cfg.insert.vectors_path.display()
            )
        })?;
    let query_data =
        load_matrix(&cfg.query.vectors_path, cfg.query.corpus_size).with_context(|| {
            format!(
                "load query vectors from {}",
                cfg.query.vectors_path.display()
            )
        })?;

    if insert_data.dim != query_data.dim {
        bail!(
            "dimension mismatch: insert dim={} query dim={}",
            insert_data.dim,
            query_data.dim
        );
    }

    run_workload(cfg, insert_data, query_data).await
}

// Spawn all insert/query workers behind one shared barrier so issue timing starts together.
async fn run_workload(
    cfg: Arc<Config>,
    insert_data: MatrixData,
    query_data: MatrixData,
) -> Result<()> {
    let total_clients = cfg.insert.clients + cfg.query.clients;
    if total_clients == 0 {
        bail!("at least one insert or query client is required");
    }

    let barrier = Arc::new(Barrier::new(total_clients));
    let mut set = JoinSet::new();

    for assignment in build_assignments(insert_data.rows, cfg.insert.clients) {
        let cfg = Arc::clone(&cfg);
        let barrier = Arc::clone(&barrier);
        let data = insert_data.clone();
        set.spawn(async move { run_insert_client(cfg, data, assignment, barrier).await });
    }

    for assignment in build_assignments(query_data.rows, cfg.query.clients) {
        let cfg = Arc::clone(&cfg);
        let barrier = Arc::clone(&barrier);
        let data = query_data.clone();
        set.spawn(async move { run_query_client(cfg, data, assignment, barrier).await });
    }

    let mut first_err: Option<anyhow::Error> = None;
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                if first_err.is_none() {
                    first_err = Some(err);
                }
            }
            Err(err) => {
                if first_err.is_none() {
                    first_err = Some(anyhow!("worker join error: {err}"));
                }
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }
    Ok(())
}

// Drive one insert worker over its assigned row range and buffer a JSONL timeline for the worker.
async fn run_insert_client(
    cfg: Arc<Config>,
    data: MatrixData,
    assignment: Assignment,
    barrier: Arc<Barrier>,
) -> Result<()> {
    let client = build_qdrant_client(&cfg, Role::Insert, assignment.client_id)?;
    let log_path = cfg
        .output_dir
        .join(format!("insert_client_{:03}.jsonl", assignment.client_id));
    let mut pacer = Pacer::new(
        cfg.insert.mode,
        derive_per_client_rate(cfg.insert.ops_per_sec, cfg.insert.clients),
    );
    let mut picker = BatchPicker::new(&cfg.insert.batch, Role::Insert, assignment.client_id);
    let mut records = Vec::with_capacity(record_capacity(assignment));

    log_worker_startup(&cfg, Role::Insert, assignment);
    // Hold until every insert/query worker has finished setup so the workload starts together.
    barrier.wait().await;

    let view = data
        .data
        .slice(s![assignment.start_row..assignment.end_row, ..]);
    let mut row = 0usize;
    let mut op_index = 0usize;
    while row < view.len_of(Axis(0)) {
        let batch_size = picker.next().min(view.len_of(Axis(0)) - row);
        pacer.wait(op_index).await;

        // Slice the already-loaded matrix directly so steady-state work avoids extra copies until RPC serialization.
        let batch_view = view.slice(s![row..row + batch_size, ..]);
        let global_start_row = assignment.start_row + row;
        let ids = (0..batch_size)
            .map(|offset| cfg.insert_start_id + (global_start_row + offset) as u64)
            .collect::<Vec<_>>();
        let points = batch_view
            .outer_iter()
            .zip(ids.iter().copied())
            .map(|(vector, id)| PointStruct::new(id, vector.to_vec(), Payload::default()))
            .collect::<Vec<_>>();

        // IDs are derived from row offsets so repeated runs with the same offset stay deterministic.
        let request = UpsertPointsBuilder::new(&cfg.collection_name, points);
        let issued_at_ns = now_unix_ns();
        let started = Instant::now();
        let result = tokio::time::timeout(cfg.rpc_timeout, client.upsert_points(request)).await;
        let completed_at_ns = now_unix_ns();
        let duration_ns = started.elapsed().as_nanos() as i64;

        // Record every attempted operation in memory, then flush once when the worker ends.
        let record = match result {
            Ok(Ok(_)) => InsertLogRecord {
                client_role: Role::Insert,
                client_id: assignment.client_id,
                op_index,
                issued_at_ns,
                completed_at_ns,
                duration_ns,
                batch_start_row: global_start_row,
                batch_end_row: global_start_row + batch_size,
                inserted_id_start: ids[0],
                inserted_id_end_exclusive: ids[ids.len() - 1] + 1,
                status: "ok",
                error: None,
            },
            Ok(Err(err)) => {
                let record = InsertLogRecord {
                    client_role: Role::Insert,
                    client_id: assignment.client_id,
                    op_index,
                    issued_at_ns,
                    completed_at_ns,
                    duration_ns,
                    batch_start_row: global_start_row,
                    batch_end_row: global_start_row + batch_size,
                    inserted_id_start: ids[0],
                    inserted_id_end_exclusive: ids[ids.len() - 1] + 1,
                    status: "error",
                    error: Some(err.to_string()),
                };
                records.push(record);
                write_jsonl(&log_path, &records)?;
                return Err(err.into());
            }
            Err(_) => {
                let record = InsertLogRecord {
                    client_role: Role::Insert,
                    client_id: assignment.client_id,
                    op_index,
                    issued_at_ns,
                    completed_at_ns,
                    duration_ns,
                    batch_start_row: global_start_row,
                    batch_end_row: global_start_row + batch_size,
                    inserted_id_start: ids[0],
                    inserted_id_end_exclusive: ids[ids.len() - 1] + 1,
                    status: "error",
                    error: Some(format!("operation timed out after {:?}", cfg.rpc_timeout)),
                };
                records.push(record);
                write_jsonl(&log_path, &records)?;
                bail!("insert client {} timed out", assignment.client_id);
            }
        };

        records.push(record);
        row += batch_size;
        op_index += 1;
    }

    write_jsonl(&log_path, &records)?;
    log_worker_done(Role::Insert, assignment, op_index);
    Ok(())
}

// Drive one query worker over its assigned row range and buffer the returned IDs/scores for later analysis.
async fn run_query_client(
    cfg: Arc<Config>,
    data: MatrixData,
    assignment: Assignment,
    barrier: Arc<Barrier>,
) -> Result<()> {
    let client = build_qdrant_client(&cfg, Role::Query, assignment.client_id)?;
    let log_path = cfg
        .output_dir
        .join(format!("query_client_{:03}.jsonl", assignment.client_id));
    let mut pacer = Pacer::new(
        cfg.query.mode,
        derive_per_client_rate(cfg.query.ops_per_sec, cfg.query.clients),
    );
    let mut picker = BatchPicker::new(&cfg.query.batch, Role::Query, assignment.client_id);
    let mut records = Vec::with_capacity(record_capacity(assignment));
    let search_params = SearchParamsBuilder::default()
        .hnsw_ef(cfg.ef_search)
        .build();

    log_worker_startup(&cfg, Role::Query, assignment);
    // Hold until every insert/query worker has finished setup so the workload starts together.
    barrier.wait().await;

    let view = data
        .data
        .slice(s![assignment.start_row..assignment.end_row, ..]);
    let mut row = 0usize;
    let mut op_index = 0usize;
    while row < view.len_of(Axis(0)) {
        let batch_size = picker.next().min(view.len_of(Axis(0)) - row);
        pacer.wait(op_index).await;

        let batch_view = view.slice(s![row..row + batch_size, ..]);
        let global_start_row = assignment.start_row + row;
        let query_row_indices =
            (global_start_row..global_start_row + batch_size).collect::<Vec<_>>();

        // Build one nearest-neighbor query per row in this batch so the JSON log can map results back to source rows.
        let queries = batch_view
            .outer_iter()
            .map(|vector| {
                QueryPointsBuilder::new(&cfg.collection_name)
                    .query(Query::new_nearest(vector.to_vec()))
                    .params(search_params.clone())
                    .limit(cfg.top_k)
                    .build()
            })
            .collect::<Vec<QueryPoints>>();
        let request = QueryBatchPointsBuilder::new(&cfg.collection_name, queries).build();

        let issued_at_ns = now_unix_ns();
        let started = Instant::now();
        let result = tokio::time::timeout(cfg.rpc_timeout, client.query_batch(request)).await;
        let completed_at_ns = now_unix_ns();
        let duration_ns = started.elapsed().as_nanos() as i64;

        let record = match result {
            Ok(Ok(response)) => {
                let (result_ids, result_scores) = response
                    .result
                    .into_iter()
                    .map(|set| {
                        let ids = set.result.iter().map(extract_point_id).collect::<Vec<_>>();
                        let scores = set
                            .result
                            .iter()
                            .map(|point| point.score)
                            .collect::<Vec<_>>();
                        (ids, scores)
                    })
                    .unzip();

                QueryLogRecord {
                    client_role: Role::Query,
                    client_id: assignment.client_id,
                    op_index,
                    issued_at_ns,
                    completed_at_ns,
                    duration_ns,
                    query_start_row: global_start_row,
                    query_end_row: global_start_row + batch_size,
                    query_row_indices,
                    result_ids,
                    result_scores,
                    status: "ok",
                    error: None,
                }
            }
            Ok(Err(err)) => {
                let record = QueryLogRecord {
                    client_role: Role::Query,
                    client_id: assignment.client_id,
                    op_index,
                    issued_at_ns,
                    completed_at_ns,
                    duration_ns,
                    query_start_row: global_start_row,
                    query_end_row: global_start_row + batch_size,
                    query_row_indices,
                    result_ids: Vec::new(),
                    result_scores: Vec::new(),
                    status: "error",
                    error: Some(err.to_string()),
                };
                records.push(record);
                write_jsonl(&log_path, &records)?;
                return Err(err.into());
            }
            Err(_) => {
                let record = QueryLogRecord {
                    client_role: Role::Query,
                    client_id: assignment.client_id,
                    op_index,
                    issued_at_ns,
                    completed_at_ns,
                    duration_ns,
                    query_start_row: global_start_row,
                    query_end_row: global_start_row + batch_size,
                    query_row_indices,
                    result_ids: Vec::new(),
                    result_scores: Vec::new(),
                    status: "error",
                    error: Some(format!("operation timed out after {:?}", cfg.rpc_timeout)),
                };
                records.push(record);
                write_jsonl(&log_path, &records)?;
                bail!("query client {} timed out", assignment.client_id);
            }
        };

        records.push(record);
        row += batch_size;
        op_index += 1;
    }

    write_jsonl(&log_path, &records)?;
    log_worker_done(Role::Query, assignment, op_index);
    Ok(())
}

// Merge CLI overrides with the existing environment-variable conventions used by the Qdrant scripts.
fn parse_config() -> Result<Config> {
    let args = parse_args()?;
    let mode =
        parse_mode(value_with_args(&args, "mode", &["MODE"]).unwrap_or_else(|| "max".to_string()))?;
    let insert_mode = parse_mode(
        value_with_args(&args, "insert-mode", &["MIXED_INSERT_MODE"]).unwrap_or_else(
            || match mode {
                RunMode::Max => "max".to_string(),
                RunMode::Rate => "rate".to_string(),
            },
        ),
    )?;
    let query_mode = parse_mode(
        value_with_args(&args, "query-mode", &["MIXED_QUERY_MODE"]).unwrap_or_else(|| match mode {
            RunMode::Max => "max".to_string(),
            RunMode::Rate => "rate".to_string(),
        }),
    )?;

    let n_workers = optional_usize(value_with_args(&args, "n-workers", &["N_WORKERS"]))?;
    let insert_clients = optional_usize(value_with_args(
        &args,
        "insert-clients",
        &["MIXED_INSERT_CLIENTS"],
    ))?
    .unwrap_or(1);
    let query_clients = optional_usize(value_with_args(
        &args,
        "query-clients",
        &["MIXED_QUERY_CLIENTS"],
    ))?
    .unwrap_or(1);

    let insert = RoleConfig {
        clients: insert_clients,
        clients_per_worker: None,
        corpus_size: optional_usize(value_with_args(
            &args,
            "insert-corpus-size",
            &["MIXED_INSERT_CORPUS_SIZE"],
        ))?,
        vectors_path: PathBuf::from(required_string(value_with_args(
            &args,
            "insert-vectors",
            &["MIXED_INSERT_DATA_FILEPATH"],
        ))?),
        batch: parse_batch_config(
            &args,
            "insert",
            &["MIXED_INSERT_BATCH_SIZE"],
            &["INSERT_BATCH_MIN"],
            &["INSERT_BATCH_MAX"],
        )?,
        mode: insert_mode,
        ops_per_sec: optional_f64(value_with_args(
            &args,
            "insert-ops-per-sec",
            &["INSERT_OPS_PER_SEC"],
        ))?
        .unwrap_or(0.0),
        balance_strategy: value_with_args(
            &args,
            "insert-balance-strategy",
            &["INSERT_BALANCE_STRATEGY"],
        )
        .unwrap_or_else(|| "NO_BALANCE".to_string()),
    };

    let query = RoleConfig {
        clients: query_clients,
        clients_per_worker: None,
        corpus_size: optional_usize(value_with_args(
            &args,
            "query-corpus-size",
            &["MIXED_QUERY_CORPUS_SIZE"],
        ))?,
        vectors_path: PathBuf::from(required_string(value_with_args(
            &args,
            "query-vectors",
            &["MIXED_QUERY_DATA_FILEPATH"],
        ))?),
        batch: parse_batch_config(
            &args,
            "query",
            &["MIXED_QUERY_BATCH_SIZE"],
            &["QUERY_BATCH_MIN"],
            &["QUERY_BATCH_MAX"],
        )?,
        mode: query_mode,
        ops_per_sec: optional_f64(value_with_args(
            &args,
            "query-ops-per-sec",
            &["QUERY_OPS_PER_SEC"],
        ))?
        .unwrap_or(0.0),
        balance_strategy: value_with_args(
            &args,
            "query-balance-strategy",
            &["QUERY_BALANCE_STRATEGY"],
        )
        .unwrap_or_else(|| "NO_BALANCE".to_string()),
    };

    validate_role_config(Role::Insert, &insert)?;
    validate_role_config(Role::Query, &query)?;

    Ok(Config {
        collection_name: value_with_args(&args, "collection", &["COLLECTION_NAME"])
            .context("missing COLLECTION_NAME")?,
        insert_start_id: optional_u64(value_with_args(
            &args,
            "insert-start-id",
            &["INSERT_START_ID"],
        ))?
        .unwrap_or(0),
        output_dir: PathBuf::from(required_string(value_with_args(
            &args,
            "output-dir",
            &["RESULT_PATH"],
        ))?),
        insert,
        query,
        top_k: optional_u64(value_with_args(&args, "top-k", &["TOP_K"]))?.unwrap_or(10),
        ef_search: optional_u64(value_with_args(
            &args,
            "ef-search",
            &["QUERY_EF_SEARCH", "EF_SEARCH"],
        ))?
        .unwrap_or(64),
        rpc_timeout: optional_duration(value_with_args(&args, "rpc-timeout", &["RPC_TIMEOUT"]))?
            .unwrap_or_else(|| Duration::from_secs(600)),
        qdrant_url: value_with_args(&args, "qdrant-url", &["QDRANT_URL"]),
        registry_path: PathBuf::from(
            value_with_args(&args, "registry-path", &["QDRANT_REGISTRY_PATH"])
                .unwrap_or_else(|| "ip_registry.txt".to_string()),
        ),
        n_workers,
    })
}

// Centralize per-role validation so the worker loops can assume a consistent config shape.
fn validate_role_config(role: Role, cfg: &RoleConfig) -> Result<()> {
    if cfg.clients == 0 {
        return Ok(());
    }
    if let Some(corpus_size) = cfg.corpus_size {
        if corpus_size == 0 {
            bail!(
                "{}-corpus-size must be positive when {} clients are configured",
                role.as_str(),
                role.as_str()
            );
        }
    }
    validate_batch_config(role, &cfg.batch)?;
    match cfg.mode {
        RunMode::Max => {}
        RunMode::Rate => {
            if cfg.ops_per_sec <= 0.0 {
                bail!(
                    "{}-ops-per-sec must be positive when {}-mode=rate",
                    role.as_str(),
                    role.as_str()
                );
            }
        }
    }
    Ok(())
}

fn validate_batch_config(role: Role, cfg: &BatchConfig) -> Result<()> {
    if cfg.fixed == 0 {
        bail!("{}-batch-size must be positive", role.as_str());
    }
    match (cfg.min, cfg.max) {
        (None, None) => Ok(()),
        (Some(min), Some(max)) => {
            if min == 0 || max == 0 {
                bail!("{} batch min/max must be positive when set", role.as_str());
            }
            if min > max {
                bail!("{} batch min must be <= max", role.as_str());
            }
            Ok(())
        }
        _ => bail!(
            "{} batch min/max must both be set or both be unset",
            role.as_str()
        ),
    }
}

// Read a 2D float32 .npy file and trim it to the requested corpus size up front.
fn load_matrix(path: &Path, corpus_size: Option<usize>) -> Result<MatrixData> {
    let data: Array2<f32> = read_npy(path)?;
    let (rows, dim) = data.dim();
    let corpus_size = corpus_size.unwrap_or(rows);
    if corpus_size > rows {
        bail!(
            "requested corpus size {} exceeds available rows {} in {}",
            corpus_size,
            rows,
            path.display()
        );
    }
    let sliced = data.slice(s![0..corpus_size, ..]).to_owned();
    Ok(MatrixData {
        rows: corpus_size,
        dim,
        data: Arc::new(sliced),
    })
}

// Split a corpus into deterministic contiguous client-owned slices.
fn build_assignments(total_rows: usize, clients: usize) -> Vec<Assignment> {
    (0..clients)
        .map(|client_id| {
            let (start_row, end_row) = split_range(total_rows, clients, client_id);
            Assignment {
                client_id,
                start_row,
                end_row,
            }
        })
        .collect()
}

// Distribute any remainder to earlier clients so assignments cover the corpus exactly once.
fn split_range(total: usize, parts: usize, idx: usize) -> (usize, usize) {
    if parts == 0 || idx >= parts {
        return (0, 0);
    }
    let base = total / parts;
    let rem = total % parts;
    if idx < rem {
        let start = idx * (base + 1);
        (start, start + base + 1)
    } else {
        let start = rem * (base + 1) + (idx - rem) * base;
        (start, start + base)
    }
}

// Use the owned row span as a cheap upper bound for buffered operation records.
fn record_capacity(assignment: Assignment) -> usize {
    assignment
        .end_row
        .saturating_sub(assignment.start_row)
        .max(1)
}

// Persist one worker's buffered timeline only at worker completion or failure to minimize measurement overhead.
fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

// Resolve the endpoint for this logical worker and build its dedicated Qdrant client.
fn build_qdrant_client(cfg: &Config, role: Role, client_id: usize) -> Result<Qdrant> {
    let url = match &cfg.qdrant_url {
        Some(url) => url.clone(),
        None => resolve_qdrant_url(cfg, role, client_id)?,
    };
    Ok(Qdrant::from_url(&url).timeout(cfg.rpc_timeout).build()?)
}

// Reuse the repo's existing balancing conventions when routing workers to Qdrant nodes.
fn resolve_qdrant_url(cfg: &Config, role: Role, client_id: usize) -> Result<String> {
    let role_cfg = match role {
        Role::Insert => &cfg.insert,
        Role::Query => &cfg.query,
    };
    let strategy = role_cfg.balance_strategy.trim();
    let target = match strategy {
        "NO_BALANCE" => 0,
        "WORKER_BALANCE" => target_worker_id(
            cfg.n_workers,
            role_cfg.clients_per_worker,
            role_cfg.clients,
            client_id,
        )?,
        other => bail!("unsupported {} balance strategy {}", role.as_str(), other),
    };
    let endpoint = read_endpoint_line(&cfg.registry_path, target + 1)?.with_context(|| {
        format!(
            "{} missing line {}",
            cfg.registry_path.display(),
            target + 1
        )
    })?;
    Ok(format!("http://{}:{}", endpoint.ip, endpoint.port - 1))
}

// Map a logical client to its destination worker under WORKER_BALANCE.
fn target_worker_id(
    n_workers: Option<usize>,
    clients_per_worker: Option<usize>,
    total_clients: usize,
    client_id: usize,
) -> Result<usize> {
    let n_workers = n_workers.context("N_WORKERS is required for WORKER_BALANCE")?;
    if n_workers == 0 {
        bail!("N_WORKERS must be positive");
    }
    if let Some(cpw) = clients_per_worker {
        if cpw == 0 {
            bail!("clients-per-worker must be positive");
        }
        return Ok((client_id / cpw).min(n_workers - 1));
    }
    if total_clients == 0 {
        bail!("total clients must be positive");
    }
    Ok(((client_id * n_workers) / total_clients).min(n_workers - 1))
}

// Read one line from the existing registry file format: rank,ip,port.
fn read_endpoint_line(path: &Path, line_num: usize) -> Result<Option<Endpoint>> {
    let reader = BufReader::new(File::open(path)?);
    let line = match reader.lines().nth(line_num - 1) {
        Some(line) => line?,
        None => return Ok(None),
    };
    let mut parts = line.split(',');
    parts.next();
    let ip = parts
        .next()
        .ok_or_else(|| anyhow!("missing IP in {}", path.display()))?
        .trim()
        .to_string();
    let port = parts
        .next()
        .ok_or_else(|| anyhow!("missing port in {}", path.display()))?
        .trim()
        .parse::<u16>()
        .with_context(|| format!("invalid port in {}", path.display()))?;
    Ok(Some(Endpoint { ip, port }))
}

fn log_worker_startup(cfg: &Config, role: Role, assignment: Assignment) {
    let target_desc = match &cfg.qdrant_url {
        Some(url) => format!("url={url}"),
        None => match resolve_qdrant_url(cfg, role, assignment.client_id) {
            Ok(url) => format!("url={url}"),
            Err(err) => format!("target-error={err}"),
        },
    };
    println!(
        "worker_ready role={} client={} rows=[{}, {}) row_count={} {}",
        role.as_str(),
        assignment.client_id,
        assignment.start_row,
        assignment.end_row,
        assignment.end_row.saturating_sub(assignment.start_row),
        target_desc
    );
}

fn log_worker_done(role: Role, assignment: Assignment, ops: usize) {
    println!(
        "worker_done role={} client={} rows=[{}, {}) ops={}",
        role.as_str(),
        assignment.client_id,
        assignment.start_row,
        assignment.end_row,
        ops
    );
}

// Normalize Qdrant point IDs into an integer form that is easy to log and post-process.
fn extract_point_id(point: &ScoredPoint) -> i64 {
    point
        .id
        .as_ref()
        .and_then(|id| id.point_id_options.as_ref())
        .map(|id| match id {
            PointIdOptions::Num(num) => *num as i64,
            PointIdOptions::Uuid(_) => -1,
        })
        .unwrap_or(-1)
}

// Convert a global per-role target into the rate one worker should attempt to maintain.
fn derive_per_client_rate(total_ops_per_sec: f64, clients: usize) -> f64 {
    if clients == 0 {
        return 0.0;
    }
    total_ops_per_sec / clients as f64
}

struct Pacer {
    mode: RunMode,
    rate: f64,
    start: Instant,
}

// The pacer spaces operation issue times evenly in rate mode and becomes a no-op in max mode.
impl Pacer {
    fn new(mode: RunMode, rate: f64) -> Self {
        Self {
            mode,
            rate,
            start: Instant::now(),
        }
    }

    async fn wait(&mut self, op_index: usize) {
        if self.mode != RunMode::Rate || self.rate <= 0.0 {
            return;
        }
        let target = self.start + Duration::from_secs_f64(op_index as f64 / self.rate);
        let now = Instant::now();
        if target > now {
            sleep_until(target.into()).await;
        }
    }
}

struct BatchPicker {
    cfg: BatchConfig,
    state: u64,
}

// Batch sizes can be fixed or deterministically pseudo-random per role/client pair.
impl BatchPicker {
    fn new(cfg: &BatchConfig, role: Role, client_id: usize) -> Self {
        let role_bias = match role {
            Role::Insert => 1u64,
            Role::Query => 10_001u64,
        };
        Self {
            cfg: cfg.clone(),
            state: client_id as u64 + role_bias,
        }
    }

    fn next(&mut self) -> usize {
        match (self.cfg.min, self.cfg.max) {
            (Some(min), Some(max)) if min < max => {
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let span = max - min + 1;
                min + (self.state as usize % span)
            }
            (Some(min), Some(_)) => min,
            _ => self.cfg.fixed,
        }
    }
}

fn now_unix_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos() as i64
}

// Support a simple --key value CLI because most real inputs still come from environment variables.
fn parse_args() -> Result<HashMap<String, String>> {
    let mut args = env::args().skip(1);
    let mut out = HashMap::new();
    while let Some(arg) = args.next() {
        if !arg.starts_with("--") {
            bail!("unexpected argument {arg}");
        }
        let key = arg.trim_start_matches("--").to_string();
        let value = args
            .next()
            .with_context(|| format!("missing value for --{key}"))?;
        out.insert(key, value);
    }
    Ok(out)
}

// CLI wins over environment variables; otherwise use the first non-empty env value in the fallback list.
fn value_with_args(
    args: &HashMap<String, String>,
    arg_name: &str,
    env_names: &[&str],
) -> Option<String> {
    args.get(arg_name)
        .cloned()
        .or_else(|| {
            env_names
                .iter()
                .find_map(|name| env::var(name).ok().map(|v| v.trim().to_string()))
        })
        .filter(|v| !v.is_empty())
}

// Each role supports either a fixed batch size or a bounded random batch-size range.
fn parse_batch_config(
    args: &HashMap<String, String>,
    role_name: &str,
    fixed_envs: &[&str],
    min_envs: &[&str],
    max_envs: &[&str],
) -> Result<BatchConfig> {
    let fixed = optional_usize(value_with_args(
        args,
        &format!("{role_name}-batch-size"),
        fixed_envs,
    ))?
    .unwrap_or(1);
    let min = optional_usize(value_with_args(
        args,
        &format!("{role_name}-batch-min"),
        min_envs,
    ))?;
    let max = optional_usize(value_with_args(
        args,
        &format!("{role_name}-batch-max"),
        max_envs,
    ))?;
    Ok(BatchConfig { fixed, min, max })
}

fn required_string(value: Option<String>) -> Result<String> {
    value.context("missing required configuration")
}

fn required_usize(value: Option<String>) -> Result<usize> {
    optional_usize(value)?.context("missing required integer configuration")
}

fn optional_usize(value: Option<String>) -> Result<Option<usize>> {
    value
        .map(|raw| {
            raw.parse::<usize>()
                .with_context(|| format!("expected integer, got {raw:?}"))
        })
        .transpose()
}

fn optional_u64(value: Option<String>) -> Result<Option<u64>> {
    value
        .map(|raw| {
            raw.parse::<u64>()
                .with_context(|| format!("expected integer, got {raw:?}"))
        })
        .transpose()
}

fn optional_f64(value: Option<String>) -> Result<Option<f64>> {
    value
        .map(|raw| {
            raw.parse::<f64>()
                .with_context(|| format!("expected float, got {raw:?}"))
        })
        .transpose()
}

fn optional_duration(value: Option<String>) -> Result<Option<Duration>> {
    value.map(|raw| parse_duration(&raw)).transpose()
}

fn parse_mode(raw: String) -> Result<RunMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "max" => Ok(RunMode::Max),
        "rate" => Ok(RunMode::Rate),
        other => bail!("unsupported mode {other}"),
    }
}

// Keep duration parsing small and shell-friendly for env vars like 500ms, 30s, or 5m.
fn parse_duration(raw: &str) -> Result<Duration> {
    if let Some(ms) = raw.strip_suffix("ms") {
        return Ok(Duration::from_millis(ms.parse::<u64>()?));
    }
    if let Some(s) = raw.strip_suffix('s') {
        return Ok(Duration::from_secs_f64(s.parse::<f64>()?));
    }
    if let Some(m) = raw.strip_suffix('m') {
        return Ok(Duration::from_secs_f64(m.parse::<f64>()? * 60.0));
    }
    if let Some(h) = raw.strip_suffix('h') {
        return Ok(Duration::from_secs_f64(h.parse::<f64>()? * 3600.0));
    }
    Ok(Duration::from_secs(raw.parse::<u64>()?))
}
