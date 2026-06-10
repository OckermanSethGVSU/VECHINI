use anyhow::{bail, Context};
use chrono::Utc;
use csv::WriterBuilder;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use ndarray_npy::{read_npy, write_npy};
use qdrant_client::qdrant::{
    point_id::PointIdOptions, CountPointsBuilder, GetPointsBuilder, PointId, PointStruct, Query,
    QueryBatchPointsBuilder, QueryPoints, QueryPointsBuilder, ScoredPoint, SearchParamsBuilder,
    UpsertPointsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::{mem, str};
use std::time::Instant;
use tokio::sync::{Barrier, Mutex};
use tokio::time::{sleep, Duration};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActiveTask {
    // Insert vectors into the target collection.
    Upload,
    // Issue nearest-neighbor queries against the target collection.
    Query,
}

impl ActiveTask {
    // Environment variables are namespaced differently for insert vs query mode.
    fn as_env_prefix(self) -> &'static str {
        match self {
            Self::Upload => "INSERT",
            Self::Query => "QUERY",
        }
    }

    // Output timing files use lowercase prefixes.
    fn as_file_prefix(self) -> &'static str {
        match self {
            Self::Upload => "insert",
            Self::Query => "query",
        }
    }
}

#[derive(Clone, Debug)]
struct RunConfig {
    active_task: ActiveTask,
    collection_name: String,
    n_workers: usize,
    total_clients: usize,
    explicit_total_clients: bool,
    clients_per_worker: usize,
    corpus_size: Option<usize>,
    batch_size: usize,
    balance_strategy: String,
    npy_path: String,
    streaming_reads: bool,
    debug_results: bool,
    ef_search: u64,
    top_k: usize,
}

#[derive(Clone, Debug)]
struct NpyMetadata {
    // Byte offset where the raw ndarray payload starts after the .npy header.
    data_offset: u64,
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
enum InputData {
    // Original behavior: load the requested corpus eagerly into process memory.
    Eager(Array2<f32>),
    // Streaming behavior: keep only file metadata in memory and read batches on demand.
    Streaming(NpyMetadata),
}

impl InputData {
    fn dims(&self) -> (usize, usize) {
        match self {
            Self::Eager(data) => data.dim(),
            Self::Streaming(meta) => (meta.rows, meta.cols),
        }
    }
}

#[derive(Debug)]
struct Endpoint {
    // Registry-provided node address.
    ip: String,
    // Registry-provided service port.
    port: u16,
}

#[inline]
pub fn even_chunk(part_idx: usize, parts: usize, n_total: usize) -> (usize, usize) {
    // Divide `n_total` items into nearly equal contiguous chunks.
    assert!(parts > 0, "parts must be > 0");
    assert!(part_idx < parts, "part_idx out of range");

    let base = n_total / parts;
    let rem = n_total % parts;
    let start = part_idx * base + part_idx.min(rem);
    let extra = if part_idx < rem { 1 } else { 0 };
    let end = start + base + extra;
    (start, end)
}

#[inline]
pub fn range_for_rank(
    rank: usize,
    n_workers: usize,
    clients_per_worker: usize,
    n_rows_total: usize,
) -> (usize, usize) {
    // First split rows across workers, then split each worker's share across its clients.
    assert!(n_workers > 0, "n_workers must be > 0");
    let world = n_workers * clients_per_worker;
    assert!(rank < world, "rank out of range");

    let worker_id = rank / clients_per_worker;
    let client_id = rank % clients_per_worker;

    let (w_start, w_end) = even_chunk(worker_id, n_workers, n_rows_total);
    let w_len = w_end - w_start;
    let (c_off_start, c_off_end) = even_chunk(client_id, clients_per_worker, w_len);

    let start = w_start + c_off_start;
    let end = w_start + c_off_end;
    (start, end)
}

#[inline]
pub fn range_for_rank_total_clients(rank: usize, total_clients: usize, n_rows_total: usize) -> (usize, usize) {
    even_chunk(rank, total_clients, n_rows_total)
}

#[inline]
pub fn worker_for_rank(rank: usize, n_workers: usize, clients_per_worker: usize) -> anyhow::Result<usize> {
    if clients_per_worker == 0 {
        bail!("clients_per_worker must be positive");
    }
    let worker_id = rank / clients_per_worker;
    if worker_id >= n_workers {
        bail!(
            "rank {} maps to worker {}, but only {} workers are available",
            rank,
            worker_id,
            n_workers
        );
    }
    Ok(worker_id)
}

fn read_endpoint_line(path: &str, line_num: usize) -> io::Result<Option<Endpoint>> {
    let reader = BufReader::new(File::open(path)?);

    let line = match reader.lines().nth(line_num - 1) {
        Some(line) => line?,
        None => return Ok(None),
    };

    let mut parts = line.split(',');
    parts.next();

    let ip = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing IP"))?
        .to_string();

    let port: u16 = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing port"))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid port"))?;

    Ok(Some(Endpoint { ip, port }))
}

fn env_var(name: &str) -> Option<String> {
    env::var(name).ok().map(|value| value.trim().to_string())
}

fn runtime_state_path(name: &str) -> String {
    match env_var("RUNTIME_STATE_DIR") {
        Some(dir) if !dir.is_empty() => format!("{dir}/{name}"),
        _ => format!("./runtime_state/{name}"),
    }
}

// Allow multiple aliases for the same config knob and return the first populated one.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| env_var(name).filter(|value| !value.is_empty()))
}

// Used for required integer settings such as worker counts and batch sizes.
fn parse_required_usize(names: &[&str], label: &str) -> anyhow::Result<usize> {
    let raw = first_env(names)
        .with_context(|| format!("missing environment variable for {label}: tried {:?}", names))?;
    raw.parse()
        .with_context(|| format!("{label} must be a valid integer, got {raw:?}"))
}

// Keep bool parsing intentionally narrow so typos fail closed to false.
fn parse_optional_bool(names: &[&str]) -> bool {
    first_env(names)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True" | "yes" | "YES"))
        .unwrap_or(false)
}

// Optional integer parsing for knobs like ef_search.
fn parse_optional_u64(names: &[&str]) -> anyhow::Result<Option<u64>> {
    match first_env(names) {
        Some(raw) => raw
            .parse()
            .map(Some)
            .with_context(|| format!("expected integer for {:?}, got {raw:?}", names)),
        None => Ok(None),
    }
}

fn parse_optional_usize(names: &[&str]) -> anyhow::Result<Option<usize>> {
    match first_env(names) {
        Some(raw) => raw
            .parse()
            .map(Some)
            .with_context(|| format!("expected integer for {:?}, got {raw:?}", names)),
        None => Ok(None),
    }
}

fn infer_active_task() -> anyhow::Result<ActiveTask> {
    let active_task = first_env(&["ACTIVE_TASK", "TASK"])
        .context("missing ACTIVE_TASK/TASK; set ACTIVE_TASK=INSERT or ACTIVE_TASK=QUERY")?;
    parse_active_task(&active_task)
}

fn parse_active_task(raw: &str) -> anyhow::Result<ActiveTask> {
    match raw.trim().to_ascii_uppercase().as_str() {
        "INSERT" => Ok(ActiveTask::Upload),
        "QUERY" => Ok(ActiveTask::Query),
        other => bail!("unsupported ACTIVE_TASK/TASK value: {other} (expected INSERT or QUERY)"),
    }
}

fn should_mark_workflow(task: ActiveTask) -> bool {
    env::var("TASK")
        .map(|value| value.trim().eq_ignore_ascii_case(task.as_env_prefix()))
        .unwrap_or(true)
}

fn load_config() -> anyhow::Result<RunConfig> {
    // Build the run configuration entirely from environment variables so the binary can be
    // launched from the existing shell scripts without changing its CLI.
    let active_task = infer_active_task()?;
    let collection_name =
        first_env(&["COLLECTION_NAME"]).context("missing COLLECTION_NAME")?;
    let n_workers = parse_required_usize(&["N_WORKERS"], "N_WORKERS")?;

    let clients_key = format!("{}_CLIENTS_PER_WORKER", active_task.as_env_prefix());
    let batch_key = format!("{}_BATCH_SIZE", active_task.as_env_prefix());
    let balance_key = format!("{}_BALANCE_STRATEGY", active_task.as_env_prefix());

    let (total_clients, explicit_total_clients, clients_per_worker) = match active_task {
        ActiveTask::Upload => {
            let clients_per_worker =
                parse_required_usize(&[clients_key.as_str()], "clients_per_worker")?;
            (n_workers * clients_per_worker, false, clients_per_worker)
        }
        ActiveTask::Query => {
            let configured_clients_per_worker =
                parse_optional_usize(&[clients_key.as_str()])?;
            if let Some(total_query_clients) = parse_optional_usize(&["TOTAL_QUERY_CLIENTS"])? {
                let clients_per_worker = configured_clients_per_worker.context(
                    "QUERY_CLIENTS_PER_WORKER is required when TOTAL_QUERY_CLIENTS is set",
                )?;
                if total_query_clients == 0 {
                    bail!("TOTAL_QUERY_CLIENTS must be positive");
                }
                if clients_per_worker == 0 {
                    bail!("QUERY_CLIENTS_PER_WORKER must be positive");
                }
                if total_query_clients > n_workers * clients_per_worker {
                    bail!(
                        "TOTAL_QUERY_CLIENTS={} exceeds capacity N_WORKERS * QUERY_CLIENTS_PER_WORKER = {}",
                        total_query_clients,
                        n_workers * clients_per_worker
                    );
                }
                (total_query_clients, true, clients_per_worker)
            } else {
                let clients_per_worker = configured_clients_per_worker
                    .context("missing environment variable for clients_per_worker: tried [\"QUERY_CLIENTS_PER_WORKER\"]")?;
                (n_workers * clients_per_worker, false, clients_per_worker)
            }
        }
    };
    let corpus_size = match active_task {
        ActiveTask::Upload => parse_optional_usize(&["INSERT_CORPUS_SIZE"])?,
        ActiveTask::Query => parse_optional_usize(&["QUERY_CORPUS_SIZE"])?,
    };
    let batch_size = parse_required_usize(&[batch_key.as_str()], "batch_size")?;

    let balance_strategy =
        first_env(&[balance_key.as_str()]).with_context(|| format!("missing {balance_key}"))?;

    let npy_path = match active_task {
        ActiveTask::Upload => first_env(&["INSERT_DATA_FILEPATH", "INSERT_FILEPATH"])
            .context("missing insert filepath env: tried INSERT_DATA_FILEPATH, INSERT_FILEPATH")?,
        ActiveTask::Query => first_env(&["QUERY_DATA_FILEPATH", "QUERY_FILEPATH"])
            .context("missing query filepath env: tried QUERY_DATA_FILEPATH, QUERY_FILEPATH")?,
    };

    let debug_results = matches!(active_task, ActiveTask::Query)
        && parse_optional_bool(&["QUERY_DEBUG_RESULTS"]);
    let streaming_reads = match active_task {
        ActiveTask::Upload => parse_optional_bool(&["INSERT_STREAMING", "STREAMING"]),
        ActiveTask::Query => parse_optional_bool(&["QUERY_STREAMING", "STREAMING"]),
    };
    let ef_search = if matches!(active_task, ActiveTask::Query) {
        parse_optional_u64(&["HNSW_EF_SEARCH"])?.unwrap_or(64)
    } else {
        64
    };
    let top_k = if matches!(active_task, ActiveTask::Query) {
        parse_optional_usize(&["QUERY_TOP_K", "TOP_K"])?.unwrap_or(10)
    } else {
        10
    };

    Ok(RunConfig {
        active_task,
        collection_name,
        n_workers,
        total_clients,
        explicit_total_clients,
        clients_per_worker,
        corpus_size,
        batch_size,
        balance_strategy,
        npy_path,
        streaming_reads,
        debug_results,
        ef_search,
        top_k,
    })
}

fn resolve_qdrant_url(
    target: usize,
    balance_strategy: &str,
) -> anyhow::Result<(usize, String)> {
    // Choose the target registry entry based on the requested balancing strategy.
    let target = match balance_strategy {
        "NO_BALANCE" => 0,
        "WORKER_BALANCE" => target,
        _ => bail!("unknown balance strategy: {balance_strategy}"),
    };

    let endpoint = read_endpoint_line("ip_registry.txt", target + 1)?
        .with_context(|| format!("ip_registry.txt missing line {}", target + 1))?;
    let qdrant_url = format!("http://{}:{}", endpoint.ip, endpoint.port - 1);
    Ok((target, qdrant_url))
}

fn parse_npy_metadata(path: &str) -> anyhow::Result<NpyMetadata> {
    // Parse only the .npy header so streaming mode can seek directly to row ranges later.
    let mut file = File::open(path).with_context(|| format!("failed to open npy file {path}"))?;

    let mut magic = [0_u8; 6];
    file.read_exact(&mut magic)?;
    if magic != *b"\x93NUMPY" {
        bail!("{path} is not a valid .npy file");
    }

    let mut version = [0_u8; 2];
    file.read_exact(&mut version)?;
    let header_len = match version[0] {
        1 => {
            let mut len = [0_u8; 2];
            file.read_exact(&mut len)?;
            u16::from_le_bytes(len) as usize
        }
        2 | 3 => {
            let mut len = [0_u8; 4];
            file.read_exact(&mut len)?;
            u32::from_le_bytes(len) as usize
        }
        other => bail!("unsupported npy version {}.{}", other, version[1]),
    };

    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = str::from_utf8(&header_bytes)
        .context("npy header was not valid utf-8/ascii")?
        .trim();

    // Streaming mode only supports the same layout assumptions the eager path relies on:
    // a 2D, C-order float32 matrix.
    let descr = parse_npy_header_string(header, "descr")?;
    if !matches!(descr, "<f4" | "=f4" | "f4") {
        bail!("unsupported npy dtype {descr:?}; expected little-endian float32");
    }

    if parse_npy_header_bool(header, "fortran_order")? {
        bail!("unsupported npy layout; only C-order arrays are supported");
    }

    let shape_raw = parse_npy_header_tuple(header, "shape")?;
    let dims: Vec<usize> = shape_raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid npy shape component {part:?}"))
        })
        .collect::<anyhow::Result<_>>()?;

    if dims.len() != 2 {
        bail!("expected a 2D npy array, got shape {:?}", dims);
    }

    let rows = dims[0];
    let cols = dims[1];
    let data_offset = file.stream_position()?;
    let expected_len = data_offset
        + (rows as u64)
            .checked_mul(cols as u64)
            .and_then(|n| n.checked_mul(mem::size_of::<f32>() as u64))
            .context("npy payload size overflow")?;
    let actual_len = file.metadata()?.len();
    if actual_len < expected_len {
        bail!(
            "npy file appears truncated: expected at least {expected_len} bytes, got {actual_len}"
        );
    }

    Ok(NpyMetadata {
        data_offset,
        rows,
        cols,
    })
}

// Minimal parser for quoted scalar fields in the Python dict-like .npy header.
fn parse_npy_header_string<'a>(header: &'a str, key: &str) -> anyhow::Result<&'a str> {
    let needle = format!("'{key}'");
    let start = header
        .find(&needle)
        .with_context(|| format!("npy header missing {key:?}"))?;
    let rest = &header[start + needle.len()..];
    let colon = rest
        .find(':')
        .with_context(|| format!("npy header missing ':' after {key:?}"))?;
    let rest = rest[colon + 1..].trim_start();
    let quote = rest
        .chars()
        .next()
        .with_context(|| format!("npy header missing value for {key:?}"))?;
    if quote != '\'' && quote != '"' {
        bail!("npy header {key:?} value was not a quoted string");
    }
    let value = &rest[1..];
    let end = value
        .find(quote)
        .with_context(|| format!("npy header unterminated string for {key:?}"))?;
    Ok(&value[..end])
}

// The shape field is a tuple literal, so extract the text inside the parentheses.
fn parse_npy_header_tuple<'a>(header: &'a str, key: &str) -> anyhow::Result<&'a str> {
    let needle = format!("'{key}'");
    let start = header
        .find(&needle)
        .with_context(|| format!("npy header missing {key:?}"))?;
    let rest = &header[start + needle.len()..];
    let colon = rest
        .find(':')
        .with_context(|| format!("npy header missing ':' after {key:?}"))?;
    let rest = rest[colon + 1..].trim_start();
    let open = rest
        .find('(')
        .with_context(|| format!("npy header missing tuple for {key:?}"))?;
    let after_open = &rest[open + 1..];
    let close = after_open
        .find(')')
        .with_context(|| format!("npy header missing ')' for {key:?}"))?;
    Ok(&after_open[..close])
}

// Parse boolean fields explicitly so we do not accidentally match unrelated text.
fn parse_npy_header_bool(header: &str, key: &str) -> anyhow::Result<bool> {
    let needle = format!("'{key}'");
    let start = header
        .find(&needle)
        .with_context(|| format!("npy header missing {key:?}"))?;
    let rest = &header[start + needle.len()..];
    let colon = rest
        .find(':')
        .with_context(|| format!("npy header missing ':' after {key:?}"))?;
    let rest = rest[colon + 1..].trim_start();

    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        bail!("npy header {key:?} value was not a boolean");
    }
}

fn resolve_corpus_size(config: &RunConfig, n_rows: usize) -> anyhow::Result<usize> {
    // Missing corpus size means "use the full matrix". Otherwise honor the caller's corpus cap.
    let corpus_size = config.corpus_size.unwrap_or(n_rows);
    if corpus_size > n_rows {
        bail!(
            "{} corpus size {} exceeds npy row count {}",
            config.active_task.as_env_prefix(),
            corpus_size,
            n_rows
        );
    }
    Ok(corpus_size)
}

fn read_rows_from_npy(
    file: &mut File,
    meta: &NpyMetadata,
    start_row: usize,
    row_count: usize,
) -> anyhow::Result<Vec<f32>> {
    // Read exactly one logical row range from the .npy payload so streaming mode
    // only keeps the current batch resident in memory.
    let bytes_per_row = meta
        .cols
        .checked_mul(mem::size_of::<f32>())
        .context("row size overflow")?;
    let byte_count = row_count
        .checked_mul(bytes_per_row)
        .context("batch byte count overflow")?;
    let byte_offset = (start_row as u64)
        .checked_mul(bytes_per_row as u64)
        .context("batch offset overflow")?;
    let offset = meta
        .data_offset
        .checked_add(byte_offset)
        .context("npy data offset overflow")?;

    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0_u8; byte_count];
    file.read_exact(&mut bytes)?;

    let mut data = Vec::with_capacity(row_count * meta.cols);
    for chunk in bytes.chunks_exact(mem::size_of::<f32>()) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(data)
}

fn numeric_point_id(point_id: &Option<PointId>) -> i64 {
    match point_id
        .as_ref()
        .and_then(|id| id.point_id_options.as_ref())
    {
        Some(PointIdOptions::Num(value)) => *value as i64,
        Some(PointIdOptions::Uuid(_)) | None => -1,
    }
}

fn write_query_result_row(
    query_result_ids: &mut Array2<i64>,
    row_idx: usize,
    scored_points: &[ScoredPoint],
    top_k: usize,
) {
    for col_idx in 0..top_k {
        query_result_ids[(row_idx, col_idx)] = scored_points
            .get(col_idx)
            .map(|point| numeric_point_id(&point.id))
            .unwrap_or(-1);
    }
}

type QueryResultChunk = Option<(usize, Array2<i64>)>;

fn merge_query_result_chunks(
    n_rows_total: usize,
    top_k: usize,
    query_chunks: Vec<(usize, Array2<i64>)>,
) -> Array2<i64> {
    let mut query_result_ids = Array2::from_elem((n_rows_total, top_k), -1_i64);
    for (start_row, chunk) in query_chunks {
        let rows = chunk.dim().0;
        query_result_ids
            .slice_mut(s![start_row..start_row + rows, ..])
            .assign(&chunk);
    }
    query_result_ids
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create shared process-wide coordination primitives up front.
    let config = Arc::new(load_config()?);
    let barrier = Arc::new(Barrier::new(config.total_clients));
    let lock = Arc::new(Mutex::new(()));
    let mut handles = Vec::new();

    // Select either eager load or the low-memory streaming reader.
    let input = if config.streaming_reads {
        let meta = parse_npy_metadata(&config.npy_path)?;
        let corpus_size = resolve_corpus_size(&config, meta.rows)?;
        println!(
            "ACTIVE_TASK={} streaming {}, shape = {}x{}",
            config.active_task.as_env_prefix(),
            config.npy_path,
            corpus_size,
            meta.cols
        );
        // Clamp the visible row count to the requested corpus size without copying data.
        Arc::new(InputData::Streaming(NpyMetadata {
            rows: corpus_size,
            ..meta
        }))
    } else {
        let data: Array2<f32> = read_npy(&config.npy_path).context("failed to read npy file")?;
        let corpus_size = resolve_corpus_size(&config, data.dim().0)?;
        let data = data.slice(s![0..corpus_size, ..]).to_owned();
        let (n_rows_total, dim) = data.dim();
        println!(
            "ACTIVE_TASK={} loaded {}, shape = {}x{}",
            config.active_task.as_env_prefix(),
            config.npy_path,
            n_rows_total,
            dim
        );
        Arc::new(InputData::Eager(data))
    };
    let (n_rows_total, _) = input.dims();
    let mut query_chunks: Vec<(usize, Array2<i64>)> = Vec::new();

    // Spawn one async task per logical client, skipping empty slices.
    for rank in 0..config.total_clients {
        let barrier = Arc::clone(&barrier);
        let lock = Arc::clone(&lock);
        let input_data = Arc::clone(&input);
        let config = Arc::clone(&config);
        let (start, end) = if config.explicit_total_clients {
            range_for_rank_total_clients(rank, config.total_clients, n_rows_total)
        } else {
            range_for_rank(rank, config.n_workers, config.clients_per_worker, n_rows_total)
        };
        if start == end {
            continue;
        }

        let handle = tokio::spawn(async move {
            worker(rank, config, input_data, barrier, lock).await
        });
        handles.push(handle);
    }

    for handle in handles {
        if let Some(chunk) = handle.await?? {
            query_chunks.push(chunk);
        }
    }

    if matches!(config.active_task, ActiveTask::Query) {
        let query_result_ids = merge_query_result_chunks(n_rows_total, config.top_k, query_chunks);
        write_npy("query_result_ids.npy", &query_result_ids)?;
    }

    Ok(())
}

async fn worker(
    rank: usize,
    config: Arc<RunConfig>,
    input_data: Arc<InputData>,
    barrier: Arc<Barrier>,
    lock: Arc<Mutex<()>>,
) -> anyhow::Result<QueryResultChunk> {
    // Each async task computes its own slice and target endpoint from the global config.
    let target_worker = worker_for_rank(rank, config.n_workers, config.clients_per_worker)?;
    let (target, qdrant_url) =
        resolve_qdrant_url(target_worker, &config.balance_strategy)?;
    let (n_rows_total, _dim) = input_data.dims();
    let (start_slice, end_slice) = if config.explicit_total_clients {
        range_for_rank_total_clients(rank, config.total_clients, n_rows_total)
    } else {
        range_for_rank(rank, config.n_workers, config.clients_per_worker, n_rows_total)
    };

    let client = Qdrant::from_url(&qdrant_url)
        .timeout(Duration::from_secs(9999))
        .build()?;
    client.collection_info(&config.collection_name).await?;

    println!(
        "Rank {} task={} connecting to Qdrant at {} with offset {}-{}, worker target {}, and batch size {}",
        rank,
        config.active_task.as_env_prefix(),
        qdrant_url,
        start_slice,
        end_slice,
        target,
        config.batch_size
    );

    match input_data.as_ref() {
        InputData::Eager(data) => {
            // Reuse the preloaded matrix and hand an ndarray view into the task-specific runner.
            let view = data.slice(s![start_slice..end_slice, ..]);

            match config.active_task {
                ActiveTask::Upload => {
                    run_upload(
                        rank,
                        &client,
                        &config.collection_name,
                        view,
                        start_slice,
                        n_rows_total,
                        config.batch_size,
                        barrier,
                        lock,
                        config.active_task,
                    )
                    .await?;
                    Ok(None)
                }
                ActiveTask::Query => {
                    run_query(
                        rank,
                        &client,
                        &config.collection_name,
                        view,
                        start_slice,
                        config.batch_size,
                        config.top_k,
                        config.debug_results,
                        config.ef_search,
                        barrier,
                        lock,
                        config.active_task,
                    )
                    .await
                }
            }
        }
        InputData::Streaming(meta) => {
            // Streaming mode opens the file independently in each task so reads stay local to the
            // task and no shared mutable file handle is required.
            match config.active_task {
                ActiveTask::Upload => {
                    run_upload_streaming(
                        rank,
                        &client,
                        &config.collection_name,
                        meta,
                        &config.npy_path,
                        start_slice,
                        end_slice,
                        n_rows_total,
                        config.batch_size,
                        barrier,
                        lock,
                        config.active_task,
                    )
                    .await?;
                    Ok(None)
                }
                ActiveTask::Query => {
                    run_query_streaming(
                        rank,
                        &client,
                        &config.collection_name,
                        meta,
                        &config.npy_path,
                        start_slice,
                        end_slice,
                        config.batch_size,
                        config.top_k,
                        config.debug_results,
                        config.ef_search,
                        barrier,
                        lock,
                        config.active_task,
                    )
                    .await
                }
            }
        }
    }
}

struct TimeFiles {
    // Shared CSV summary file.
    csv: String,
    // Per-rank batch preparation timing arrays.
    batch_npy: String,
    // Per-rank main operation timing arrays.
    main_npy: String,
    // Per-rank end-to-end per-batch timing arrays.
    op_npy: String,
    // Optional extra timing arrays used only by upload mode.
    extra_npy: Option<String>,
}

fn time_files(task: ActiveTask) -> TimeFiles {
    // Keep file naming identical across eager and streaming implementations.
    let prefix = task.as_file_prefix();
    match task {
        ActiveTask::Upload => TimeFiles {
            csv: format!("{prefix}_times.csv"),
            batch_npy: format!("{prefix}_batch_construction_times"),
            main_npy: format!("{prefix}_upload_times"),
            op_npy: format!("{prefix}_op_times"),
            extra_npy: Some(format!("{prefix}_count_times")),
        },
        ActiveTask::Query => TimeFiles {
            csv: format!("{prefix}_times.csv"),
            batch_npy: format!("{prefix}_batch_construction_times"),
            main_npy: format!("{prefix}_query_times"),
            op_npy: format!("{prefix}_op_times"),
            extra_npy: None,
        },
    }
}

async fn run_upload(
    rank: usize,
    client: &Qdrant,
    collection_name: &str,
    view: ArrayView2<'_, f32>,
    offset: usize,
    n_rows_total: usize,
    batch_size: usize,
    barrier: Arc<Barrier>,
    lock: Arc<Mutex<()>>,
    task: ActiveTask,
) -> anyhow::Result<()> {

    barrier.wait().await;

    let mut batch_idx = 0;
    let mut elapsed_process_times = Vec::new();
    let mut elapsed_upload_times = Vec::new();
    let mut elapsed_op_times = Vec::new();
    let mut elapsed_count_times = Vec::new();

    let mark_workflow = should_mark_workflow(task);
    if mark_workflow && rank == 0 {
        let workflow_start = runtime_state_path("workflow_start.txt");
        File::create(&workflow_start)
            .with_context(|| format!("failed to create runtime state marker {workflow_start}"))?;
        sleep(Duration::from_secs(3)).await;
    }
    barrier.wait().await;

    let start = Utc::now();
    let start_loop = Instant::now();

    // Build one upsert request per chunk of the rank-local ndarray view.
    for chunk in view.axis_chunks_iter(Axis(0), batch_size) {
        let start_batch = Instant::now();
        let points: Vec<PointStruct> = chunk
            .outer_iter()
            .enumerate()
            .map(|(i, row)| {
                let id = ((batch_idx * batch_size) + i + offset) as u64;
                PointStruct::new(id, row.to_vec(), Payload::default())
            })
            .collect();

        let batch = UpsertPointsBuilder::new(collection_name, points).wait(false);
        let start_upload = Instant::now();
        client.upsert_points(batch).await?;
        let end_upload = Instant::now();

        elapsed_process_times.push(start_upload.duration_since(start_batch).as_secs_f64());
        elapsed_upload_times.push(end_upload.duration_since(start_upload).as_secs_f64());
        elapsed_op_times.push(end_upload.duration_since(start_batch).as_secs_f64());
        batch_idx += 1;
    }

    let end = Utc::now();
    let end_loop = Instant::now();
    let expected = n_rows_total as u64;

    // Rank 0 waits for the collection count to reach the expected total before all ranks
    // record the searchable timestamp.
    barrier.wait().await;
    if rank == 0 {
        loop {
            let start_count = Instant::now();
            let count = client
                .count(
                    CountPointsBuilder::new(collection_name)
                        .exact(true)
                        .timeout(9999),
                )
                .await?
                .result
                .context("count response missing result")?
                .count;
            let end_count = Instant::now();
            elapsed_count_times.push(end_count.duration_since(start_count).as_secs_f64());

            if count == expected {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    barrier.wait().await;
    let searchable = Instant::now();
    let global_end = Utc::now();

    if mark_workflow && rank == 0 {
        let workflow_stop = runtime_state_path("workflow_stop.txt");
        File::create(&workflow_stop)
            .with_context(|| format!("failed to create runtime state marker {workflow_stop}"))?;
        sleep(Duration::from_secs(3)).await;
    }

    // Sanity-check that the final point this rank wrote is visible.
    let point_id = (offset + view.dim().0 - 1) as u64;
    let resp = client
        .get_points(GetPointsBuilder::new(collection_name, vec![point_id.into()]))
        .await?
        .result;
    let sanity_check = (!resp.is_empty()).to_string();

    let files = time_files(task);
    if rank == 0 {
        write_upload_header(&files.csv, &lock).await?;
    }
    barrier.wait().await;
    
    write_upload_summary(
        rank,
        &files.csv,
        &lock,
        sanity_check,
        start,
        end,
        global_end,
        start_loop,
        end_loop,
        searchable,
    )
    .await?;

    write_npy(
        &format!("{}_rank_{}.npy", files.batch_npy, rank),
        &Array1::from(elapsed_process_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.main_npy, rank),
        &Array1::from(elapsed_upload_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.op_npy, rank),
        &Array1::from(elapsed_op_times),
    )?;
    if let Some(extra_prefix) = files.extra_npy {
        write_npy(
            &format!("{}_rank_{}.npy", extra_prefix, rank),
            &Array1::from(elapsed_count_times),
        )?;
    }

    Ok(())
}

async fn write_upload_header(csv_path: &str, lock: &Mutex<()>) -> anyhow::Result<()> {
    // Serialize header creation so only rank 0 truncates and initializes the CSV once.
    let _guard = lock.lock().await;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(csv_path)?;
    let mut w = WriterBuilder::new().has_headers(false).from_writer(file);

    w.write_record([
        "rank",
        "sanity_check",
        "loop_duration_s",
        "wait_period_s",
        "total_s",
        "start_loop_utc",
        "end_loop_utc",
        "global_end_utc",
    ])?;
    Ok(())
}

async fn write_upload_summary(
    rank: usize,
    csv_path: &str,
    lock: &Mutex<()>,
    sanity_check: String,
    start: chrono::DateTime<Utc>,
    end: chrono::DateTime<Utc>,
    global_end: chrono::DateTime<Utc>,
    start_loop: Instant,
    end_loop: Instant,
    searchable: Instant,
) -> anyhow::Result<()> {
    // Append one summary row per rank after the shared header has been written.
    let loop_duration = end_loop.duration_since(start_loop).as_secs_f64().to_string();
    let wait_period = searchable.duration_since(end_loop).as_secs_f64().to_string();
    let total = searchable.duration_since(start_loop).as_secs_f64().to_string();

    let _guard = lock.lock().await;
    let file = OpenOptions::new().create(true).append(true).open(csv_path)?;
    let mut w = WriterBuilder::new().has_headers(false).from_writer(file);

    w.write_record([
        rank.to_string(),
        sanity_check,
        loop_duration,
        wait_period,
        total,
        start.to_rfc3339(),
        end.to_rfc3339(),
        global_end.to_rfc3339(),
    ])?;
    Ok(())
}

async fn run_query(
    rank: usize,
    client: &Qdrant,
    collection_name: &str,
    view: ArrayView2<'_, f32>,
    start_slice: usize,
    batch_size: usize,
    top_k: usize,
    debug_results: bool,
    ef_search: u64,
    barrier: Arc<Barrier>,
    lock: Arc<Mutex<()>>,
    task: ActiveTask,
) -> anyhow::Result<QueryResultChunk> {

    barrier.wait().await;

    let mut batch_idx = 0;
    let mut elapsed_process_times = Vec::new();
    let mut elapsed_query_times = Vec::new();
    let mut elapsed_op_times = Vec::new();
    let mut printed_debug_results = false;
    let mut query_result_ids = Array2::from_elem((view.dim().0, top_k), -1_i64);
    let search_params = SearchParamsBuilder::default().hnsw_ef(ef_search).build();

    let mark_workflow = should_mark_workflow(task);
    if mark_workflow && rank == 0 {
        let workflow_start = runtime_state_path("workflow_start.txt");
        File::create(&workflow_start)
            .with_context(|| format!("failed to create runtime state marker {workflow_start}"))?;
        sleep(Duration::from_secs(3)).await;
    }
    barrier.wait().await;

    let start = Utc::now();
    let start_loop = Instant::now();

    // Query either one vector at a time or one batch request at a time, matching the
    // behavior selected by the configured batch size.
    for chunk in view.axis_chunks_iter(Axis(0), batch_size) {
        let start_batch = Instant::now();

        if batch_size == 1 {
            let query = QueryPointsBuilder::new(collection_name)
                .query(Query::new_nearest(chunk.row(0).to_vec()))
                .params(search_params.clone())
                .limit(top_k as u64);
            let start_query = Instant::now();
            let response = client.query(query).await?;
            let end_query = Instant::now();

            write_query_result_row(&mut query_result_ids, batch_idx, &response.result, top_k);

            if debug_results && rank == 0 && !printed_debug_results {
                println!(
                    "DEBUG rank {rank}: single query returned {} points",
                    response.result.len()
                );
                for (idx, point) in response.result.iter().take(3).enumerate() {
                    println!(
                        "DEBUG rank {rank}: result[{idx}] id={:?} score={:?} payload={:?}",
                        point.id, point.score, point.payload
                    );
                }
                printed_debug_results = true;
            }

            elapsed_process_times.push(start_query.duration_since(start_batch).as_secs_f64());
            elapsed_query_times.push(end_query.duration_since(start_query).as_secs_f64());
            elapsed_op_times.push(end_query.duration_since(start_batch).as_secs_f64());
        } else {
            let queries: Vec<QueryPoints> = chunk
                .outer_iter()
                .map(|row| {
                    QueryPointsBuilder::new(collection_name)
                        .query(Query::new_nearest(row.to_vec()))
                        .params(search_params.clone())
                        .limit(top_k as u64)
                        .build()
                })
                .collect();

            let batch_query = QueryBatchPointsBuilder::new(collection_name, queries)
                .timeout(999)
                .build();
            let start_query = Instant::now();
            let response = client.query_batch(batch_query).await?;
            let end_query = Instant::now();

            for (local_idx, batch_result) in response.result.iter().enumerate() {
                write_query_result_row(
                    &mut query_result_ids,
                    batch_idx * batch_size + local_idx,
                    &batch_result.result,
                    top_k,
                );
            }

            if debug_results && rank == 0 && !printed_debug_results {
                println!(
                    "DEBUG rank {rank}: batch query returned {} result sets",
                    response.result.len()
                );
                if let Some(first_query) = response.result.first() {
                    println!(
                        "DEBUG rank {rank}: first query returned {} points",
                        first_query.result.len()
                    );
                    for (idx, point) in first_query.result.iter().take(3).enumerate() {
                        println!(
                            "DEBUG rank {rank}: first_query.result[{idx}] id={:?} score={:?} payload={:?}",
                            point.id, point.score, point.payload
                        );
                    }
                }
                printed_debug_results = true;
            }

            elapsed_process_times.push(start_query.duration_since(start_batch).as_secs_f64());
            elapsed_query_times.push(end_query.duration_since(start_query).as_secs_f64());
            elapsed_op_times.push(end_query.duration_since(start_batch).as_secs_f64());
        }

        batch_idx += 1;
    }

    let end = Utc::now();
    let end_loop = Instant::now();
    barrier.wait().await;
    let searchable = Instant::now();
    let global_end = Utc::now();

    if mark_workflow && rank == 0 {
        let workflow_stop = runtime_state_path("workflow_stop.txt");
        File::create(&workflow_stop)
            .with_context(|| format!("failed to create runtime state marker {workflow_stop}"))?;
        sleep(Duration::from_secs(3)).await;
    }

    let files = time_files(task);
    if rank == 0 {
        write_query_header(&files.csv, &lock).await?;
    }
    barrier.wait().await;

    write_query_summary(
        rank,
        &files.csv,
        &lock,
        start,
        end,
        global_end,
        start_loop,
        end_loop,
        searchable,
    )
    .await?;

    write_npy(
        &format!("{}_rank_{}.npy", files.batch_npy, rank),
        &Array1::from(elapsed_process_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.main_npy, rank),
        &Array1::from(elapsed_query_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.op_npy, rank),
        &Array1::from(elapsed_op_times),
    )?;

    Ok(Some((start_slice, query_result_ids)))
}

async fn run_upload_streaming(
    rank: usize,
    client: &Qdrant,
    collection_name: &str,
    meta: &NpyMetadata,
    npy_path: &str,
    start_slice: usize,
    end_slice: usize,
    n_rows_total: usize,
    batch_size: usize,
    barrier: Arc<Barrier>,
    lock: Arc<Mutex<()>>,
    task: ActiveTask,
) -> anyhow::Result<()> {
    // Streaming upload mirrors `run_upload`, but fetches each batch directly from the source file.
    barrier.wait().await;

    let mut file = File::open(npy_path).with_context(|| format!("failed to open npy file {npy_path}"))?;
    let mut elapsed_process_times = Vec::new();
    let mut elapsed_upload_times = Vec::new();
    let mut elapsed_op_times = Vec::new();
    let mut elapsed_count_times = Vec::new();

    let mark_workflow = should_mark_workflow(task);
    if mark_workflow && rank == 0 {
        let workflow_start = runtime_state_path("workflow_start.txt");
        File::create(&workflow_start)
            .with_context(|| format!("failed to create runtime state marker {workflow_start}"))?;
        sleep(Duration::from_secs(3)).await;
    }
    barrier.wait().await;

    let start = Utc::now();
    let start_loop = Instant::now();

    let mut batch_start = start_slice;
    while batch_start < end_slice {
        let rows_in_batch = (end_slice - batch_start).min(batch_size);
        // Pull only the next batch worth of vectors from disk, then drop it after the request.
        let batch_data = read_rows_from_npy(&mut file, meta, batch_start, rows_in_batch)?;

        let start_batch = Instant::now();
        // Convert the raw float buffer into the same point payloads used by eager mode.
        let points: Vec<PointStruct> = batch_data
            .chunks_exact(meta.cols)
            .enumerate()
            .map(|(i, row)| {
                let id = (batch_start + i) as u64;
                PointStruct::new(id, row.to_vec(), Payload::default())
            })
            .collect();

        let batch = UpsertPointsBuilder::new(collection_name, points).wait(false);
        let start_upload = Instant::now();
        client.upsert_points(batch).await?;
        let end_upload = Instant::now();

        elapsed_process_times.push(start_upload.duration_since(start_batch).as_secs_f64());
        elapsed_upload_times.push(end_upload.duration_since(start_upload).as_secs_f64());
        elapsed_op_times.push(end_upload.duration_since(start_batch).as_secs_f64());

        batch_start += rows_in_batch;
    }

    let end = Utc::now();
    let end_loop = Instant::now();
    let expected = n_rows_total as u64;

    // Keep upload completion timing semantics identical to eager mode.
    barrier.wait().await;
    if rank == 0 {
        loop {
            let start_count = Instant::now();
            let count = client
                .count(
                    CountPointsBuilder::new(collection_name)
                        .exact(true)
                        .timeout(9999),
                )
                .await?
                .result
                .context("count response missing result")?
                .count;
            let end_count = Instant::now();
            elapsed_count_times.push(end_count.duration_since(start_count).as_secs_f64());

            if count == expected {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    barrier.wait().await;
    let searchable = Instant::now();
    let global_end = Utc::now();

    if mark_workflow && rank == 0 {
        let workflow_stop = runtime_state_path("workflow_stop.txt");
        File::create(&workflow_stop)
            .with_context(|| format!("failed to create runtime state marker {workflow_stop}"))?;
        sleep(Duration::from_secs(3)).await;
    }

    // Sanity-check the final point for this streamed slice.
    let point_id = (end_slice - 1) as u64;
    let resp = client
        .get_points(GetPointsBuilder::new(collection_name, vec![point_id.into()]))
        .await?
        .result;
    let sanity_check = (!resp.is_empty()).to_string();

    let files = time_files(task);
    if rank == 0 {
        write_upload_header(&files.csv, &lock).await?;
    }
    barrier.wait().await;

    write_upload_summary(
        rank,
        &files.csv,
        &lock,
        sanity_check,
        start,
        end,
        global_end,
        start_loop,
        end_loop,
        searchable,
    )
    .await?;

    write_npy(
        &format!("{}_rank_{}.npy", files.batch_npy, rank),
        &Array1::from(elapsed_process_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.main_npy, rank),
        &Array1::from(elapsed_upload_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.op_npy, rank),
        &Array1::from(elapsed_op_times),
    )?;
    if let Some(extra_prefix) = files.extra_npy {
        write_npy(
            &format!("{}_rank_{}.npy", extra_prefix, rank),
            &Array1::from(elapsed_count_times),
        )?;
    }

    Ok(())
}

async fn run_query_streaming(
    rank: usize,
    client: &Qdrant,
    collection_name: &str,
    meta: &NpyMetadata,
    npy_path: &str,
    start_slice: usize,
    end_slice: usize,
    batch_size: usize,
    top_k: usize,
    debug_results: bool,
    ef_search: u64,
    barrier: Arc<Barrier>,
    lock: Arc<Mutex<()>>,
    task: ActiveTask,
    ) -> anyhow::Result<QueryResultChunk> {
    
    // Streaming query mirrors `run_query`, but materializes only one batch of vectors at a time.
    barrier.wait().await;

    let mut file =
        File::open(npy_path).with_context(|| format!("failed to open npy file {npy_path}"))?;
    let mut elapsed_process_times = Vec::new();
    let mut elapsed_query_times = Vec::new();
    let mut elapsed_op_times = Vec::new();
    let mut printed_debug_results = false;
    let mut query_result_ids = Array2::from_elem((end_slice - start_slice, top_k), -1_i64);
    let search_params = SearchParamsBuilder::default().hnsw_ef(ef_search).build();

    let mark_workflow = should_mark_workflow(task);
    if mark_workflow && rank == 0 {
        let workflow_start = runtime_state_path("workflow_start.txt");
        File::create(&workflow_start)
            .with_context(|| format!("failed to create runtime state marker {workflow_start}"))?;
        sleep(Duration::from_secs(3)).await;
    }
    barrier.wait().await;

    let start = Utc::now();
    let start_loop = Instant::now();

    let mut batch_start = start_slice;
    while batch_start < end_slice {
        let rows_in_batch = (end_slice - batch_start).min(batch_size);
        // Query mode reuses the same streaming reader so only one query batch is materialized.
        let batch_data = read_rows_from_npy(&mut file, meta, batch_start, rows_in_batch)?;
        let start_batch = Instant::now();

        // Preserve the original RPC selection semantics: use single-query mode only when the
        // configured batch size is 1, even if the final partial batch contains one row.
        if batch_size == 1 {
            let query = QueryPointsBuilder::new(collection_name)
                .query(Query::new_nearest(batch_data))
                .params(search_params.clone())
                .limit(top_k as u64);
            let start_query = Instant::now();
            let response = client.query(query).await?;
            let end_query = Instant::now();

            write_query_result_row(
                &mut query_result_ids,
                batch_start - start_slice,
                &response.result,
                top_k,
            );

            if debug_results && rank == 0 && !printed_debug_results {
                println!(
                    "DEBUG rank {rank}: single query returned {} points",
                    response.result.len()
                );
                for (idx, point) in response.result.iter().take(3).enumerate() {
                    println!(
                        "DEBUG rank {rank}: result[{idx}] id={:?} score={:?} payload={:?}",
                        point.id, point.score, point.payload
                    );
                }
                printed_debug_results = true;
            }

            elapsed_process_times.push(start_query.duration_since(start_batch).as_secs_f64());
            elapsed_query_times.push(end_query.duration_since(start_query).as_secs_f64());
            elapsed_op_times.push(end_query.duration_since(start_batch).as_secs_f64());
        } else {
            // Build a batch query request from the streamed rows exactly as the eager path does.
            let queries: Vec<QueryPoints> = batch_data
                .chunks_exact(meta.cols)
                .map(|row| {
                    QueryPointsBuilder::new(collection_name)
                        .query(Query::new_nearest(row.to_vec()))
                        .params(search_params.clone())
                        .limit(top_k as u64)
                        .build()
                })
                .collect();

            let batch_query = QueryBatchPointsBuilder::new(collection_name, queries)
                .timeout(999)
                .build();
            let start_query = Instant::now();
            let response = client.query_batch(batch_query).await?;
            let end_query = Instant::now();

            for (local_idx, batch_result) in response.result.iter().enumerate() {
                write_query_result_row(
                    &mut query_result_ids,
                    batch_start - start_slice + local_idx,
                    &batch_result.result,
                    top_k,
                );
            }

            if debug_results && rank == 0 && !printed_debug_results {
                println!(
                    "DEBUG rank {rank}: batch query returned {} result sets",
                    response.result.len()
                );
                if let Some(first_query) = response.result.first() {
                    println!(
                        "DEBUG rank {rank}: first query returned {} points",
                        first_query.result.len()
                    );
                    for (idx, point) in first_query.result.iter().take(3).enumerate() {
                        println!(
                            "DEBUG rank {rank}: first_query.result[{idx}] id={:?} score={:?} payload={:?}",
                            point.id, point.score, point.payload
                        );
                    }
                }
                printed_debug_results = true;
            }

            elapsed_process_times.push(start_query.duration_since(start_batch).as_secs_f64());
            elapsed_query_times.push(end_query.duration_since(start_query).as_secs_f64());
            elapsed_op_times.push(end_query.duration_since(start_batch).as_secs_f64());
        }

        batch_start += rows_in_batch;
    }

    let end = Utc::now();
    let end_loop = Instant::now();
    barrier.wait().await;
    let searchable = Instant::now();
    let global_end = Utc::now();

    if mark_workflow && rank == 0 {
        let workflow_stop = runtime_state_path("workflow_stop.txt");
        File::create(&workflow_stop)
            .with_context(|| format!("failed to create runtime state marker {workflow_stop}"))?;
        sleep(Duration::from_secs(3)).await;
    }

    let files = time_files(task);
    if rank == 0 {
        write_query_header(&files.csv, &lock).await?;
    }
    barrier.wait().await;

    write_query_summary(
        rank,
        &files.csv,
        &lock,
        start,
        end,
        global_end,
        start_loop,
        end_loop,
        searchable,
    )
    .await?;

    write_npy(
        &format!("{}_rank_{}.npy", files.batch_npy, rank),
        &Array1::from(elapsed_process_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.main_npy, rank),
        &Array1::from(elapsed_query_times),
    )?;
    write_npy(
        &format!("{}_rank_{}.npy", files.op_npy, rank),
        &Array1::from(elapsed_op_times),
    )?;

    Ok(Some((start_slice, query_result_ids)))
}


async fn write_query_header(csv_path: &str, lock: &Mutex<()>) -> anyhow::Result<()> {
    // Serialize header creation so only rank 0 initializes the query CSV.
    let _guard = lock.lock().await;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(csv_path)?;
    let mut w = WriterBuilder::new().has_headers(false).from_writer(file);

    w.write_record([
        "rank",
        "loop_duration_s",
        "wait_period_s",
        "total_s",
        "start_loop_utc",
        "end_loop_utc",
        "global_end_utc",
    ])?;
    Ok(())
}

async fn write_query_summary(
    rank: usize,
    csv_path: &str,
    lock: &Mutex<()>,
    start: chrono::DateTime<Utc>,
    end: chrono::DateTime<Utc>,
    global_end: chrono::DateTime<Utc>,
    start_loop: Instant,
    end_loop: Instant,
    searchable: Instant,
) -> anyhow::Result<()> {
    // Append one query timing summary row for this rank.
    let loop_duration = end_loop.duration_since(start_loop).as_secs_f64().to_string();
    let wait_period = searchable.duration_since(end_loop).as_secs_f64().to_string();
    let total = searchable.duration_since(start_loop).as_secs_f64().to_string();

    let _guard = lock.lock().await;
    let file = OpenOptions::new().create(true).append(true).open(csv_path)?;
    let mut w = WriterBuilder::new().has_headers(false).from_writer(file);

    w.write_record([
        rank.to_string(),
        loop_duration,
        wait_period,
        total,
        start.to_rfc3339(),
        end.to_rfc3339(),
        global_end.to_rfc3339(),
    ])?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "multi_client_op_{name}_{}_{}_test.npy",
            std::process::id(),
            nanos
        ))
    }

    fn cleanup_file(path: &Path) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn even_chunk_balances_remainder_across_early_partitions() {
        assert_eq!(even_chunk(0, 3, 10), (0, 4));
        assert_eq!(even_chunk(1, 3, 10), (4, 7));
        assert_eq!(even_chunk(2, 3, 10), (7, 10));
    }

    #[test]
    fn range_for_rank_splits_rows_by_worker_then_client() {
        let ranges: Vec<(usize, usize)> = (0..4).map(|rank| range_for_rank(rank, 2, 2, 10)).collect();
        assert_eq!(ranges, vec![(0, 3), (3, 5), (5, 8), (8, 10)]);
    }

    #[test]
    fn range_for_rank_total_clients_splits_across_only_active_clients() {
        let ranges: Vec<(usize, usize)> =
            (0..2).map(|rank| range_for_rank_total_clients(rank, 2, 10)).collect();
        assert_eq!(ranges, vec![(0, 5), (5, 10)]);
    }

    #[test]
    fn worker_for_rank_uses_clients_per_worker_as_worker_capacity() {
        assert_eq!(worker_for_rank(0, 4, 1).expect("rank 0"), 0);
        assert_eq!(worker_for_rank(1, 4, 1).expect("rank 1"), 1);
        assert_eq!(worker_for_rank(2, 4, 2).expect("rank 2"), 1);
    }

    #[test]
    fn parse_npy_metadata_reads_shape_and_payload_offset() {
        let path = unique_test_path("metadata");
        let data = array![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        write_npy(&path, &data).expect("write test npy");

        let meta = parse_npy_metadata(path.to_str().expect("utf8 path")).expect("parse metadata");
        assert_eq!(meta.rows, 2);
        assert_eq!(meta.cols, 3);
        assert!(meta.data_offset > 0);

        cleanup_file(&path);
    }

    #[test]
    fn read_rows_from_npy_returns_exact_requested_rows() {
        let path = unique_test_path("stream_rows");
        let data = array![
            [10.0_f32, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
            [40.0, 41.0, 42.0]
        ];
        write_npy(&path, &data).expect("write test npy");

        let path_str = path.to_str().expect("utf8 path");
        let meta = parse_npy_metadata(path_str).expect("parse metadata");
        let mut file = File::open(path_str).expect("open test npy");
        let rows = read_rows_from_npy(&mut file, &meta, 1, 2).expect("read streamed rows");

        assert_eq!(rows, vec![20.0, 21.0, 22.0, 30.0, 31.0, 32.0]);

        cleanup_file(&path);
    }

    #[test]
    fn parse_npy_metadata_rejects_fortran_order_arrays() {
        let path = unique_test_path("fortran_order");
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0_u8; 24]);
        fs::write(&path, bytes).expect("write handcrafted npy");

        let err = parse_npy_metadata(path.to_str().expect("utf8 path")).expect_err("fortran order should fail");
        assert!(err.to_string().contains("only C-order arrays are supported"));

        cleanup_file(&path);
    }

    #[test]
    fn parse_npy_metadata_rejects_non_f32_arrays() {
        let path = unique_test_path("dtype");
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0_u8; 48]);
        fs::write(&path, bytes).expect("write handcrafted npy");

        let err = parse_npy_metadata(path.to_str().expect("utf8 path")).expect_err("non-f32 dtype should fail");
        assert!(err.to_string().contains("expected little-endian float32"));

        cleanup_file(&path);
    }

    #[test]
    fn parse_npy_metadata_rejects_truncated_payloads() {
        let path = unique_test_path("truncated");
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }\n";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0_u8; 8]);
        fs::write(&path, bytes).expect("write truncated npy");

        let err = parse_npy_metadata(path.to_str().expect("utf8 path")).expect_err("truncated payload should fail");
        assert!(err.to_string().contains("appears truncated"));

        cleanup_file(&path);
    }

    #[test]
    fn validate_corpus_size_rejects_requests_larger_than_file() {
        let config = RunConfig {
            active_task: ActiveTask::Upload,
            n_workers: 1,
            total_clients: 1,
            explicit_total_clients: false,
            clients_per_worker: 1,
            corpus_size: 11,
            batch_size: 1,
            balance_strategy: "NO_BALANCE".to_string(),
            npy_path: "unused.npy".to_string(),
            streaming_reads: false,
            debug_results: false,
            ef_search: 64,
            top_k: 10,
        };

        let err = validate_corpus_size(&config, 10).expect_err("oversized corpus should fail");
        assert!(err.to_string().contains("exceeds npy row count 10"));
    }

    #[test]
    fn merge_query_result_chunks_places_rank_local_slices_at_global_offsets() {
        let chunk0 = array![[10_i64, 11_i64], [12_i64, 13_i64]];
        let chunk1 = array![[20_i64, 21_i64], [22_i64, 23_i64], [24_i64, 25_i64]];
        let merged = merge_query_result_chunks(5, 2, vec![(0, chunk0), (2, chunk1)]);

        assert_eq!(
            merged,
            array![
                [10_i64, 11_i64],
                [12_i64, 13_i64],
                [20_i64, 21_i64],
                [22_i64, 23_i64],
                [24_i64, 25_i64]
            ]
        );
    }
}
