import bisect
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

DEFAULT_VISIBILITY_LAGS_MS = [0.0, 1.0, 5.0, 25.0, 100.0]
RECALL_CANDIDATE_CHUNK_SIZE = 16384
POST_QUERY_THROUGHPUT_INTERVAL_S = 3.0


@dataclass(frozen=True)
class InsertEvent:
    client_id: int
    op_index: int
    issued_at_ns: int
    completed_at_ns: int
    batch_start_row: int
    batch_end_row: int
    inserted_id_start: int
    inserted_id_end_exclusive: int
    status: str


@dataclass(frozen=True)
class QueryEvent:
    client_id: int
    op_index: int
    issued_at_ns: int
    completed_at_ns: int
    query_start_row: int
    query_end_row: int
    query_row_indices: List[int]
    result_ids: List[List[int]]
    result_scores: List[List[float]]
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge mixed Milvus logs into a single timeline and compute recall under visibility assumptions."
    )
    parser.add_argument("--log-dir", required=True, help="Directory containing mixed runner JSONL logs")
    parser.add_argument("--insert-vectors", default=None, help="Insert vector matrix (.npy)")
    parser.add_argument("--query-vectors", default=None, help="Query vector matrix (.npy)")
    parser.add_argument("--init-vectors", default=None, help="Optional initial visible vector matrix (.npy)")
    parser.add_argument(
        "--insert-max-rows",
        type=int,
        default=None,
        help="Optional row limit for insert-vectors to match workload corpus slicing.",
    )
    parser.add_argument(
        "--query-max-rows",
        type=int,
        default=None,
        help="Optional row limit for query-vectors to match workload corpus slicing.",
    )
    parser.add_argument(
        "--init-max-rows",
        type=int,
        default=None,
        help="Optional row limit for init-vectors to match workload corpus slicing.",
    )
    parser.add_argument(
        "--metric",
        default="l2",
        choices=("dot", "cosine", "l2"),
        help="Distance/similarity metric for exact recall reconstruction",
    )
    parser.add_argument(
        "--visibility-lag-ms",
        type=float,
        nargs="*",
        default=DEFAULT_VISIBILITY_LAGS_MS,
        help="Insert visibility lag assumptions in milliseconds. Defaults to 0, 1, 5, 25, 100, 500, 1000, 5000, 10000.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k. By default it is inferred from the query logs.",
    )
    parser.add_argument(
        "--timeline-out",
        default=None,
        help="Output path for merged timeline JSONL. Defaults to <log-dir>/global_timeline.jsonl",
    )
    parser.add_argument(
        "--recall-detail-out",
        default=None,
        help="Output path for per-query recall JSONL. Defaults to <log-dir>/recall_details.jsonl",
    )
    parser.add_argument(
        "--recall-summary-out",
        default=None,
        help="Output path for aggregated recall CSV. Defaults to <log-dir>/recall_summary.csv",
    )
    parser.add_argument(
        "--throughput-summary-out",
        default=None,
        help="Output path for throughput CSV. Defaults to <log-dir>/throughput_summary.csv",
    )
    parser.add_argument(
        "--init-id-offset",
        type=int,
        default=0,
        help="ID offset for optional init vectors. Defaults to 0.",
    )
    parser.add_argument(
        "--insert-id-offset",
        type=int,
        default=0,
        help="ID offset for insert vectors. Defaults to 0, matching the current mixed runner.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Worker threads for recall reconstruction. Query events are processed independently.",
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip recall reconstruction and .npy inputs; only write throughput and the global timeline.",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Only write throughput outputs and the global timeline; equivalent to --skip-recall.",
    )
    args = parser.parse_args()
    if args.throughput_only:
        args.skip_recall = True
    if not args.skip_recall:
        if not args.insert_vectors:
            parser.error("--insert-vectors is required unless --skip-recall is set")
        if not args.query_vectors:
            parser.error("--query-vectors is required unless --skip-recall is set")
    return args


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_events(log_dir: Path) -> Tuple[List[InsertEvent], List[QueryEvent]]:
    insert_events: List[InsertEvent] = []
    query_events: List[QueryEvent] = []

    for path in sorted(log_dir.glob("insert_client_*.jsonl")):
        for record in load_jsonl(path):
            insert_events.append(
                InsertEvent(
                    client_id=int(record["client_id"]),
                    op_index=int(record["op_index"]),
                    issued_at_ns=int(record["issued_at_ns"]),
                    completed_at_ns=int(record.get("completed_at_ns", record["issued_at_ns"])),
                    batch_start_row=int(record["batch_start_row"]),
                    batch_end_row=int(record["batch_end_row"]),
                    inserted_id_start=int(record["inserted_id_start"]),
                    inserted_id_end_exclusive=int(record["inserted_id_end_exclusive"]),
                    status=str(record["status"]),
                )
            )

    for path in sorted(log_dir.glob("query_client_*.jsonl")):
        for record in load_jsonl(path):
            query_events.append(
                QueryEvent(
                    client_id=int(record["client_id"]),
                    op_index=int(record["op_index"]),
                    issued_at_ns=int(record["issued_at_ns"]),
                    completed_at_ns=int(record.get("completed_at_ns", record["issued_at_ns"])),
                    query_start_row=int(record["query_start_row"]),
                    query_end_row=int(record["query_end_row"]),
                    query_row_indices=[int(v) for v in record["query_row_indices"]],
                    result_ids=[[int(v) for v in row] for row in record["result_ids"]],
                    result_scores=[[float(v) for v in row] for row in record["result_scores"]],
                    status=str(record["status"]),
                )
            )

    insert_events.sort(key=lambda event: (event.completed_at_ns, 0, event.client_id, event.op_index))
    query_events.sort(key=lambda event: (event.completed_at_ns, 1, event.client_id, event.op_index))
    return insert_events, query_events


def build_timeline(insert_events: List[InsertEvent], query_events: List[QueryEvent]) -> List[dict]:
    combined: List[dict] = []
    for event in insert_events:
        combined.append(
            {
                "issued_at_ns": event.issued_at_ns,
                "completed_at_ns": event.completed_at_ns,
                "op_type": "insert",
                "client_id": event.client_id,
                "op_index": event.op_index,
                "batch_start_row": event.batch_start_row,
                "batch_end_row": event.batch_end_row,
                "inserted_id_start": event.inserted_id_start,
                "inserted_id_end_exclusive": event.inserted_id_end_exclusive,
                "status": event.status,
            }
        )
    for event in query_events:
        combined.append(
            {
                "issued_at_ns": event.issued_at_ns,
                "completed_at_ns": event.completed_at_ns,
                "op_type": "query",
                "client_id": event.client_id,
                "op_index": event.op_index,
                "query_start_row": event.query_start_row,
                "query_end_row": event.query_end_row,
                "query_row_indices": event.query_row_indices,
                "result_ids": event.result_ids,
                "status": event.status,
            }
        )

    combined.sort(key=lambda record: (record["completed_at_ns"], 0 if record["op_type"] == "insert" else 1, record["client_id"], record["op_index"]))
    for sequence_id, record in enumerate(combined):
        record["sequence_id"] = sequence_id
    return combined


def compute_observed_rates(insert_events: List[InsertEvent], query_events: List[QueryEvent]) -> dict:
    timestamps: List[int] = []
    total_insert_ops = 0
    total_insert_vectors = 0
    total_query_ops = 0
    total_query_vectors = 0

    for event in insert_events:
        if event.status != "ok":
            continue
        timestamps.append(event.completed_at_ns)
        total_insert_ops += 1
        total_insert_vectors += event.batch_end_row - event.batch_start_row

    for event in query_events:
        if event.status != "ok":
            continue
        timestamps.append(event.completed_at_ns)
        total_query_ops += 1
        total_query_vectors += event.query_end_row - event.query_start_row

    if not timestamps:
        duration_s = 0.0
    else:
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        duration_s = max((max_ts - min_ts) / 1_000_000_000.0, 0.0)

    if duration_s <= 0:
        query_ops_per_sec = 0.0
        query_vectors_per_sec = 0.0
        insert_ops_per_sec = 0.0
        insert_vectors_per_sec = 0.0
    else:
        query_ops_per_sec = total_query_ops / duration_s
        query_vectors_per_sec = total_query_vectors / duration_s
        insert_ops_per_sec = total_insert_ops / duration_s
        insert_vectors_per_sec = total_insert_vectors / duration_s

    return {
        "observed_duration_s": duration_s,
        "total_query_ops": total_query_ops,
        "total_query_vectors": total_query_vectors,
        "total_insert_ops": total_insert_ops,
        "total_insert_vectors": total_insert_vectors,
        "achieved_query_ops_per_sec": query_ops_per_sec,
        "achieved_query_vectors_per_sec": query_vectors_per_sec,
        "achieved_insert_ops_per_sec": insert_ops_per_sec,
        "achieved_insert_vectors_per_sec": insert_vectors_per_sec,
    }


def last_successful_query_issue_ns(query_events: List[QueryEvent]) -> int | None:
    successful_issues = [event.issued_at_ns for event in query_events if event.status == "ok"]
    if not successful_issues:
        return None
    return max(successful_issues)


def recall_insert_row_limit(insert_events: List[InsertEvent], query_events: List[QueryEvent], visibility_lags_ms: List[float]) -> int:
    last_query_issue_ns = last_successful_query_issue_ns(query_events)
    if last_query_issue_ns is None:
        return 0

    min_lag_ns = int(round(min(visibility_lags_ms or [0.0]) * 1_000_000.0))
    visibility_cutoff_ns = last_query_issue_ns - min_lag_ns

    max_visible_row = 0
    for event in insert_events:
        if event.status != "ok":
            continue
        if event.completed_at_ns <= visibility_cutoff_ns:
            max_visible_row = max(max_visible_row, event.batch_end_row)
    return max_visible_row


def _compute_rate_row(
    scope: str,
    role_name: str,
    client_id: str,
    timestamps: List[int],
    op_count: int,
    vector_count: int,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    window_label: str | None = None,
    duration_override_s: float | None = None,
) -> dict:
    if duration_override_s is not None:
        duration_s = max(duration_override_s, 0.0)
    elif timestamps:
        duration_s = max((max(timestamps) - min(timestamps)) / 1_000_000_000.0, 0.0)
    else:
        duration_s = 0.0

    if duration_s <= 0.0:
        ops_per_sec = 0.0
        vectors_per_sec = 0.0
    else:
        ops_per_sec = op_count / duration_s
        vectors_per_sec = vector_count / duration_s

    return {
        "scope": scope,
        "role": role_name,
        "client_id": client_id,
        "window_label": window_label if window_label is not None else scope,
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "observed_duration_s": duration_s,
        "op_count": op_count,
        "vector_count": vector_count,
        "ops_per_sec": ops_per_sec,
        "vectors_per_sec": vectors_per_sec,
    }


def compute_throughput_rows(insert_events: List[InsertEvent], query_events: List[QueryEvent]) -> List[dict]:
    rows: List[dict] = []

    insert_ok = [event for event in insert_events if event.status == "ok"]
    query_ok = [event for event in query_events if event.status == "ok"]
    last_query_completed_ns = max((event.completed_at_ns for event in query_ok), default=None)
    first_mixed_completed_ns = min(
        [event.completed_at_ns for event in insert_ok] + [event.completed_at_ns for event in query_ok],
        default=None,
    )

    rows.append(
        _compute_rate_row(
            "aggregate",
            "insert",
            "all",
            [event.completed_at_ns for event in insert_ok],
            len(insert_ok),
            sum(event.batch_end_row - event.batch_start_row for event in insert_ok),
        )
    )
    rows.append(
        _compute_rate_row(
            "aggregate",
            "query",
            "all",
            [event.completed_at_ns for event in query_ok],
            len(query_ok),
            sum(event.query_end_row - event.query_start_row for event in query_ok),
        )
    )

    if last_query_completed_ns is not None:
        mixed_insert = [event for event in insert_ok if event.completed_at_ns <= last_query_completed_ns]
        rows.append(
            _compute_rate_row(
                "mixed_window",
                "insert",
                "all",
                [event.completed_at_ns for event in mixed_insert],
                len(mixed_insert),
                sum(event.batch_end_row - event.batch_start_row for event in mixed_insert),
                window_start_ns=first_mixed_completed_ns,
                window_end_ns=last_query_completed_ns,
                window_label="before_last_query_completion",
                duration_override_s=(
                    max((last_query_completed_ns - first_mixed_completed_ns) / 1_000_000_000.0, 0.0)
                    if first_mixed_completed_ns is not None
                    else None
                ),
            )
        )

        post_query_insert = [event for event in insert_ok if event.completed_at_ns > last_query_completed_ns]
        interval_ns = int(round(POST_QUERY_THROUGHPUT_INTERVAL_S * 1_000_000_000.0))
        interval_index = 0
        interval_start_ns = last_query_completed_ns
        last_post_query_insert_ns = max((event.completed_at_ns for event in post_query_insert), default=None)
        while last_post_query_insert_ns is not None and interval_start_ns < last_post_query_insert_ns:
            interval_end_ns = interval_start_ns + interval_ns
            interval_events = [
                event
                for event in post_query_insert
                if interval_start_ns < event.completed_at_ns <= interval_end_ns
            ]
            if interval_events:
                rows.append(
                    _compute_rate_row(
                        "post_query_interval",
                        "insert",
                        "all",
                        [event.completed_at_ns for event in interval_events],
                        len(interval_events),
                        sum(event.batch_end_row - event.batch_start_row for event in interval_events),
                        window_start_ns=interval_start_ns,
                        window_end_ns=interval_end_ns,
                        window_label=f"post_query_{interval_index}",
                        duration_override_s=POST_QUERY_THROUGHPUT_INTERVAL_S,
                    )
                )
            else:
                rows.append(
                    _compute_rate_row(
                        "post_query_interval",
                        "insert",
                        "all",
                        [],
                        0,
                        0,
                        window_start_ns=interval_start_ns,
                        window_end_ns=interval_end_ns,
                        window_label=f"post_query_{interval_index}",
                        duration_override_s=POST_QUERY_THROUGHPUT_INTERVAL_S,
                    )
                )
            interval_index += 1
            interval_start_ns = interval_end_ns

    insert_by_client: Dict[int, List[InsertEvent]] = {}
    for event in insert_ok:
        insert_by_client.setdefault(event.client_id, []).append(event)
    for client_id, events in sorted(insert_by_client.items()):
        rows.append(
            _compute_rate_row(
                "per_client",
                "insert",
                str(client_id),
                [event.completed_at_ns for event in events],
                len(events),
                sum(event.batch_end_row - event.batch_start_row for event in events),
            )
        )

    query_by_client: Dict[int, List[QueryEvent]] = {}
    for event in query_ok:
        query_by_client.setdefault(event.client_id, []).append(event)
    for client_id, events in sorted(query_by_client.items()):
        rows.append(
            _compute_rate_row(
                "per_client",
                "query",
                str(client_id),
                [event.completed_at_ns for event in events],
                len(events),
                sum(event.query_end_row - event.query_start_row for event in events),
            )
        )

    return rows


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def load_matrix(path: Path, label: str, max_rows: int | None) -> np.ndarray:
    matrix = np.asarray(np.load(path.expanduser()), dtype=np.float32)
    if matrix.ndim != 2:
        raise SystemExit(f"{label} must be a 2D matrix")
    if max_rows is None:
        return matrix
    if max_rows < 0:
        raise SystemExit(f"{label} row limit must be non-negative")
    if max_rows > matrix.shape[0]:
        raise SystemExit(
            f"{label} row limit {max_rows} exceeds available rows {matrix.shape[0]} in {path}"
        )
    return matrix[:max_rows]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def score_vectors(metric: str, queries: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if metric == "dot":
        return queries @ candidates.T
    if metric == "cosine":
        return normalize_rows(queries) @ normalize_rows(candidates).T
    if metric == "l2":
        diff = queries[:, None, :] - candidates[None, :, :]
        return np.sum(diff * diff, axis=2)
    raise ValueError(f"unsupported metric {metric}")


def topk_indices(metric: str, scores: np.ndarray, k: int) -> np.ndarray:
    if scores.shape[1] == 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)

    k = min(k, scores.shape[1])
    if metric == "l2":
        part = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(part_scores, axis=1)
        return np.take_along_axis(part, order, axis=1)

    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def infer_topk(query_events: List[QueryEvent]) -> int:
    for event in query_events:
        if event.result_ids:
            return len(event.result_ids[0])
    raise ValueError("could not infer top-k from query logs")


def build_insert_visibility_index(insert_events: List[InsertEvent], insert_row_count: int) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    completed_times: List[int] = []
    row_offsets = [0]
    row_blocks: List[np.ndarray] = []
    row_visibility_order = np.full((insert_row_count,), -1, dtype=np.int64)

    for event in insert_events:
        if event.status != "ok":
            continue
        completed_times.append(event.completed_at_ns)
        # Mixed inserts advance through the insert matrix in row order, so each event contributes
        # one contiguous block. Keeping the blocks in completion order lets later queries take a
        # prefix of "all visible rows" for any visibility lag assumption.
        block = np.arange(event.batch_start_row, event.batch_end_row, dtype=np.int64)
        row_blocks.append(block)
        start_order = row_offsets[-1]
        row_visibility_order[block] = np.arange(start_order, start_order + block.size, dtype=np.int64)
        row_offsets.append(row_offsets[-1] + (event.batch_end_row - event.batch_start_row))

    all_rows = np.concatenate(row_blocks) if row_blocks else np.empty((0,), dtype=np.int64)
    return completed_times, np.asarray(row_offsets, dtype=np.int64), all_rows, row_visibility_order


def build_visible_insert_rows(
    completed_times: List[int],
    row_offsets: np.ndarray,
    all_rows: np.ndarray,
    query_time_ns: int,
    lag_ns: int,
) -> np.ndarray:
    cutoff = query_time_ns - lag_ns
    visible_event_count = bisect.bisect_right(completed_times, cutoff)
    visible_row_count = int(row_offsets[visible_event_count])
    return all_rows[:visible_row_count]


def visible_insert_count(completed_times: List[int], row_offsets: np.ndarray, query_time_ns: int, lag_ns: int) -> int:
    cutoff = query_time_ns - lag_ns
    visible_event_count = bisect.bisect_right(completed_times, cutoff)
    return int(row_offsets[visible_event_count])


def init_topk_state(metric: str, n_queries: int, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    fill_value = np.inf if metric == "l2" else -np.inf
    scores = np.full((n_queries, top_k), fill_value, dtype=np.float32)
    ids = np.full((n_queries, top_k), -1, dtype=np.int64)
    return scores, ids


def merge_topk(
    metric: str,
    current_scores: np.ndarray,
    current_ids: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_scores.shape[1] == 0:
        return current_scores, current_ids

    # Keep only the best top-k seen so far by merging the current running winners with the
    # winners from the next candidate chunk.
    combined_scores = np.concatenate((current_scores, candidate_scores), axis=1)
    combined_ids = np.concatenate((current_ids, candidate_ids), axis=1)
    if metric == "l2":
        order = np.argsort(combined_scores, axis=1)[:, :top_k]
    else:
        order = np.argsort(-combined_scores, axis=1)[:, :top_k]
    return (
        np.take_along_axis(combined_scores, order, axis=1),
        np.take_along_axis(combined_ids, order, axis=1),
    )


def update_topk_with_candidates(
    metric: str,
    query_batch: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_ids: np.ndarray,
    current_scores: np.ndarray,
    current_ids: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_vectors.shape[0] == 0:
        return current_scores, current_ids

    # Score only this chunk, reduce it to chunk-local top-k, then merge with the running top-k.
    # This avoids materializing a full query x visible_corpus score matrix.
    scores = score_vectors(metric, query_batch, candidate_vectors)
    chunk_topk = topk_indices(metric, scores, top_k)
    top_scores = np.take_along_axis(scores, chunk_topk, axis=1)
    top_ids = np.take_along_axis(np.broadcast_to(candidate_ids, scores.shape), chunk_topk, axis=1)
    return merge_topk(metric, current_scores, current_ids, top_scores, top_ids, top_k)


def topk_state_to_lists(scores: np.ndarray, ids: np.ndarray) -> Tuple[List[List[int]], List[List[float]]]:
    valid = ids >= 0
    all_ids: List[List[int]] = []
    all_scores: List[List[float]] = []
    for row_valid, row_ids, row_scores in zip(valid, ids, scores):
        all_ids.append([int(value) for value in row_ids[row_valid]])
        all_scores.append([float(value) for value in row_scores[row_valid]])
    return all_ids, all_scores


def is_id_visible_under_lag(
    point_id: int,
    init_id_offset: int,
    init_count: int,
    insert_id_offset: int,
    insert_row_count: int,
    insert_visible_count: int,
    row_visibility_order: np.ndarray,
) -> bool:
    if init_count > 0 and init_id_offset <= point_id < init_id_offset + init_count:
        return True
    insert_row = point_id - insert_id_offset
    if insert_row < 0 or insert_row >= insert_row_count:
        return False
    visible_order = int(row_visibility_order[insert_row])
    return visible_order >= 0 and visible_order < insert_visible_count


def compute_recall_details(
    insert_events: List[InsertEvent],
    query_events: List[QueryEvent],
    insert_vectors: np.ndarray,
    query_vectors: np.ndarray,
    init_vectors: np.ndarray | None,
    metric: str,
    top_k: int,
    visibility_lags_ms: List[float],
    init_id_offset: int,
    insert_id_offset: int,
    threads: int = 1,
) -> List[dict]:
    lag_specs = [(lag_ms, int(round(lag_ms * 1_000_000.0))) for lag_ms in sorted(set(visibility_lags_ms))]
    completed_times, row_offsets, all_visible_rows, row_visibility_order = build_insert_visibility_index(
        insert_events,
        insert_vectors.shape[0],
    )

    init_count = 0 if init_vectors is None else init_vectors.shape[0]
    init_ids = np.arange(init_id_offset, init_id_offset + (0 if init_vectors is None else init_vectors.shape[0]), dtype=np.int64)
    insert_ids = np.arange(insert_id_offset, insert_id_offset + insert_vectors.shape[0], dtype=np.int64)

    def process_query_event(event: QueryEvent) -> dict | None:
        if event.status != "ok":
            return None

        query_batch = query_vectors[event.query_start_row:event.query_end_row]
        actual_ids = [list(row) for row in event.result_ids]
        actual_scores = [list(row) for row in event.result_scores]

        record: dict = {
            "issued_at_ns": event.issued_at_ns,
            "completed_at_ns": event.completed_at_ns,
            "client_id": event.client_id,
            "op_index": event.op_index,
            "query_start_row": event.query_start_row,
            "query_end_row": event.query_end_row,
            "queries": [],
        }

        visible_count_cache: Dict[int, int] = {}
        exact_cache: Dict[int, Tuple[List[List[int]], List[List[float]]]] = {}
        lag_targets = []
        for lag_ms, lag_ns in lag_specs:
            visible_count = visible_insert_count(completed_times, row_offsets, event.issued_at_ns, lag_ns)
            visible_count_cache[lag_ns] = visible_count
            lag_targets.append((visible_count, lag_ns))
        # Larger lags expose fewer completed inserts. Sorting by visible prefix length means we can
        # extend one running top-k state instead of recomputing each lag from scratch.
        lag_targets.sort(key=lambda item: item[0])

        running_scores, running_ids = init_topk_state(metric, query_batch.shape[0], top_k)
        if init_vectors is not None and init_vectors.shape[0] > 0:
            # Init vectors are visible for every lag assumption, so incorporate them once up front.
            for start in range(0, init_vectors.shape[0], RECALL_CANDIDATE_CHUNK_SIZE):
                end = min(start + RECALL_CANDIDATE_CHUNK_SIZE, init_vectors.shape[0])
                running_scores, running_ids = update_topk_with_candidates(
                    metric,
                    query_batch,
                    init_vectors[start:end],
                    init_ids[start:end],
                    running_scores,
                    running_ids,
                    top_k,
                )

        processed_visible_count = 0
        for target_visible_count, lag_ns in lag_targets:
            # Each lag only adds the newly visible suffix beyond the previous lag's prefix.
            while processed_visible_count < target_visible_count:
                next_count = min(processed_visible_count + RECALL_CANDIDATE_CHUNK_SIZE, target_visible_count)
                visible_chunk_rows = all_visible_rows[processed_visible_count:next_count]
                running_scores, running_ids = update_topk_with_candidates(
                    metric,
                    query_batch,
                    insert_vectors[visible_chunk_rows],
                    insert_ids[visible_chunk_rows],
                    running_scores,
                    running_ids,
                    top_k,
                )
                processed_visible_count = next_count

            # Snapshot the running exact top-k once this lag's visibility boundary has been reached.
            exact_cache[lag_ns] = topk_state_to_lists(running_scores.copy(), running_ids.copy())

        for local_idx, query_row in enumerate(event.query_row_indices):
            query_entry = {
                "query_row_index": query_row,
                "actual_result_ids": actual_ids[local_idx],
                "actual_result_scores": actual_scores[local_idx],
                "assumptions": {},
            }
            actual_set = set(actual_ids[local_idx][:top_k])
            for lag_ms, lag_ns in lag_specs:
                exact_ids, exact_scores = exact_cache[lag_ns]
                expected = exact_ids[local_idx]
                visible_actual_pairs = [
                    (int(point_id), float(score))
                    for point_id, score in zip(actual_ids[local_idx][:top_k], actual_scores[local_idx][:top_k])
                    if is_id_visible_under_lag(
                        int(point_id),
                        init_id_offset,
                        init_count,
                        insert_id_offset,
                        insert_vectors.shape[0],
                        visible_count_cache[lag_ns],
                        row_visibility_order,
                    )
                ]
                visible_actual_ids = [point_id for point_id, _score in visible_actual_pairs]
                visible_actual_scores = [_score for _point_id, _score in visible_actual_pairs]
                visible_actual_set = set(visible_actual_ids)
                overlap = len(actual_set.intersection(expected[:top_k]))
                visible_overlap = len(visible_actual_set.intersection(expected[:top_k]))
                denom = min(top_k, len(expected)) if expected else top_k
                recall = float(overlap / denom) if denom > 0 else 1.0
                visible_precision = (
                    float(visible_overlap / len(visible_actual_ids))
                    if visible_actual_ids
                    else None
                )
                assumption_name = f"lag_ms_{lag_ms:g}"
                query_entry["assumptions"][assumption_name] = {
                    "visible_insert_count": visible_count_cache[lag_ns],
                    "expected_result_ids": expected,
                    "expected_result_scores": exact_scores[local_idx],
                    "recall_at_k": recall,
                    "overlap_count": overlap,
                    "visible_actual_result_ids": visible_actual_ids,
                    "visible_actual_result_scores": visible_actual_scores,
                    "visible_overlap_count": visible_overlap,
                    "visible_precision_at_k": visible_precision,
                    "actual_too_new_count": len(actual_ids[local_idx][:top_k]) - len(visible_actual_ids),
                    "expected_missing_count": max(len(expected[:top_k]) - visible_overlap, 0),
                }
            record["queries"].append(query_entry)
        return record

    successful_events = [event for event in query_events if event.status == "ok"]
    if threads <= 1 or len(successful_events) <= 1:
        return [detail for detail in (process_query_event(event) for event in query_events) if detail is not None]

    details: List[dict] = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for detail in executor.map(process_query_event, successful_events):
            if detail is not None:
                details.append(detail)
    return details


def write_recall_summary(path: Path, details: List[dict]) -> None:
    assumption_recalls: Dict[str, List[float]] = {}
    assumption_visible_precisions: Dict[str, List[float]] = {}
    assumption_too_new_counts: Dict[str, List[int]] = {}
    query_count = 0
    for event in details:
        for query in event["queries"]:
            query_count += 1
            for assumption_name, payload in query["assumptions"].items():
                assumption_recalls.setdefault(assumption_name, []).append(payload["recall_at_k"])
                visible_precision = payload["visible_precision_at_k"]
                if visible_precision is not None:
                    assumption_visible_precisions.setdefault(assumption_name, []).append(visible_precision)
                assumption_too_new_counts.setdefault(assumption_name, []).append(payload["actual_too_new_count"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "assumption",
            "query_count",
            "mean_recall_at_k",
            "mean_visible_precision_at_k",
            "mean_actual_too_new_count",
            "total_actual_too_new_count",
        ])
        for assumption_name, recalls in sorted(assumption_recalls.items()):
            visible_precisions = assumption_visible_precisions.get(assumption_name, [])
            too_new_counts = assumption_too_new_counts[assumption_name]
            writer.writerow([
                assumption_name,
                query_count,
                float(np.mean(recalls)) if recalls else 0.0,
                float(np.mean(visible_precisions)) if visible_precisions else 0.0,
                float(np.mean(too_new_counts)) if too_new_counts else 0.0,
                int(np.sum(too_new_counts)) if too_new_counts else 0,
            ])


def write_throughput_summary(path: Path, throughput_rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scope",
            "role",
            "client_id",
            "window_label",
            "window_start_ns",
            "window_end_ns",
            "observed_duration_s",
            "op_count",
            "vector_count",
            "ops_per_sec",
            "vectors_per_sec",
        ])
        for row in throughput_rows:
            writer.writerow([
                row["scope"],
                row["role"],
                row["client_id"],
                row["window_label"],
                row["window_start_ns"],
                row["window_end_ns"],
                row["observed_duration_s"],
                row["op_count"],
                row["vector_count"],
                row["ops_per_sec"],
                row["vectors_per_sec"],
            ])


def main() -> None:
    args = parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    timeline_out = Path(args.timeline_out).expanduser().resolve() if args.timeline_out else log_dir / "global_timeline.jsonl"
    recall_detail_out = Path(args.recall_detail_out).expanduser().resolve() if args.recall_detail_out else log_dir / "recall_details.jsonl"
    recall_summary_out = Path(args.recall_summary_out).expanduser().resolve() if args.recall_summary_out else log_dir / "recall_summary.csv"
    throughput_summary_out = Path(args.throughput_summary_out).expanduser().resolve() if args.throughput_summary_out else log_dir / "throughput_summary.csv"

    insert_events, query_events = load_events(log_dir)
    if not insert_events and not query_events:
        raise SystemExit(f"no mixed runner logs found in {log_dir}")

    timeline = build_timeline(insert_events, query_events)
    write_jsonl(timeline_out, timeline)
    observed_rates = compute_observed_rates(insert_events, query_events)
    throughput_rows = compute_throughput_rows(insert_events, query_events)
    write_throughput_summary(throughput_summary_out, throughput_rows)

    print(f"Wrote timeline to {timeline_out}")
    print(f"Wrote throughput summary to {throughput_summary_out}")
    if args.skip_recall:
        return

    recall_insert_max_rows = recall_insert_row_limit(insert_events, query_events, args.visibility_lag_ms)

    requested_insert_max_rows = args.insert_max_rows
    if requested_insert_max_rows is None:
        effective_insert_max_rows = recall_insert_max_rows
    else:
        effective_insert_max_rows = min(requested_insert_max_rows, recall_insert_max_rows)

    recall_insert_events = [
        event for event in insert_events
        if event.status != "ok" or event.batch_start_row < effective_insert_max_rows
    ]

    insert_vectors = load_matrix(Path(args.insert_vectors), "insert-vectors", effective_insert_max_rows)
    query_vectors = load_matrix(Path(args.query_vectors), "query-vectors", args.query_max_rows)
    init_vectors = None
    if args.init_vectors:
        init_vectors = load_matrix(Path(args.init_vectors), "init-vectors", args.init_max_rows)

    top_k = args.top_k if args.top_k is not None else infer_topk(query_events)

    details = compute_recall_details(
        insert_events=recall_insert_events,
        query_events=query_events,
        insert_vectors=insert_vectors,
        query_vectors=query_vectors,
        init_vectors=init_vectors,
        metric=args.metric,
        top_k=top_k,
        visibility_lags_ms=args.visibility_lag_ms,
        init_id_offset=args.init_id_offset,
        insert_id_offset=args.insert_id_offset,
        threads=args.threads,
    )
    write_jsonl(recall_detail_out, details)
    write_recall_summary(recall_summary_out, details)

    print(f"Wrote recall details to {recall_detail_out}")
    print(f"Wrote recall summary to {recall_summary_out}")


if __name__ == "__main__":
    main()
