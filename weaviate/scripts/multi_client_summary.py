import csv
import os
import re
from pathlib import Path

import numpy as np


SUMMARY_HEADER = [
    "run",
    "batch_size",
    "rank",
    "operation",
    "total",
    "mean",
    "std",
    "p99",
    "rank_op/s",
    "rank_v/s",
]


def parse_batch_sizes(raw_value):
    value = raw_value.strip()
    if not value:
        raise ValueError("empty batch size")

    try:
        batch_size = int(value)
        if batch_size <= 0:
            raise ValueError(f"invalid batch size {raw_value!r}")
        return [batch_size]
    except ValueError:
        pass

    cleaned = value.strip("()[]")
    fields = [field for field in cleaned.replace(",", " ").split() if field]
    if not fields:
        raise ValueError(f"invalid batch size list {raw_value!r}")

    batch_sizes = []
    for field in fields:
        batch_size = int(field)
        if batch_size <= 0:
            raise ValueError(f"invalid batch size entry {field!r} in {raw_value!r}")
        batch_sizes.append(batch_size)

    return batch_sizes


def env_required(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"{name} is required")
    return value.strip()


def env_optional_int(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value.strip())


def resolve_client_npy_path(run_dir, filename, active_task):
    task_upper = active_task.upper()
    if task_upper == "QUERY":
        path = run_dir / "queryNPY" / filename
    elif task_upper == "INSERT":
        path = run_dir / "uploadNPY" / filename
    else:
        raise ValueError(f"unsupported ACTIVE_TASK for client timing summary: {active_task}")

    if path.exists():
        return path

    raise FileNotFoundError(f"missing client timing npy file: {path}")


def summarize_npy(path, run_label, rank, name, batch_size, vector_count):
    arr = np.load(path) * 1000
    total_ms = np.sum(arr)
    total_s = total_ms / 1000
    if total_s == 0:
        op_rate = 0.0
        vector_rate = 0.0
    else:
        op_rate = len(arr) / total_s
        vector_rate = vector_count / total_s
    return [
        run_label,
        batch_size,
        rank,
        name,
        total_ms,
        np.mean(arr),
        np.std(arr),
        np.percentile(arr, 99),
        op_rate,
        vector_rate,
    ], arr


def ag_stats(run_label, batch_size, rank, name, arr, total_time, corpus_size):
    return [
        run_label,
        batch_size,
        rank,
        name,
        np.sum(arr),
        np.mean(arr),
        np.std(arr),
        np.percentile(arr, 99),
        len(arr) / total_time,
        corpus_size / total_time,
    ]


def load_client_rows(times_path):
    rows = {}
    with times_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"worker", "client", "global_client", "start_idx", "end_idx", "shared_loop_start_to_searchable"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns in {times_path}: {sorted(missing)}")

        for row in reader:
            worker = int(row["worker"])
            client = int(row["client"])
            rows[(worker, client)] = {
                "global_client": int(row["global_client"]),
                "start_idx": int(row["start_idx"]),
                "end_idx": int(row["end_idx"]),
                "shared_loop_start_to_searchable": float(row["shared_loop_start_to_searchable"]),
            }

    if not rows:
        raise ValueError(f"no client timing rows found in {times_path}")

    return rows


def discover_runs(base_dir, active_task, batch_sizes):
    if len(batch_sizes) == 1:
        return [(base_dir, batch_sizes[0], f"batch_{batch_sizes[0]}")]

    if active_task != "QUERY":
        raise ValueError("batch size sweeps are only supported for QUERY")

    runs = []
    for idx, batch_size in enumerate(batch_sizes):
        run_dir = base_dir / f"query_batch_{batch_size}_run_{idx:02d}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"missing sweep directory: {run_dir}")
        runs.append((run_dir, batch_size, run_dir.name))

    return runs


def resolve_corpus_size(active_task):
    explicit = env_optional_int(f"{active_task}_CORPUS_SIZE")
    if explicit is not None:
        return explicit

    data_path = Path(env_required(f"{active_task}_DATA_FILEPATH"))
    arr = np.load(data_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D npy input at {data_path}, got shape {arr.shape}")
    return int(arr.shape[0])


def summarize_run(run_dir, batch_size, run_label, corpus_size, active_task):
    summary_rows = []
    all_prep = []
    all_upload = []
    all_op = []
    client_rows = load_client_rows(run_dir / f"{active_task.lower()}_times.csv")

    for worker, client in discover_worker_clients(run_dir, active_task):
        client_row = client_rows.get((worker, client))
        if client_row is None:
            raise ValueError(
                f"missing timing CSV row for worker={worker} client={client} in {run_dir / f'{active_task.lower()}_times.csv'}"
            )
        global_client = client_row["global_client"]
        vector_count = client_row["end_idx"] - client_row["start_idx"]
        prep, prep_arr = summarize_npy(
            resolve_client_npy_path(
                run_dir,
                f"batch_construction_times_w{worker}_c{client}.npy",
                active_task,
            ),
            run_label,
            global_client,
            "prep",
            batch_size,
            vector_count,
        )
        upload, upload_arr = summarize_npy(
            resolve_client_npy_path(
                run_dir,
                f"upload_times_w{worker}_c{client}.npy",
                active_task,
            ),
            run_label,
            global_client,
            "upload",
            batch_size,
            vector_count,
        )
        op, op_arr = summarize_npy(
            resolve_client_npy_path(
                run_dir,
                f"op_times_w{worker}_c{client}.npy",
                active_task,
            ),
            run_label,
            global_client,
            "op",
            batch_size,
            vector_count,
        )
        summary_rows.extend([prep, upload, op])
        all_prep.append(prep_arr)
        all_upload.append(upload_arr)
        all_op.append(op_arr)

    stacked_prep = np.concatenate(all_prep)
    stacked_upload = np.concatenate(all_upload)
    stacked_op = np.concatenate(all_op)
    aggregate_time = min(row["shared_loop_start_to_searchable"] for row in client_rows.values())

    summary_rows.extend(
        [
            ag_stats(run_label, batch_size, "all", "prep", stacked_prep, aggregate_time, corpus_size),
            ag_stats(run_label, batch_size, "all", "upload", stacked_upload, aggregate_time, corpus_size),
            ag_stats(run_label, batch_size, "all", "op", stacked_op, aggregate_time, corpus_size),
        ]
    )

    with (run_dir / f"{active_task.lower()}_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(summary_rows)

    return summary_rows


def discover_worker_clients(run_dir, active_task):
    task_upper = active_task.upper()
    if task_upper == "QUERY":
        npy_dir = run_dir / "queryNPY"
    elif task_upper == "INSERT":
        npy_dir = run_dir / "uploadNPY"
    else:
        raise ValueError(f"unsupported ACTIVE_TASK for client timing summary: {active_task}")

    pattern = re.compile(r"op_times_w(\d+)_c(\d+)\.npy$")
    worker_clients = []
    for path in sorted(npy_dir.glob("op_times_w*_c*.npy")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        worker_clients.append((int(match.group(1)), int(match.group(2))))

    if not worker_clients:
        raise FileNotFoundError(f"no client timing npy files found under {npy_dir}")

    return worker_clients


def main():
    active_task = os.getenv("ACTIVE_TASK", "").strip().upper()
    if not active_task:
        raise ValueError("ACTIVE_TASK is required")

    corpus_size = resolve_corpus_size(active_task)
    batch_sizes = parse_batch_sizes(os.getenv(f"{active_task}_BATCH_SIZE", ""))

    base_dir = Path(".")
    runs = discover_runs(base_dir, active_task, batch_sizes)

    combined_rows = []
    for run_dir, batch_size, run_label in runs:
        combined_rows.extend(
            summarize_run(run_dir, batch_size, run_label, corpus_size, active_task)
        )

    with (base_dir / f"{active_task.lower()}_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(combined_rows)


if __name__ == "__main__":
    main()
