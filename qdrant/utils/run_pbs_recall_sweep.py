#!/usr/bin/env python3
"""Coordinate filesystem-backed Qdrant recall sweep workers on PBS systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUNNER_PATH = SCRIPT_PATH.with_name("run_local_recall_sweep.py")


def load_local_runner():
    spec = importlib.util.spec_from_file_location(
        "qdrant_local_recall_sweep", LOCAL_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LOCAL_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


local = load_local_runner()


@dataclass(frozen=True)
class QueueSettings:
    root: Path
    sif: Path
    scratch_root: Path
    heartbeat_seconds: int
    stale_after_seconds: int
    worker_max_units: int
    stop_before_seconds: int
    account: str
    queue: str
    walltime: str
    place: str
    platform: str
    apptainer_args: tuple[str, ...]
    setup_commands: tuple[str, ...]
    queue_candidates: tuple[str, ...]
    queue_limits: tuple[int, ...]
    queue_queued_limits: tuple[int, ...]
    submit_username: str
    watch_poll_seconds: int
    aggregate_on_complete: bool


QUEUE_DIRS = (
    "pending",
    "claimed",
    "completed",
    "failed",
    "heartbeats",
    "units",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and execute independent Qdrant sweep work units using "
            "atomic renames on a shared filesystem."
        )
    )
    parser.add_argument("config", type=Path, help="TOML sweep configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create pending work units")
    prepare.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing queue state before creating units",
    )

    worker = subparsers.add_parser("worker", help="Claim and execute work units")
    worker.add_argument("--worker-id", default="", help="Unique worker identifier")
    worker.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Maximum units to execute; zero means until walltime or queue exhaustion",
    )

    subparsers.add_parser("status", help="Print queue state counts")
    watch = subparsers.add_parser(
        "watch", help="Continuously submit workers until the sweep finishes"
    )
    watch.add_argument(
        "--poll-seconds",
        type=int,
        default=None,
        help="Override queue polling interval",
    )

    requeue = subparsers.add_parser(
        "requeue-stale", help="Return stale claimed units to pending"
    )
    subparsers.add_parser(
        "requeue-failed", help="Return failed work units to pending"
    )
    requeue.add_argument(
        "--stale-after",
        type=int,
        default=None,
        help="Override heartbeat staleness threshold in seconds",
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="Combine per-unit result CSVs"
    )
    aggregate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Aggregate CSV path; defaults to <queue>/results.csv",
    )
    return parser.parse_args()


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_queue_settings(
    config: dict[str, Any], config_dir: Path
) -> QueueSettings:
    table = config.get("pbs")
    if not isinstance(table, dict):
        raise ValueError("PBS sweep config requires a [pbs] table")
    root = resolve_path(str(table.get("queue_dir", "../pbs_sweep_queue")), config_dir)
    sif_value = str(table.get("qdrant_sif", "")).strip()
    if not sif_value:
        raise ValueError("pbs.qdrant_sif is required")
    scratch_default = os.environ.get("TMPDIR", "/tmp")
    apptainer_args = table.get("apptainer_args", [])
    if not isinstance(apptainer_args, list) or not all(
        isinstance(value, str) for value in apptainer_args
    ):
        raise ValueError("pbs.apptainer_args must be a TOML string array")
    setup_commands = table.get("setup_commands", [])
    if not isinstance(setup_commands, list) or not all(
        isinstance(value, str) for value in setup_commands
    ):
        raise ValueError("pbs.setup_commands must be a TOML string array")
    queue_candidates = table.get("queue_candidates", [])
    if not isinstance(queue_candidates, list) or not all(
        isinstance(value, str) and value.strip() for value in queue_candidates
    ):
        raise ValueError("pbs.queue_candidates must be a TOML string array")
    if not queue_candidates and str(table.get("queue", "")).strip():
        queue_candidates = [str(table["queue"]).strip()]
    queue_limits = table.get("queue_limits", [])
    if not isinstance(queue_limits, list):
        raise ValueError("pbs.queue_limits must be a TOML integer array")
    queue_limits = [
        positive_int(value, "pbs.queue_limits") for value in queue_limits
    ]
    if queue_candidates and len(queue_candidates) != len(queue_limits):
        raise ValueError(
            "pbs.queue_candidates and pbs.queue_limits must have equal lengths"
        )
    queue_queued_limits = table.get("queue_queued_limits", queue_limits)
    if not isinstance(queue_queued_limits, list):
        raise ValueError("pbs.queue_queued_limits must be a TOML integer array")
    queue_queued_limits = [
        positive_int(value, "pbs.queue_queued_limits")
        for value in queue_queued_limits
    ]
    if queue_candidates and len(queue_candidates) != len(queue_queued_limits):
        raise ValueError(
            "pbs.queue_candidates and pbs.queue_queued_limits must have equal lengths"
        )
    return QueueSettings(
        root=root,
        sif=resolve_path(sif_value, config_dir),
        scratch_root=resolve_path(
            str(table.get("scratch_root", scratch_default)), config_dir
        ),
        heartbeat_seconds=positive_int(
            table.get("heartbeat_seconds", 30), "pbs.heartbeat_seconds"
        ),
        stale_after_seconds=positive_int(
            table.get("stale_after_seconds", 900), "pbs.stale_after_seconds"
        ),
        worker_max_units=nonnegative_int(
            table.get("worker_max_units", 0), "pbs.worker_max_units"
        ),
        stop_before_seconds=positive_int(
            table.get("stop_before_seconds", 300), "pbs.stop_before_seconds"
        ),
        account=str(table.get("account", "")).strip(),
        queue=str(table.get("queue", "")).strip(),
        walltime=str(table.get("walltime", "01:00:00")).strip(),
        place=str(table.get("place", "excl")).strip(),
        platform=str(table.get("platform", "")).strip(),
        apptainer_args=tuple(apptainer_args),
        setup_commands=tuple(setup_commands),
        queue_candidates=tuple(queue_candidates),
        queue_limits=tuple(queue_limits),
        queue_queued_limits=tuple(queue_queued_limits),
        submit_username=(
            str(table.get("submit_username", "")).strip()
            or os.environ.get("USER", "")
        ),
        watch_poll_seconds=positive_int(
            table.get("watch_poll_seconds", 30), "pbs.watch_poll_seconds"
        ),
        aggregate_on_complete=bool(table.get("aggregate_on_complete", True)),
    )


def load_context(config_path: Path):
    resolved = config_path.expanduser().resolve()
    with resolved.open("rb") as handle:
        config = tomllib.load(handle)
    config_dir = resolved.parent
    run_settings = local.load_settings(config, config_dir, False)
    datasets = local.load_datasets(config, config_dir)
    sweep = local.load_sweep(config)
    variants = local.load_quantization_variants(config)
    queue = load_queue_settings(config, config_dir)
    if max(sweep["top_k"]) > min(
        dataset.ground_truth_meta.columns for dataset in datasets
    ):
        raise ValueError("sweep.top_k exceeds at least one ground-truth matrix width")
    return resolved, config, run_settings, datasets, sweep, variants, queue


def ensure_queue_dirs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in QUEUE_DIRS:
        (root / name).mkdir(exist_ok=True)


def unit_id(dataset: str, segments: int, quantization: str) -> str:
    return local.safe_name(f"{dataset}__segments_{segments}__{quantization}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prepared_config(queue: QueueSettings, config_path: Path) -> None:
    manifest_path = queue.root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"queue is not prepared: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_digest = manifest.get("config_sha256", "")
    actual_digest = file_sha256(config_path.expanduser().resolve())
    if expected_digest != actual_digest:
        raise RuntimeError(
            "sweep config changed after queue preparation; rerun prepare "
            "(use --reset only if existing queue state should be discarded)"
        )


def all_state_paths(root: Path, identifier: str) -> list[Path]:
    paths: list[Path] = []
    for state in ("pending", "claimed", "completed", "failed"):
        paths.extend((root / state).glob(f"{identifier}*.json"))
    return paths


def write_worker_script(
    config_path: Path, queue: QueueSettings
) -> None:
    directives = ["#!/bin/bash", "#PBS -l select=1"]
    if queue.walltime:
        directives.append(f"#PBS -l walltime={queue.walltime}")
    if queue.place:
        directives.append(f"#PBS -l place={queue.place}")
    if queue.queue:
        directives.append(f"#PBS -q {queue.queue}")
    if queue.account:
        directives.append(f"#PBS -A {queue.account}")
    directives.extend(
        [
            "#PBS -N qdrant-sweep-worker",
            "",
            "set -euo pipefail",
            'cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"',
        ]
    )
    directives.extend(queue.setup_commands)
    directives.append(
        f"python3 {shell_quote(SCRIPT_PATH)} {shell_quote(config_path)} worker"
    )
    path = queue.root / "worker.pbs.sh"
    path.write_text("\n".join(directives) + "\n", encoding="utf-8")
    path.chmod(0o755)


def shell_quote(value: Path | str) -> str:
    import shlex

    return shlex.quote(str(value))


def prepare_queue(config_path: Path, reset: bool) -> int:
    (
        resolved_config,
        _,
        _,
        datasets,
        sweep,
        variants,
        queue,
    ) = load_context(config_path)
    if reset and queue.root.exists():
        shutil.rmtree(queue.root)
    ensure_queue_dirs(queue.root)
    config_digest = file_sha256(resolved_config)
    manifest_path = queue.root / "manifest.json"
    if manifest_path.is_file() and not reset:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_digest:
            raise RuntimeError(
                "sweep config changed after queue preparation; use prepare --reset "
                "to discard old queue state before preparing the new sweep"
            )
    created = 0
    existing = 0
    for dataset in datasets:
        for segments in sweep["number_of_segments"]:
            for variant in variants:
                identifier = unit_id(dataset.name, segments, variant.name)
                payload = {
                    "schema_version": 1,
                    "unit_id": identifier,
                    "dataset": dataset.name,
                    "number_of_segments": segments,
                    "quantization_variant": variant.name,
                    "config_path": str(resolved_config),
                    "config_sha256": config_digest,
                    "created_at": time.time(),
                }
                if all_state_paths(queue.root, identifier):
                    existing += 1
                    continue
                atomic_write_json(queue.root / "pending" / f"{identifier}.json", payload)
                created += 1
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "config_path": str(resolved_config),
            "config_sha256": config_digest,
            "work_unit": "dataset + number_of_segments + quantization_variant",
            "created_at": time.time(),
        },
    )
    write_worker_script(resolved_config, queue)
    print(f"queue: {queue.root}")
    print(f"created units: {created}")
    print(f"existing units: {existing}")
    print(f"query configurations per unit: {len(local.effective_query_pairs(sweep)) * len(sweep['hnsw_m']) * len(sweep['ef_construct'])}")
    print(f"worker script: {queue.root / 'worker.pbs.sh'}")
    return 0


def default_worker_id() -> str:
    job_id = os.environ.get("PBS_JOBID", "interactive").split(".", 1)[0]
    return local.safe_name(f"{socket.gethostname()}_{job_id}_{os.getpid()}")


def claim_unit(queue: QueueSettings, worker_id: str) -> Path | None:
    for pending in sorted((queue.root / "pending").glob("*.json")):
        claimed = queue.root / "claimed" / f"{pending.stem}__{worker_id}.json"
        try:
            os.rename(pending, claimed)
        except FileNotFoundError:
            continue
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        payload.update(
            {
                "worker_id": worker_id,
                "claimed_at": time.time(),
                "pbs_job_id": os.environ.get("PBS_JOBID", ""),
                "hostname": socket.gethostname(),
            }
        )
        atomic_write_json(claimed, payload)
        return claimed
    return None


class Heartbeat:
    def __init__(
        self, queue: QueueSettings, identifier: str, worker_id: str
    ) -> None:
        self.path = queue.root / "heartbeats" / f"{identifier}.json"
        self.interval = queue.heartbeat_seconds
        self.payload = {
            "unit_id": identifier,
            "worker_id": worker_id,
            "pbs_job_id": os.environ.get("PBS_JOBID", ""),
            "hostname": socket.gethostname(),
        }
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _write(self) -> None:
        atomic_write_json(self.path, {**self.payload, "heartbeat_at": time.time()})

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._write()

    def __enter__(self):
        self._write()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 1)
        self.path.unlink(missing_ok=True)


class WorkUnitDeferred(RuntimeError):
    """Raised when a worker reaches its walltime reserve."""


def parse_walltime_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid PBS walltime {value!r}; expected HH:MM:SS")
    hours, minutes, seconds = map(int, parts)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid PBS walltime {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def pbs_remaining_seconds(queue: QueueSettings) -> int:
    configured = parse_walltime_seconds(queue.walltime)
    job_id = os.environ.get("PBS_JOBID", "").strip()
    if not job_id or shutil.which("qstat") is None:
        return configured
    try:
        result = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return configured
    if result.returncode != 0:
        return configured
    allocated_match = re.search(
        r"Resource_List\.walltime\s*=\s*(\d+:\d{2}:\d{2})",
        result.stdout,
    )
    used_match = re.search(
        r"resources_used\.walltime\s*=\s*(\d+:\d{2}:\d{2})",
        result.stdout,
    )
    allocated = (
        parse_walltime_seconds(allocated_match.group(1))
        if allocated_match
        else configured
    )
    used = parse_walltime_seconds(used_match.group(1)) if used_match else 0
    return max(0, allocated - used)


class WorkerDeadline:
    def __init__(self, queue: QueueSettings) -> None:
        self.reserve_seconds = queue.stop_before_seconds
        remaining = pbs_remaining_seconds(queue)
        usable = max(0, remaining - self.reserve_seconds)
        self.deadline = time.monotonic() + usable
        self.initial_remaining = remaining

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self) -> None:
        if self.seconds_left() <= 0:
            raise WorkUnitDeferred(
                "PBS walltime reserve reached; returning work unit to pending"
            )

    def command_timeout(self) -> float:
        self.check()
        return max(1.0, self.seconds_left())


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_deadline_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    deadline: WorkerDeadline,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=local.command_env(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=deadline.command_timeout())
        except subprocess.TimeoutExpired as exc:
            terminate_process_group(process)
            raise WorkUnitDeferred(
                "PBS walltime reserve reached during command: "
                f"{' '.join(command)}"
            ) from exc
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: "
            f"{' '.join(command)}; see {log_path}"
        )


class ApptainerQdrant:
    def __init__(
        self,
        queue: QueueSettings,
        run_settings,
        scratch: Path,
        log_path: Path,
        deadline: WorkerDeadline,
    ) -> None:
        self.queue = queue
        self.settings = run_settings
        self.scratch = scratch
        self.log_path = log_path
        self.deadline = deadline
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None

    def __enter__(self):
        self.deadline.check()
        if not self.queue.sif.is_file():
            raise RuntimeError(f"Qdrant SIF not found: {self.queue.sif}")
        if shutil.which("apptainer") is None:
            raise RuntimeError("apptainer is required on the worker node")
        for name in ("storage", "config", "snapshots"):
            (self.scratch / name).mkdir(parents=True, exist_ok=True)
        command = [
            "apptainer",
            "run",
            "--cleanenv",
            "--writable-tmpfs",
            "--bind",
            f"{self.scratch / 'storage'}:/qdrant/storage",
            "--bind",
            f"{self.scratch / 'config'}:/qdrant/config/local",
            "--bind",
            f"{self.scratch / 'snapshots'}:/qdrant/snapshots",
            *self.queue.apptainer_args,
            str(self.queue.sif),
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        for name in (
            "NO_PROXY",
            "no_proxy",
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
        ):
            env[name] = ""
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.scratch,
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            health_url = (
                f"http://{self.settings.host}:{self.settings.http_port}/healthz"
            )
            for _ in range(120):
                self.deadline.check()
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Qdrant Apptainer process exited; see {self.log_path}"
                    )
                try:
                    with urllib.request.urlopen(health_url, timeout=2) as response:
                        if response.status == 200:
                            return self
                except OSError:
                    time.sleep(1)
            raise RuntimeError(f"Qdrant did not become healthy at {health_url}")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is not None:
            terminate_process_group(self.process)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __exit__(self, *_):
        self.stop()


def find_named(items, name: str, label: str):
    for item in items:
        if item.name == name:
            return item
    raise ValueError(f"unknown {label} in work unit: {name}")


def unit_result_row(
    dataset,
    variant,
    run_settings,
    segments: int,
    segment_size_kb: int,
    actual_segments: Any,
    hnsw_m: int,
    ef_construct: int,
    ef_search: int,
    top_k: int,
    insert_time: float,
    index_time: float,
    query_dir: Path,
) -> dict[str, Any]:
    data_size_bytes = (
        dataset.data_meta.rows
        * dataset.data_meta.columns
        * dataset.data_meta.item_size
    )
    return {
        "run_key": local.make_run_key(
            dataset,
            segments,
            variant,
            hnsw_m,
            ef_construct,
            ef_search,
            top_k,
        ),
        "status": "failed",
        "dataset": dataset.name,
        "data_file": dataset.data,
        "query_file": dataset.queries,
        "ground_truth_file": dataset.ground_truth,
        "corpus_size": dataset.data_meta.rows,
        "query_count": dataset.query_meta.rows,
        "vector_dim": dataset.data_meta.columns,
        "data_size_bytes": data_size_bytes,
        "distance_metric": dataset.distance_metric,
        "qdrant_image": run_settings.image,
        "number_of_segments": segments,
        "segment_size_kb": segment_size_kb,
        "actual_segments": actual_segments,
        "quantization_variant": variant.name,
        "quantization": variant.quantization_type,
        "quantization_always_ram": variant.always_ram,
        "quantization_scalar_quantile": variant.scalar_quantile,
        "quantization_binary_encoding": variant.binary_encoding,
        "quantization_product_compression": variant.product_compression,
        "quantization_turbo_bits": variant.turbo_bits,
        "hnsw_m": hnsw_m,
        "ef_construct": ef_construct,
        "ef_search": ef_search,
        "top_k": top_k,
        "top_k_execution_mode": "INDIVIDUAL",
        "insert_time_s": insert_time,
        "index_time_s": index_time,
        "result_dir": query_dir,
    }


def execute_unit(
    payload: dict[str, Any],
    run_settings,
    datasets,
    sweep,
    variants,
    queue: QueueSettings,
    deadline: WorkerDeadline,
) -> None:
    deadline.check()
    dataset = find_named(datasets, payload["dataset"], "dataset")
    variant = find_named(
        variants, payload["quantization_variant"], "quantization variant"
    )
    segments = int(payload["number_of_segments"])
    identifier = payload["unit_id"]
    unit_dir = queue.root / "units" / identifier
    unit_dir.mkdir(parents=True, exist_ok=True)
    results_csv = unit_dir / "results.csv"
    completed = local.completed_run_keys(results_csv)
    run_settings = replace(
        run_settings,
        output_dir=unit_dir,
        results_csv=results_csv,
        image=f"apptainer:{queue.sif}",
        host="127.0.0.1",
    )
    scratch = queue.scratch_root / f"qdrant_sweep_{identifier}_{os.getpid()}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    data_size_bytes = (
        dataset.data_meta.rows
        * dataset.data_meta.columns
        * dataset.data_meta.item_size
    )
    segment_size_kb = local.math.ceil(data_size_bytes / segments / 1024)
    collection_name = local.safe_name(f"pbs_{identifier}")[:200]
    build_dir = unit_dir / "build"
    local.write_registry(build_dir / "ip_registry.txt", run_settings)
    base_env = local.base_environment(
        run_settings,
        dataset,
        collection_name,
        segments,
        segment_size_kb,
        variant,
        sweep["top_k"],
    )
    query_pairs = local.effective_query_pairs(sweep)

    try:
        with ApptainerQdrant(
            queue, run_settings, scratch, unit_dir / "qdrant.log", deadline
        ):
            initial_m = sweep["hnsw_m"][0]
            initial_ef_construct = sweep["ef_construct"][0]
            insert_env = dict(base_env)
            insert_env.update(
                {
                    "ACTIVE_TASK": "INSERT",
                    "HNSW_M": str(initial_m),
                    "HNSW_EF_CONSTRUCTION": str(initial_ef_construct),
                }
            )
            local.write_run_config(build_dir, insert_env)
            run_deadline_command(
                [sys.executable, str(local.CONFIGURE_COLLECTION)],
                build_dir,
                insert_env,
                build_dir / "configure.log",
                deadline,
            )
            (build_dir / "ready.flag").unlink(missing_ok=True)
            run_deadline_command(
                [str(run_settings.batch_client)],
                build_dir,
                insert_env,
                build_dir / "insert.log",
                deadline,
            )
            insert_time = local.read_rank_zero_total(build_dir / "insert_times.csv")

            for hnsw_m in sweep["hnsw_m"]:
                for ef_construct in sweep["ef_construct"]:
                    deadline.check()
                    pending = [
                        pair
                        for pair in query_pairs
                        if local.make_run_key(
                            dataset,
                            segments,
                            variant,
                            hnsw_m,
                            ef_construct,
                            *pair,
                        )
                        not in completed
                    ]
                    if not pending:
                        continue
                    index_dir = (
                        unit_dir
                        / f"m_{hnsw_m}"
                        / f"ef_construct_{ef_construct}"
                    )
                    local.write_registry(index_dir / "ip_registry.txt", run_settings)
                    index_env = dict(base_env)
                    index_env.update(
                        {
                            "ACTIVE_TASK": "INDEX",
                            "HNSW_M": str(hnsw_m),
                            "HNSW_EF_CONSTRUCTION": str(ef_construct),
                        }
                    )
                    local.write_run_config(index_dir, index_env)
                    run_deadline_command(
                        [sys.executable, str(local.BUILD_INDEX)],
                        index_dir,
                        index_env,
                        index_dir / "index.log",
                        deadline,
                    )
                    index_time = float(
                        (index_dir / "index_time.txt").read_text(encoding="utf-8")
                    )
                    info = local.query_collection_info(run_settings, collection_name)
                    actual_segments = info.get("segments_count", "")

                    for ef_search, top_k in pending:
                        deadline.check()
                        query_dir = (
                            index_dir
                            / f"ef_search_{ef_search}"
                            / f"top_k_{top_k}"
                        )
                        local.write_registry(
                            query_dir / "ip_registry.txt", run_settings
                        )
                        query_env = dict(index_env)
                        query_env.update(
                            {
                                "ACTIVE_TASK": "QUERY",
                                "HNSW_EF_SEARCH": str(ef_search),
                                "TOP_K": str(top_k),
                            }
                        )
                        local.write_run_config(query_dir, query_env)
                        row = unit_result_row(
                            dataset,
                            variant,
                            run_settings,
                            segments,
                            segment_size_kb,
                            actual_segments,
                            hnsw_m,
                            ef_construct,
                            ef_search,
                            top_k,
                            insert_time,
                            index_time,
                            query_dir,
                        )
                        try:
                            run_deadline_command(
                                [str(run_settings.batch_client)],
                                query_dir,
                                query_env,
                                query_dir / "query.log",
                                deadline,
                            )
                            row["query_time_s"] = local.read_rank_zero_total(
                                query_dir / "query_times.csv"
                            )
                            run_deadline_command(
                                [
                                    sys.executable,
                                    str(local.COMPUTE_RECALL),
                                    str(dataset.ground_truth),
                                    str(query_dir / "query_result_ids.npy"),
                                    str(top_k),
                                    "--output",
                                    str(query_dir / "recall.csv"),
                                ],
                                query_dir,
                                query_env,
                                query_dir / "recall.log",
                                deadline,
                            )
                            row.update(
                                local.read_single_csv_row(query_dir / "recall.csv")
                            )
                            row["status"] = "success"
                            completed.add(row["run_key"])
                        except WorkUnitDeferred:
                            raise
                        except Exception as exc:
                            row["error"] = str(exc)
                            local.append_result(results_csv, row)
                            raise
                        local.append_result(results_csv, row)
                        print(
                            f"{identifier}: {row['run_key']} "
                            f"recall={row['mean_recall_at_k']}",
                            flush=True,
                        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def finish_claim(
    claimed: Path,
    queue: QueueSettings,
    state: str,
    error: str = "",
) -> None:
    payload = json.loads(claimed.read_text(encoding="utf-8"))
    payload.update(
        {
            "finished_at": time.time(),
            "final_state": state,
            "error": error,
        }
    )
    atomic_write_json(claimed, payload)
    destination = queue.root / state / f"{payload['unit_id']}.json"
    os.replace(claimed, destination)


def release_claim(claimed: Path, queue: QueueSettings, reason: str) -> None:
    payload = json.loads(claimed.read_text(encoding="utf-8"))
    payload.update(
        {
            "released_at": time.time(),
            "release_reason": reason,
        }
    )
    atomic_write_json(claimed, payload)
    destination = queue.root / "pending" / f"{payload['unit_id']}.json"
    os.replace(claimed, destination)


def run_worker(config_path: Path, worker_id: str, max_units: int | None) -> int:
    (
        _,
        _,
        run_settings,
        datasets,
        sweep,
        variants,
        queue,
    ) = load_context(config_path)
    ensure_queue_dirs(queue.root)
    verify_prepared_config(queue, config_path)
    worker_id = local.safe_name(worker_id or default_worker_id())
    maximum = max_units if max_units is not None else queue.worker_max_units
    nonnegative_int(maximum, "--max-units")
    deadline = WorkerDeadline(queue)
    print(
        f"Worker walltime remaining at startup: {deadline.initial_remaining}s; "
        f"reserve: {deadline.reserve_seconds}s",
        flush=True,
    )
    processed = 0
    while maximum == 0 or processed < maximum:
        try:
            deadline.check()
        except WorkUnitDeferred:
            print("Walltime reserve reached before claiming another unit.")
            break
        claimed = claim_unit(queue, worker_id)
        if claimed is None:
            print("No pending work units.")
            break
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        identifier = payload["unit_id"]
        print(f"Claimed {identifier}", flush=True)
        with Heartbeat(queue, identifier, worker_id):
            try:
                execute_unit(
                    payload,
                    run_settings,
                    datasets,
                    sweep,
                    variants,
                    queue,
                    deadline,
                )
            except WorkUnitDeferred as exc:
                release_claim(claimed, queue, str(exc))
                print(
                    f"Released {identifier} back to pending: {exc}",
                    flush=True,
                )
                return 0
            except Exception as exc:
                finish_claim(claimed, queue, "failed", str(exc))
                print(f"Failed {identifier}: {exc}", file=sys.stderr, flush=True)
                return 1
            else:
                finish_claim(claimed, queue, "completed")
                print(f"Completed {identifier}", flush=True)
        processed += 1
    return 0


def queue_status(queue: QueueSettings) -> int:
    ensure_queue_dirs(queue.root)
    for state in ("pending", "claimed", "completed", "failed"):
        count = sum(1 for _ in (queue.root / state).glob("*.json"))
        print(f"{state}: {count}")
    print(
        "heartbeats: "
        f"{sum(1 for _ in (queue.root / 'heartbeats').glob('*.json'))}"
    )
    return 0


def queue_state_counts(queue: QueueSettings) -> dict[str, int]:
    return {
        state: sum(1 for _ in (queue.root / state).glob("*.json"))
        for state in ("pending", "claimed", "completed", "failed")
    }


def queue_matches(actual: str, requested: str) -> bool:
    if requested == "debug":
        return actual == "debug"
    if requested == "debug-scaling":
        return actual == "debug-scaling" or actual.startswith("debug-s")
    return actual == requested


def parse_qstat_occupancy(output: str, queue_names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    occupancy = {
        name: {"jobs": 0, "queued": 0}
        for name in queue_names
    }
    for line in output.splitlines():
        fields = line.split()
        if not fields or not fields[0][:1].isdigit() or len(fields) < 4:
            continue
        actual_queue = fields[2]
        state = fields[-2]
        for name in queue_names:
            if queue_matches(actual_queue, name):
                occupancy[name]["jobs"] += 1
                if state == "Q":
                    occupancy[name]["queued"] += 1
                break
    return occupancy


def qstat_occupancy(queue: QueueSettings) -> dict[str, dict[str, int]]:
    if shutil.which("qstat") is None:
        raise RuntimeError("qstat is required for the watch command")
    if not queue.submit_username:
        raise ValueError("pbs.submit_username is required for the watch command")
    result = subprocess.run(
        ["qstat", "-u", queue.submit_username],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"qstat failed for {queue.submit_username}: {result.stderr.strip()}"
        )
    return parse_qstat_occupancy(result.stdout, queue.queue_candidates)


def submit_worker(queue: QueueSettings, queue_name: str) -> str:
    if shutil.which("qsub") is None:
        raise RuntimeError("qsub is required for the watch command")
    worker_script = queue.root / "worker.pbs.sh"
    if not worker_script.is_file():
        raise RuntimeError(f"missing worker script: {worker_script}; run prepare")
    result = subprocess.run(
        ["qsub", "-q", queue_name, worker_script.name],
        cwd=queue.root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"qsub failed for queue {queue_name}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


class WatchLock:
    def __init__(self, queue: QueueSettings) -> None:
        self.path = queue.root / "watch.lock"

    def __enter__(self):
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise RuntimeError(
                f"another watcher appears active: {self.path}; "
                "remove it only after confirming that watcher has stopped"
            ) from exc
        atomic_write_json(
            self.path / "owner.json",
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_at": time.time(),
            },
        )
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.path, ignore_errors=True)


def watch_queue(queue: QueueSettings, poll_seconds: int | None) -> int:
    if not queue.queue_candidates:
        raise ValueError(
            "pbs.queue_candidates and pbs.queue_limits are required for watch"
        )
    interval = poll_seconds or queue.watch_poll_seconds
    positive_int(interval, "--poll-seconds")
    ensure_queue_dirs(queue.root)
    manifest = queue.root / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"queue is not prepared: missing {manifest}")

    with WatchLock(queue):
        while True:
            requeue_stale(queue, queue.stale_after_seconds, quiet=True)
            counts = queue_state_counts(queue)
            if counts["failed"]:
                print(
                    f"Stopping watcher: {counts['failed']} failed unit(s). "
                    "Inspect failed/ and run requeue-failed after correcting the issue.",
                    file=sys.stderr,
                )
                return 1
            if counts["pending"] == 0 and counts["claimed"] == 0:
                print(
                    f"Sweep complete: {counts['completed']} units completed.",
                    flush=True,
                )
                if queue.aggregate_on_complete:
                    aggregate_results(queue, None)
                return 0

            occupancy = qstat_occupancy(queue)
            occupancy_text = []
            submitted = 0
            remaining_pending = counts["pending"]
            for index, queue_name in enumerate(queue.queue_candidates):
                jobs = occupancy[queue_name]["jobs"]
                queued = occupancy[queue_name]["queued"]
                job_limit = queue.queue_limits[index]
                queued_limit = queue.queue_queued_limits[index]
                occupancy_text.append(
                    f"{queue_name}=jobs:{jobs}/{job_limit},"
                    f"queued:{queued}/{queued_limit}"
                )
                openings = max(
                    0,
                    min(job_limit - jobs, queued_limit - queued),
                )
                for _ in range(min(openings, remaining_pending)):
                    job_id = submit_worker(queue, queue_name)
                    submitted += 1
                    remaining_pending -= 1
                    jobs += 1
                    queued += 1
                    print(
                        f"Submitted worker {job_id} to {queue_name}",
                        flush=True,
                    )
            print(
                "Queue occupancy: "
                + " ".join(occupancy_text)
                + f" | units pending={counts['pending']} "
                f"claimed={counts['claimed']} completed={counts['completed']} "
                f"submitted={submitted}",
                flush=True,
            )
            time.sleep(interval)


def requeue_stale(
    queue: QueueSettings, stale_after: int | None, quiet: bool = False
) -> int:
    threshold = stale_after or queue.stale_after_seconds
    positive_int(threshold, "--stale-after")
    now = time.time()
    requeued = 0
    for claimed in sorted((queue.root / "claimed").glob("*.json")):
        payload = json.loads(claimed.read_text(encoding="utf-8"))
        identifier = payload["unit_id"]
        heartbeat = queue.root / "heartbeats" / f"{identifier}.json"
        timestamp = heartbeat.stat().st_mtime if heartbeat.exists() else claimed.stat().st_mtime
        if now - timestamp <= threshold:
            continue
        destination = queue.root / "pending" / f"{identifier}.json"
        try:
            os.rename(claimed, destination)
        except FileNotFoundError:
            continue
        heartbeat.unlink(missing_ok=True)
        requeued += 1
        if not quiet:
            print(f"Requeued stale unit: {identifier}")
    if not quiet:
        print(f"requeued: {requeued}")
    return 0


def requeue_failed(queue: QueueSettings) -> int:
    requeued = 0
    for failed in sorted((queue.root / "failed").glob("*.json")):
        payload = json.loads(failed.read_text(encoding="utf-8"))
        destination = queue.root / "pending" / f"{payload['unit_id']}.json"
        try:
            os.rename(failed, destination)
        except FileNotFoundError:
            continue
        requeued += 1
        print(f"Requeued failed unit: {payload['unit_id']}")
    print(f"requeued: {requeued}")
    return 0


def aggregate_results(queue: QueueSettings, output: Path | None) -> int:
    destination = (
        output.expanduser().resolve()
        if output is not None
        else queue.root / "results.csv"
    )
    source_paths = sorted((queue.root / "units").glob("*/results.csv"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=local.RESULT_FIELDS)
        writer.writeheader()
        for path in source_paths:
            with path.open(newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    writer.writerow(
                        {field: row.get(field, "") for field in local.RESULT_FIELDS}
                    )
                    rows += 1
    print(f"wrote {destination} with {rows} rows")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare_queue(args.config, args.reset)

    _, _, _, _, _, _, queue = load_context(args.config)
    if args.command == "worker":
        return run_worker(args.config, args.worker_id, args.max_units)
    if args.command == "status":
        return queue_status(queue)
    if args.command == "watch":
        verify_prepared_config(queue, args.config)
        return watch_queue(queue, args.poll_seconds)
    if args.command == "requeue-stale":
        return requeue_stale(queue, args.stale_after)
    if args.command == "requeue-failed":
        return requeue_failed(queue)
    if args.command == "aggregate":
        return aggregate_results(queue, args.output)
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
