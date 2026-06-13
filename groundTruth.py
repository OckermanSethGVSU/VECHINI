import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("Missing dependency: numpy") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("Missing dependency: torch") from exc

try:
    from mpi4py import MPI
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("Missing dependency: mpi4py") from exc


@dataclass
class BatchRequest:
    batch_id: int
    start: int
    stop: int
    ids_send: np.ndarray
    dist_send: np.ndarray
    ids_recv: np.ndarray | None
    dist_recv: np.ndarray | None
    requests: list[Any]
    compute_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact multi-GPU MPI ground-truth computation."
    )
    parser.add_argument("--dataset", required=True, help="Dataset .npy file, shape [N, D].")
    parser.add_argument("--queries", required=True, help="Query .npy file, shape [Q, D].")
    parser.add_argument("--output", required=True, help="Output .npz file written by rank 0.")
    parser.add_argument("--k", type=int, required=True, help="Number of nearest neighbors.")
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=("cosine", "dot", "ip", "euclidean", "l2"),
        help=(
            "Distance metric. 'dot' and 'ip' select maximum raw inner product; "
            "'euclidean' and 'l2' select minimum squared-L2 distance. "
            "Default: cosine."
        ),
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=1024,
        help="Number of queries processed per GPU batch.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=3,
        help="Maximum outstanding MPI gather batches.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Use cuda for GPU execution or cpu for local debugging.",
    )
    parser.add_argument(
        "--normalize-queries-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For cosine only, normalize all queries once at startup instead "
            "of per batch."
        ),
    )
    parser.add_argument(
        "--deterministic-ties",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guarantee distance-then-ID ordering for exact equal-distance ties.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print rank-level timing information.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=0.0,
        help="Rank 0 prints progress at most this often in seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def local_rank(comm: MPI.Comm) -> int:
    for key in (
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "PMI_LOCAL_RANK",
    ):
        value = os.environ.get(key)
        if value is not None:
            return int(value)

    shared = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        return shared.Get_rank()
    finally:
        shared.Free()


def shard_bounds(total_rows: int, rank: int, size: int) -> tuple[int, int]:
    base = total_rows // size
    rem = total_rows % size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def load_array(path: str, mmap_mode: str | None = None) -> np.ndarray:
    arr = np.load(path, mmap_mode=mmap_mode)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2-D .npy array, got shape {arr.shape}")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def normalize_numpy_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    np.maximum(norms, np.finfo(np.float32).eps, out=norms)
    return arr / norms


def normalize_torch_rows(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tensor, p=2, dim=1, eps=1e-12)


def select_device(args: argparse.Namespace, comm: MPI.Comm) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("CUDA was requested but no CUDA devices are visible")

    gpu_id = local_rank(comm) % gpu_count
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def load_dataset_shard(
    path: str,
    rank: int,
    size: int,
    device: torch.device,
    metric: str,
) -> tuple[torch.Tensor, int, int, int, int]:
    dataset = load_array(path, mmap_mode="r")
    total_rows, dim = dataset.shape
    start, stop = shard_bounds(total_rows, rank, size)
    if start == stop:
        raise ValueError(
            f"Rank {rank} received an empty dataset shard. "
            f"Use at most {total_rows} ranks for this dataset."
        )

    # The slice is contiguous by construction, which keeps Lustre reads large.
    shard = np.asarray(dataset[start:stop], dtype=np.float32, order="C")
    shard_t = torch.from_numpy(shard).to(device=device, non_blocking=False)
    if metric == "cosine":
        shard_t = normalize_torch_rows(shard_t)
    return shard_t, start, stop, total_rows, dim


def local_topk(
    query_batch: np.ndarray,
    shard: torch.Tensor,
    shard_start: int,
    k: int,
    device: torch.device,
    metric: str,
    queries_already_normalized: bool,
    deterministic_ties: bool,
) -> tuple[np.ndarray, np.ndarray]:
    query_t = torch.from_numpy(np.asarray(query_batch, dtype=np.float32, order="C"))
    query_t = query_t.to(device=device, non_blocking=False)
    if metric == "cosine" and not queries_already_normalized:
        query_t = normalize_torch_rows(query_t)

    local_k = min(k, shard.shape[0])
    with torch.no_grad():
        products = query_t @ shard.T
        if metric == "euclidean":
            query_norms = torch.sum(query_t * query_t, dim=1, keepdim=True)
            shard_norms = torch.sum(shard * shard, dim=1).unsqueeze(0)
            distances = torch.clamp(
                query_norms + shard_norms - 2.0 * products,
                min=0.0,
            )
            ranking_scores = -distances
        else:
            ranking_scores = products

        values, indices = torch.topk(
            ranking_scores,
            local_k,
            dim=1,
            largest=True,
            sorted=False,
        )
        if deterministic_ties:
            fix_local_ties(ranking_scores, values, indices, local_k)
        if metric == "cosine":
            distances = 1.0 - values
        else:
            distances = -values

    batch_rows = query_batch.shape[0]
    ids_out = np.full((batch_rows, k), -1, dtype=np.int64)
    dist_out = np.full((batch_rows, k), np.inf, dtype=np.float32)

    ids = (indices + shard_start).to(dtype=torch.int64).cpu().numpy()
    dist = distances.to(dtype=torch.float32).cpu().numpy()
    ids_out[:, :local_k] = ids
    dist_out[:, :local_k] = dist
    return ids_out, dist_out


def fix_local_ties(
    scores: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    local_k: int,
) -> None:
    """Ensure local candidates include lowest IDs for exact boundary ties.

    torch.topk does not define which equal-valued items it returns. For exact
    ground truth with ID tie-breaking, that matters only when the kth local
    score is tied with candidates outside the returned set.
    """
    for row in range(scores.shape[0]):
        threshold = values[row].min()
        selected_equal = torch.count_nonzero(values[row] == threshold)
        total_equal = torch.count_nonzero(scores[row] == threshold)
        if total_equal <= selected_equal:
            continue

        better_idx = torch.nonzero(scores[row] > threshold, as_tuple=False).flatten()
        equal_idx = torch.nonzero(scores[row] == threshold, as_tuple=False).flatten()
        needed_equal = local_k - better_idx.numel()
        if needed_equal <= 0:
            continue

        # equal_idx is ascending by local/global ID for a contiguous shard.
        chosen = torch.cat((better_idx, equal_idx[:needed_equal]))
        indices[row, :] = chosen
        values[row, :] = scores[row, chosen]


def post_gather(
    comm: MPI.Comm,
    batch_id: int,
    start: int,
    stop: int,
    ids_send: np.ndarray,
    dist_send: np.ndarray,
    compute_seconds: float,
) -> BatchRequest:
    rank = comm.Get_rank()
    size = comm.Get_size()
    ids_recv = None
    dist_recv = None
    if rank == 0:
        dist_recv = np.empty((size, *dist_send.shape), dtype=np.float32)
        ids_recv = np.empty((size, *ids_send.shape), dtype=np.int64)

    # All ranks post collectives in the same order for every batch.
    req_dist = comm.Igather(dist_send, dist_recv, root=0)
    req_ids = comm.Igather(ids_send, ids_recv, root=0)
    return BatchRequest(
        batch_id=batch_id,
        start=start,
        stop=stop,
        ids_send=ids_send,
        dist_send=dist_send,
        ids_recv=ids_recv,
        dist_recv=dist_recv,
        requests=[req_dist, req_ids],
        compute_seconds=compute_seconds,
    )


def merge_batch(
    batch: BatchRequest,
    result_ids: np.ndarray,
    result_distances: np.ndarray,
    k: int,
) -> float:
    if batch.ids_recv is None or batch.dist_recv is None:
        return 0.0

    merge_start = time.perf_counter()
    rows = batch.stop - batch.start
    candidates_ids = batch.ids_recv.transpose(1, 0, 2).reshape(rows, -1)
    candidates_dist = batch.dist_recv.transpose(1, 0, 2).reshape(rows, -1)

    for row in range(rows):
        ids = candidates_ids[row]
        distances = candidates_dist[row]
        valid = ids >= 0
        ids = ids[valid]
        distances = distances[valid]
        order = np.lexsort((ids, distances))[:k]
        result_ids[batch.start + row, :] = ids[order]
        result_distances[batch.start + row, :] = distances[order]

    return time.perf_counter() - merge_start


def complete_ready_batches(
    inflight: list[BatchRequest],
    result_ids: np.ndarray | None,
    result_distances: np.ndarray | None,
    k: int,
    force_one: bool = False,
) -> tuple[float, int]:
    merge_seconds = 0.0
    completed = 0

    index = 0
    while index < len(inflight):
        batch = inflight[index]
        ready = MPI.Request.Testall(batch.requests)
        if not ready and force_one and completed == 0:
            MPI.Request.Waitall(batch.requests)
            ready = True

        if ready:
            if result_ids is not None and result_distances is not None:
                merge_seconds += merge_batch(batch, result_ids, result_distances, k)
            inflight.pop(index)
            completed += 1
        else:
            index += 1

    return merge_seconds, completed


def print_progress(
    completed_batches: int,
    completed_queries: int,
    total_batches: int,
    total_queries: int,
    inflight_count: int,
) -> None:
    percent = 100.0 * completed_queries / max(total_queries, 1)
    print(
        "progress batches={completed_batches}/{total_batches} "
        "queries={completed_queries}/{total_queries} "
        "({percent:.1f}%) inflight={inflight_count}".format(
            completed_batches=completed_batches,
            total_batches=total_batches,
            completed_queries=completed_queries,
            total_queries=total_queries,
            percent=percent,
            inflight_count=inflight_count,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    metric = {
        "cosine": "cosine",
        "dot": "dot",
        "ip": "dot",
        "euclidean": "euclidean",
        "l2": "euclidean",
    }[args.metric]
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.max_inflight <= 0:
        raise ValueError("--max-inflight must be positive")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    device = select_device(args, comm)

    t0 = time.perf_counter()
    queries = load_array(args.queries, mmap_mode=None)
    if metric == "cosine" and args.normalize_queries_once:
        queries = normalize_numpy_rows(np.asarray(queries, dtype=np.float32, order="C"))
    else:
        queries = np.asarray(queries, dtype=np.float32, order="C")

    shard, shard_start, shard_stop, total_rows, dataset_dim = load_dataset_shard(
        args.dataset, rank, size, device, metric
    )
    if queries.shape[1] != dataset_dim:
        raise ValueError(
            f"Dimension mismatch: queries have dim {queries.shape[1]}, "
            f"dataset has dim {dataset_dim}"
        )
    if args.k > total_rows:
        raise ValueError(f"--k={args.k} exceeds dataset rows={total_rows}")

    if rank == 0:
        result_ids = np.empty((queries.shape[0], args.k), dtype=np.int64)
        result_distances = np.empty((queries.shape[0], args.k), dtype=np.float32)
    else:
        result_ids = None
        result_distances = None

    load_seconds = time.perf_counter() - t0
    compute_seconds = 0.0
    merge_seconds = 0.0
    completed_batches = 0
    completed_queries = 0
    total_batches = (queries.shape[0] + args.query_batch_size - 1) // args.query_batch_size
    last_progress = time.perf_counter()
    inflight: list[BatchRequest] = []

    for batch_id, start in enumerate(range(0, queries.shape[0], args.query_batch_size)):
        while len(inflight) >= args.max_inflight:
            merged, completed = complete_ready_batches(
                inflight, result_ids, result_distances, args.k, force_one=True
            )
            merge_seconds += merged
            completed_batches += completed
            completed_queries = min(completed_batches * args.query_batch_size, queries.shape[0])

        stop = min(start + args.query_batch_size, queries.shape[0])
        compute_start = time.perf_counter()
        ids_send, dist_send = local_topk(
            queries[start:stop],
            shard,
            shard_start,
            args.k,
            device,
            metric,
            metric == "cosine" and args.normalize_queries_once,
            args.deterministic_ties,
        )
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        batch_compute_seconds = time.perf_counter() - compute_start
        compute_seconds += batch_compute_seconds

        inflight.append(
            post_gather(
                comm,
                batch_id,
                start,
                stop,
                ids_send,
                dist_send,
                batch_compute_seconds,
            )
        )
        merged, completed = complete_ready_batches(
            inflight, result_ids, result_distances, args.k, force_one=False
        )
        merge_seconds += merged
        completed_batches += completed
        completed_queries = min(completed_batches * args.query_batch_size, queries.shape[0])
        now = time.perf_counter()
        if (
            rank == 0
            and args.progress_interval > 0
            and now - last_progress >= args.progress_interval
        ):
            print_progress(
                completed_batches,
                completed_queries,
                total_batches,
                queries.shape[0],
                len(inflight),
            )
            last_progress = now

    while inflight:
        merged, completed = complete_ready_batches(
            inflight, result_ids, result_distances, args.k, force_one=True
        )
        merge_seconds += merged
        completed_batches += completed
        completed_queries = min(completed_batches * args.query_batch_size, queries.shape[0])
        now = time.perf_counter()
        if (
            rank == 0
            and args.progress_interval > 0
            and now - last_progress >= args.progress_interval
        ):
            print_progress(
                completed_batches,
                completed_queries,
                total_batches,
                queries.shape[0],
                len(inflight),
            )
            last_progress = now

    if rank == 0 and args.progress_interval > 0:
        print_progress(
            completed_batches,
            queries.shape[0],
            total_batches,
            queries.shape[0],
            len(inflight),
        )

    if rank == 0:
        np.savez(
            args.output,
            ids=result_ids,
            distances=result_distances,
            dataset=args.dataset,
            queries=args.queries,
            k=np.array(args.k, dtype=np.int64),
            metric=np.array(metric),
            world_size=np.array(size, dtype=np.int64),
        )

    total_seconds = time.perf_counter() - t0
    if args.profile:
        print(
            "rank={rank} device={device} shard=[{start},{stop}) "
            "load={load:.3f}s compute={compute:.3f}s merge={merge:.3f}s total={total:.3f}s".format(
                rank=rank,
                device=device,
                start=shard_start,
                stop=shard_stop,
                load=load_seconds,
                compute=compute_seconds,
                merge=merge_seconds,
                total=total_seconds,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
