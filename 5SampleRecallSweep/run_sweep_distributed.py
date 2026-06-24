#!/home/seth-ockerman/Documents/basicEnv/bin/python3
"""
Distributed 5-cluster Qdrant recall sweep with work-stealing.

Initial partition (by M value, minimises index rebuilds):
  Cluster 1 (sample1) → M=16,  all efConstruct                  28 triples
  Cluster 2 (sample2) → M=32,  all efConstruct                  28 triples
  Cluster 3 (sample3) → M=64,  all efConstruct                  28 triples
  Cluster 4 (sample4) → M=128, efConstruct ∈ {128, 256}         14 triples
  Cluster 5 (sample5) → M=128, efConstruct ∈ {512, 1024}        14 triples

Work-stealing: once a cluster exhausts its own triples it atomically claims
unclaimed triples from the shared pool so no cluster sits idle.  Results
from stolen work land in the stealing cluster's CSV (same sample, different
M/efConstruct settings).

Results use the same CSV/JSONL/npy schema as run_sweep.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from run_sweep import (
    TOP_K, HNSW_M_VALUES, EF_CONSTRUCT_VALUES, QUANTIZATION_VARIANTS,
    N_QUERIES, QUERY_COMBOS, RESULT_FIELDS,
    DEFAULT_ENV_FILE, QUERY_EMBEDDINGS_FILE, GROUND_TRUTH_DIR,
    QuantizationVariant, ClusterConfig,
    load_clusters, build_client, discover_collection, get_distance_metric,
    update_and_wait, run_queries_all_ef, compute_recall,
    make_run_key, successful_run_keys, append_result,
)

SWEEP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SWEEP_DIR / "results"

# All triples in canonical order (M outer → efc middle → quant inner)
ALL_TRIPLES: list[tuple[int, int, QuantizationVariant]] = list(
    itertools.product(HNSW_M_VALUES, EF_CONSTRUCT_VALUES, QUANTIZATION_VARIANTS)
)


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------
def assign_combos_to_clusters(
    clusters: list[ClusterConfig],
) -> dict[int, list[tuple[int, int, QuantizationVariant]]]:
    """
    Partition all 112 triples across clusters by M value.

      cluster 1 → M=16,  all efConstruct
      cluster 2 → M=32,  all efConstruct
      cluster 3 → M=64,  all efConstruct
      cluster 4 → M=128, efConstruct ∈ {128, 256}
      cluster 5 → M=128, efConstruct ∈ {512, 1024}
    """
    efc_lo = set(EF_CONSTRUCT_VALUES[:2])  # 128, 256
    m_to_cluster = {16: 1, 32: 2, 64: 3}

    assignment: dict[int, list] = {c.idx: [] for c in clusters}
    for m, efc, quant in ALL_TRIPLES:
        if m in m_to_cluster:
            assignment[m_to_cluster[m]].append((m, efc, quant))
        elif efc in efc_lo:
            assignment[4].append((m, efc, quant))
        else:
            assignment[5].append((m, efc, quant))
    return assignment


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distributed 5-cluster Qdrant recall sweep with work-stealing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", type=Path, default=DEFAULT_ENV_FILE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"Loading cluster configs from {args.env}")
    clusters = load_clusters(args.env)

    print("Building clients…")
    clients: dict[int, Any] = {c.idx: build_client(c) for c in clusters}

    collection_name = discover_collection(clients[1], args.collection)
    print(f"Collection: {collection_name!r}")
    for c in clusters[1:]:
        discover_collection(clients[c.idx], collection_name)

    distance_metric = get_distance_metric(clients[1], collection_name)
    print(f"Distance metric: {distance_metric}")

    print(f"Loading queries from {QUERY_EMBEDDINGS_FILE.name}…")
    queries = np.load(QUERY_EMBEDDINGS_FILE).astype(np.float32)[:N_QUERIES]
    n_queries, dim = queries.shape
    print(f"  {n_queries} queries, dim={dim}")

    print("Loading ground truth…")
    gt_arrays: dict[int, np.ndarray] = {}
    for c in clusters:
        gt = np.load(c.ground_truth_path, mmap_mode="r")
        gt_arrays[c.idx] = gt
        print(f"  {c.sample}: {gt.shape} {gt.dtype}")

    combo_assignment = assign_combos_to_clusters(clusters)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_csvs: dict[int, Path] = {
        c.idx: args.output_dir / f"results_{c.sample}.csv" for c in clusters
    }
    cluster_completed: dict[int, set[str]] = {
        c.idx: set() if args.no_resume else successful_run_keys(cluster_csvs[c.idx])
        for c in clusters
    }

    # Work-stealing: shared claimed set — a triple is claimed once a worker
    # starts on it so no two clusters duplicate the same index rebuild.
    claim_lock = threading.Lock()
    claimed: set[tuple[int, int, str]] = set()

    def try_claim(m: int, efc: int, quant: QuantizationVariant) -> bool:
        key = (m, efc, quant.name)
        with claim_lock:
            if key in claimed:
                return False
            claimed.add(key)
            return True

    print(f"\nDistributed sweep plan ({len(ALL_TRIPLES)} total triples, "
          f"{len(QUERY_COMBOS)} query combos each):")
    for c in clusters:
        my_combos = combo_assignment[c.idx]
        # Count only runs relevant to this cluster's assigned triples and QUERY_COMBOS
        done_count = sum(
            1 for m, efc, quant in my_combos
            for efs, os in QUERY_COMBOS
            if make_run_key(c.sample, quant, m, efc, efs, os, TOP_K) in cluster_completed[c.idx]
        )
        total_runs = len(my_combos) * len(QUERY_COMBOS)
        m_vals = sorted({m for m, _, _ in my_combos})
        efc_vals = sorted({efc for _, efc, _ in my_combos})
        print(
            f"  {c.sample}: M={m_vals} efc={efc_vals} "
            f"→ {len(my_combos)} triples × {len(QUERY_COMBOS)} combos = {total_runs} runs  "
            f"({done_count}/{total_runs} done)"
        )

    if args.dry_run:
        print("\n[dry-run] Exiting.")
        return

    print_lock = threading.Lock()

    def base_row(
        c: ClusterConfig,
        quant: QuantizationVariant,
        m: int,
        ef_construct: int,
        ef_search: int,
        oversampling: float,
        run_key: str,
        update_time: float | str,
    ) -> dict[str, Any]:
        return {
            "run_key": run_key,
            "cluster_idx": c.idx,
            "sample": c.sample,
            "cluster_url": c.url,
            "collection_name": collection_name,
            "query_count": n_queries,
            "vector_dim": dim,
            "distance_metric": distance_metric,
            "quantization_variant": quant.name,
            "quantization_type": quant.quantization_type,
            "quantization_always_ram": str(quant.always_ram),
            "quantization_turbo_bits": quant.turbo_bits,
            "quantization_binary_encoding": quant.binary_encoding,
            "hnsw_m": m,
            "ef_construct": ef_construct,
            "ef_search": ef_search,
            "oversampling": oversampling,
            "top_k": TOP_K,
            "update_time_s": update_time,
        }

    def cluster_worker(c: ClusterConfig) -> None:
        client = clients[c.idx]
        gt = gt_arrays[c.idx]
        csv_path = cluster_csvs[c.idx]
        done: set[str] = set(cluster_completed[c.idx])

        recall_dir = args.output_dir / "per_query_recall" / c.sample
        recall_dir.mkdir(parents=True, exist_ok=True)
        build_log_path = args.output_dir / f"build_log_{c.sample}.jsonl"

        def log_build(entry: dict[str, Any]) -> None:
            with build_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        def emit(
            row: dict[str, Any],
            per_query: np.ndarray | None = None,
            *,
            mark_done: bool = False,
        ) -> None:
            append_result(csv_path, row)
            if mark_done and row.get("status") == "success":
                done.add(str(row["run_key"]))
                if per_query is not None:
                    np.save(recall_dir / f"{row['run_key']}.npy", per_query)
            key = row.get("run_key", "")
            with print_lock:
                if row.get("status") == "success":
                    r = row.get("mean_recall_at_k", "?")
                    r_str = f"{r:.4f}" if isinstance(r, float) else str(r)
                    print(f"  [c{c.idx}] {key}: recall={r_str}")
                else:
                    print(f"  [c{c.idx}] {key}: FAILED – {row.get('error', '')}")

        def process_triple(m: int, ef_construct: int, quant: QuantizationVariant) -> None:
            needs = [
                (efs, os) for efs, os in QUERY_COMBOS
                if make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K) not in done
            ]
            if not needs:
                return

            try:
                update_time, col_snap = update_and_wait(
                    client, collection_name, m, ef_construct, quant
                )
                log_build({
                    "event": "update_ok", "sample": c.sample,
                    "m": m, "ef_construct": ef_construct, "quant": quant.name,
                    "update_time_s": update_time, "collection_state": col_snap,
                })
                with print_lock:
                    indexed = col_snap.get("indexed_vectors_count")
                    total = col_snap.get("points_count")
                    print(
                        f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: "
                        f"ready in {update_time:.1f}s  indexed={indexed}/{total}"
                    )
            except Exception as exc:
                log_build({
                    "event": "update_failed", "sample": c.sample,
                    "m": m, "ef_construct": ef_construct, "quant": quant.name,
                    "error": str(exc),
                })
                with print_lock:
                    print(f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: update FAILED – {exc}")
                for efs, os in needs:
                    run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                    emit({
                        **base_row(c, quant, m, ef_construct, efs, os, run_key, ""),
                        "status": "failed",
                        "error": f"index update failed: {exc}",
                    })
                return

            try:
                all_rids, total_qtime = run_queries_all_ef(
                    client, collection_name, queries, needs, TOP_K, args.batch_size
                )
            except Exception as exc:
                for efs, os in needs:
                    run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                    emit({
                        **base_row(c, quant, m, ef_construct, efs, os, run_key, update_time),  # type: ignore[possibly-unbound]
                        "status": "failed", "error": f"query failed: {exc}",
                    })
                return

            with print_lock:
                print(
                    f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: "
                    f"{len(needs)} combos queried in {total_qtime:.1f}s"
                )

            for efs, os in needs:
                short = sum(1 for r in all_rids[(efs, os)] if len(r) < TOP_K)
                if short:
                    with print_lock:
                        print(f"  [c{c.idx}] WARNING: {short} queries returned <{TOP_K} at efs={efs} os={os}")

            qtime_per_combo = total_qtime / len(needs)
            for efs, os in needs:
                run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                brow = base_row(c, quant, m, ef_construct, efs, os, run_key, update_time)
                try:
                    recall_stats, per_query = compute_recall(all_rids[(efs, os)], gt, TOP_K)
                    emit(
                        {**brow, "status": "success", "query_time_s": qtime_per_combo, **recall_stats},
                        per_query=per_query,
                        mark_done=True,
                    )
                except Exception as exc:
                    emit({**brow, "status": "failed", "error": f"recall failed: {exc}"})

        # ── Phase 1: own assigned triples ────────────────────────────────────
        my_combos = combo_assignment[c.idx]
        for m, efc, quant in my_combos:
            if try_claim(m, efc, quant):
                process_triple(m, efc, quant)

        # ── Phase 2: work-stealing — grab any unclaimed triple ───────────────
        with print_lock:
            print(f"  [c{c.idx}] own triples done, entering steal mode")
        for m, efc, quant in ALL_TRIPLES:
            if try_claim(m, efc, quant):
                process_triple(m, efc, quant)
        with print_lock:
            print(f"  [c{c.idx}] all work exhausted")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        futures = [pool.submit(cluster_worker, c) for c in clusters]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    print(f"\nDone.  Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
