#!/bin/bash
#
# Install shard-builder artifacts for one rank, per runtime_state/install_map.tsv
# (rank<TAB>shard, written by make_install_map.py from the cluster's own install-plan).
#
# Runs on the head node: ARTIFACT_DIR requires STORAGE_MEDIUM=lustre (validated at
# submit time), so every rank's storage tree under ./qdrantDir is on the shared
# filesystem and a move is a metadata-only rename from any node.
#
# INSTALL_MODE=move (default) renames the whole shard directory into place -- one
# atomic rename per shard, seconds for any size, and naturally resumable: a re-run
# finds the artifact gone and the target populated, and skips. It CONSUMES the
# artifact, so a backup of ARTIFACT_DIR must exist. INSTALL_MODE=copy preserves the
# artifacts at the cost of copying every byte.
#
# Only the shards the map assigns to this rank are touched. Every other shard keeps
# the stub directory qdrant created (replica_state.json + shard_config.json) -- those
# stubs are required, and installing a shard on a peer that does not own it makes
# writes diverge silently.

set -euo pipefail

RANK=${1:?usage: install_shards.sh <rank>}
ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR is required}
COLLECTION_NAME=${COLLECTION_NAME:?COLLECTION_NAME is required}
INSTALL_MODE=${INSTALL_MODE:-move}
MAP=${INSTALL_MAP:-./runtime_state/install_map.tsv}
TARGET_BASE=${TARGET_BASE:-./qdrantDir}

[[ -f "$MAP" ]] || { echo "install_shards: no install map at $MAP" >&2; exit 1; }

mapfile -t shards < <(awk -F'\t' -v r="$RANK" '$1 == r {print $2}' "$MAP")

# A rank can legitimately own no shards (more nodes than shards); it still needs its
# done flag so launch.sh relaunches it.
for shard in "${shards[@]}"; do
    src="$ARTIFACT_DIR/shard_$shard"
    dst="$TARGET_BASE/data/node$RANK/collections/$COLLECTION_NAME/$shard"

    if [[ -d "$src" ]]; then
        [[ -d "$dst" ]] || { echo "install_shards: rank $RANK: target shard dir $dst does not exist -- was the collection created?" >&2; exit 1; }
        if [[ "$INSTALL_MODE" == "move" ]]; then
            # Replace the empty shard qdrant created with the artifact, as one rename.
            rm -rf "${dst:?}"
            mv "$src" "$dst"
        else
            rm -rf "${dst:?}"/*
            cp -a "$src/." "$dst/"
        fi
        echo "install_shards: rank $RANK: installed shard $shard ($INSTALL_MODE)"
    elif [[ -d "$dst/wal" ]]; then
        # Artifact gone and the target carries a real shard (stubs have no wal/):
        # a previous run already installed it.
        echo "install_shards: rank $RANK: shard $shard already installed, skipping"
    else
        echo "install_shards: rank $RANK: artifact $src is missing and $dst holds no shard -- artifacts consumed by an earlier run? Re-provision ARTIFACT_DIR from the backup." >&2
        exit 1
    fi
done

touch "./runtime_state/install_done_${RANK}.txt"
