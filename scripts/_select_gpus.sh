#!/usr/bin/env bash
# _select_gpus.sh — helper for parallel-eval scripts.
#
# Usage (source into your script):
#   source "$SCRIPT_DIR/_select_gpus.sh"
#   GPU_LIST=$(select_free_gpus N)      # e.g. "3,5,7,2"
#
# Queries nvidia-smi for (memory.used, utilization.gpu), sorts ASCENDING
# by both (least-used first), takes top N, returns comma-separated index
# list. Snapshots once — caller dispatches processes to the picked GPUs
# before their memory shows up in nvidia-smi (avoids double-allocation).
#
# Falls back silently to sequential 0..N-1 if nvidia-smi is unavailable
# or returns garbage (e.g. no NVIDIA driver).

select_free_gpus() {
    local n="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        # fallback: 0..n-1
        seq 0 $((n - 1)) | paste -sd,
        return
    fi

    # Columns: index, memory.used (MiB), utilization.gpu (%)
    # Sort: memory ascending, then utilization ascending.
    local list
    list=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
            | sort -t, -k2n -k3n \
            | head -n "$n" \
            | awk -F, '{gsub(/ /,""); print $1}' \
            | paste -sd,)

    if [[ -z "$list" ]]; then
        # nvidia-smi ran but empty — fallback
        seq 0 $((n - 1)) | paste -sd,
    else
        echo "$list"
    fi
}
