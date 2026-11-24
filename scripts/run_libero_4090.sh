#!/usr/bin/env bash
# Run in an existing isolated environment. This entry downloads no files.
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: $0 {probe|run} RUNTIME_ROOT OUTPUT [run options]" >&2
    exit 2
fi
mode=$1
runtime=$(realpath "$2")
output=$(realpath -m "$3")
shift 3
source=$(cd "$(dirname "$0")/.." && pwd)
python="$runtime/venv/bin/python"
assets="$runtime/assets"
config=${FPO_CONFIG:-"$source/configs/libero_4090_p1.json"}
checkpoint="$assets/checkpoints/pi0_libero"
case "$mode" in
    probe) module=scripts.probe_libero_4090 ;;
    run) module=scripts.reproduce_libero_4090 ;;
    *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac
[[ -x "$python" && -r "$config" && -d "$source/openpi/src" && -d "$source/LIBERO/libero/libero" ]]
[[ $(cat "$assets/openpi.commit") == a6d2400d2534ce32e7bdf8747709b97aaef8ec04 ]]
[[ $(cat "$assets/LIBERO.commit") == 87edbd1b823e464bc66557d213cc45e9f88aa543 ]]
[[ -s "$assets/big_vision/paligemma_tokenizer.model" && -s "$assets/nvidia-egl.json" ]]
if [[ $mode == run ]]; then
    if [[ ${FPO_CHECKPOINT_WAIT_SECONDS:-0} != 0 ]]; then
        [[ $FPO_CHECKPOINT_WAIT_SECONDS =~ ^[1-9][0-9]*$ ]]
        deadline=$((SECONDS + FPO_CHECKPOINT_WAIT_SECONDS))
        while [[ ! -s "$checkpoint/checkpoint_receipt.json" ]]; do
            if [[ -e "$output/STOP" ]]; then
                echo "Stopped before GPU launch"
                exit 0
            fi
            if (( SECONDS >= deadline )); then
                echo "Official checkpoint verification did not finish before the wait deadline" >&2
                exit 1
            fi
            if [[ -n ${FPO_DOWNLOAD_PID:-} ]]; then
                state=$(ps -o stat= -p "$FPO_DOWNLOAD_PID" || true)
                if [[ -z $state || $state == Z* ]]; then
                    echo "Checkpoint downloader exited without an integrity receipt" >&2
                    exit 1
                fi
            fi
            printf '%s waiting for validated checkpoint (%s staged)\n' "$(date -Is)" "$(du -sh "$checkpoint" | cut -f1)"
            sleep 30
        done
    fi
    [[ -s "$checkpoint/checkpoint_receipt.json" ]]
    revision=${FPO_SOURCE_REVISION:-}
    if [[ -z $revision ]]; then
        revision=$(git -C "$source" rev-parse HEAD)
    fi
    set -- --checkpoint "$checkpoint" --receipt "$checkpoint/checkpoint_receipt.json" --source-revision "$revision" "$@"
fi
cd "$source"
mkdir -p "$(dirname "$output")" "$runtime/libero-config" "$runtime/jax-cache"
exec 9>"$runtime/gpu.lock"
flock -n 9 || { echo "Another FPO GPU stage holds the lock" >&2; exit 1; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "Existing GPU compute processes detected" >&2
    exit 1
fi
export PYTHONPATH="$source:$source/openpi/src:$source/openpi/packages/openpi-client/src:$source/LIBERO"
export OPENPI_DATA_HOME="$assets" LIBERO_CONFIG_PATH="$runtime/libero-config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$assets/nvidia-egl.json"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$runtime/jax-cache" XDG_CACHE_HOME="$runtime/cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ ! -e "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
    export FPO_LIBERO_ROOT="$source/LIBERO/libero/libero"
    "$python" - <<'PY'
import os
from pathlib import Path

import yaml

root = Path(os.environ["FPO_LIBERO_ROOT"])
paths = dict(benchmark_root=str(root), assets=str(root / "assets"), bddl_files=str(root / "bddl_files"),
             init_states=str(root / "init_files"), datasets=str(root.parent / "datasets"))
Path(os.environ["LIBERO_CONFIG_PATH"], "config.yaml").write_text(yaml.safe_dump(paths))
PY
fi
resources="$output.resources.$(date -u +%Y%m%dT%H%M%SZ).csv"
printf 'utc,gpu_mib,gpu_total_mib,gpu_util_pct,cgroup_ram_bytes\n' > "$resources"
(
    # The sampler must not hold the lock after the Python process exits.
    exec 9>&-
    while :; do
        gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
        [[ -r /sys/fs/cgroup/memory.current ]] && ram=$(cat /sys/fs/cgroup/memory.current) || ram=
        printf '%s,%s,%s\n' "$(date -u +%FT%TZ)" "$gpu" "$ram"
        sleep 2
    done
) >> "$resources" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true' EXIT
"$python" -u -m "$module" --config "$config" --output "$output" "$@"
