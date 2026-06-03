#!/bin/bash
#SBATCH --job-name=grpo-visual-obs-4b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --output=/mnt/data/sgsilva/logs/grpo_logs/slurm-%j.out
#SBATCH --error=/mnt/data/sgsilva/logs/grpo_logs/slurm-%j.err

set -euo pipefail
SCRIPT_PATH="/home/sgsilva/nemo-rl-vlm/examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh"

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p /mnt/data/sgsilva/logs/grpo_logs
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash "$SCRIPT_PATH" --worker
    exit $?
fi

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
# Use local IP directly so Ray GCS connection works regardless of hostname resolution
export RAY_ADDRESS="$(hostname -I | awk '{print $1}'):6379"
export NODE_RANK=$SLURM_NODEID

GPUS_PER_NODE=8
export GPUS_PER_NODE

echo "Environment check:"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  RAY_ADDRESS: $RAY_ADDRESS"

# 2. SET UP PATHS AND ENVIRONMENT
cd /home/sgsilva/nemo-rl-vlm/

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 3. APPLICATION-SPECIFIC VARIABLES
export HF_HOME="/mnt/data/shared/cache"
export HF_DATASETS_CACHE="/mnt/data/shared/cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset UV_CACHE_DIR

# Expose only free GPUs to avoid conflicting with other users' processes on shared nodes.
# Detect free GPUs (>=200 GiB free) and expose only those.
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2 > 200000 {printf "%s,", $1}' | sed 's/,$//')
if [ -n "$FREE_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$FREE_GPUS"
    GPUS_PER_NODE=$(echo "$FREE_GPUS" | tr ',' '\n' | wc -l)
    export GPUS_PER_NODE
    echo "  CUDA_VISIBLE_DEVICES set to free GPUs: $CUDA_VISIBLE_DEVICES"
    echo "  GPUS_PER_NODE updated to: $GPUS_PER_NODE"
else
    echo "  WARNING: No GPUs with >200 GiB free found, using all GPUs"
fi

# nemo-rl-vlm-grpo-home-venv is built for B300/CUDA 13 (torch 2.10+cu130, transformer-engine,
# ray, and vllm 0.16 compiled with TORCH_CUDA_ARCH_LIST="10.0a" for sm_100a kernels).
# The sibling nemo-rl-vlm-home-venv has vllm built WITHOUT that arch flag => "no kernel image
# available" on B300, so we must point at the grpo venv here.
# _env_builder builds a venv named after each Ray worker class under NEMO_RL_VENV_DIR, which
# DEFAULTS to $GIT_ROOT/venvs/ (see nemo_rl/distributed/worker_groups.py:485). The working SFT
# launchers do NOT override NEMO_RL_VENV_DIR — we match that. Overriding it to /mnt/data broke
# Ray's worker bootstrap: the ReplayBuffer/Megatron actors crashed at start with
# "ModuleNotFoundError: No module named 'ray'" (raylet spawns .venv/default_worker.py but the
# worker venv under the custom dir wasn't on the resolved path). Using the default venvs/ dir
# fixes it (same as the SFT). We still pre-seed ONLY the vLLM worker classes there as symlinks to
# grpo-home-venv (they need the sm_100 vllm + _triton_alloc_fix.pth); Megatron/ReplayBuffer get
# normal _env_builder real venvs in venvs/ (one-time, cached) — and those bootstrap ray correctly.
unset NEMO_RL_VENV_DIR
NEMO_RL_VENV_DIR_RESOLVED="/home/sgsilva/nemo-rl-vlm/venvs"
WORKER_VLLM_VENV="/home/sgsilva/nemo-rl-vlm-grpo-home-venv"
mkdir -p "$NEMO_RL_VENV_DIR_RESOLVED"
for WORKER_VENV_NAME in \
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker" \
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker" ; do
    WORKER_VENV_TARGET="$NEMO_RL_VENV_DIR_RESOLVED/$WORKER_VENV_NAME"
    # Force-replace: if it's a real dir (from a previous _env_builder run), or a symlink pointing
    # somewhere else (e.g. the stale CUDA-12-arch home venv), delete and re-create the symlink.
    if [ -e "$WORKER_VENV_TARGET" ] || [ -L "$WORKER_VENV_TARGET" ]; then
        if [ "$(readlink -f "$WORKER_VENV_TARGET")" != "$(readlink -f "$WORKER_VLLM_VENV")" ]; then
            echo "Replacing stale worker venv at $WORKER_VENV_TARGET -> $WORKER_VLLM_VENV"
            rm -rf "$WORKER_VENV_TARGET"
        fi
    fi
    if [ ! -e "$WORKER_VENV_TARGET" ]; then
        ln -sf "$WORKER_VLLM_VENV" "$WORKER_VENV_TARGET"
    fi
    echo "Worker venv [$(basename "$WORKER_VENV_NAME")]: $(readlink -f "$WORKER_VENV_TARGET")"
done

export PYTHONPATH="/home/sgsilva/nemo-rl-vlm/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${PYTHONPATH:-}"
RAY_CMD="./.venv/bin/python -m ray.scripts.scripts"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth0

export RAY_TMPDIR="/tmp/ray_${USER}/ray__${SLURM_JOB_ID}_${NODE_RANK}"
mkdir -p "$RAY_TMPDIR"
echo "  RAY_TMPDIR: $RAY_TMPDIR"
export TMPDIR="$RAY_TMPDIR"
export RAY_START_TIMEOUT_SECONDS=300
export RAY_gcs_server_request_timeout_seconds=120
export RAY_raylet_heartbeat_timeout_milliseconds=90000
export RAY_num_heartbeats_timeout=50
export RAY_raylet_client_num_connect_attempts=20
export RAY_gcs_rpc_server_reconnect_timeout_s=120
export RAY_TIMEOUT_MS=300000
export RAY_REDIS_START_RETRIES=20

# 4. START RAY CLUSTER
echo "Starting Ray cluster on node $NODE_RANK ($GPUS_PER_NODE GPUs). Master: $MASTER_ADDR:$MASTER_PORT"

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    $RAY_CMD start --head --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray head started"
    sleep 30
else
    echo "=== Starting Ray WORKER node ==="
    sleep 15
    $RAY_CMD start --address=$MASTER_ADDR:6379 --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray worker $NODE_RANK connected"
    sleep 5
fi

# 4.5 GPU PREFLIGHT (head node only)
if [ "$NODE_RANK" -eq 0 ]; then
    EXPECTED_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
    echo "Waiting for Ray to report ${EXPECTED_GPUS} GPUs..."

    for attempt in $(seq 1 30); do
        CURRENT_GPUS=$(./.venv/bin/python - <<'PY'
import os, ray
try:
    ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, ignore_reinit_error=True)
    print(int(ray.cluster_resources().get("GPU", 0)))
    ray.shutdown()
except Exception:
    print(0)
PY
)
        echo "  GPU preflight ${attempt}/30: ${CURRENT_GPUS}/${EXPECTED_GPUS}"
        if [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo "GPU preflight passed."
            break
        fi
        if [ "$attempt" -eq 30 ]; then
            echo "GPU preflight failed: expected ${EXPECTED_GPUS}, got ${CURRENT_GPUS}"
            $RAY_CMD status || true
            exit 1
        fi
        sleep 10
    done
fi

# 5. EXECUTE TRAINING (head node only)
if [ "$NODE_RANK" -eq 0 ]; then
    CONFIG="${GRPO_CONFIG:-examples/configs/recipes/vlm/vlm_grpo_qwen35_4b_visual_obs_cat.yaml}"
    echo "=== Starting GRPO (visual-obs 4B) — config: $CONFIG ==="

    LOG_DIR="/mnt/data/sgsilva/logs/grpo_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/grpo_visual_obs_cat_4b_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "Training started at $(date)" | tee -a "$LOG_FILE"

    .venv/bin/python examples/run_vlm_grpo.py \
        --config "$CONFIG" \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?
    echo "=== Training exit code: $TRAINING_EXIT_CODE ===" | tee -a "$LOG_FILE"

    $RAY_CMD stop
    exit $TRAINING_EXIT_CODE
else
    echo "=== Worker $NODE_RANK waiting for training to complete ==="
    while $RAY_CMD status > /dev/null 2>&1; do
        sleep 30
    done
    echo "=== Worker $NODE_RANK shutting down ==="
    $RAY_CMD stop
fi
