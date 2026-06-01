#!/bin/bash
#SBATCH --job-name=grpo-visual-obs-4b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --output=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.out
#SBATCH --error=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.err

set -euo pipefail
SCRIPT_PATH="/home/sgsilva/nemo-rl-vlm/examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh"

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p /home/sgsilva/nemo-rl-vlm/slurm_logs
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash "$SCRIPT_PATH" --worker
    exit $?
fi

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
export RAY_ADDRESS="${MASTER_ADDR}:6379"
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
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset UV_CACHE_DIR

export PYTHONPATH="/home/sgsilva/nemo-rl-vlm/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${PYTHONPATH:-}"
RAY_CMD="./.venv/bin/python -m ray.scripts.scripts"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

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
    sleep 20
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
ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
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

    LOG_DIR="/home/sgsilva/nemo-rl-vlm/slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/grpo_visual_obs_cat_4b_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "Training started at $(date)" | tee -a "$LOG_FILE"

    uv run python examples/run_vlm_grpo.py \
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
