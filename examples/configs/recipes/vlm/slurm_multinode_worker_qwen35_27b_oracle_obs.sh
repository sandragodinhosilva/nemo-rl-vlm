#!/bin/bash
#SBATCH --job-name=sft-vlm-qwen35-27b-oracle-obs
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --exclude=worker-30,worker-31
#SBATCH --output=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.out
#SBATCH --error=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.err

set -euo pipefail
SCRIPT_PATH="/home/sgsilva/nemo-rl-vlm/examples/configs/recipes/vlm/slurm_multinode_worker_qwen35_27b_oracle_obs.sh"

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p /home/sgsilva/nemo-rl-vlm/slurm_logs
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash "$SCRIPT_PATH" --worker
    exit $?
fi

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
export RAY_ADDRESS="${MASTER_ADDR}:6379"
export NODE_RANK=$SLURM_NODEID            # The rank of the current node (0 or 1)

# This variable holds the number of GPUs per node
GPUS_PER_NODE=8
export GPUS_PER_NODE

echo "Environment check:"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  RAY_ADDRESS: $RAY_ADDRESS"

# 2. SET UP PATHS AND ENVIRONMENT
cd /home/sgsilva/nemo-rl-vlm/

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 3. SET APPLICATION-SPECIFIC VARIABLES (with fix for cache)
export HF_HOME="/mnt/data/shared/cache" # Use HF_HOME instead of the deprecated TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="/mnt/data/shared/cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Force Python to show all output immediately
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Disable Ray's automatic UV venv creation to avoid dependency conflicts
# NeMo RL handles venv creation separately to avoid contention
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

# Unset UV_CACHE_DIR to avoid cache conflicts
unset UV_CACHE_DIR

# Add Megatron-Bridge to Python path
export PYTHONPATH="/home/sgsilva/nemo-rl-vlm/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${PYTHONPATH:-}"
RAY_CMD="./.venv/bin/python -m ray.scripts.scripts"

# Help PyTorch find CUDA runtime
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# --- NCCL DEBUGGING (IMPROVED) ---
# Create a dedicated directory for NCCL logs inside your main slurm_logs
# NCCL_LOG_DIR="/home/sgsilva/nemo-rl-vlm/slurm_logs/nccl_logs_${SLURM_JOB_ID}"
# mkdir -p "$NCCL_LOG_DIR"

# Set NCCL debug level and specify a UNIQUE, ABSOLUTE path for the log file
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_ALL
# export NCCL_DEBUG_FILE="${NCCL_LOG_DIR}/nccl_debug_node${SLURM_NODEID}_$(hostname -s).log"
# echo "  NCCL Log File: ${NCCL_DEBUG_FILE}"
# --- END NCCL DEBUGGING ---

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth0

HOSTNAME_SHORT=$(hostname -s)
export RAY_TMPDIR="/tmp/ray_${USER}/ray__${SLURM_JOB_ID}_${NODE_RANK}"
mkdir -p "$RAY_TMPDIR"
echo "  RAY_TMPDIR: $RAY_TMPDIR (node-specific, isolated)"
export TMPDIR="$RAY_TMPDIR"
export RAY_START_TIMEOUT_SECONDS=300  # 5 minutes instead of 30 seconds
export RAY_gcs_server_request_timeout_seconds=120
export RAY_raylet_heartbeat_timeout_milliseconds=90000  # 90 seconds
export RAY_num_heartbeats_timeout=50
export RAY_raylet_client_num_connect_attempts=20
export RAY_gcs_rpc_server_reconnect_timeout_s=120

# Ray cluster formation timeouts
export RAY_TIMEOUT_MS=300000  # 5 minutes
export RAY_REDIS_START_RETRIES=20

# 4. START RAY CLUSTER
echo "Starting Ray cluster setup on node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    $RAY_CMD start --head --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray head started successfully"

    # Wait for initialization
    echo "Waiting for Ray to initialize..."
    sleep 20

else
    echo "=== Starting Ray WORKER node ==="
    # Wait for head node to be ready
    sleep 15

    $RAY_CMD start --address=$MASTER_ADDR:6379 --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 5
fi


# 4.5 VERIFY RAY REGISTERED ALL EXPECTED GPUS BEFORE TRAINING
if [ "$NODE_RANK" -eq 0 ]; then
    EXPECTED_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
    echo "Waiting for Ray to report ${EXPECTED_GPUS} GPUs across ${SLURM_NNODES} nodes..."

    for attempt in $(seq 1 30); do
        CURRENT_GPUS=$(./.venv/bin/python - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)
        echo "  Ray GPU preflight ${attempt}/30: ${CURRENT_GPUS}/${EXPECTED_GPUS} GPUs visible"

        if [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo "Ray resource preflight passed."
            break
        fi

        if [ "$attempt" -eq 30 ]; then
            echo "Ray resource preflight failed: expected ${EXPECTED_GPUS} GPUs, but only ${CURRENT_GPUS} were visible."
            $RAY_CMD status || true
            exit 1
        fi

        sleep 10
    done

    echo "Ray cluster resources snapshot:"
    ./.venv/bin/python - <<'PY'
import os
import pprint
import ray

ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False, ignore_reinit_error=True)
pprint.pprint(ray.cluster_resources())
ray.shutdown()
PY
fi


# 5. EXECUTE THE TRAINING JOB (only on head node)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL VLM SFT Training ==="

    # Create a detailed log with timestamps and node info
    LOG_DIR="/home/sgsilva/nemo-rl-vlm/slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_qwen35_27b_oracle_obs_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "📊 Starting training on node $NODE_RANK at $(date)" | tee -a "$LOG_FILE"

    # Run with detailed logging and real-time output
    uv run python examples/run_vlm_sft.py \
        --config examples/configs/sft_vlm_qwen35_27b_oracle_obs_megatron.yaml \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?

    echo "=== Training completed with exit code $TRAINING_EXIT_CODE ==="

    # Shutdown Ray cluster
    echo "=== Shutting down Ray cluster ==="
    $RAY_CMD stop

    exit $TRAINING_EXIT_CODE
else
    echo "=== Worker node $NODE_RANK waiting for training to complete ==="

    # Worker nodes wait for the training to complete
    while $RAY_CMD status > /dev/null 2>&1; do
        sleep 30
    done

    echo "=== Worker node $NODE_RANK shutting down ==="
    $RAY_CMD stop
fi
