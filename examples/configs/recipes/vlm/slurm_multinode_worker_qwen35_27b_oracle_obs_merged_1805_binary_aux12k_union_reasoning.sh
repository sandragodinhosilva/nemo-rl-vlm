#!/bin/bash
#SBATCH --job-name=sft-vlm-qwen35-27b-obs-1805-bin-aux12k-union-reas
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=192
#SBATCH --exclude=worker-30,worker-31
#SBATCH --output=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.out
#SBATCH --error=/home/sgsilva/nemo-rl-vlm/slurm_logs/slurm-%j.err

set -euo pipefail
SCRIPT_PATH="/home/sgsilva/nemo-rl-vlm/examples/configs/recipes/vlm/slurm_multinode_worker_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union_reasoning.sh"

if [[ "${1:-}" != "--worker" ]]; then
    mkdir -p /home/sgsilva/nemo-rl-vlm/slurm_logs
    srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash "$SCRIPT_PATH" --worker
    exit $?
fi

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

cd /home/sgsilva/nemo-rl-vlm/

if ! command -v uv &> /dev/null; then
    pip install uv
fi

export HF_HOME="/mnt/data/shared/cache"
export HF_DATASETS_CACHE="/mnt/data/shared/cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
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
export TMPDIR="$RAY_TMPDIR"
export RAY_START_TIMEOUT_SECONDS=300
export RAY_gcs_server_request_timeout_seconds=120
export RAY_raylet_heartbeat_timeout_milliseconds=90000
export RAY_num_heartbeats_timeout=50
export RAY_raylet_client_num_connect_attempts=20
export RAY_gcs_rpc_server_reconnect_timeout_s=120
export RAY_TIMEOUT_MS=300000
export RAY_REDIS_START_RETRIES=20

echo "Starting Ray cluster setup on node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    $RAY_CMD start --head --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray head started successfully"
    sleep 20
else
    echo "=== Starting Ray WORKER node ==="
    sleep 15
    $RAY_CMD start --address=$MASTER_ADDR:6379 --disable-usage-stats --num-gpus="$GPUS_PER_NODE"
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 5
fi

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

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL VLM SFT Training ==="

    LOG_DIR="/home/sgsilva/nemo-rl-vlm/slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union_reasoning_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "Starting training on node $NODE_RANK at $(date)" | tee -a "$LOG_FILE"

    /home/sgsilva/nemo-rl-vlm/.venv/bin/python examples/run_vlm_sft.py \
        --config examples/configs/sft_vlm_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union_reasoning_megatron.yaml \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?
    echo "=== Training completed with exit code $TRAINING_EXIT_CODE ==="
    $RAY_CMD stop
    exit $TRAINING_EXIT_CODE
else
    echo "=== Worker node $NODE_RANK waiting for training to complete ==="
    while $RAY_CMD status > /dev/null 2>&1; do
        sleep 30
    done
    echo "=== Worker node $NODE_RANK shutting down ==="
    $RAY_CMD stop
fi
