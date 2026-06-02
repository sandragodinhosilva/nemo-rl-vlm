#!/bin/bash
# SFT → export → serve → eval autopilot for the qwen3.5-4b 2705-reasoning run.
#
# USAGE
# -----
# Interactive (srun --pty, current pattern):
#   srun --nodes=1 --gres=gpu:8 -c 192 --mem=1500G --job-name sft-vlm --pty bash -i
#   # then on the worker:
#   bash /home/sgsilva/nemo-rl-vlm/scripts/run_sft_2705_reasoning_autopilot.sh 2>&1 | tee /mnt/data/sgsilva/tmp/autopilot_2705_reasoning.log
#
# Non-interactive (sbatch — write a tiny wrapper):
#   cat > /tmp/sub.sbatch <<'EOF'
#   #!/bin/bash
#   #SBATCH --nodes=1
#   #SBATCH --gres=gpu:8
#   #SBATCH -c 192
#   #SBATCH --mem=1500G
#   #SBATCH --job-name=sft-2705-reasoning
#   #SBATCH --output=/mnt/data/sgsilva/tmp/autopilot_2705_reasoning.%j.log
#   bash /home/sgsilva/nemo-rl-vlm/scripts/run_sft_2705_reasoning_autopilot.sh
#   EOF
#   sbatch /tmp/sub.sbatch
#
# Stage-skip flags (set to 1 to skip):
#   SKIP_SFT=1     skip stage 1
#   SKIP_EXPORT=1  skip stage 2
#   SKIP_SERVE=1   skip stage 3 (also skips eval)
#   SKIP_EVAL=1    skip stage 4
#
# Recovery: each stage writes a sentinel under SENTINEL_DIR. A re-run skips
# stages whose sentinel already exists. Delete a sentinel to re-run that stage.

set -euo pipefail

# ─── config ────────────────────────────────────────────────────────────────
CONFIG=/home/sgsilva/nemo-rl-vlm/examples/configs/sft_vlm_qwen35_4b_mcqa_video_3d_2705_reasoning_local_megatron.yaml
NEMO_DIR=/home/sgsilva/nemo-rl-vlm
NEMO_PY=${NEMO_DIR}/.venv/bin/python
EVAL_PY=/home/sgsilva/vlm-post-training-home-venv/bin/python
EVAL_HARNESS=/home/sgsilva/vlm-post-training/aux_tasks/sft/eval_multimodal_post_sft.sh
SERVE_SCRIPT=/home/sgsilva/vlm-evaluation/start_vllm_server.sh
EXPORT_SCRIPT=${NEMO_DIR}/scripts/export_all_checkpoints.sh

CKPT_DIR=/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_mcqa_video_3d_2705_reasoning
HF_BASE=/mnt/data/sgsilva/models
HF_PREFIX=qwen35-4b-mcqa-video-3d-2705-reasoning

# The eval harness reads the test set as a JSONL file (not an arrow dir);
# this is the 504-row post-regen test set promoted from q3d_2705_regen.
TEST_JSONL=/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_3d_2705/test_video_native.jsonl
RESULTS_DIR=/mnt/data/sgsilva/results/mcqa_video_3d_2705_reasoning
SENTINEL_DIR=${RESULTS_DIR}/autopilot_sentinels

SERVE_PORT=${SERVE_PORT:-8001}
SERVE_TP=${SERVE_TP:-8}
SERVE_MAXLEN=${SERVE_MAXLEN:-65536}    # match training max_total_sequence_length
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-16384}   # see feedback_eval_maxtokens_context

mkdir -p "${RESULTS_DIR}" "${SENTINEL_DIR}"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()   { echo "[$(stamp)] $*"; }
fail()  { log "FATAL: $*"; exit 1; }

sentinel_ok() { [[ -f "${SENTINEL_DIR}/$1.done" ]]; }
mark_done()   { date -u +"%Y-%m-%dT%H:%M:%SZ" > "${SENTINEL_DIR}/$1.done"; }

# ─── stage 0: precheck ─────────────────────────────────────────────────────
log "host=$(hostname)  gpus_visible=${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader || \
  fail "nvidia-smi failed — are we on a GPU node?"

[[ -f "${CONFIG}" ]]      || fail "missing config ${CONFIG}"
[[ -x "${NEMO_PY}" ]]     || fail "missing nemo venv python ${NEMO_PY}"
[[ -x "${EVAL_PY}" ]]     || fail "missing eval venv python ${EVAL_PY}"
[[ -f "${EVAL_HARNESS}" ]]|| fail "missing eval harness ${EVAL_HARNESS}"
[[ -f "${SERVE_SCRIPT}" ]]|| fail "missing serve script"
[[ -f "${EXPORT_SCRIPT}" ]]|| fail "missing export script"
[[ -f "${TEST_JSONL}" ]]  || fail "missing test JSONL ${TEST_JSONL}"

log "precheck OK; results -> ${RESULTS_DIR}"

# ─── stage 1: verify config + train ────────────────────────────────────────
if [[ "${SKIP_SFT:-0}" == "1" ]]; then
  log "[1/4] SKIP_SFT=1 — skipping training"
elif sentinel_ok 01_sft; then
  log "[1/4] sentinel found — skipping training (delete ${SENTINEL_DIR}/01_sft.done to re-run)"
else
  log "[1/4] verifying config (no --fix; expect green)"
  cd "${NEMO_DIR}"
  "${NEMO_PY}" scripts/verify_and_fix_config.py --config "${CONFIG}" || \
    fail "config verification failed — re-run with --fix manually before retrying"
  log "[1/4] launching SFT"
  "${NEMO_PY}" examples/run_vlm_sft.py --config "${CONFIG}" || fail "SFT exited non-zero"
  mark_done 01_sft
fi

# ─── stage 2: export all checkpoints to HF ─────────────────────────────────
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  log "[2/4] SKIP_EXPORT=1 — skipping export"
elif sentinel_ok 02_export; then
  log "[2/4] sentinel found — skipping export"
else
  log "[2/4] exporting checkpoints from ${CKPT_DIR} to ${HF_BASE}/${HF_PREFIX}*"
  [[ -d "${CKPT_DIR}" ]] || fail "checkpoint dir missing ${CKPT_DIR}"
  bash "${EXPORT_SCRIPT}" "${CKPT_DIR}" "${HF_BASE}" "${HF_PREFIX}" || \
    fail "export script failed"
  mark_done 02_export
fi

# ─── identify the latest exported HF model dir ─────────────────────────────
# export_all_checkpoints.sh names dirs like ${HF_PREFIX}-step<N>; pick highest N.
pick_latest_hf() {
  ls -d "${HF_BASE}/${HF_PREFIX}"*step* 2>/dev/null | \
    awk -F'step' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-
}
LATEST_HF=$(pick_latest_hf || true)
[[ -n "${LATEST_HF}" ]] || fail "no exported HF checkpoint found under ${HF_BASE}/${HF_PREFIX}*step*"
log "latest HF checkpoint: ${LATEST_HF}"

# ─── stage 3: serve vLLM ───────────────────────────────────────────────────
if [[ "${SKIP_SERVE:-0}" == "1" ]]; then
  log "[3/4] SKIP_SERVE=1 — skipping serve+eval"
  exit 0
fi

if sentinel_ok 03_serve_ready; then
  log "[3/4] serve sentinel found — assuming a server is already running on :${SERVE_PORT}"
else
  log "[3/4] starting vLLM server (ENABLE_THINKING=1, port=${SERVE_PORT})"
  # Reasoning model: must serve thinking-on. Logs go next to sentinels.
  SERVE_LOG="${SENTINEL_DIR}/serve_${SERVE_PORT}.log"
  ENABLE_THINKING=1 QWEN35_VENV=/home/sgsilva/vlm-post-training-home-venv \
    nohup bash "${SERVE_SCRIPT}" "${LATEST_HF}" "${SERVE_TP}" "${SERVE_MAXLEN}" "${SERVE_PORT}" \
    > "${SERVE_LOG}" 2>&1 &
  SERVE_PID=$!
  echo "${SERVE_PID}" > "${SENTINEL_DIR}/serve.pid"
  log "[3/4] serve pid=${SERVE_PID}, log=${SERVE_LOG}"
  log "[3/4] waiting for /v1/models on :${SERVE_PORT} (up to 30 min)"
  for i in $(seq 1 180); do
    if curl -s --max-time 3 "http://localhost:${SERVE_PORT}/v1/models" | grep -q '"id"'; then
      log "[3/4] server up after ~$((i*10))s"
      mark_done 03_serve_ready
      break
    fi
    sleep 10
  done
  sentinel_ok 03_serve_ready || fail "server did not come up in 30 min; see ${SERVE_LOG}"
fi

# ─── stage 4: evaluate ─────────────────────────────────────────────────────
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  log "[4/4] SKIP_EVAL=1 — skipping eval"
  exit 0
fi

STEP=$(echo "${LATEST_HF}" | awk -F'step' '{print $NF}')
RUN_TAG="${HF_PREFIX}-step${STEP}"

log "[4/4] evaluating ${LATEST_HF} on 2705 test (504 rows, video-only) via eval_multimodal_post_sft.sh"
# The harness handles the MCQA flow (uses eval_mcqa_video_baseline.py under
# the hood), per-template stratification, naming policy, and writes results
# under EVAL_OUTPUT_ROOT/<tag>/.
# --skip-text/--skip-image: this is a video-only run (no text or image leg).
# VIDEO_JSONL env override points at our 504-row 2705 test JSONL.
EVAL_OUTPUT_ROOT="${RESULTS_DIR}" \
VIDEO_JSONL="${TEST_JSONL}" \
PYTHON="${EVAL_PY}" \
  bash "${EVAL_HARNESS}" \
    --model "${LATEST_HF}" \
    --tag "${RUN_TAG}" \
    --run-id "step${STEP}" \
    --base-model qwen3.5-4b \
    --train-group-id 2705-reasoning \
    --eval-family 2705-test-504 \
    --api-base "http://localhost:${SERVE_PORT}" \
    --server-url "http://localhost:${SERVE_PORT}" \
    --enable-thinking true \
    --max-tokens "${EVAL_MAX_TOKENS}" \
    --temperature 0.4 \
    --top-p 0.95 \
    --top-k 20 \
    --max-concurrency 20 \
    --skip-text \
    --skip-image || fail "eval harness failed"
mark_done 04_eval

log "DONE. results under: ${RESULTS_DIR}/"
log "to kill the serve later: kill \$(cat ${SENTINEL_DIR}/serve.pid)"
