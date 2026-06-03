# GRPO for Visual Observations — Setup Reference

**Date:** 2026-06-01  
**Branch:** `sft-mcqa-video-3d-2605`  
**Goal:** Train a 4B model with GRPO on the stage-1 visual observations task (categorical, 1105 dataset) using an ordinal-distance reward.

---

## 1. What was done

### 1.1 Synced pmartins modular environment

The old monolithic `thrive_vlm_environment.py` was replaced with the modular version from `/mnt/data/pmartins/grpo_environments/`. These files were copied into `nemo_rl/environments/`:

| File | Purpose |
|---|---|
| `thrive_vlm_environment.py` | Main environment class; dispatches by `task_type` |
| `thrive_vlm_rep_rewards.py` | Per-rep severity rewards |
| `thrive_vlm_full_exercise_rewards.py` | Full-exercise Q1–Q9 rewards |
| `thrive_vlm_aux_rewards.py` | MCQ, exercise-name, keypoint rewards |
| `thrive_vlm_comparison_rewards.py` | Comparison/verdict rewards |
| `thrive_vlm_reward_utils.py` | Shared utilities |
| `thrive_vlm_sdar_utils.py` | SDAR utilities |

### 1.2 New visual-obs reward module

**File:** `nemo_rl/environments/thrive_vlm_visual_obs_rewards.py`

**Reward logic:** Ordinal-distance over categorical options.

For each of the (up to 7) questions in a `[VISUAL OBSERVATIONS]` block:

```
score_i = 1 - |pred_idx - gt_idx| / (n_options - 1)
```

- Options are looked up from `visual_observations_categorical.json` by `exercise_id`
- Binary questions (2 options) collapse to exact-match (0 or 1)
- Predictions that are not valid options score 0 (not nearest-neighbour)
- Final reward = mean across all GT questions
- Capped at `MAX_QUESTIONS_PER_EXERCISE = 7`

Schema file: `/home/sgsilva/vlm-post-training/visual_observations_categorical.json`

**Smoke tests (all pass):**
- Perfect match → 1.0
- Blank prediction → 0.0
- One step off on a 5-option question (out of 7) → 0.964

### 1.3 Environment wiring

**`nemo_rl/environments/thrive_vlm_environment.py`**
- Added import of `compute_visual_obs_reward`
- `ThriveVLMVerifyWorker.__init__`: added `self.visual_obs_config: dict = {}`
- `verify()`: added `exercise_ids` parameter; sets `self.visual_obs_config = {"exercise_id": ...}` per sample
- `_compute_reward()` / `_compute_reward_with_details()`: dispatches `task_type == "visual_obs"` before other types
- `ThriveVLMEnvironment.step()`: extracts `exercise_ids` from metadata, chunks and passes to `verify.remote()`

**`nemo_rl/data/utils.py`**
- Added `"visual-obs"` to the VLM env routing whitelist (prevents override to generic `"vlm"`)

**`nemo_rl/environments/utils.py`**
- Registered `"visual-obs"` in `ENV_REGISTRY` → `ThriveVLMEnvironment`

### 1.4 Dataset formatter

**`nemo_rl/data/datasets/response_datasets/thrive_vlm_grpo.py`**

`format_thrive_vlm_grpo_dataset()` now detects visual-obs samples by `schema == "categorical"` and emits:
```python
task_type = "visual_obs"
extra_env_info = {
    "ground_truth": assistant_content,   # the [VISUAL OBSERVATIONS] block
    "task_type": "visual_obs",
    "exercise_id": str(example["exercise_id"]),
}
```
`exercise_id` flows through `extra_env_info` → `step()` metadata → `verify()` → reward function.

---

## 2. Configs

### 2.1 SFT — 4B oracle-obs-cat

**File:** `examples/configs/sft_vlm_qwen35_4b_oracle_obs_cat_local_megatron.yaml`

| Key | Value |
|---|---|
| Base model | `/mnt/data/shared/models/Qwen3.5-4B` (via `sft_vlm_qwen35_4b_local_common_megatron.yaml`) |
| Train data | `/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed_visual_obs/categorical/train` (3,814 samples) |
| Val data | `…/categorical/test` (1,181 samples) |
| Epochs | 3 |
| GBS | 32, MBS 1 |
| Steps/epoch | ~119 (3814 / 32) |
| Total steps | ~357 |
| Save period | 119 (1 ckpt/epoch) |
| LR decay iters | 357 |
| LR warmup iters | 36 |
| Checkpoint dir | `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_oracle_obs_cat_1105` |

### 2.2 GRPO — 4B visual-obs-cat

**File:** `examples/configs/recipes/vlm/vlm_grpo_qwen35_4b_visual_obs_cat.yaml`

| Key | Value |
|---|---|
| Starting model | `/mnt/data/sgsilva/models/qwen35-4b-oracle-obs-cat-1105-step357` (exported SFT ckpt) |
| Train data | `…/1105_not_reviewed_visual_obs/categorical/train` |
| Val data | `…/categorical/test` |
| env_name | `"visual-obs"` |
| dataset_name | `"thrive-vlm"` (uses `ThriveVLMGRPODataset`) |
| num_prompts_per_step | 16 |
| num_generations_per_prompt | 16 |
| val_period | 500 |
| Parallelism | megatron TP=2, vLLM TP=1 (8 GPUs colocated); GBS 16, MBS 1, logprob_batch_size 4 |
| max_new_tokens | 2048 |
| Checkpoint dir | `/mnt/data/sgsilva/checkpoints/grpo_visual_obs_cat_1105_4b` |
| Log dir | `/mnt/data/sgsilva/logs/grpo_logs/grpo_visual_obs_cat_1105_4b` |

### 2.3 SLURM launcher

**File:** `examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh`

- 1 node, uses free GPUs only (auto-detected, see §5)
- vLLM generation workers use a pre-built sub-venv symlinked under `NEMO_RL_VENV_DIR` (see §5)
- Logs written to `/mnt/data/sgsilva/logs/grpo_logs/` (NOT `/home`)
- Always pass `--worker` when running directly (no SLURM)

---

## 3. Launch sequence

### Step 1 — Run the 4B SFT

```bash
cd /home/sgsilva/nemo-rl-vlm

# Verify and fix step math against actual dataset size
.venv/bin/python scripts/verify_and_fix_config.py \
  --config examples/configs/sft_vlm_qwen35_4b_oracle_obs_cat_local_megatron.yaml \
  --fix

# Launch SFT (~357 steps, ~3 epochs)
.venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_oracle_obs_cat_1105_local_megatron.yaml
```

### Step 2 — Export the best SFT checkpoint to HF

```bash
bash scripts/export_all_checkpoints.sh \
  /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_oracle_obs_cat_1105 \
  /mnt/data/sgsilva/models \
  qwen35-4b-oracle-obs-cat
```

This produces `/mnt/data/sgsilva/models/qwen35-4b-oracle-obs-cat-step<N>/`.

### Step 3 — Update GRPO config model_name

In `examples/configs/recipes/vlm/vlm_grpo_qwen35_4b_visual_obs_cat.yaml`, set:
```yaml
policy:
  model_name: /mnt/data/sgsilva/models/qwen35-4b-oracle-obs-cat-step<N>
```

### Step 4 — Launch GRPO

**Prerequisite:** Complete §5 (environment setup) before the first launch. Needs a node with
**8 free GPUs** (the configs are sized for 8-GPU colocated).

Two ways to target the node — **always resolve it, never hardcode `worker-N`** (allocation moves):

**(a) Launch into an existing idle allocation** (recommended — a `bash` placeholder job holding 8 GPUs):
```bash
cd /home/sgsilva/nemo-rl-vlm
JOBID=<your idle 8-GPU allocation>   # e.g. from `squeue -u $USER`

# Smoketest (5 steps) — success = "Step 1/5" in the log:
GRPO_CONFIG=examples/configs/recipes/vlm/vlm_grpo_qwen35_4b_visual_obs_cat_smoketest.yaml \
srun --jobid="$JOBID" --ntasks=1 --ntasks-per-node=1 \
  bash examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh --worker \
  2>&1 | tee /mnt/data/sgsilva/tmp/smoketest.log

# Full run (omit GRPO_CONFIG → defaults to the full config):
srun --jobid="$JOBID" --ntasks=1 --ntasks-per-node=1 \
  bash examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh --worker
```

**(b) Already interactively on the node** — set the SLURM vars by hand:
```bash
export SLURM_JOB_NODELIST=$(hostname)   # resolve, never hardcode
export SLURM_NODEID=0; export SLURM_NNODES=1; export SLURM_JOB_ID=grpo_run_001
bash examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh --worker \
  2>&1 | tee /mnt/data/sgsilva/logs/grpo_logs/grpo_run_001.log
```

> `ray stop` between attempts is safe — it only kills *your* Ray procs. The launcher's free-GPU
> auto-detect (§5.5) keeps you off GPUs other users occupy; never `kill` their processes.

---

## 4. Key design decisions

| Decision | Rationale |
|---|---|
| Ordinal-distance reward | Gives partial credit for near-misses on ordered options (e.g. "large lift" vs "moderate lift"); smooth gradient signal vs binary exact-match |
| Cap at 7 questions | Maximum per-exercise question count; keeps reward scale consistent across exercises |
| Zero credit for out-of-vocabulary predictions | Model must learn to stay within the option set; avoids rewarding hallucinated text that happens to be close to GT |
| `task_type = "visual_obs"` detection via `schema == "categorical"` | Transparent mixing with other task types (rep/full_exercise/mcqa) in future multi-task GRPO |
| `exercise_id` threaded through metadata | Required for schema lookup; cannot be a global config because different samples have different exercises |
| Separate env name `"visual-obs"` | Keeps reward logic isolated from `"thrive-vlm"` rep/full-exercise env config; allows different `num_workers`, future judge config, etc. |
| Start from 4B (not 27B) | Sanity-check run; 4B fits on 1 node and validates reward signal before scaling |

---

## 5. Environment Setup for GRPO on B300 / CUDA 13

**Updated: 2026-06-02**

The cluster uses **NVIDIA B300 GPUs (compute capability 10.0a / sm_100a)** with **CUDA 13 only** (`libcudart.so.13`). The repo-local `.venv` works for SFT because it uses torch 2.10+cu130, but vLLM needs additional setup.

### 5.1 Venv roles

| Venv | Purpose | Touch? |
|---|---|---|
| `/home/sgsilva/nemo-rl-vlm/.venv` | SFT + GRPO orchestration (Megatron, Ray, trainer) | ✅ normal use |
| `/home/sgsilva/nemo-rl-vlm-home-venv` | General-purpose home venv (eval, serving) | ❌ do not modify |
| `/home/sgsilva/nemo-rl-vlm-grpo-home-venv` | **vLLM generation workers for GRPO** — torch 2.10+cu130 + vllm built with `TORCH_CUDA_ARCH_LIST="10.0a"` | ✅ only this one for GRPO vllm fixes |

### 5.2 Building the GRPO worker venv (first time only)

If `nemo-rl-vlm-grpo-home-venv` does not have vllm installed:

```bash
cd /home/sgsilva/nemo-rl-vlm
LOG=/mnt/data/sgsilva/tmp/vllm_grpo_venv_build.log
# setsid+nohup so a transient cluster hiccup / shell exit doesn't SIGKILL the build (see §5.4):
setsid nohup env TORCH_CUDA_ARCH_LIST="10.0a" \
  UV_PROJECT_ENVIRONMENT=/home/sgsilva/nemo-rl-vlm-grpo-home-venv \
  uv sync --locked --extra vllm > "$LOG" 2>&1 < /dev/null &
```

Takes ~30-45 min (vllm sm_100a kernels are the long pole). Verify when done:
```bash
/home/sgsilva/nemo-rl-vlm-grpo-home-venv/bin/python -c "import vllm; print(vllm.__version__)"
# Should print 0.16.x
strings /home/sgsilva/nemo-rl-vlm-grpo-home-venv/lib/python3.12/site-packages/vllm/_C.*.so \
  | grep "sm_10"
# Should show sm_100a
```

### 5.3 Worker venv setup — symlink ONLY the vLLM classes, in the DEFAULT venv dir

Two hard-won rules, both critical:

**(a) Use the DEFAULT `NEMO_RL_VENV_DIR` (`$GIT_ROOT/venvs/`). Do NOT override it to `/mnt/data`.**
NeMo-RL's `create_local_venv_on_each_node` (`worker_groups.py:485`) builds per-node-local worker
venvs there. Overriding `NEMO_RL_VENV_DIR=/mnt/data/...` (NFS) **breaks Ray's worker bootstrap**:
non-vLLM actors (`ReplayBuffer`, `MegatronPolicyWorker`) crash at start with
`ModuleNotFoundError: No module named 'ray'` (raylet runs `.venv/default_worker.py` but the
worker venv under the custom NFS dir isn't on the resolved path), and the async loop then spins
forever on `buffer_filled_ratio=0/4`. The working SFT launchers never set `NEMO_RL_VENV_DIR`.
**Fix: `unset NEMO_RL_VENV_DIR`; use `/home/sgsilva/nemo-rl-vlm/venvs/`.**

**(b) Symlink ONLY the vLLM worker classes to `grpo-home-venv` — let the others build normally.**
```
/home/sgsilva/nemo-rl-vlm/venvs/
  nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker            → grpo-home-venv  (symlink)
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker → grpo-home-venv  (symlink)
  nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker  (real _env_builder venv)
  nemo_rl.algorithms.async_utils.ReplayBuffer                                (real _env_builder venv, async only)
```
Only the vLLM workers need the sm_100 vllm + `_triton_alloc_fix.pth`, so only they get the symlink.
**Do NOT symlink `MegatronPolicyWorker`/`ReplayBuffer` to grpo-home-venv** — even though it has ray,
the symlink confuses Ray's worker bootstrap (same `No module named 'ray'`). They need their own real
`_env_builder` venvs (one-time, cached) which bootstrap ray correctly. The launcher symlinks both vLLM
classes on every run (deleting any stale real dir / wrong-target symlink) and leaves the rest alone.

### 5.4 Known issues and workarounds

| Issue | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vllm` | vllm not installed in any worker venv | Build it into `grpo-home-venv` (§5.2) |
| `libcudart.so.12 not found` | B300 only ships CUDA 13 (`libcudart.so.13`); old venvs were cu12 | Use torch 2.10+**cu130** venv (`grpo-home-venv`) |
| deep_gemm build: `cuda/std/utility: No such file or directory` | CUDA 13 moved libcudacxx headers under `cccl/` | `pyproject.toml` `extra-build-variables`: `deep_gemm`/`deep_ep` get `CFLAGS/CXXFLAGS = "-I/usr/local/cuda/include/cccl"` |
| `CUDA error: no kernel image for device` | vllm compiled without `sm_100a` kernels (arch flag not set during build) | Build `grpo-home-venv`'s vllm with `TORCH_CUDA_ARCH_LIST="10.0a"` (§5.2); point worker symlink at it (§5.3) |
| `torch.linalg.cholesky` missing (hit via vllm `get_overridable_functions()`) | **NOT a torch 2.10 API change.** The old `nemo-rl-vlm-home-venv` had a **corrupted `torch.linalg`** — only 7 attrs, missing cholesky/svd/qr/inv/solve. A healthy install (the SFT `.venv` and `grpo-home-venv`) has all 54. | **Root fix:** use `grpo-home-venv` (linalg intact). `TORCHDYNAMO_DISABLE=1` stays in the launcher as defensive cover but is not what resolves cholesky. |
| `networkx.lazy_imports` missing | Corrupted networkx in worker venv | Reinstall networkx inside the affected venv |
| `RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set` (in `fla/ops/solve_tril.py` → `chunk_gated_delta_rule`) | **The real generation blocker.** Qwen3.5-4B is a Qwen3-Next **hybrid / Gated-DeltaNet linear-attention** model; its bundled FLA Triton kernel needs a Triton 3.6 scratch allocator that this vllm 0.16-dev build never registers. Fires on **every** generation (even plain text), independent of attention backend. | Register a torch-backed Triton allocator at interpreter startup. Installed in `grpo-home-venv` as `site-packages/_triton_alloc_fix.py` + `_triton_alloc_fix.pth` (auto-imported by `site`, so every Ray-spawned vLLM worker gets it). See §5.4.1. |
| `BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'o_data_type'` | FlashInfer 0.5.3 is incompatible with vllm 0.16-dev's FlashInfer call signature | Force the FlashAttention backend: `vllm_kwargs.attention_backend: "FLASH_ATTN"` in the GRPO configs |
| `RuntimeError: aot_compile is not supported by the current configuration … torch: 2.10.0+cu130` (async EngineCore fails to start) | **CONFIRMED hard limitation.** The async EngineCore AOT-compiles CUDA graphs when `enforce_eager: false` (the reference value); `vllm/compilation/wrapper.py:201` requires a torch with `aot_compile`, which **torch 2.10+cu130 does not have**. So the reference's `enforce_eager: false` **cannot run on B300** — verified by direct repro (all EngineCores crash at init). | Set `vllm_cfg.enforce_eager: true` (the **only forced deviation** from the reference). Loses CUDA-graph capture; async_grpo/async_engine still work in eager mode. |
| `AssertionError: response_length=N > max_model_len=M` (sync `vllm_worker.py:719`) | `max_new_tokens` (e.g. 65536) exceeds `max_model_len` (= `max_total_sequence_length`, 32784) → a generation can overshoot the window by ≥1 token. The **sync/colocated** worker asserts on this; the **async** worker does not. | On **colocated** configs keep `max_new_tokens` below `max_total_sequence_length` (e.g. ≤ ~30000, leaving room for the prompt). The async (reference-style) configs keep 65536 since the async path tolerates it. |
| `Using TRTLLM prefill attention (auto-detected)` then Triton allocator crash | vllm auto-selects FlashInfer TRTLLM prefill when `kv_cache_dtype: auto`; TRTLLM kernel also hits the unallocated-Triton path | Same `attention_config` fix above (FLASH_ATTN bypasses all FlashInfer paths) |
| **Run exits cleanly (code 0) after "SETUP COMPLETE"/"Using GRPO advantage estimator" but runs 0 steps** (no `Epoch`/`Step` banner) | **Not a bug — a stale checkpoint resume.** A prior run saved a `step_N` ckpt to the same `checkpointing.checkpoint_dir`; the next run resumes `total_steps=N`, so `while total_steps < max_num_steps` is immediately false (e.g. resumed `step_5` with `max_num_steps: 5`). Look for `loading distributed checkpoint from …/step_N` in the log. | Delete the stale checkpoint dir (or bump `max_num_steps` / point `checkpoint_dir` somewhere fresh) before re-running a smoketest: `rm -rf <checkpoint_dir>/step_*` |
| Async config runs 0 steps and exits 0 | The async path needs `run_vlm_grpo.py` to dispatch to `async_grpo_train` (the old VLM entry point only called the sync `grpo_train`). | Resolved by the origin/main merge — current `run_vlm_grpo.py` branches on `async_grpo.enabled` and calls `async_grpo_train`. (Also requires `enforce_eager: true`, above.) |
| `Free memory < desired utilization` on `cuda:0` | Shared node: other users' processes occupy some GPUs; with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + TP=1 all vllm workers pile onto `cuda:0` | Auto-detect free GPUs (>200 GiB) → `CUDA_VISIBLE_DEVICES` + `GPUS_PER_NODE` (§5.5) |
| `raylet started with N GPUs but CUDA_VISIBLE_DEVICES has M` | Ray `--num-gpus` must match the `CUDA_VISIBLE_DEVICES` count | `GPUS_PER_NODE` recomputed from the detected free-GPU list |
| `cluster.gpus_per_node` / `colocated.resources.gpus_per_node` mismatch | YAML GPU count must match the launcher's detected free-GPU count | On a fully-free 8-GPU node both are set to **8** (colocated); the launcher's auto-detect must yield 8 (no other-user procs) |
| `ModuleNotFoundError: No module named 'ray'` from raylet (ReplayBuffer/Megatron worker crashes at start) → async loop spins `buffer_filled_ratio=0/4` forever | **`NEMO_RL_VENV_DIR` was overridden to `/mnt/data` (NFS).** `create_local_venv_on_each_node` (`worker_groups.py:485`) expects the **default node-local `$GIT_ROOT/venvs/`**; building worker venvs on the NFS path breaks Ray's per-node worker bootstrap. The working SFT launchers never override it. | **`unset NEMO_RL_VENV_DIR`; use default `venvs/`** (§5.3). Symlink only the vLLM classes there; let Megatron/ReplayBuffer build normal real venvs. |
| `RuntimeError: pidfd_getfd: Operation not permitted` in `update_weights_via_ipc_zmq` → `ZMQ communication timeout after 120000ms` (colocated weight refit dies) | **`PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True` breaks colocated CUDA-IPC weight transfer.** Expandable-segments tensors use a VM-based allocator that can't be exported via CUDA IPC handles, so the Megatron→vLLM weight stream fails. (`pidfd_getfd` itself is *not* blocked on any node — verified.) The worker-27 PASS used `False`; setting `True` as an OOM fix caused this. | Keep `megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"` for **colocated** runs. It is **mutually exclusive** with colocated IPC weight transfer. |
| `torch.OutOfMemoryError` in Megatron `backward_step` (colocated, after generation; vLLM sleep-mode still holds ~8.5 GiB/GPU, ~15 GiB reserved-but-unallocated) | Long sequences (`max_total_sequence_length: 32784`) → large backward activations; colocated leaves little headroom. | Set **`megatron_cfg.activation_checkpointing: true`** (recompute activations in backward — big memory saving, allocator-agnostic so it's IPC-safe). Do **NOT** use `expandable_segments:True` for this in colocated mode (breaks IPC, above). |

> **Build robustness:** run the `uv sync` build under `setsid nohup … < /dev/null &` and log to
> `/mnt/data/sgsilva/tmp/`. The cluster has had transient memory pressure that silently
> SIGKILLs a detached build mid-vllm-compile (no error line — log just stops after
> "Built transformer-engine"). uv caches the deep-ep/deep-gemm/TE wheels, so a relaunch
> resumes straight at vllm.

#### 5.4.1 Triton allocator fix (the generation blocker)

Qwen3.5-4B uses Gated-DeltaNet linear attention; its FLA Triton kernels need a Triton 3.6
scratch allocator vllm never sets. We register one at interpreter startup so **every** process
that uses the worker venv (including Ray-spawned vLLM workers) gets it automatically — no
NeMo-RL code change. Two files in `grpo-home-venv/lib/python3.12/site-packages/`:

`_triton_alloc_fix.py`:
```python
def _install():
    try:
        import torch, triton
    except Exception:
        return
    if not hasattr(triton, "set_allocator"):
        return
    def _alloc(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    try:
        triton.set_allocator(_alloc)
    except Exception:
        pass
_install()
```
`_triton_alloc_fix.pth` (one line, auto-imported by `site` at startup):
```
import _triton_alloc_fix
```
Verify: `grpo-home-venv/bin/python -c "import triton; from triton.runtime import _allocation; print(_allocation._allocator.get())"`
should print a function (not `NullAllocator`). Standalone-verified: vLLM generates text on B300
with this in place, both single-process and V1-multiprocessing.

### 5.5 Shared-node checklist before launch

```bash
# 1. Check free GPUs — need 8 free (>200 GiB each) for the colocated configs
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits

# 2. Confirm no stale Ray from previous runs (only kills YOUR ray procs)
.venv/bin/python -m ray.scripts.scripts stop --force

# 3. Confirm grpo-home-venv has vllm AND sm_100a kernels
/home/sgsilva/nemo-rl-vlm-grpo-home-venv/bin/python -c "import vllm; print(vllm.__version__)"
strings /home/sgsilva/nemo-rl-vlm-grpo-home-venv/lib/python3.12/site-packages/vllm/_C.*.so | grep -m1 sm_100a
```

### 5.6 Config changes vs original

These differ from the original design in `examples/configs/recipes/vlm/vlm_grpo_qwen35_4b_thrive.yaml`:

| Setting | Original | Current (B300) |
|---|---|---|
| `vllm_cfg.enforce_eager` | `false` | `true` (avoids some torch.compile paths) |
| `cluster.gpus_per_node` | `8` | `8` on a fully-free node (colocated); auto-detect must yield 8. On a partially-occupied node, set both this and `colocated.resources.gpus_per_node` to the free-GPU count. |
| `colocated.resources.gpus_per_node` | `4` | `8` (vLLM + policy share all 8 GPUs; megatron TP=2, vLLM TP=1) |
| `log_dir` | `/home/sgsilva/nemo-rl-vlm/logs_grpo/` | `/mnt/data/sgsilva/logs/grpo_logs/` |
| Launcher env | (none) | `TORCHDYNAMO_DISABLE=1`, `CUDA_VISIBLE_DEVICES` auto-set |
| Worker vllm venv | `.venv` (cu12, SFT) | `nemo-rl-vlm-grpo-home-venv` (cu130 + sm_100a vllm) via symlink |

### 5.7 Status (2026-06-02)

**Working / resolved:**
- [x] SFT working end-to-end
- [x] Launcher boots Ray, GPU preflight passes (8/8)
- [x] vllm sm_100 build in `grpo-home-venv` — done & verified (`import vllm` = 0.16.0rc2.dev433+cu130; `cuobjdump _C.abi3.so` shows 49 `sm_100` kernels). First build was SIGKILLed by transient cluster memory pressure; rebuilt under `setsid nohup`.
- [x] vLLM workers initialize and load model config + checkpoint (Megatron policy + reference)
- [x] `torch.linalg.cholesky` — root-caused (corrupt linalg in the *old* `home-venv`, not a torch API change); fixed by using `grpo-home-venv`
- [x] deep_gemm cccl-include build failure — fixed in `pyproject.toml`
- [x] Shared-node GPU contention — launcher auto-detects free GPUs
- [x] Worker venv symlink → `grpo-home-venv` (with stale-symlink guard)
- [x] Configs scaled to 8-GPU colocated
- [x] **Generation works on B300** — root-caused the `solve_tril` Triton allocator crash (Qwen3-Next GDN linear attention) and fixed via `_triton_alloc_fix.pth` (§5.4.1) + `attention_backend: FLASH_ATTN` (avoids FlashInfer 0.5.3 `o_data_type`). Standalone-verified, single-proc and V1-multiprocessing.
- [x] **Colocated GRPO smoketest PASSED end-to-end** — `Training exit code: 0`; 5 steps, generation→reward→policy-update→validation→checkpoint, `Avg Reward 0.0960`, zero errors (worker-27, 8-GPU colocated). The proven path.
- [x] Launcher symlinks **both** sync + async worker venv classes to `grpo-home-venv` (§5.3).
- [x] **Merged origin/main** (resolved 5 conflicts; visual-obs wiring re-verified post-merge: formatter still emits `task_type=visual_obs` + `exercise_id`). Pushed to the `sandragodinhosilva` fork only (never SWORD `origin`). The merge also brought in `run_vlm_grpo.py`'s `async_grpo_train` dispatch.
- [x] **Renamed `thrive_vlm_*` env modules → `visual_obs_*`** to avoid colliding with origin/main's env refactor (it deleted `thrive_vlm_environment.py` → generic `vlm_environment.py`). Class names + the `visual-obs` env key unchanged.
- [x] Fixed `extract_debug_info` — added the `visual_obs` branch (+ `exercise_ids` param); debug print now shows `[visual_obs]` with real per-question scores instead of `[repetition]`.
- [x] Identified the **stale-checkpoint-resume 0-steps trap** (§5.4) — a prior `step_N` ckpt in the `checkpoint_dir` makes a re-run resume at `total_steps=N` and exit 0 without training. Clear `checkpoint_dir/step_*` between runs.
- [x] **Worker venv-dir root cause fixed (§5.3)** — the custom `NEMO_RL_VENV_DIR=/mnt/data` (NFS) broke Ray's worker bootstrap (`ModuleNotFoundError: ray` for ReplayBuffer/Megatron). Reverted to default `venvs/`; symlink only the vLLM classes.
- [x] **Reward pipeline CONFIRMED end-to-end** — a real visual-obs rollout scored `Final Reward: 0.9286` with the correct `[visual_obs]` label (not `[repetition]`) — validates `compute_visual_obs_reward` routing, the `extract_debug_info` fix, AND the merged formatter's `exercise_id` threading, all on B300.

**Run mode — colocated is the validated path (USE THIS).** Final working colocated config:
`enforce_eager: true`, `max_new_tokens: 16384` (< 32784 ceiling), `attention_backend: FLASH_ATTN`,
`PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False` (IPC-compatible),
`megatron_cfg.activation_checkpointing: true` (OOM fix), default `venvs/` dir, `_triton_alloc_fix.pth`.
- [x] **Colocated GRPO trains AND learns end-to-end on B300.** Full-config smoketest on the SFT
  checkpoint: initial validation `Avg Reward 0.7433` (per-question `[visual_obs]` ordinal reward, range
  0.54–0.93), then after 1 GRPO step **`Avg Reward 0.8659`** — i.e. the policy improved. Passed both
  failure gates: IPC weight refit (no pidfd/timeout) and Megatron backward (no OOM).
- [x] **Reward verified per-question** — unit-tested on real exercise schema (exercise 10001, 7 Qs):
  each question scored by ordinal distance within its own option set (1-step→0.75, farthest→0.0,
  exact→1.0), aggregate = exact mean. `reward_details` (with `per_question`) is logged to
  `train_data_step*.jsonl` for the `tools/grpo_dashboard.py` dashboard.
- [x] **Full 238-step run** on the SFT checkpoint launched (`num_prompts_per_step: 16` ×
  `num_generations_per_prompt: 8` = 128 rollouts/step for ~2× speed; `val_at_start` + `val_at_end`
  for a before/after GRPO delta).
- [~] **Async** (reference-style, non-colocated): infra all works (venv bootstrap fixed, EngineCores
  start with `enforce_eager: true`, ReplayBuffer + AsyncTrajectoryCollector run, scored a real
  `0.9286`), but generation at `max_new_tokens: 65536` is too slow to fill the buffer practically.
  **Deferred — use colocated.**

**Target node:** launch into an idle 8-GPU allocation via `srun --jobid=<JOBID> --overlap` (the
`--overlap` flag is required to attach to an allocation already holding a `bash` placeholder task).
Never hardcode the node — resolve with `$(hostname)`/`scontrol`/`squeue` and confirm 8 free GPUs first.

### 5.8 Config variants: proven (colocated) vs experimental (async)

Four GRPO configs exist, all sharing `attention_backend: "FLASH_ATTN"`, `mm_processor_kwargs`,
`gpu_memory_utilization: 0.9`:

| Config | Mode | max_new_tokens | Status |
|---|---|---|---|
| `vlm_grpo_qwen35_4b_visual_obs_cat_smoketest.yaml` | colocated, sync, 8 GPU | ≤32784 (sync assert) | ✅ proven (exit 0) |
| `vlm_grpo_qwen35_4b_visual_obs_cat.yaml` | colocated, sync, 8 GPU (SFT ckpt) | ≤32784 | ✅ proven layout |
| `vlm_grpo_qwen35_4b_visual_obs_cat_async_smoketest.yaml` | **async**, non-colocated (4 gen / 4 train) | 65536 (= ref) | ⏳ validating |
| `vlm_grpo_qwen35_4b_visual_obs_cat_async.yaml` | **async**, non-colocated (SFT ckpt) | 65536 (= ref) | ⏳ experimental |

**Colocated (proven):** vLLM + Megatron trainer time-share all 8 GPUs (`colocated.enabled: true`,
`async_engine: false`, `async_grpo.enabled: false`). Simplest; validated end-to-end. NB: the sync
worker asserts `response_length ≤ max_model_len`, so `max_new_tokens` must stay < `max_total_sequence_length`
(32784) — do **not** copy the reference's 65536 onto a colocated config (§5.4).

**Async (mirrors reference `vlm_grpo_qwen35_4b_thrive.yaml`):** generation and training run
concurrently on separate GPU pools — `colocated.enabled: false`, `async_engine: true`,
`async_grpo.enabled: true` (+ `in_flight_weight_updates`), `loss_fn.use_importance_sampling_correction: true`,
`ratio_clip_max: 0.27`, `max_new_tokens: 65536`. **These match the reference on every knob EXCEPT
`enforce_eager`**, which must be `true` on B300 (the reference's `false` crashes the async EngineCore —
torch 2.10+cu130 has no `aot_compile`; §5.4). That single deviation is forced, not a choice.
