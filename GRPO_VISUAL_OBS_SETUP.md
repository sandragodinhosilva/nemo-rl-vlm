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

### 5.3 Worker venv symlink (BOTH sync and async classes)

`_env_builder` looks for a venv named after the Ray worker **class**. The colocated (sync) path
uses `VllmGenerationWorker`; the async (non-colocated) path uses `VllmAsyncGenerationWorker`.
The launcher symlinks **both** names to `grpo-home-venv` on every run:
```
/mnt/data/sgsilva/tmp/nemo-rl-ray-venvs/
  nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker            → grpo-home-venv
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker → grpo-home-venv
```
**Why both:** if the async class is *not* symlinked, `_env_builder` builds a fresh venv for it —
which (a) takes ~30 min and (b) lacks `_triton_alloc_fix.pth`, so generation crashes on B300 with
the GDN/`solve_tril` allocator error (§5.4.1). Symlinking both to `grpo-home-venv` (which has the
sm_100 vllm + the `.pth`) makes either run mode start instantly with the fix in place.
`grpo-home-venv` imports `AsyncLLMEngine`/`AsyncLLM` cleanly, so it serves the async worker too.

If a stale real dir or wrong-target symlink exists, the launcher deletes and re-creates it.

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
| `RuntimeError: aot_compile is not supported by the current configuration … torch: 2.10.0+cu130` (async EngineCore fails to start) | The **async** EngineCore tries to AOT-compile CUDA graphs when `enforce_eager: false`; torch 2.10+cu130 has no `aot_compile`. Colocated avoids it because it runs `enforce_eager: true`. | Set `vllm_cfg.enforce_eager: true` in the async configs too (loses the CUDA-graph speedup; async_grpo/async_engine still work in eager mode) |
| `Using TRTLLM prefill attention (auto-detected)` then Triton allocator crash | vllm auto-selects FlashInfer TRTLLM prefill when `kv_cache_dtype: auto`; TRTLLM kernel also hits the unallocated-Triton path | Same `attention_config` fix above (FLASH_ATTN bypasses all FlashInfer paths) |
| `Free memory < desired utilization` on `cuda:0` | Shared node: other users' processes occupy some GPUs; with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` + TP=1 all vllm workers pile onto `cuda:0` | Auto-detect free GPUs (>200 GiB) → `CUDA_VISIBLE_DEVICES` + `GPUS_PER_NODE` (§5.5) |
| `raylet started with N GPUs but CUDA_VISIBLE_DEVICES has M` | Ray `--num-gpus` must match the `CUDA_VISIBLE_DEVICES` count | `GPUS_PER_NODE` recomputed from the detected free-GPU list |
| `cluster.gpus_per_node` / `colocated.resources.gpus_per_node` mismatch | YAML GPU count must match the launcher's detected free-GPU count | On a fully-free 8-GPU node both are set to **8** (colocated); the launcher's auto-detect must yield 8 (no other-user procs) |

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

**Pending:**
- [ ] Async (experimental) smoketest validation — in progress; GPU preflight passes, no venv rebuild. Verifies `async_grpo` + `async_engine` + non-colocated on B300.
- [ ] Full GRPO run on the SFT checkpoint — colocated config is ready now; async after its smoketest validates.
- [ ] (cosmetic) `extract_debug_info` has no `visual_obs` branch → debug print mislabels visual-obs samples as `[repetition]` with `effectiveness=None`. The **actual** reward path (`verify()`→`_compute_reward_with_details`) routes correctly (reward 0.096 ≠ 0). Print-only bug.

**Target node:** launch into an idle 8-GPU allocation via `srun --jobid=<JOBID> --overlap` (the
`--overlap` flag is required to attach to an allocation already holding a `bash` placeholder task).
Never hardcode the node — resolve with `$(hostname)`/`scontrol`/`squeue` and confirm 8 free GPUs first.

### 5.8 Config variants: proven (colocated) vs experimental (async)

Four GRPO configs exist; **all** share the reference generation settings (`max_new_tokens: 65536`,
`attention_backend: "FLASH_ATTN"`, `mm_processor_kwargs`, `gpu_memory_utilization: 0.9`):

| Config | Mode | Status |
|---|---|---|
| `vlm_grpo_qwen35_4b_visual_obs_cat_smoketest.yaml` | colocated, sync, 8 GPU | ✅ proven (exit 0) |
| `vlm_grpo_qwen35_4b_visual_obs_cat.yaml` | colocated, sync, 8 GPU (SFT ckpt) | ✅ proven layout — ready for full run |
| `vlm_grpo_qwen35_4b_visual_obs_cat_async_smoketest.yaml` | **async**, non-colocated (4 gen / 4 train) | ⏳ experimental — validating |
| `vlm_grpo_qwen35_4b_visual_obs_cat_async.yaml` | **async**, non-colocated (SFT ckpt) | ⏳ experimental |

**Colocated (proven):** vLLM + Megatron trainer time-share all 8 GPUs (`colocated.enabled: true`,
`async_engine: false`, `async_grpo.enabled: false`). Simplest; validated end-to-end.

**Async (experimental, mirrors reference `vlm_grpo_qwen35_4b_thrive.yaml`):** generation and
training run concurrently on separate GPU pools — `colocated.enabled: false`, `async_engine: true`,
`async_grpo.enabled: true` (+ `in_flight_weight_updates`), and `loss_fn.use_importance_sampling_correction: true`
with `ratio_clip_max: 0.27` (rollouts are slightly off-policy). Higher GPU utilization; needed its
own smoketest on B300 before trusting a full run, and the async worker-class venv symlink fix (§5.3).
