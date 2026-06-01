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
| Starting model | `TODO: set to exported SFT checkpoint after SFT run` |
| Train data | `…/1105_not_reviewed_visual_obs/categorical/train` |
| Val data | `…/categorical/test` |
| env_name | `"visual-obs"` |
| dataset_name | `"thrive-vlm"` (uses `ThriveVLMGRPODataset`) |
| num_prompts_per_step | 16 |
| num_generations_per_prompt | 8 |
| val_period | 500 |
| TP | 2, GBS 16, MBS 1, logprob_batch_size 4, gen_batch_size 16 |
| max_new_tokens | 512 |
| Checkpoint dir | `/mnt/data/sgsilva/results/grpo_visual_obs_cat_1105_4b` |
| Log dir | `/home/sgsilva/nemo-rl-vlm/logs_grpo/grpo_visual_obs_cat_1105_4b` |

### 2.3 SLURM launcher

**File:** `examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh`

- 1 node, 8 GPUs
- Uses `UV_PROJECT_ENVIRONMENT=/home/sgsilva/nemo-rl-vlm/.venv`
- vLLM generation workers build their own sub-venv via `uv run --extra vllm` (separate from the SFT venv)

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

```bash
# Via SLURM (single node):
sbatch --wrap="srun --ntasks-per-node=1 \
  examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh"

# Or directly on a node:
bash examples/configs/recipes/vlm/slurm_multinode_worker_grpo_oracle_obs_cat_4b.sh
```

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
