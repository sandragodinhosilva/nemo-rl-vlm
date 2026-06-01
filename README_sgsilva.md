# README_sgsilva

This file documents the `nemo-rl-vlm` changes currently in place for my local use case: SFT for text, video, and image workflows, with emphasis on Qwen3.5/Qwen3-VL style models and local datasets under `/mnt/data/shared/vlm/data`.

## Scope

This repo is currently being used for three SFT paths:
- text SFT on transcript-based auxiliary datasets
- video SFT on `thrive-vlm-sft` datasets
- image or mixed VLM SFT through the same `thrive-vlm-sft` pipeline used by the existing VLM recipes

The text path is the most validated one so far. The transcript Qwen3.5-4B run completed successfully on 1 node / 8 GPUs.

## Working Environment

Primary working repo:
- `/home/sgsilva/nemo-rl-vlm`

Working virtual environment:
- repo-local `.venv`

Related operator environments:
- `/home/sgsilva/vlm-post-training-home-venv`
- `/home/sgsilva/qwen3.5-serving-home-venv`

Working launch pattern:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_sft.py --config <CONFIG>
```

Checkpoint rule:
- keep `checkpointing.checkpoint_dir` under `/mnt/data/sgsilva/checkpoints` so intermediate checkpoints land on the shared data volume, not inside the repo

For multimodal VLM SFT, use:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py --config <CONFIG>
```

Compatibility note:
- prefer repo-local `.venv/bin/python` in this repo
- older references to `/home/sgsilva/nemo-rl-vlm-home-venv` should be treated as obsolete

Environment provenance:
- Python: `3.12`
- current home-scoped NeMo env confirmed with Python `3.12.3`
- env created by `uv`
- dependency source of truth:
  - `pyproject.toml`
  - `uv.lock`
  - submodule / workspace state under `3rdparty/`

Important pinned versions from `uv.lock`:
- `torch==2.10.0`
- `torchvision==0.25.0`
- `triton==3.4.0`
- `ray==2.49.2`
- `datasets==4.6.1`
- `omegaconf==2.3.0`
- `transformers==5.3.0`
- `flash-attn==2.8.3`
- `nvidia-ml-py==13.590.48`

Current status as of `2026-04-07`:
- use `/home/sgsilva/nemo-rl-vlm` for repo paths and the repo-local `.venv` for execution
- package inventory check on the repo `.venv`:
  - `276` installed distributions in both envs
  - `0` version differences across installed packages

Recommended launch:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_text_aux_qa_2403_mcqa_eval_megatron.yaml
```

Checkpoint location policy:
- intermediate and final training checkpoints should be saved under `/mnt/data/sgsilva/checkpoints`
- for the validated text config `examples/configs/sft_vlm_qwen35_4b_text_aux_qa_2403_mcqa_eval_megatron.yaml`, the configured path is:
  - `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_text_aux_qa_2403_mcqa_eval`
- verify this in config files through `checkpointing.checkpoint_dir`

## Main Files Added or Modified For This Use Case

Configs:
- `examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml`
- `examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml`
- `examples/configs/sft_vlm_qwen35_4b_video_all_samples_2403_megatron.yaml`
- `examples/configs/sft_vlm_qwen3-vl-4b_megatron.yaml`
- `examples/configs/sft_vlm_4b_4epoch_text_sft.yaml`

Code paths:
- `examples/run_sft.py`
- `examples/run_vlm_sft.py`
- `nemo_rl/data/datasets/response_datasets/oai_format_dataset.py`
- `nemo_rl/data/llm_message_utils.py`
- `scripts/verify_and_fix_config.py`

Related transcript dataset documentation lives outside this repo:
- `/mnt/data/sgsilva/vlm-post-training/aux_tasks/docs/11_TRANSCRIPT_DATASET_FORMAT_PILOT.md`

## Modifications In Place

### 1. Text SFT: transcript datasets with OpenAI-format JSONL

The text transcript use case is based on:
- `/mnt/data/shared/vlm/data/text_aux_datasets/patient_qa_open_postsession_1903`
- `/mnt/data/shared/vlm/data/text_aux_datasets/patient_qa_mcqa_postsession_1903`

Config added:
- `examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml`

Key changes made for this path:
- `openai_format` dataset loading now accepts both:
  - native JSON arrays in `messages`
  - legacy string-encoded JSON stored inside `messages`
- the normalized loader now keeps only the SFT-relevant fields needed downstream, which avoids HuggingFace dataset concatenation failures caused by mismatched metadata columns across datasets
- empty Qwen non-thinking wrappers are stripped during formatting when they are template scaffolding only:
  - `<think></think>`
  - `<thinking></thinking>`
- non-empty reasoning content is preserved
- the transcript config uses full `data:` override and full `logger:` override because base `sft.yaml` assumes a different config structure for `data.train`

Text SFT command:
```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python examples/run_sft.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml
```

Status:
- validated end to end
- successful run completed for 4 epochs on 1 node / 8 GPUs
- final validation loss observed during the successful run: `1.0155`

### 1b. Text SFT: `exercise_instructions_natural_2403/openai_sft`

This repo also now has a local text SFT path for:
- original HF dataset root:
  - `/mnt/data/shared/vlm/data/text_aux_datasets/exercise_instructions_2403`
- final training dataset:
  - `/mnt/data/shared/vlm/data/text_aux_datasets/exercise_instructions_natural_2403/openai_sft`

Important dataset details:
- the original `exercise_instructions_2403` root is a HuggingFace `save_to_disk()` dataset with only a `train` split
- the final dataset used for SFT is the derived OpenAI-format export under `openai_sft`
- `openai_sft` already provides:
  - `train.jsonl`
  - `validation.jsonl`
- assistant messages in those files already contain empty think wrappers:
  - `<think>\n\n</think>\n\n`
- `enable_thinking` stays disabled in config:
  - `policy.tokenizer.chat_template_kwargs.enable_thinking: false`

Configs added:
- original-format local config:
  - `examples/configs/sft_vlm_qwen35_4b_text_aux_exercise_instructions_2403_megatron.yaml`
- final local config used for training:
  - `examples/configs/sft_vlm_qwen35_4b_text_aux_exercise_instructions_2403_empty_think_megatron.yaml`

Notes:
- the final config uses `openai_format`
- train data path:
  - `/mnt/data/shared/vlm/data/text_aux_datasets/exercise_instructions_natural_2403/openai_sft/train.jsonl`
- validation data path:
  - `/mnt/data/shared/vlm/data/text_aux_datasets/exercise_instructions_natural_2403/openai_sft/validation.jsonl`
- checkpoint root:
  - `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_text_aux_exercise_instructions_2403_empty_think`

Launch command:
```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python examples/run_sft.py --config examples/configs/sft_vlm_qwen35_4b_text_aux_exercise_instructions_2403_empty_think_megatron.yaml
```

Observed local outcome:
- training completed and produced checkpoints:
  - `step_4`
  - `step_8`
  - `step_12`
- all three checkpoints were exported successfully to Hugging Face format under:
  - `/mnt/data/sgsilva/models/qwen35-4b-exercise-instructions-natural-2403-empty-think-step4`
  - `/mnt/data/sgsilva/models/qwen35-4b-exercise-instructions-natural-2403-empty-think-step8`
  - `/mnt/data/sgsilva/models/qwen35-4b-exercise-instructions-natural-2403-empty-think-step12`

Recommended next step after export:
- serve one exported checkpoint with `vllm`
- run a small smoke test on exercise-instruction prompts
- if needed, add a dedicated evaluator for this dataset family, since the canonical text evaluator does not yet register `exercise_instructions_natural_2403`

### 2. Video SFT: active `2403` thrive-vlm path

The active local video line is the promoted `2403` family under:
- `/mnt/data/shared/vlm/data/video_aux_datasets`

For current video SFT, use the `thrive-vlm-sft` dataset path and launch through `run_vlm_sft.py`.
`run_vlm_sft.py` is a thin wrapper around `run_sft.py` that sets `is_vlm=True`, which is what selects the multimodal processor path instead of the plain tokenizer path.

Current relevant configs:
- MCQA-only:
  - `examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml`
- full grouped video mix:
  - `examples/configs/sft_vlm_qwen35_4b_video_all_samples_2403_megatron.yaml`
- grouped family configs:
  - `examples/configs/sft_vlm_qwen35_4b_video_open_aux_2403_megatron.yaml`
  - `examples/configs/sft_vlm_qwen35_4b_video_knowledge_2403_megatron.yaml`
  - `examples/configs/sft_vlm_qwen35_4b_video_reference_2403_megatron.yaml`
  - `examples/configs/sft_vlm_qwen35_4b_video_recognition_2403_megatron.yaml`

Current active dataset interfaces:
- `mcqa_video_2403`:
  - train root: `/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_2403`
  - train rows: `7205`
  - native validation split lives under the same root
- non-MCQA promoted roots:
  - `/mnt/data/shared/vlm/data/video_aux_datasets/body_region_2403/train` (`1183`)
  - `/mnt/data/shared/vlm/data/video_aux_datasets/breathing_guidance_natural_2403/train` (`1183`)
  - `/mnt/data/shared/vlm/data/video_aux_datasets/description_generation_natural_2403/train` (`434`)
  - `/mnt/data/shared/vlm/data/video_aux_datasets/exercise_name_identification_natural_2403/train` (`1183`)
  - `/mnt/data/shared/vlm/data/video_aux_datasets/muscles_involved_natural_2403/train` (`1038`)
- grouped full-video train total:
  - `12226`

Important behavior in `examples/run_sft.py` already in place:
- if the dataset task resolves to `thrive-vlm`, the loader attaches a `datum_preprocessor`
- that preprocessor uses `format_thrive_vlm_dataset(..., return_pil=True)`
- this is the multimodal path used to materialize image/video inputs for training and validation
- `need_to_flip=true` is consumed in this path and mirrored via `ImageOps.mirror()` before frames/images are passed to the processor

Current config contract for promoted `2403` video roots:
- for non-MCQA promoted roots:
  - use `data_path: <root>/train`
  - keep `dataset_name: thrive-vlm-sft`
  - keep `split: train`
  - set `split_validation_size: 0.05`
  - set `seed: ${sft.seed}`
  - keep `data.validation: null` so validation is derived from train
- for `mcqa_video_2403`:
  - keep `data_path: /mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_2403`
  - keep `split: train`
  - keep `data.validation: null`
  - rely on the native `validation/` split exposed by the dataset root

Important dataset-interface note:
- promoted `2403` video roots also expose `openai_sft/`, but the active runtime training path should still use the dataset-root interface expected by `thrive-vlm-sft`
- for current video SFT, do not switch the training configs over to `openai_format` JSONL

Recommended MCQA video launch command:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py --config examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml
```

Fresh-run option:
- add `--ignore-previous` to force a new run instead of resuming from the latest checkpoint under the configured `checkpointing.checkpoint_dir`
- this writes checkpoints to a fresh timestamped directory derived from the configured checkpoint root, so older checkpoints are left in place

Example fresh MCQA video launch:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml \
  --ignore-previous
```

Recommended full grouped video launch command:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py --config examples/configs/sft_vlm_qwen35_4b_video_all_samples_2403_megatron.yaml
```

Recommended smoke test before a long run:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml \
  sft.max_num_steps=2 \
  sft.val_period=1 \
  sft.val_batches=1 \
  checkpointing.enabled=false
```

Expected debug / sanity signal:
- the user message should contain ordered visual inputs plus one text prompt
- training should reach the first step and validation without a media-format crash
- for the canonical MCQA line, `video_pad` or "Video tokens" may remain `0` because the working path injects frames as ordered image items rather than a native video token stream

Canonical eval input for video MCQA:
- `/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_2403/test_video_native.jsonl`

Canonical eval flip rule:
- pass `--flip-horizontal` when evaluating the canonical MCQA JSONL
- the local post-SFT helper in `/mnt/data/sgsilva/vlm-post-training/aux_tasks/sft/eval_video_post_sft.sh` already does this for the canonical path

Mixed-modality caveat relevant to video:
- the current trainer concatenates raw datasets before task-specific preprocessing
- current text/image/video source datasets do not all share one compatible raw HuggingFace schema
- for first-pass mixed-modality aux-task SFT, do not point one config directly at separate text/image/video sources
- instead, materialize one merged HF dataset root first, then train that merged root through `thrive-vlm-sft`

Status:
- active `2403` configs are in place
- the repo path for video and VLM SFT is in place
- the grouped `2403` video contract is the one that should be treated as canonical now

### 3. Image SFT: same VLM pipeline as video

There is not yet a dedicated local `README_sgsilva`-specific image config added for one image dataset, but the repo path is already aligned for image SFT through the same VLM machinery.

Relevant references:
- `examples/configs/sft_vlm_qwen3-vl-4b_megatron.yaml`
- `examples/configs/recipes/vlm/sft_qwen35_4b_thrive.yaml`
- `examples/configs/recipes/vlm/sft_qwen3_vl_4b_thrive.yaml`
- `examples/configs/recipes/vlm/sft_qwen3_vl_8b_thrive.yaml`

What this means in practice:
- image-only and mixed image/video datasets should go through `run_vlm_sft.py`
- multimodal datasets should use a dataset format compatible with `thrive-vlm-sft`
- the same multimodal preprocessing path in `examples/run_sft.py` handles both image and video content when routed through `thrive-vlm`

Status:
- image-capable configs and code paths exist
- image SFT is supported through the VLM pipeline
- a dedicated local image config for one specific aux dataset may still need to be created depending on the target dataset

### 4. Config verification script

Added script:
- `scripts/verify_and_fix_config.py`

This script now supports both:
- legacy HF dataset roots using `data.dataset_name`
- multi-source configs using `data.train` and `data.validation`
- JSONL-based `openai_format` datasets

What it checks:
- train size
- validation size
- steps per epoch
- total steps
- warmup steps
- whether `save_period`, `lr_decay_iters`, and `lr_warmup_iters` match the actual dataset size and batch size

Usage:
```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml
```

Auto-fix mode:
```bash
python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml --fix
```

## Session Log: 2026-04-06 — Video SFT Debugging and Fixes

**Date**: 2026-04-06

This section documents all problems encountered and changes made during this session while getting video SFT working on `all_samples_2403`.

### Problem 1: `deep-ep` venv build failure (DTensorPolicyWorkerV2)

**Symptom**: `NRL_FORCE_REBUILD_VENVS=true` prefetch reported `DTensorPolicyWorkerV2` as failed.

**Root cause**: CUDA 13 moved libcudacxx headers under `cccl/` subdirectory. `deep-ep`'s C++ build step includes `/usr/local/cuda/include` but not `/usr/local/cuda/include/cccl`, so `#include "cuda/std/tuple"` fails with `fatal error: cuda/std/tuple: No such file or directory`.

**Fix**: Added `deep_ep = { CFLAGS = "-I/usr/local/cuda/include/cccl", CXXFLAGS = "-I/usr/local/cuda/include/cccl" }` to `[tool.uv.extra-build-variables]` in `pyproject.toml`.

**File changed**: `pyproject.toml` line ~216.

---

### Removed from this note

The trial-specific Ray and cross-directory issues from `2026-04-06` were removed from this README.
They were tied to launching from the wrong location and mixing paths during migration, not to the final working setup.

The current documented baseline is simpler:
- use the repo-local `.venv`
- run from the repo you actually intend to use
- treat `/home/sgsilva/nemo-rl-vlm` as the primary working repo

---

### Problem 6: `shape mismatch: value tensor of shape [78897, 2560] cannot be broadcast to indexing result of shape [30420, 2560]`

**Symptom**: After workers initialized and model loaded, the first forward pass crashed in `modelling_qwen3_vl/model.py:371`:
```
combined_embeddings[vision_mask] = vision_embeds
RuntimeError: shape mismatch: value tensor of shape [78897, 2560] cannot be broadcast to indexing result of shape [30420, 2560]
```

**Root cause**: `converter_type: "Qwen2ForCausalLM"` in the base config `sft_vlm_qwen35_4b_local_common_megatron.yaml` loads the plain text Megatron bridge (`Qwen2ForCausalLM` → text-only GPT model). For Qwen3.5-4B VLM (`Qwen3_5ForConditionalGeneration`), the correct bridge is `Qwen35VLBridge` which sets `image_token_id=248056` and `video_token_id=248057` (the actual Qwen3.5 token IDs). The wrong bridge defaulted to `image_token_id=151655` (Qwen2-VL token ID), so `reorganize_inputs` found 0 image positions via `vision_mask` while `vision_embeds` had actual visual data — causing the count mismatch.

**Fix**: Changed `converter_type` in `examples/configs/sft_vlm_qwen35_4b_local_common_megatron.yaml`:
```yaml
# Before (wrong — text-only bridge):
converter_type: "Qwen2ForCausalLM"

# After (correct — Qwen3.5 VLM bridge):
converter_type: "Qwen3_5ForConditionalGeneration"
```

**File changed**: `examples/configs/sft_vlm_qwen35_4b_local_common_megatron.yaml` line 39.

**Note**: The recipe configs under `examples/configs/recipes/vlm/sft_qwen35_*.yaml` all also use `"Qwen2ForCausalLM"` — these would need the same fix if used for VLM (video/image) SFT. For text-only SFT they are fine.

---

### Problem 7: Horizontal flip verification

**Question**: Do all video datasets in `all_samples_2403` flip correctly?

**Finding**:
- `mcqa_video_2403`: `need_to_flip=True` for all samples. Uses `video_frames` field. Flip applied correctly via `ImageOps.mirror()` in `format_thrive_vlm_dataset`.
- `breathing_guidance_natural_2403`, `body_region_2403`, `description_generation_natural_2403`, `exercise_name_identification_natural_2403`, `muscles_involved_natural_2403`: these datasets do NOT have `need_to_flip` field (returns `False`). They use the `image` field (list of frame paths). No flip applied — correct behavior since these datasets are already in the right orientation.

**Status**: Flip handling is correct for all current `all_samples_2403` datasets.

---

## Known Constraints

### Qwen3.5 non-thinking mode

For the transcript text use case we explicitly use:
- `policy.tokenizer.chat_template_kwargs.enable_thinking: false`

The repo now normalizes empty reasoning wrappers inserted by the template when needed, instead of supervising on empty reasoning tags.

### Sequence packing with Qwen3.5 Megatron

For Qwen3.5-4B in this Megatron stack, sequence packing should remain disabled.

Observed failure when enabled:
- `GDN does not support packed sequence for now.`

So for the working transcript config:
- `sequence_packing.enabled: false`

This constraint is especially important for Qwen3.5-based text SFT, and likely relevant to the Qwen3.5 multimodal path as well.

### Checkpointing overhead on small datasets

For the successful transcript run, checkpointing dominated step time.

Observed example:
- total step time: about `48.44s`
- checkpointing: about `43.52s`
- policy training: about `2.10s`

Implication:
- for small datasets, save frequency should usually be reduced
- future transcript reruns should likely save only once at the end unless intermediate checkpoints are actually needed

## Inference And Checkpoint Export

### What training produces

For the current local SFT use case, checkpoints are being saved under the configured `checkpointing.checkpoint_dir`.

Current local examples:
- text transcript SFT:
  - `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts`
- video SFT:
  - `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_mcqa_video_2403`
  - `/mnt/data/sgsilva/checkpoints/sft_qwen35_4b_video_all_samples_2403`

For Megatron SFT runs, the saved training weights are not yet a plain Hugging Face checkpoint directory ready for `from_pretrained(...)` or `vllm serve`.

The training output structure typically contains step folders such as:
- `tmp_step_<N>/`
- `step_<N>/`

Inside each saved step, the weights live under:
- `policy/weights/`

For Megatron exports, the converter expects the Megatron iteration subdirectory, typically:
- `policy/weights/iter_0000000`

### Export path for inference

For the Qwen3.5 Megatron SFT runs in this repo, the intended export path is:
1. train with `run_sft.py` or `run_vlm_sft.py`
2. pick the saved checkpoint step directory
3. convert the Megatron checkpoint to Hugging Face format
4. use the exported HF directory for inference, evaluation, or serving

The repo already includes the single-checkpoint export script for Megatron checkpoints:
- `examples/converters/convert_megatron_to_hf.py`

There is also a repo-local bulk export helper for this use case:
- `scripts/export_all_checkpoints.sh`

There is also a DCP export script:
- `examples/converters/convert_dcp_to_hf.py`

For the current transcript and local video configs, the relevant export path is the Megatron one, not the DCP one.

### Example: export one transcript SFT checkpoint to HF

After a successful transcript SFT run, export one checkpoint with:

```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py   --config /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts/step_68/config.yaml   --megatron-ckpt-path /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts/step_68/policy/weights/iter_0000000   --hf-ckpt-path /mnt/data/sgsilva/models/qwen35-4b-patient-qa-postsession-transcripts-step68
```

If the final saved directory is `tmp_step_68` rather than `step_68`, use that path instead.

### Export all transcript checkpoints

The repo now includes a bulk export helper that follows the old `nvidia-rl` style.

Default models output directory:
- `/mnt/data/sgsilva/models`

Example:

```bash
cd /home/sgsilva/nemo-rl-vlm
bash scripts/export_all_checkpoints.sh   /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts   ""   qwen35-4b-patient-qa-postsession-transcripts
```

This exports all saved `step_*` checkpoints to:
- `/mnt/data/sgsilva/models/qwen35-4b-patient-qa-postsession-transcripts-step17`
- `/mnt/data/sgsilva/models/qwen35-4b-patient-qa-postsession-transcripts-step34`
- `/mnt/data/sgsilva/models/qwen35-4b-patient-qa-postsession-transcripts-step51`
- `/mnt/data/sgsilva/models/qwen35-4b-patient-qa-postsession-transcripts-step68`

If the output directory argument is omitted, the script uses `/mnt/data/sgsilva/models` by default.

### Example: export video SFT checkpoint to HF

```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py   --config /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_mcqa_video_2403/<STEP_DIR>/config.yaml   --megatron-ckpt-path /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_mcqa_video_2403/<STEP_DIR>/policy/weights/iter_0000000   --hf-ckpt-path /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_mcqa_video_2403/hf_<STEP_DIR>
```

Replace `<STEP_DIR>` with the actual saved step directory name.

### Inference after export

Once exported to Hugging Face format, there are two practical options.

Option 1: Hugging Face / Transformers inference
- best for quick local checks or writing a custom prompt script
- use the exported HF directory as `model_name_or_path`

Typical pattern:
```python
from transformers import AutoProcessor, AutoTokenizer

model_path = "/path/to/exported/hf_checkpoint"
```

For text-only Qwen3.5 runs, `AutoTokenizer` is the relevant tokenizer path.
For image/video-capable Qwen3-VL style runs, use the appropriate processor/tokenizer expected by the model family.

Option 2: vLLM serving or offline inference
- best when you want high-throughput generation or an API-style endpoint
- point `vllm serve` or your vLLM script to the exported HF checkpoint directory

Example pattern:
```bash
uv run --extra vllm vllm serve /path/to/exported/hf_checkpoint
```

### Evaluation path in this repo

The repo's documented evaluation flow is:
1. export to HF
2. run `examples/run_eval.py` against the exported HF directory

Example:
```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
uv run python examples/run_eval.py generation.model_name=/path/to/exported/hf_checkpoint
```

### Practical notes for this use case

- For the transcript text run, inference should be done from the exported HF checkpoint, not directly from the Megatron step folder.
- For multimodal image/video models, the safest first step is also to export to HF and then test inference from the exported checkpoint.
- If you want to keep provenance clean, create one exported HF directory per saved step, for example:
  - `hf_step_68`
  - `hf_step_136`
- For small SFT runs where only the final checkpoint matters, reducing checkpoint frequency will simplify later export and inference.

## Practical Launch Summary

Text transcripts:
```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python examples/run_sft.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml
```

Video VLM:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python examples/run_vlm_sft.py --config examples/configs/sft_vlm_qwen35_4b_mcqa_video_2403_local_megatron.yaml
```

Bulk checkpoint export:
```bash
cd /home/sgsilva/nemo-rl-vlm
bash scripts/export_all_checkpoints.sh   /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts   ""   qwen35-4b-patient-qa-postsession-transcripts
```

Config verification:
```bash
cd /home/sgsilva/nemo-rl-vlm
.venv/bin/python scripts/verify_and_fix_config.py --config <CONFIG>
```

Important config-shape note:
- `verify_and_fix_config.py` validates dataset sizes and schedule fields, but it does not guarantee that Hydra/OmegaConf inheritance will merge cleanly at runtime.
- For multi-dataset SFT configs that replace the base single-dataset `data.train` / `data.validation` structure with lists, set `data._override_: true`.
- This is required for configs such as `examples/configs/sft_vlm_qwen35_4b_image_all_tasks_3103_megatron.yaml`, which otherwise can pass the checker but still fail in `examples/run_sft.py` with `Cannot merge DictConfig with ListConfig`.

## Notes For Future Extension

If additional aux datasets are added next, prefer this pattern:
- text datasets: `openai_format` JSONL via `run_sft.py`
- image/video datasets: `thrive-vlm-sft` via `run_vlm_sft.py`
- keep Qwen3.5 `enable_thinking: false` unless a reasoning dataset explicitly needs reasoning supervision
- validate schedule fields with `scripts/verify_and_fix_config.py`
- be careful with sequence packing on Qwen3.5 Megatron

## Image SFT Smoke Test

For the current image `3103/openai_sft/*.jsonl` exports, use the VLM entrypoint:

```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python examples/run_vlm_sft.py --config examples/configs/sft_vlm_qwen35_4b_image_all_tasks_3103_megatron.yaml
```

Why:
- the active image SFT datasets are multimodal `openai_format`
- user `messages[*].content` is a list containing an `image_url` block plus a text block
- `run_vlm_sft.py` routes through `run_sft.py` with `is_vlm=True`, so it uses the processor path instead of the plain tokenizer path

Before launching a long run, use this smoke test:

```bash
cd /home/sgsilva/nemo-rl-vlm
source .venv/bin/activate
python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_image_all_tasks_3103_megatron.yaml \
  data.debug_max_samples=1 \
  data.debug_decode_chars=300
```

Expected debug signal in stdout:
- `DEBUG: Tokenized sample inspection`
- for a user message:
  - `content=list`
  - `image_items=1`
  - `text_items=1`

That is the simplest confirmation that image content is being carried into the SFT preprocessing path.

Additional mixed-modality debug flags in `examples/run_sft.py`:

- `+data.debug_one_sample_per_modality=true`
  - scans until it finds one `text`, one `image`, and one `video` sample
- `+data.debug_one_sample_per_task_type=true`
  - scans until it finds one representative sample for each raw `source_dataset` value
  - for `/mnt/data/shared/vlm/data/merged_aux_datasets/reas_mix_image10k_text5k_video10k_0804`, the task-type field is `source_dataset`
- `+data.debug_decode_chars=-1`
  - disables decoded text truncation and prints the full decoded preview
- `+data.debug_stop_after_samples=true`
  - exits cleanly after the requested debug scan completes
  - for per-modality scans, it now waits until all requested modalities are found before exiting

Per-modality debug example for the reasoning mix:

```bash
cd /home/sgsilva/nemo-rl-vlm
unset RAY_ADDRESS
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_reas_mix_image10k_text5k_video10k_0804_megatron.yaml \
  +data.debug_one_sample_per_modality=true \
  +data.debug_decode_chars=-1 \
  +data.debug_stop_after_samples=true \
  checkpointing.enabled=false
```

What the per-modality debug header shows:

- `modality=text|image|video`
- `source_dataset=...`
- `source_modality=...`
- `reasoning=true|false`
- for video samples: `need_to_flip=true|false`

Notes:

- `source_dataset` is preserved from the raw THRIVE row and is the practical task-type label for merged aux datasets
- `reasoning=true` means the assistant target already contains a `<think>` or `<thinking>` block in the raw training sample
- multimodal user text is now forced onto a new line after image/video content in `nemo_rl/data/llm_message_utils.py`, so decoded previews are easier to read

Per-task-type debug example:

```bash
cd /home/sgsilva/nemo-rl-vlm
unset RAY_ADDRESS
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_reas_mix_image10k_text5k_video10k_0804_megatron.yaml \
  +data.debug_one_sample_per_task_type=true \
  +data.debug_decode_chars=-1 \
  +data.debug_stop_after_samples=true \
  checkpointing.enabled=false
```

Save debug output to a file while also seeing it live:

```bash
cd /home/sgsilva/nemo-rl-vlm
unset RAY_ADDRESS
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python examples/run_vlm_sft.py \
  --config examples/configs/sft_vlm_qwen35_4b_reas_mix_image10k_text5k_video10k_0804_megatron.yaml \
  +data.debug_one_sample_per_modality=true \
  +data.debug_decode_chars=-1 \
  +data.debug_stop_after_samples=true \
  checkpointing.enabled=false \
  2>&1 | tee /home/sgsilva/vlm_debug_samples.txt
```

Notes on Hydra override syntax:

- use `foo.bar=value` only for keys that already exist in the config
- use `+foo.bar=value` when adding debug-only keys that are not declared in the YAML
- example:
  - `+data.debug_one_sample_per_modality=true`

## THRIVE Dataset Format And Config

High-level pipeline summary:

Your workflow is a production-grade Supervised Fine-Tuning (SFT) pipeline centered on Qwen-3.5-4B, sharded via Tensor Parallelism (`TP=4`) within the NVIDIA NeMo RL ecosystem to support a massive 32k context window. You are training on a balanced multimodal mix of about 25k samples across image, video, and text using the custom `thrive-vlm-sft` / THRIVE-VLM data path, which handles preprocessing details such as video-frame geometry normalization and horizontal mirroring for `need_to_flip` samples. The run is orchestrated through Ray and Hydra, uses `bfloat16` precision with micro batch size `1`, and relies on token-level debug inspection to verify that the vision pathway is wired correctly and that the model is learning from visual context rather than text-only artifacts.

Data strategy: `Thrive-VLM` multimodal mix

- the active reasoning-mix workflow is built around a custom THRIVE loader path (`ThriveVLMDataset` / `thrive-vlm-sft`) over a mixed dataset made of roughly `10k` image samples, `5k` text-only samples, and `10k` video samples
- the dataset contract is OpenAI-style chat data (`role=user`, `role=assistant`) while still injecting raw multimodal payloads directly into the user turn
- for image rows, the user turn carries image items plus text
- for video rows, the user turn carries video frame lists or video content plus text
- the formatting path uses raw PIL images / PIL frame lists during preprocessing rather than flattening everything into pure text ahead of time
- video preprocessing includes a geometry-normalization safety layer that resizes within-sample frame-size drift to a common spatial size before the Qwen processor sees the frames
- hardware-specific correction is handled through `need_to_flip=true`, which mirrors frames/images for datasets captured with mirrored camera mounting

Training infrastructure

- stack: NVIDIA NeMo RL + Ray + Hydra
- Ray is used to stand up or attach to the local/distributed compute cluster and to manage worker placement
- Hydra-style overrides are used at launch time for debugging and config mutation
- the current Megatron/Qwen3.5 4B SFT runs use `bfloat16`, micro batch size `1`, and tensor parallelism `4`

The mixed image/video/text SFT roots used here are loaded through the `thrive-vlm-sft` response-dataset path and then formatted by `nemo_rl/data/datasets/response_datasets/thrive_vlm.py`.

Practical raw row fields in the merged HF dataset:

- `messages`
  - the base chat structure used for training
  - user content may already contain text items and sometimes placeholder `image_url` fields
- `image` or `video_frames`
  - modality-bearing fields used by the THRIVE formatter to inject multimodal items into the user turn
- `need_to_flip`
  - important for many video examples; the THRIVE formatter mirrors frames/images when this is `true`
- `source_dataset`
  - the most useful task-type label for merged roots
  - examples: `task1_3103`, `task4a_3103`, `mcqa_video_0804`, `clinical_reasoning_natural_2403`
- `source_modality`
  - raw modality label, typically `Image`, `Video`, or `Text`
- `source_split`
  - original split label from the materialized source
- `question_category`
  - populated for some text/patient-QA style subsets
- `reasoning_teacher_prompt`, `reasoning_teacher_system_message`, `reasoning_teacher_model`, `reasoning_raw_response`
  - present in reasoning-oriented merged datasets and useful for provenance/debugging
- exercise metadata when available:
  - `exercise_code`
  - `exercise_name`
  - `body_region`
  - `variant`
  - `session_mode`
  - `pain_bucket`

THRIVE formatting behavior:

- if `video_frames` or `video` is present, the sample is treated as `video`
- else if `image` or `images` is present, the sample is treated as `image`
- else the sample is treated as `text`
- multimodal content is injected into the user turn before the text item
- user multimodal text is normalized in `nemo_rl/data/llm_message_utils.py` so the decoded preview starts on a new line after vision content
- for reasoning debug, `reasoning=true` means the assistant target already contains a `<think>` or `<thinking>` block in the raw training sample

Config fields that matter most for these runs:

- `data.train[].dataset_name: thrive-vlm-sft`
  - keep this for the THRIVE multimodal dataset interface
- `data.train[].data_path`
  - points to the merged HF dataset root on disk
  - example: `/mnt/data/shared/vlm/data/merged_aux_datasets/reas_mix_image10k_text5k_video10k_0804`
- `data.default.processor: sft_processor`
  - keeps the SFT processor path active
- `data.default.skip_missing_local_media_filter: true`
  - avoids hard failures when promoted roots contain stale local media references
- `policy.tokenizer.chat_template_kwargs.enable_thinking`
  - `true` for reasoning-supervised configs
  - `false` for non-reasoning configs
- `policy.megatron_cfg.tensor_model_parallel_size`
  - your visible GPU count must be compatible with this
  - for the current 4-way Qwen3.5 4B Megatron configs, use 4 GPUs
- `checkpointing.checkpoint_dir`
  - keep this outside the repo, typically under `/mnt/data/sgsilva/checkpoints`
- `logger.log_dir`
  - normal experiment logs under the repo-local `logs/` tree

Example reasoning-mix config characteristics:

- `examples/configs/sft_vlm_qwen35_4b_reas_mix_image10k_text5k_video10k_0804_megatron.yaml`
  - uses `data_path: /mnt/data/shared/vlm/data/merged_aux_datasets/reas_mix_image10k_text5k_video10k_0804`
  - sets `policy.tokenizer.chat_template_kwargs.enable_thinking: true`
  - uses `policy.megatron_cfg.tensor_model_parallel_size: 4`

Operational notes:

- `run_vlm_sft.py` is a thin wrapper over `run_sft.py` with `is_vlm=True`
- when debugging locally, `unset RAY_ADDRESS` avoids accidentally attaching to an existing Ray cluster
- if you redirect output to a file, current debug output is line-buffered so progress and sample dumps should appear immediately
