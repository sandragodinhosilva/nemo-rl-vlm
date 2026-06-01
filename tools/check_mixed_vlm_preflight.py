#!/usr/bin/env python3
"""Preflight checks for mixed-modality VLM SFT datasets/configs.

This script validates the materialized dataset contract before training:
- assistant messages are non-empty and do not carry a redundant leading empty
  think wrapper before a real reasoning trace
- user messages retain text
- image rows expose image content to the tokenizer path
- video rows expose video content with preserved metadata
- a representative mixed batch passes Qwen bridge input reorganization
- visual token mask count matches expected merged visual embed count
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


THINK_BLOCK_RE = re.compile(r"<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>", re.DOTALL | re.IGNORECASE)
LEADING_EMPTY_THINK_BEFORE_REAL_RE = re.compile(
    r"^\s*<(?:think|thinking)>\s*</(?:think|thinking)>\s*(?=<(?:think|thinking)>\s*\S)",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="NeMo SFT config path")
    parser.add_argument(
        "--mixed-batch-size",
        type=int,
        default=16,
        help="Representative mixed batch size for bridge audit",
    )
    return parser.parse_args()


def _count_modality(row: dict[str, Any]) -> str:
    source_modality = str(row.get("source_modality") or "")
    if source_modality:
        return source_modality
    if row.get("video_frames"):
        return "Video"
    if row.get("image"):
        return "Image"
    return "Text"


def _find_first_indices(ds: Dataset, *, per_modality: int) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {"Text": [], "Image": [], "Video": []}
    for idx, row in enumerate(ds):
        modality = _count_modality(dict(row))
        if modality in found and len(found[modality]) < per_modality:
            found[modality].append(idx)
        if all(len(v) >= per_modality for v in found.values()):
            break
    return found


def _user_text_present(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and str(item.get("text") or "").strip():
                    return True
    return False


def _assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
            return "".join(texts)
    return ""


def _assistant_has_reasoning_trace(messages: list[dict[str, Any]]) -> bool:
    return bool(THINK_BLOCK_RE.search(_assistant_text(messages)))


def _assistant_has_leading_empty_wrapper_before_real_trace(messages: list[dict[str, Any]]) -> bool:
    return bool(LEADING_EMPTY_THINK_BEFORE_REAL_RE.search(_assistant_text(messages)))


def _image_item_count(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image", "image_url"}:
                    url = ""
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = str(image_url.get("url") or "")
                    if url:
                        count += 1
    return count


def _build_representative_indices(ds: Dataset, batch_size: int) -> list[int]:
    per_modality = max(1, batch_size // 4)
    found = _find_first_indices(ds, per_modality=per_modality)
    combined = found["Text"][:per_modality] + found["Image"][:per_modality] + found["Video"][:per_modality]
    # Fill any remainder with the next available rows in order.
    seen = set(combined)
    idx = 0
    while len(combined) < batch_size and idx < len(ds):
        if idx not in seen:
            combined.append(idx)
            seen.add(idx)
        idx += 1
    return combined


def main() -> int:
    args = parse_args()

    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from examples.run_sft import setup_data
    from nemo_rl.data.datasets.response_datasets.thrive_vlm import format_thrive_vlm_dataset
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        batched_message_log_to_flat_message,
        get_formatted_message_log,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import reorganize_inputs

    register_omegaconf_resolvers()
    config = load_config(args.config)
    train_entries = config["data"]["train"]
    if isinstance(train_entries, dict):
        train_entries = [train_entries]
    if len(train_entries) != 1:
        raise SystemExit("Preflight currently expects a single merged train root in config.data.train")

    data_path = Path(str(train_entries[0]["data_path"]))
    raw_ds = load_from_disk(str(data_path))["train"]
    from transformers import AutoProcessor

    model_name = str(config["policy"].get("model_name"))
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.eos_token = processor.tokenizer.eos_token
    processor.bos_token = processor.tokenizer.bos_token
    tokenizer = processor
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and hasattr(tokenizer, "tokenizer"):
        pad_token_id = getattr(tokenizer.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        raise SystemExit("Unable to resolve pad_token_id from processor/tokenizer")
    task_spec = TaskDataSpec()

    summary: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "data_path": str(data_path),
        "row_count": len(raw_ds),
    }

    raw_modality_counts = Counter(_count_modality(dict(row)) for row in raw_ds)
    summary["raw_modality_counts"] = dict(raw_modality_counts)

    firsts = _find_first_indices(raw_ds, per_modality=2)
    summary["representative_indices"] = firsts

    sample_checks: list[dict[str, Any]] = []
    for modality in ("Text", "Image", "Video"):
        for idx in firsts[modality]:
            row = dict(raw_ds[idx])
            messages = row["messages"]
            check = {
                "idx": idx,
                "source_modality": _count_modality(row),
                "source_dataset": row.get("source_dataset"),
                "assistant_non_empty": bool(_assistant_text(messages).strip()),
                "assistant_has_reasoning_trace": _assistant_has_reasoning_trace(messages),
                "assistant_has_leading_empty_wrapper_before_real_trace": _assistant_has_leading_empty_wrapper_before_real_trace(messages),
                "user_text_present": _user_text_present(messages),
                "raw_image_items": _image_item_count(messages),
                "raw_video_frames_count": len(row.get("video_frames") or []),
                "need_to_flip": bool(row.get("need_to_flip", False)),
                "fps": float(row.get("fps") or 0.0),
                "frame_indices_head": list((row.get("frame_indices") or [])[:8]),
            }

            formatted = format_thrive_vlm_dataset(row, return_pil=True)
            message_log = get_formatted_message_log(
                formatted["messages"],
                tokenizer,
                task_spec,
                add_bos_token=False,
                add_eos_token=True,
                add_generation_prompt=False,
            )
            image_token_id = 248056
            video_token_id = 248057
            check["parsed_image_token_count"] = sum(
                int((m["token_ids"] == image_token_id).sum().item()) for m in message_log
            )
            check["parsed_video_token_count"] = sum(
                int((m["token_ids"] == video_token_id).sum().item()) for m in message_log
            )
            multimodal_shapes = {}
            for m in message_log:
                for key, value in m.items():
                    if key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                        multimodal_shapes[key] = tuple(value.as_tensor().shape)
            check["parsed_multimodal_shapes"] = multimodal_shapes
            sample_checks.append(check)

    summary["sample_checks"] = sample_checks

    batch_indices = _build_representative_indices(raw_ds, args.mixed_batch_size)
    message_logs = []
    for idx in batch_indices:
        row = dict(raw_ds[idx])
        formatted = format_thrive_vlm_dataset(row, return_pil=True)
        message_logs.append(
            get_formatted_message_log(
                formatted["messages"],
                tokenizer,
                task_spec,
                add_bos_token=False,
                add_eos_token=True,
                add_generation_prompt=False,
            )
        )
    add_loss_mask_to_message_log(message_logs, roles_to_train_on=["assistant"])
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": pad_token_id},
        make_sequence_length_divisible_by=config["policy"]["make_sequence_length_divisible_by"],
    )
    mm = flat.get_multimodal_dict(as_tensors=True)
    vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
        input_ids=flat["token_ids"],
        pixel_values=mm.get("pixel_values"),
        pixel_values_videos=mm.get("pixel_values_videos"),
        image_grid_thw=mm.get("image_grid_thw"),
        video_grid_thw=mm.get("video_grid_thw"),
        image_token_id=248056,
        video_token_id=248057,
        square_merge_size=4,
    )
    expected_visual_embeds = int((vision_grid_thw.prod(dim=1) // 4).sum().item()) if vision_grid_thw is not None else 0
    actual_visual_slots = int(vision_mask.sum().item()) if vision_mask is not None else 0
    summary["mixed_batch_check"] = {
        "batch_indices": batch_indices,
        "input_ids_shape": tuple(flat["token_ids"].shape),
        "input_lengths_shape": tuple(input_lengths.shape),
        "multimodal_shapes": {k: tuple(v.shape) for k, v in mm.items()},
        "vision_data_shape": tuple(vision_data.shape) if vision_data is not None else None,
        "vision_grid_shape": tuple(vision_grid_thw.shape) if vision_grid_thw is not None else None,
        "expected_visual_embeds": expected_visual_embeds,
        "actual_visual_slots": actual_visual_slots,
        "match": expected_visual_embeds == actual_visual_slots,
    }

    failures: list[str] = []
    for item in sample_checks:
        if not item["assistant_non_empty"]:
            failures.append(f"idx={item['idx']} missing assistant text")
        if item["assistant_has_leading_empty_wrapper_before_real_trace"]:
            failures.append(f"idx={item['idx']} assistant has redundant leading empty think wrapper")
        if not item["user_text_present"]:
            failures.append(f"idx={item['idx']} missing user text")
        modality = item["source_modality"]
        if modality == "Image" and item["parsed_image_token_count"] <= 0:
            failures.append(f"idx={item['idx']} image row produced no image tokens")
        if modality == "Video" and item["parsed_video_token_count"] <= 0:
            failures.append(f"idx={item['idx']} video row produced no video tokens")
        if modality == "Text" and (item["parsed_image_token_count"] > 0 or item["parsed_video_token_count"] > 0):
            failures.append(f"idx={item['idx']} text row produced visual tokens")
    if not summary["mixed_batch_check"]["match"]:
        failures.append("mixed batch bridge audit: visual slot count mismatch")

    summary["failures"] = failures
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
