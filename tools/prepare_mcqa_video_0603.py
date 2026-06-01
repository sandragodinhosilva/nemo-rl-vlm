#!/usr/bin/env python3
"""
Convert v12 prepared MCQA JSONL files into video-native samples for NeMo-RL SFT.

Supports both prepared shapes:
  1) older frame-list samples where `messages[*].content` contains `{"image": ...}` items
  2) newer Thrive-style samples with top-level `video_frames` and string-only messages

Outputs keep the same QA text/answer but normalize media to top-level fields:
  - video_frames: list[str]
  - video: video directory path (from sample.video or metadata.video_path)
  - fps: float
  - num_frames: int
  - need_to_flip: bool

It writes:
  1) HF DatasetDict (train/validation) to --output-dir
  2) video-native test JSONL to --output-test-jsonl for baseline eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


DEFAULT_TRAIN_JSONL = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_train_prepared.jsonl"
)
DEFAULT_VAL_JSONL = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_val_prepared.jsonl"
)
DEFAULT_TEST_JSONL = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_test_prepared.jsonl"
)
DEFAULT_OUTPUT_DIR = "/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_0603"
DEFAULT_OUTPUT_TEST_JSONL = (
    "/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_0603/test_video_native.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            text = str(item["text"]).strip()
            if text:
                text_parts.append(text)
        elif item.get("type") == "text":
            text = str(item.get("text", "")).strip()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()


def _to_string_content(content: Any) -> str:
    text = _text_from_content(content)
    if text:
        return text
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _image_paths_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    image_paths: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image")
        if isinstance(image_path, str) and image_path.strip():
            image_paths.append(image_path)
    return image_paths


def _build_video_native_sample(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(sample.get("metadata") or {})
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("sample.messages must be a list")

    video_path = sample.get("video")
    if not isinstance(video_path, str) or not video_path:
        video_path = metadata.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("missing video path (expected sample.video or metadata.video_path)")

    fps_value = sample.get("fps", metadata.get("fps"))
    if fps_value is None:
        raise ValueError("missing fps (expected sample.fps or metadata.fps)")
    fps = float(fps_value)
    if fps <= 0:
        raise ValueError(f"invalid fps={fps}")

    num_frames_value = sample.get("num_frames", metadata.get("num_frames"))
    if num_frames_value is None:
        raise ValueError("missing num_frames (expected sample.num_frames or metadata.num_frames)")
    num_frames = int(num_frames_value)
    if num_frames <= 0:
        raise ValueError(f"invalid num_frames={num_frames}")

    output_messages: list[dict[str, Any]] = []
    user_text = ""
    video_frames = sample.get("video_frames")
    if not isinstance(video_frames, list):
        video_frames = []
    else:
        video_frames = [str(p) for p in video_frames if isinstance(p, str) and p.strip()]
    assistant_added = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            output_messages.append(
                {"role": "system", "content": _to_string_content(msg.get("content", ""))}
            )
        elif role == "user" and not user_text:
            user_text = _text_from_content(msg.get("content", ""))
            if not video_frames:
                video_frames = _image_paths_from_content(msg.get("content", ""))
        elif role == "assistant" and not assistant_added:
            output_messages.append(
                {"role": "assistant", "content": _to_string_content(msg.get("content", ""))}
            )
            assistant_added = True

    if not user_text:
        raise ValueError("missing user text in messages")
    if not video_frames:
        raise ValueError("missing user image frames in messages")
    if not assistant_added:
        raise ValueError("missing assistant message in messages")

    insert_pos = len(output_messages)
    for i, msg in enumerate(output_messages):
        if msg.get("role") == "assistant":
            insert_pos = i
            break
    output_messages.insert(insert_pos, {"role": "user", "content": user_text})

    out: dict[str, Any] = {
        "messages": output_messages,
        "metadata": metadata,
        "video_frames": video_frames,
        "video": video_path,
        "fps": fps,
        "num_frames": num_frames,
        "need_to_flip": bool(sample.get("need_to_flip", metadata.get("need_to_flip", False))),
    }
    return out


def _convert_file(path: Path, strict: bool) -> tuple[list[dict[str, Any]], int]:
    rows = _read_jsonl(path)
    converted: list[dict[str, Any]] = []
    dropped = 0
    for idx, row in enumerate(rows, start=1):
        try:
            converted.append(_build_video_native_sample(row))
        except Exception as e:  # noqa: BLE001
            if strict:
                raise ValueError(f"failed converting {path}:{idx}: {e}") from e
            dropped += 1
    return converted, dropped


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_rows(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"{name}: 0 samples")
        return
    fps_values = [float(r["fps"]) for r in rows]
    frame_values = [int(r["num_frames"]) for r in rows]
    print(
        f"{name}: {len(rows)} samples | "
        f"fps[{min(fps_values):.2f}, {max(fps_values):.2f}] | "
        f"num_frames[{min(frame_values)}, {max(frame_values)}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare mcqa_video_0603 HF dataset + video-native test JSONL"
    )
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL_JSONL)
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-test-jsonl", default=DEFAULT_OUTPUT_TEST_JSONL)
    parser.add_argument("--strict", action="store_true", help="fail on the first bad sample")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting output dir/jsonl")
    parser.add_argument("--dry-run", action="store_true", help="validate + summarize without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    test_path = Path(args.test_jsonl)
    output_dir = Path(args.output_dir)
    output_test_jsonl = Path(args.output_test_jsonl)

    for p in [train_path, val_path, test_path]:
        if not p.is_file():
            raise FileNotFoundError(f"missing input file: {p}")

    if not args.overwrite:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"output dir already exists and is not empty: {output_dir} "
                "(use --overwrite)"
            )
        if output_test_jsonl.exists():
            raise FileExistsError(f"output file already exists: {output_test_jsonl} (use --overwrite)")

    train_rows, train_dropped = _convert_file(train_path, strict=args.strict)
    val_rows, val_dropped = _convert_file(val_path, strict=args.strict)
    test_rows, test_dropped = _convert_file(test_path, strict=args.strict)

    _summarize_rows("train", train_rows)
    _summarize_rows("validation", val_rows)
    _summarize_rows("test", test_rows)
    print(f"dropped: train={train_dropped}, validation={val_dropped}, test={test_dropped}")

    if args.dry_run:
        print("dry-run enabled; no files written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
        }
    )
    ds_dict.save_to_disk(str(output_dir))

    _write_jsonl(output_test_jsonl, test_rows)
    print(f"wrote HF dataset: {output_dir}")
    print(f"wrote test JSONL: {output_test_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
