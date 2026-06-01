#!/usr/bin/env python3
"""
Run MCQA video Phase 2 stages with preflight checks.

Stages:
  - eval: baseline evaluation using tools/eval_mcqa_video_baseline.py
  - prepare: dataset conversion using tools/prepare_mcqa_video_0603.py
  - all: eval then prepare

Default behavior is dry-run (no stage execution). Pass --execute to run.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_SOURCE_TRAIN = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_train_prepared.jsonl"
)
DEFAULT_SOURCE_VAL = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_val_prepared.jsonl"
)
DEFAULT_SOURCE_TEST = (
    "/mnt/data/sgsilva/video-sft-vlm/training/splits_v12/qa_samples_v12_test_prepared.jsonl"
)
DEFAULT_DATASET_DIR = "/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_0603"
DEFAULT_TEST_VIDEO_NATIVE = (
    "/mnt/data/shared/vlm/data/video_aux_datasets/mcqa_video_0603/test_video_native.jsonl"
)
DEFAULT_EVAL_OUTPUT = "/mnt/data/sgsilva/vlm-evaluation/results/mcqa_video_0603_baseline.json"
DEFAULT_MODEL_NAME = "/mnt/data/shared/models/Qwen3.5-4B"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"


def _check_writable_target(path: Path, execute: bool) -> None:
    parent = path.parent
    if execute:
        parent.mkdir(parents=True, exist_ok=True)
    else:
        # In dry-run, allow non-existing parents and validate the nearest existing ancestor.
        probe = parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        parent = probe
    if not parent.exists():
        raise FileNotFoundError(f"no existing ancestor found for target: {path}")
    if not parent.is_dir():
        raise NotADirectoryError(f"ancestor is not a directory: {parent}")


def _jsonl_first_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"invalid row type at {path}:{line_no}; expected object")
            return row
    raise ValueError(f"no non-empty rows in {path}")


def _check_expected_keys_and_fps(path: Path) -> None:
    row = _jsonl_first_row(path)
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(messages, list):
        raise ValueError(f"{path}: missing/invalid messages list")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing/invalid metadata object")

    # Contract from README: preserve video_path/fps/num_frames from prepared samples.
    if "video_path" not in metadata:
        raise ValueError(f"{path}: metadata.video_path missing")
    fps = metadata.get("fps", row.get("fps"))
    if fps is None:
        raise ValueError(f"{path}: fps missing in metadata/top-level")
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"{path}: invalid fps={fps}")
    num_frames = metadata.get("num_frames", row.get("num_frames"))
    if num_frames is None:
        raise ValueError(f"{path}: num_frames missing in metadata/top-level")
    if int(num_frames) <= 0:
        raise ValueError(f"{path}: invalid num_frames={num_frames}")


def _check_endpoint(api_base: str, timeout_s: float = 1.5) -> None:
    parsed = urlparse(api_base)
    host = parsed.hostname
    if host is None:
        raise ValueError(f"invalid api_base: {api_base}")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    with socket.create_connection((host, port), timeout=timeout_s):
        return


def _run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _build_prepare_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prepare_mcqa_video_0603.py"),
        "--train-jsonl",
        args.source_train_jsonl,
        "--val-jsonl",
        args.source_val_jsonl,
        "--test-jsonl",
        args.source_test_jsonl,
        "--output-dir",
        args.output_dataset_dir,
        "--output-test-jsonl",
        args.output_test_jsonl,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.strict:
        cmd.append("--strict")
    return cmd


def _build_eval_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_mcqa_video_baseline.py"),
        "--test-jsonl",
        args.eval_test_jsonl,
        "--model-name",
        args.model_name,
        "--api-base",
        args.api_base,
        "--api-key",
        args.api_key,
        "--output",
        args.eval_output_json,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 MCQA video pipeline")
    parser.add_argument("--stage", choices=["eval", "prepare", "all"], default="all")
    parser.add_argument("--execute", action="store_true", help="execute commands (default is dry-run)")

    parser.add_argument("--source-train-jsonl", default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--source-val-jsonl", default=DEFAULT_SOURCE_VAL)
    parser.add_argument("--source-test-jsonl", default=DEFAULT_SOURCE_TEST)
    parser.add_argument("--output-dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-test-jsonl", default=DEFAULT_TEST_VIDEO_NATIVE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")

    parser.add_argument(
        "--eval-test-jsonl",
        default=DEFAULT_SOURCE_TEST,
        help="Input JSONL for eval stage (prepared or video-native)",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--eval-output-json", default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    print(f"mode: {'execute' if execute else 'dry-run'}")
    print(f"stage: {args.stage}")

    if args.stage in ("prepare", "all"):
        for p in [Path(args.source_train_jsonl), Path(args.source_val_jsonl), Path(args.source_test_jsonl)]:
            if not p.is_file():
                raise FileNotFoundError(f"missing input file: {p}")
            _check_expected_keys_and_fps(p)
        _check_writable_target(Path(args.output_test_jsonl), execute=execute)
        # Dataset dir is a directory target; check parent writability.
        _check_writable_target(Path(args.output_dataset_dir) / ".dummy", execute=execute)
        print("preflight(prepare): OK")

    if args.stage in ("eval", "all"):
        eval_jsonl = Path(args.eval_test_jsonl)
        if not eval_jsonl.is_file():
            raise FileNotFoundError(f"missing eval test jsonl: {eval_jsonl}")
        _check_expected_keys_and_fps(eval_jsonl)
        _check_writable_target(Path(args.eval_output_json), execute=execute)
        _check_endpoint(args.api_base)
        print("preflight(eval): OK")

    commands: list[list[str]] = []
    if args.stage in ("eval", "all"):
        commands.append(_build_eval_cmd(args))
    if args.stage in ("prepare", "all"):
        commands.append(_build_prepare_cmd(args))

    if not execute:
        print("dry-run complete; planned commands:")
        for cmd in commands:
            print(" ", " ".join(cmd))
        return 0

    for cmd in commands:
        _run_cmd(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
