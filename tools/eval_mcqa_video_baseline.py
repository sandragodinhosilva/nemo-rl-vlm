#!/usr/bin/env python3
"""
Video-native MCQA baseline evaluator.

This script evaluates MCQA JSONL samples by always sending a video payload
(`video_url`) to the model endpoint. It supports:
- prepared frame-list samples (user content contains image items)
- video-native samples (top-level "video" or "video_frames")

It preserves the same output metric schema used by legacy evaluators.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def load_jsonl(path: Path) -> List[dict]:
    samples: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_answer_letter(response: str) -> Optional[str]:
    text = response.strip()
    if text in ("A", "B", "C", "D"):
        return text
    m = re.match(r"^([A-D])[\.\)\s:]", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|correct|choose|select)\s+(?:is\s+)?([A-D])\b", text, re.I)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return None


def _natural_key(name: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _frames_from_video_dir(video_dir: str) -> List[str]:
    d = Path(video_dir)
    if not d.is_dir():
        return []
    frames = sorted(d.glob("*.webp"), key=lambda p: _natural_key(p.name))
    if not frames:
        images_dir = d / "images"
        if images_dir.is_dir():
            frames = sorted(images_dir.glob("*.webp"), key=lambda p: _natural_key(p.name))
    return [str(p) for p in frames]


def _encode_images_to_video(image_paths: List[str], fps: float, output_path: Path) -> Path:
    first = cv2.imread(image_paths[0])
    if first is None:
        raise ValueError(f"Failed to read first image: {image_paths[0]}")
    height, width, _ = first.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return output_path


def _video_data_url(video_path: Path) -> str:
    data = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    return f"data:video/mp4;base64,{data}"


def _sanitize_messages(messages: List[dict]) -> List[dict]:
    return [m for m in messages if m.get("role") != "assistant"]


def _extract_text_and_media(sample: dict) -> Tuple[str, Optional[str], List[str], float]:
    """
    Returns: (user_text, video_path, frame_paths, fps)
    """
    metadata = sample.get("metadata", {}) or {}
    fps = float(metadata.get("fps", 10.0))
    user_text_parts: List[str] = []
    frame_paths: List[str] = []
    video_path: Optional[str] = None

    if isinstance(sample.get("video"), str):
        video_path = sample["video"]
    elif isinstance(sample.get("video_frames"), list):
        frame_paths = [str(x) for x in sample["video_frames"] if isinstance(x, str)]

    messages = _sanitize_messages(sample.get("messages", []))
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            user_text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if "text" in item:
                user_text_parts.append(str(item["text"]))
            elif item.get("type") == "text" and "text" in item:
                user_text_parts.append(str(item["text"]))
            elif "image" in item:
                frame_paths.append(str(item["image"]))
            elif item.get("type") == "image" and "image" in item:
                frame_paths.append(str(item["image"]))
            elif "video" in item and video_path is None:
                video_path = str(item["video"])
            elif item.get("type") == "video" and "video" in item and video_path is None:
                video_path = str(item["video"])
            elif item.get("type") == "video_url":
                maybe_url = item.get("video_url", {}).get("url")
                if isinstance(maybe_url, str):
                    video_path = maybe_url

    if video_path and not video_path.startswith("data:") and Path(video_path).is_dir():
        frame_paths = _frames_from_video_dir(video_path)
        video_path = None

    return ("\n".join([t for t in user_text_parts if t]).strip(), video_path, frame_paths, fps)


def _build_api_messages(original_messages: List[dict], user_text: str, video_url: str) -> List[dict]:
    api_messages: List[dict] = []
    for msg in _sanitize_messages(original_messages):
        role = msg.get("role")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(str(item["text"]))
                content = "\n".join(text_parts)
            api_messages.append({"role": "system", "content": str(content)})
            continue
        if role == "user":
            api_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text or "Analyze this exercise video."},
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                }
            )
            break
    if not any(m.get("role") == "user" for m in api_messages):
        api_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "Analyze this exercise video."},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        )
    return api_messages


def compute_metrics(predictions: List[Tuple[Optional[str], str, dict]]) -> dict:
    total = len(predictions)
    correct = sum(1 for pred, gold, _ in predictions if pred == gold)
    unparsed = sum(1 for pred, _, _ in predictions if pred is None)

    tier_counts: Dict[str, Counter] = defaultdict(Counter)
    template_counts: Dict[str, Counter] = defaultdict(Counter)

    for pred, gold, meta in predictions:
        tier = meta.get("difficulty_tier", "unknown")
        template = meta.get("question_template", "unknown")
        tier_counts[tier]["total"] += 1
        template_counts[template]["total"] += 1
        if pred == gold:
            tier_counts[tier]["correct"] += 1
            template_counts[template]["correct"] += 1

    def accuracy_dict(counts: Dict[str, Counter]) -> dict:
        result = {}
        for key, c in sorted(counts.items()):
            t = c["total"]
            cr = c["correct"]
            result[key] = {
                "total": t,
                "correct": cr,
                "accuracy": round(cr / t * 100, 2) if t > 0 else 0.0,
            }
        return result

    gold_dist = Counter(gold for _, gold, _ in predictions)
    pred_dist = Counter(pred for pred, _, _ in predictions if pred is not None)

    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 2) if total > 0 else 0.0,
            "unparsed": unparsed,
            "unparsed_pct": round(unparsed / total * 100, 2) if total > 0 else 0.0,
        },
        "by_tier": accuracy_dict(tier_counts),
        "by_template": accuracy_dict(template_counts),
        "gold_answer_distribution": dict(gold_dist),
        "predicted_answer_distribution": dict(pred_dist),
    }


def run_eval(args: argparse.Namespace) -> int:
    samples = load_jsonl(Path(args.test_jsonl))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.test_jsonl}")

    if args.dry_run:
        ok_media = 0
        for sample in samples[: min(len(samples), 5)]:
            _, video_path, frame_paths, _ = _extract_text_and_media(sample)
            if (video_path and (video_path.startswith("data:") or Path(video_path).exists())) or frame_paths:
                ok_media += 1
        print("Dry-run checks:")
        print(f"  - sample_count: {len(samples)}")
        print(f"  - media_resolvable_in_first_{min(len(samples),5)}: {ok_media}")
        print(f"  - output_path: {args.output}")
        return 0

    try:
        import litellm
    except ImportError:
        print("ERROR: litellm not installed. Install it in your runtime environment.")
        return 1

    litellm.api_base = args.api_base
    litellm.api_key = args.api_key

    predictions: List[Tuple[Optional[str], str, dict]] = []
    with tempfile.TemporaryDirectory(prefix="mcqa_video_eval_") as td:
        temp_dir = Path(td)
        for i, sample in enumerate(samples):
            metadata = sample.get("metadata", {}) or {}
            gold = metadata.get("correct_answer", "?")
            user_text, video_path, frame_paths, fps = _extract_text_and_media(sample)
            try:
                if video_path and video_path.startswith("data:"):
                    video_url = video_path
                elif video_path and Path(video_path).is_file():
                    video_url = _video_data_url(Path(video_path))
                else:
                    if not frame_paths:
                        raise ValueError("No video or frames found for sample")
                    out_mp4 = temp_dir / f"sample_{i:06d}.mp4"
                    _encode_images_to_video(frame_paths, fps=fps, output_path=out_mp4)
                    video_url = _video_data_url(out_mp4)

                api_messages = _build_api_messages(sample.get("messages", []), user_text, video_url)
                response = litellm.completion(
                    model=f"openai/{args.model_name}",
                    messages=api_messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                text = response.choices[0].message.content or ""
                pred = extract_answer_letter(text)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i+1}/{len(samples)}] ERROR: {e}")
                pred = None
            predictions.append((pred, gold, metadata))
            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                running = sum(1 for p, g, _ in predictions if p == g)
                print(f"  [{i+1}/{len(samples)}] Running accuracy: {running / len(predictions) * 100:.1f}%")

    metrics = compute_metrics(predictions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model_name,
        "test_set": str(args.test_jsonl),
        "num_samples": len(samples),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n============================================================")
    print(f"EVALUATION RESULTS — {args.model_name}")
    print("============================================================")
    print(
        f"Overall: {metrics['overall']['correct']}/{metrics['overall']['total']} "
        f"({metrics['overall']['accuracy']:.1f}%)"
    )
    if metrics["overall"]["unparsed"] > 0:
        print(
            f"Unparsed: {metrics['overall']['unparsed']} "
            f"({metrics['overall']['unparsed_pct']:.1f}%)"
        )
    print(f"Saved: {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-native MCQA baseline evaluator")
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
