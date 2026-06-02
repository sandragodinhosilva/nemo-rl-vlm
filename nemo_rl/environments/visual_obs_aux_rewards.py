# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auxiliary task reward functions for Thrive VLM GRPO training.

Reward functions for MCQ, exercise name identification, keypoint prediction,
and keypoint labeling tasks. All functions are stateless and take
(response, ground_truth, config) -> (reward, details_dict).
"""

import difflib
import math
import re
from typing import Optional


# --- Task type registries ---

MCQ_TASK_TYPES = {
    "video_mcqa",
    "image_mcqa",
    "phase_sequencing_mcqa",
    "muscle_exercise_mcqa",
    "error_correction_mcqa",
    "error_recognition",
}

AUX_TASK_TYPES = MCQ_TASK_TYPES | {
    "exercise_name_identification",
    "keypoint_prediction",
    "keypoint_labeling",
}

# COCO-style per-keypoint sigmas for OKS computation.
# Standard COCO keypoints + interpolated values for hand/foot keypoints.
KEYPOINT_SIGMAS = {
    "nose": 0.026,
    "left eye": 0.025,
    "right eye": 0.025,
    "left ear": 0.035,
    "right ear": 0.035,
    "left shoulder": 0.079,
    "right shoulder": 0.079,
    "left elbow": 0.072,
    "right elbow": 0.072,
    "left wrist": 0.062,
    "right wrist": 0.062,
    "left hip": 0.107,
    "right hip": 0.107,
    "left knee": 0.087,
    "right knee": 0.087,
    "left ankle": 0.089,
    "right ankle": 0.089,
    # Interpolated from wrist/ankle sigmas for hand/foot keypoints
    "left pinky": 0.062,
    "right pinky": 0.062,
    "left index": 0.062,
    "right index": 0.062,
    "left heel": 0.089,
    "right heel": 0.089,
    "left foot index": 0.089,
    "right foot index": 0.089,
}

DEFAULT_SIGMA = 0.079  # median COCO sigma, used for unknown keypoints


# --- Helper functions ---

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _extract_mcq_answer(text: str) -> Optional[str]:
    """Extract a single MCQ answer letter (A-D) from text.

    Looks for a standalone letter at the end of the text (after think tags are stripped).
    """
    stripped = _strip_think_tags(text).strip()
    # Try last non-empty line first
    lines = [l.strip() for l in stripped.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check if it's a single letter
        match = re.match(r'^([A-Da-d])\.?$', last_line)
        if match:
            return match.group(1).upper()
    # Fallback: find any standalone letter in the stripped text
    match = re.search(r'\b([A-Da-d])\b', stripped)
    if match:
        return match.group(1).upper()
    return None


def _parse_keypoints(text: str) -> dict[str, tuple[float, float]]:
    """Parse keypoint coordinates from text.

    Expected format: `N. Body Part: <point>(x, y)</point>`
    Also handles: `N. Body Part: (x, y)` without point tags.

    Returns dict mapping normalized body part name -> (x, y).
    """
    keypoints = {}
    # Pattern with <point> tags
    pattern = r'(\d+)\.\s*(.+?):\s*(?:<point>)?\(([^,]+),\s*([^)]+)\)(?:</point>)?'
    for match in re.finditer(pattern, text):
        name = match.group(2).strip().lower()
        try:
            x = float(match.group(3).strip())
            y = float(match.group(4).strip())
            keypoints[name] = (x, y)
        except ValueError:
            continue
    return keypoints


def _parse_keypoint_labels(text: str) -> set[str]:
    """Parse keypoint labels from a numbered list.

    Expected format: `N. Body Part Name`

    Returns set of normalized body part names.
    """
    labels = set()
    pattern = r'(\d+)\.\s*(.+)'
    for match in re.finditer(pattern, text):
        name = match.group(2).strip().lower()
        if name:
            labels.add(name)
    return labels


# --- Reward functions ---

def compute_mcq_reward(
    response: str, ground_truth: str, config: dict
) -> tuple[float, dict]:
    """Compute reward for MCQ tasks.

    Components:
    1. Correctness (binary): predicted letter matches GT letter
    2. Format (continuous): has think tags + valid single-letter answer
    """
    gt_answer = _extract_mcq_answer(ground_truth)
    pred_answer = _extract_mcq_answer(response)

    # Correctness
    correctness = 1.0 if (
        pred_answer is not None
        and gt_answer is not None
        and pred_answer == gt_answer
    ) else 0.0

    # Format checks
    format_score = 0.0
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    has_valid_answer = pred_answer is not None
    if has_think:
        format_score += 0.5
    if has_valid_answer:
        format_score += 0.5

    # Combine
    cw = config.get("mcq_correctness_weight", 1.0)
    fw = config.get("mcq_format_weight", 0.0)
    total_w = cw + fw
    reward = (cw * correctness + fw * format_score) / total_w if total_w > 0 else correctness

    details = {
        "gt_answer": gt_answer,
        "pred_answer": pred_answer,
        "correctness": correctness,
        "format": format_score,
    }
    return max(0.0, min(1.0, reward)), details


def compute_exercise_name_reward(
    response: str, ground_truth: str, config: dict
) -> tuple[float, dict]:
    """Compute reward for exercise name identification.

    Components:
    1. Fuzzy match (continuous): SequenceMatcher ratio on normalized strings
    2. Exact match (binary): normalized strings are identical
    """
    gt_name = _strip_think_tags(ground_truth).strip().lower()
    pred_name = _strip_think_tags(response).strip().lower()

    # Remove common noise
    for noise in ["the exercise is", "the exercise being performed is", "this is"]:
        pred_name = pred_name.replace(noise, "").strip()

    fuzzy_score = difflib.SequenceMatcher(None, pred_name, gt_name).ratio()
    exact_match = 1.0 if pred_name == gt_name else 0.0

    # Combine
    fzw = config.get("exercise_name_fuzzy_weight", 0.8)
    exw = config.get("exercise_name_exact_weight", 0.2)
    total_w = fzw + exw
    reward = (fzw * fuzzy_score + exw * exact_match) / total_w if total_w > 0 else fuzzy_score

    details = {
        "gt_name": gt_name,
        "pred_name": pred_name,
        "fuzzy_score": fuzzy_score,
        "exact_match": exact_match,
    }
    return max(0.0, min(1.0, reward)), details


def compute_keypoint_prediction_reward(
    response: str, ground_truth: str, config: dict
) -> tuple[float, dict]:
    """Compute reward for keypoint prediction using OKS.

    Components:
    1. OKS (continuous): Object Keypoint Similarity averaged across matched keypoints
    2. Detection rate (continuous): fraction of GT keypoints found in prediction
    """
    gt_stripped = _strip_think_tags(ground_truth)
    pred_stripped = _strip_think_tags(response)

    gt_kps = _parse_keypoints(gt_stripped)
    pred_kps = _parse_keypoints(pred_stripped)

    if not gt_kps:
        return 0.0, {"error": "no GT keypoints parsed"}

    # Estimate object scale from GT keypoints (diagonal of bounding box)
    gt_coords = list(gt_kps.values())
    xs = [c[0] for c in gt_coords]
    ys = [c[1] for c in gt_coords]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    scale_sq = bbox_w * bbox_h  # area as scale²
    if scale_sq == 0:
        scale_sq = 1.0  # avoid division by zero

    # Compute per-keypoint OKS
    oks_values = []
    matched = 0
    for kp_name, gt_coord in gt_kps.items():
        if kp_name in pred_kps:
            matched += 1
            pred_coord = pred_kps[kp_name]
            d_sq = (gt_coord[0] - pred_coord[0]) ** 2 + (gt_coord[1] - pred_coord[1]) ** 2
            sigma = KEYPOINT_SIGMAS.get(kp_name, DEFAULT_SIGMA)
            oks = math.exp(-d_sq / (2 * scale_sq * sigma ** 2))
            oks_values.append(oks)
        else:
            oks_values.append(0.0)

    mean_oks = sum(oks_values) / len(oks_values) if oks_values else 0.0
    detection_rate = matched / len(gt_kps) if gt_kps else 0.0

    # Combine
    ow = config.get("kp_oks_weight", 0.7)
    dw = config.get("kp_detection_weight", 0.3)
    total_w = ow + dw
    reward = (ow * mean_oks + dw * detection_rate) / total_w if total_w > 0 else mean_oks

    details = {
        "mean_oks": mean_oks,
        "detection_rate": detection_rate,
        "gt_keypoints": len(gt_kps),
        "pred_keypoints": len(pred_kps),
        "matched": matched,
    }
    return max(0.0, min(1.0, reward)), details


def compute_keypoint_labeling_reward(
    response: str, ground_truth: str, config: dict
) -> tuple[float, dict]:
    """Compute reward for keypoint labeling.

    Components:
    1. F1 score (continuous): harmonic mean of precision and recall on label sets
    2. Exact match (binary): predicted set equals GT set
    """
    gt_stripped = _strip_think_tags(ground_truth)
    pred_stripped = _strip_think_tags(response)

    gt_labels = _parse_keypoint_labels(gt_stripped)
    pred_labels = _parse_keypoint_labels(pred_stripped)

    if not gt_labels:
        return 0.0, {"error": "no GT labels parsed"}

    # Set-based metrics
    true_positives = len(gt_labels & pred_labels)
    precision = true_positives / len(pred_labels) if pred_labels else 0.0
    recall = true_positives / len(gt_labels) if gt_labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if gt_labels == pred_labels else 0.0

    # Combine
    f1w = config.get("kl_f1_weight", 0.8)
    exw = config.get("kl_exact_weight", 0.2)
    total_w = f1w + exw
    reward = (f1w * f1 + exw * exact_match) / total_w if total_w > 0 else f1

    details = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "exact_match": exact_match,
        "gt_labels": len(gt_labels),
        "pred_labels": len(pred_labels),
    }
    return max(0.0, min(1.0, reward)), details


# --- Dispatch ---

def compute_aux_reward(
    task_type: str, response: str, ground_truth: str, config: dict
) -> tuple[float, dict]:
    """Dispatch to the appropriate aux reward function based on task type."""
    if task_type in MCQ_TASK_TYPES:
        return compute_mcq_reward(response, ground_truth, config)
    elif task_type == "exercise_name_identification":
        return compute_exercise_name_reward(response, ground_truth, config)
    elif task_type == "keypoint_prediction":
        return compute_keypoint_prediction_reward(response, ground_truth, config)
    elif task_type == "keypoint_labeling":
        return compute_keypoint_labeling_reward(response, ground_truth, config)
    else:
        raise ValueError(f"Unknown aux task type: {task_type}")
