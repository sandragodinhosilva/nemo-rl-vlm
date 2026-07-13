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

"""Repetition analysis reward functions for Thrive VLM GRPO training."""

import math
import re
from typing import Optional

from nemo_rl.environments.visual_obs_reward_utils import (
    extract_error_scores,
    extract_injury_risk,
    extract_movement_score,
    extract_section,
    normalize_error_name,
)


def compute_format_reward(response: str, gt_error_names: set[str]) -> float:
    """Compute format adherence reward for repetition responses.

    Criteria (fraction passed = reward):
    1. [MOVEMENT ANALYSIS] section present
    2. [FEEDBACK] section present
    3. [SCORES] section with parseable Effectiveness and Injury Risk
    4. [ERRORS] section with valid format
    5. Predicted error names match GT error names
    """
    checks_passed = 0
    total_checks = 5

    if extract_section(response, "MOVEMENT ANALYSIS"):
        checks_passed += 1

    if extract_section(response, "FEEDBACK"):
        checks_passed += 1

    scores_section = extract_section(response, "SCORES")
    if scores_section:
        has_eff = re.search(r'Effectiveness:\s*\d+', scores_section, re.IGNORECASE)
        has_ir = re.search(r'Injury Risk:\s*\d+', scores_section, re.IGNORECASE)
        if has_eff and has_ir:
            checks_passed += 1

    errors_section = extract_section(response, "ERRORS")
    if errors_section:
        strict_pattern = r'^[^\d:]+:\s*\d+\s*$'
        non_empty_lines = [line.strip() for line in errors_section.strip().split('\n') if line.strip()]
        if non_empty_lines and all(re.match(strict_pattern, line) for line in non_empty_lines):
            checks_passed += 1

    pred_errors = extract_error_scores(response)
    pred_error_names = {normalize_error_name(k) for k in pred_errors.keys()}
    if pred_error_names == gt_error_names:
        checks_passed += 1

    return checks_passed / total_checks


def compute_error_f1(
    gt_errors: dict[str, float],
    pred_errors: dict[str, float],
) -> float:
    """Per-rep error-detection F1 — the eval-anchored reward component
    (LOCAL-ONLY, sgsilva 2026-07-13, report 2026-07-13_grpo_vobs_tool_scaffold.md §2.1).

    MIRRORS the eval scorer's semantics (vlm-post-training
    visual_obs/score_tool_loop_batch.py::_score_row, lines ~95-113, state
    2026-07-13 — the scorer that produced every arm-comparison F1): presence =
    severity > 1, pairs iterate the GT-LISTED fields only (a predicted name
    absent from the GT block is IGNORED, exactly like the eval — hallucinated
    names are handled by the format reward's name-match check, not here), a
    missing predicted field counts as "not present" (FN if GT-present).

    Two DELIBERATE deviations from _score_row, both documented in the report:
    - no-error reps (no GT field > 1): the eval's per-rep f1 column is
      structurally 0.0 there (TP impossible) and its headline number is the
      POOLED F1, where such reps contribute only false positives. The per-rep
      reward surrogate of that pooled behavior is 1.0 iff nothing is flagged,
      else 0.0 — a flat 0.0 regardless of behavior would zero the gradient on
      every tier-0 rep.
    - no positional-zip fallback (the scorer's tolerance for name-mismatched
      outputs): as a REWARD it would be gameable — any N numbers under any
      names would pair up positionally.
    """
    gt_n = {normalize_error_name(k): v for k, v in gt_errors.items()}
    pred_n = {normalize_error_name(k): v for k, v in pred_errors.items()}
    tp = fp = fn = 0
    for name, gt_sev in gt_n.items():
        gt_present = gt_sev > 1.0
        pred_present = pred_n.get(name, 1.0) > 1.0
        if gt_present and pred_present:
            tp += 1
        elif (not gt_present) and pred_present:
            fp += 1
        elif gt_present and not pred_present:
            fn += 1
    if not any(v > 1.0 for v in gt_n.values()):
        return 1.0 if fp == 0 else 0.0
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def compute_severity_reward(
    gt_movement_score: float,
    gt_errors: dict[str, float],
    pred_movement_score: float,
    pred_errors: dict[str, float],
    distance_type: str = "linear",
    gt_injury_risk: Optional[float] = None,
    pred_injury_risk: Optional[float] = None,
) -> float:
    """Compute severity-based reward using distance metric with per-field normalization."""
    MAX_EFF_RAW = 2.0
    MAX_IR_RAW = 2.0
    MAX_SEV_RAW = 5.0

    if distance_type == "quadratic":
        def dist(a, b): return (abs(a - b)) ** 2
        def max_dist_for(m): return m ** 2
    elif distance_type == "sqrt":
        def dist(a, b): return math.sqrt(abs(a - b))
        def max_dist_for(m): return math.sqrt(m)
    else:
        def dist(a, b): return abs(a - b)
        def max_dist_for(m): return m

    total_normalized_distance = 0.0
    count = 0

    max_eff = max_dist_for(MAX_EFF_RAW)
    total_normalized_distance += dist(pred_movement_score, gt_movement_score) / max_eff
    count += 1

    if gt_injury_risk is not None and pred_injury_risk is not None:
        max_ir = max_dist_for(MAX_IR_RAW)
        total_normalized_distance += dist(pred_injury_risk, gt_injury_risk) / max_ir
        count += 1

    normalized_gt_errors = {normalize_error_name(k): v for k, v in gt_errors.items()}
    normalized_pred_errors = {normalize_error_name(k): v for k, v in pred_errors.items()}

    max_sev = max_dist_for(MAX_SEV_RAW)
    for norm_name, gt_score in normalized_gt_errors.items():
        if norm_name in normalized_pred_errors:
            total_normalized_distance += dist(normalized_pred_errors[norm_name], gt_score) / max_sev
        else:
            total_normalized_distance += 1.0
        count += 1

    if count == 0:
        return 0.0

    return max(0.0, 1.0 - total_normalized_distance / count)


def compute_severity_reward_weighted_separate_norm(
    gt_movement_score: float,
    gt_injury_risk: float,
    gt_errors: dict[str, float],
    pred_movement_score: float,
    pred_injury_risk: float,
    pred_errors: dict[str, float],
    distance_type: str = "linear",
    multiplicative: bool = False,
    error_weight: float = 1.0,
    non_error_weight: float = 0.5,
) -> float:
    """Compute severity reward with separate normalization for 'no error' vs 'has error' fields."""
    gt_errors = {normalize_error_name(k): v for k, v in gt_errors.items()}
    pred_errors = {normalize_error_name(k): v for k, v in pred_errors.items()}

    MAX_EFF_RAW = 2.0
    MAX_IR_RAW = 2.0
    MAX_SEV_RAW = 5.0

    if distance_type == "quadratic":
        def dist(a, b): return (abs(a - b)) ** 2
        def max_dist_for(m): return m ** 2
    elif distance_type == "sqrt":
        def dist(a, b): return math.sqrt(abs(a - b))
        def max_dist_for(m): return math.sqrt(m)
    else:
        def dist(a, b): return abs(a - b)
        def max_dist_for(m): return m

    no_error_dists: list[float] = []
    has_error_dists: list[float] = []

    # Effectiveness (1-3, 3=best/"no error")
    max_eff = max_dist_for(MAX_EFF_RAW)
    if gt_movement_score == 3.0:
        d = dist(pred_movement_score, 3.0) / max_eff if pred_movement_score < 3.0 else 0.0
        no_error_dists.append(d)
    else:
        has_error_dists.append(dist(pred_movement_score, gt_movement_score) / max_eff)

    # Injury Risk (1-3, 1=best/"no error")
    max_ir = max_dist_for(MAX_IR_RAW)
    if gt_injury_risk == 1.0:
        d = dist(pred_injury_risk, 1.0) / max_ir if pred_injury_risk > 1.0 else 0.0
        no_error_dists.append(d)
    else:
        has_error_dists.append(dist(pred_injury_risk, gt_injury_risk) / max_ir)

    # Error Severities (1-6, 1="not present")
    max_sev = max_dist_for(MAX_SEV_RAW)
    for error_name, gt_severity in gt_errors.items():
        pred_severity = pred_errors.get(error_name, 1.0)
        if gt_severity == 1.0:
            d = dist(pred_severity, 1.0) / max_sev if pred_severity > 1.0 else 0.0
            no_error_dists.append(d)
        else:
            has_error_dists.append(dist(pred_severity, gt_severity) / max_sev)

    if multiplicative:
        no_error_accuracy = 1.0
        for d in no_error_dists:
            no_error_accuracy *= (1.0 - d)
        has_error_accuracy = 1.0
        for d in has_error_dists:
            has_error_accuracy *= (1.0 - d)
    else:
        no_error_penalty = sum(no_error_dists) / len(no_error_dists) if no_error_dists else 0.0
        no_error_accuracy = 1.0 - no_error_penalty
        has_error_penalty = sum(has_error_dists) / len(has_error_dists) if has_error_dists else 0.0
        has_error_accuracy = 1.0 - has_error_penalty

    total_weighted_accuracy = 0.0
    total_weight = 0.0
    if no_error_dists:
        total_weighted_accuracy += non_error_weight * no_error_accuracy
        total_weight += non_error_weight
    if has_error_dists:
        total_weighted_accuracy += error_weight * has_error_accuracy
        total_weight += error_weight

    if total_weight == 0.0:
        return 0.0
    return max(0.0, total_weighted_accuracy / total_weight)


def compute_detection_correctness_severity(
    gt_movement_score: float,
    gt_injury_risk: float,
    gt_errors: dict[str, float],
    pred_movement_score: float,
    pred_injury_risk: float,
    pred_errors: dict[str, float],
    distance_type: str = "linear",
    format_reward: Optional[float] = None,
    multiplicative: bool = False,
    detection_weight: float = 0.25,
    correctness_weight: float = 0.25,
    severity_weight: float = 0.5,
    format_weight: float = 0.0,
    error_weight: float = 1.0,
    non_error_weight: float = 0.5,
) -> float:
    """Compute three-component reward: detection + correctness + severity."""
    gt_errors = {normalize_error_name(k): v for k, v in gt_errors.items()}
    pred_errors = {normalize_error_name(k): v for k, v in pred_errors.items()}

    # Detection
    detection_correct = 0
    total_error_fields = 0
    for error_name in gt_errors.keys():
        gt_has_error = gt_errors[error_name] > 1.0
        pred_has_error = pred_errors.get(error_name, 1.0) > 1.0
        if gt_has_error == pred_has_error:
            detection_correct += 1
        total_error_fields += 1
    detection_reward = detection_correct / total_error_fields if total_error_fields > 0 else 0.0

    # Correctness
    effectiveness_match = int(pred_movement_score) == int(gt_movement_score)
    injury_risk_match = int(pred_injury_risk) == int(gt_injury_risk)
    errors_match = all(
        int(pred_errors.get(name, 1.0)) == int(gt_errors[name])
        for name in gt_errors.keys()
    )
    correctness_reward = 1.0 if (effectiveness_match and injury_risk_match and errors_match) else 0.0

    # Severity
    severity_reward = compute_severity_reward_weighted_separate_norm(
        gt_movement_score, gt_injury_risk, gt_errors,
        pred_movement_score, pred_injury_risk, pred_errors,
        distance_type=distance_type, multiplicative=multiplicative,
        error_weight=error_weight, non_error_weight=non_error_weight,
    )

    # Combine
    if format_reward is not None and format_weight > 0:
        total_w = detection_weight + correctness_weight + severity_weight + format_weight
        total_reward = (
            detection_weight * detection_reward +
            correctness_weight * correctness_reward +
            severity_weight * severity_reward +
            format_weight * format_reward
        ) / total_w
    else:
        total_w = detection_weight + correctness_weight + severity_weight
        total_reward = (
            detection_weight * detection_reward +
            correctness_weight * correctness_reward +
            severity_weight * severity_reward
        ) / total_w if total_w > 0 else 0.0

    return max(0.0, min(1.0, total_reward))


def compute_rep_reward(response: str, ground_truth: str, config: dict) -> tuple[float, dict]:
    """Top-level repetition reward computation.

    Args:
        response: Model's generated response
        ground_truth: Ground truth text
        config: Dict with keys: reward_mode, detection_weight, correctness_weight,
                severity_weight, format_weight, error_weight, non_error_weight

    Returns:
        (reward, details_dict)
    """
    gt_ms = extract_movement_score(ground_truth)
    gt_ir = extract_injury_risk(ground_truth)
    gt_errors = extract_error_scores(ground_truth)
    pred_ms = extract_movement_score(response)
    pred_ir = extract_injury_risk(response)
    pred_errors = extract_error_scores(response)

    details = {
        "task_type": "repetition",
        "gt_effectiveness": gt_ms,
        "gt_injury_risk": gt_ir,
        "gt_errors": gt_errors,
        "pred_effectiveness": pred_ms,
        "pred_injury_risk": pred_ir,
        "pred_errors": pred_errors,
    }

    if gt_ms is None:
        print(f"❌ Ground truth missing movement_score. Ground truth text: {ground_truth[:200]}", flush=True)
        details["final_reward"] = 0.0
        return 0.0, details

    if pred_ms is None:
        print(f"❌ Failed to extract movement score from response: {response[:300]}", flush=True)
        details["final_reward"] = 0.0
        return 0.0, details

    reward_mode = config.get("reward_mode", "detection_correctness_severity")
    multiplicative = "_multiplicative" in reward_mode
    mode_suffix = reward_mode.replace("_multiplicative", "")
    if mode_suffix.endswith("_quadratic"):
        distance_type = "quadratic"
    elif mode_suffix.endswith("_sqrt"):
        distance_type = "sqrt"
    else:
        distance_type = "linear"

    dw = config.get("detection_weight", 0.25)
    cw = config.get("correctness_weight", 0.25)
    sw = config.get("severity_weight", 0.5)
    fw = config.get("format_weight", 0.0)
    ew = config.get("error_weight", 1.0)
    new = config.get("non_error_weight", 0.5)

    # Format reward
    format_reward = None
    if fw > 0:
        gt_error_names = {normalize_error_name(k) for k in gt_errors.keys()}
        format_reward = compute_format_reward(response, gt_error_names)
        details["format"] = format_reward

    # Compute reward
    if gt_ir is None or pred_ir is None:
        final_reward = compute_severity_reward(
            gt_ms, gt_errors, pred_ms, pred_errors,
            distance_type=distance_type, gt_injury_risk=gt_ir, pred_injury_risk=pred_ir
        )
        if format_reward is not None:
            total_w = 1.0 + fw
            final_reward = (final_reward + fw * format_reward) / total_w
    else:
        final_reward = compute_detection_correctness_severity(
            gt_ms, gt_ir, gt_errors, pred_ms, pred_ir, pred_errors,
            distance_type=distance_type, format_reward=format_reward,
            multiplicative=multiplicative,
            detection_weight=dw, correctness_weight=cw, severity_weight=sw,
            format_weight=fw, error_weight=ew, non_error_weight=new,
        )

    # Optional eval-anchored F1 component (LOCAL-ONLY, sgsilva 2026-07-13,
    # tool-rollout GRPO — report §2.1). Default f1_weight=0.0 → the blend is a
    # no-op and every existing config's reward is byte-identical.
    f1w = config.get("f1_weight", 0.0)
    if f1w > 0:
        f1 = compute_error_f1(gt_errors, pred_errors)
        details["f1"] = f1
        if gt_ir is None or pred_ir is None:
            base_w = 1.0 + (fw if format_reward is not None else 0.0)
        else:
            base_w = dw + cw + sw + (fw if (format_reward is not None and fw > 0) else 0.0)
        final_reward = (base_w * final_reward + f1w * f1) / (base_w + f1w)

    # Compute component details for logging
    if gt_ms is not None and pred_ms is not None and gt_ir is not None and pred_ir is not None:
        gt_err_norm = {normalize_error_name(k): v for k, v in gt_errors.items()}
        pred_err_norm = {normalize_error_name(k): v for k, v in pred_errors.items()}

        det_correct = sum(
            1 for name in gt_err_norm
            if (gt_err_norm[name] > 1.0) == (pred_err_norm.get(name, 1.0) > 1.0)
        )
        total_fields = len(gt_err_norm)
        details["detection"] = det_correct / total_fields if total_fields > 0 else 0.0

        details["correctness"] = 1.0 if (
            int(pred_ms) == int(gt_ms) and
            int(pred_ir) == int(gt_ir) and
            all(int(pred_err_norm.get(n, 1.0)) == int(gt_err_norm[n]) for n in gt_err_norm)
        ) else 0.0

        details["severity"] = compute_severity_reward_weighted_separate_norm(
            gt_ms, gt_ir, gt_errors, pred_ms, pred_ir, pred_errors,
            distance_type=distance_type, multiplicative=multiplicative,
            error_weight=ew, non_error_weight=new,
        )

    details["final_reward"] = final_reward
    return final_reward, details
