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

"""Full exercise analysis reward functions for Thrive VLM GRPO training."""

import math
import re
from typing import Optional

from nemo_rl.environments.thrive_vlm_reward_utils import (
    Q_ORDINALS,
    extract_injury_risk,
    extract_movement_score,
    extract_qa_answers,
    extract_section,
)


def compute_full_exercise_format_reward(response: str) -> float:
    """Compute format adherence reward for full-exercise analysis responses.

    Criteria (fraction passed = reward):
    1. [SCORES] section with parseable Effectiveness and Injury Risk
    2. [MOVEMENT ANALYSIS] section present
    3. [FEEDBACK] section present
    4. All 9 Q/A pairs present
    5. All Q/A answers are recognized ordinal values from Q_ORDINALS
    """
    checks_passed = 0
    total_checks = 5

    scores_section = extract_section(response, "SCORES")
    if scores_section:
        has_eff = re.search(r'Effectiveness:\s*\d+', scores_section, re.IGNORECASE)
        has_ir = re.search(r'Injury Risk:\s*\d+', scores_section, re.IGNORECASE)
        if has_eff and has_ir:
            checks_passed += 1

    if extract_section(response, "MOVEMENT ANALYSIS"):
        checks_passed += 1

    if extract_section(response, "FEEDBACK"):
        checks_passed += 1

    answers = extract_qa_answers(response)
    non_none = [a for a in answers if a is not None]
    all_present = len(non_none) == 9
    if all_present:
        checks_passed += 1

    if all_present:
        all_recognized = True
        for q_idx, ans in enumerate(answers):
            if ans is None:
                all_recognized = False
                break
            options = Q_ORDINALS.get(q_idx, {})
            ans_lower = ans.lower().strip()
            if ans_lower not in options and ans_lower not in ("unable to determine", "not applicable"):
                all_recognized = False
                break
        if all_recognized:
            checks_passed += 1

    return checks_passed / total_checks


def compute_full_exercise_correctness_reward(
    gt_effectiveness: float,
    gt_injury_risk: float,
    gt_answers: list[Optional[str]],
    pred_effectiveness: Optional[float],
    pred_injury_risk: Optional[float],
    pred_answers: list[Optional[str]],
) -> float:
    """Compute binary correctness reward for full exercise.

    Returns 1.0 only if ALL fields match exactly.
    """
    if pred_effectiveness is None or int(pred_effectiveness) != int(gt_effectiveness):
        return 0.0
    if pred_injury_risk is None or int(pred_injury_risk) != int(gt_injury_risk):
        return 0.0

    for q_idx, (gt_ans, pred_ans) in enumerate(zip(gt_answers, pred_answers)):
        if gt_ans is None:
            continue
        gt_lower = gt_ans.lower().strip()
        if gt_lower in ("unable to determine", "not applicable"):
            continue
        options = Q_ORDINALS.get(q_idx, {})
        if gt_lower not in options:
            continue
        if pred_ans is None:
            return 0.0
        if pred_ans.lower().strip() != gt_lower:
            return 0.0

    return 1.0


def compute_full_exercise_severity_reward(
    gt_effectiveness: float,
    gt_injury_risk: float,
    gt_answers: list[Optional[str]],
    pred_effectiveness: Optional[float],
    pred_injury_risk: Optional[float],
    pred_answers: list[Optional[str]],
    distance_type: str = "linear",
) -> float:
    """Compute continuous severity reward for full exercise analysis.

    Uses ordinal distance for effectiveness (1-3), injury risk (1-3), and Q1-Q9.
    """
    MAX_EFF_RAW = 2.0
    MAX_IR_RAW = 2.0

    def _item_reward(pred_val: float, gt_val: float, max_val: float) -> float:
        d = abs(pred_val - gt_val)
        if distance_type == "quadratic":
            d = d * d
            max_d = max_val * max_val
        elif distance_type == "sqrt":
            d = math.sqrt(d)
            max_d = math.sqrt(max_val)
        else:
            max_d = max_val
        if max_d == 0.0:
            return 1.0
        return max(0.0, 1.0 - d / max_d)

    total_reward = 0.0
    count = 0

    eff_reward = _item_reward(pred_effectiveness, gt_effectiveness, MAX_EFF_RAW) if pred_effectiveness is not None else 0.0
    total_reward += eff_reward
    count += 1

    ir_reward = _item_reward(pred_injury_risk, gt_injury_risk, MAX_IR_RAW) if pred_injury_risk is not None else 0.0
    total_reward += ir_reward
    count += 1

    for q_idx, (gt_ans, pred_ans) in enumerate(zip(gt_answers, pred_answers)):
        if gt_ans is None:
            continue
        gt_lower = gt_ans.lower().strip()
        if gt_lower in ("unable to determine", "not applicable"):
            continue

        options = Q_ORDINALS.get(q_idx, {})
        gt_ord = options.get(gt_lower)
        if gt_ord is None:
            continue

        max_ord = float(max(options.values()))

        if pred_ans is None:
            q_reward = 0.0
        else:
            pred_lower = pred_ans.lower().strip()
            if pred_lower in ("unable to determine", "not applicable"):
                q_reward = 0.0
            else:
                pred_ord = options.get(pred_lower)
                if pred_ord is None:
                    q_reward = 0.0
                else:
                    q_reward = _item_reward(float(pred_ord), float(gt_ord), max_ord)

        total_reward += q_reward
        count += 1

    if count == 0:
        return 0.0
    return total_reward / count


def compute_full_exercise_reward(response: str, ground_truth: str, config: dict) -> tuple[float, dict]:
    """Top-level full exercise reward computation.

    Args:
        config: Dict with keys: fe_correctness_weight, fe_severity_weight,
                fe_format_weight, full_exercise_reward_mode
    """
    gt_eff = extract_movement_score(ground_truth)
    gt_ir = extract_injury_risk(ground_truth)
    pred_eff = extract_movement_score(response)
    pred_ir = extract_injury_risk(response)
    gt_answers = extract_qa_answers(ground_truth)
    pred_answers = extract_qa_answers(response)

    details = {
        "task_type": "full_exercise",
        "gt_effectiveness": gt_eff,
        "gt_injury_risk": gt_ir,
        "pred_effectiveness": pred_eff,
        "pred_injury_risk": pred_ir,
        "gt_answers": gt_answers,
        "pred_answers": pred_answers,
    }

    if gt_eff is None:
        print(f"❌ Full-exercise GT missing effectiveness. GT: {ground_truth[:200]}", flush=True)
        details["final_reward"] = 0.0
        return 0.0, details
    if gt_ir is None:
        print(f"❌ Full-exercise GT missing injury_risk. GT: {ground_truth[:200]}", flush=True)
        details["final_reward"] = 0.0
        return 0.0, details

    # Resolve distance type
    mode = config.get("full_exercise_reward_mode", "vanilla")
    if mode.endswith("_quadratic"):
        dt = "quadratic"
    elif mode.endswith("_sqrt"):
        dt = "sqrt"
    else:
        dt = "linear"

    cw = config.get("fe_correctness_weight", 0.25)
    sw = config.get("fe_severity_weight", 0.5)
    fw = config.get("fe_format_weight", 0.0)

    correctness = compute_full_exercise_correctness_reward(
        gt_eff, gt_ir, gt_answers, pred_eff, pred_ir, pred_answers,
    )
    severity = compute_full_exercise_severity_reward(
        gt_eff, gt_ir, gt_answers, pred_eff, pred_ir, pred_answers,
        distance_type=dt,
    )
    details["correctness"] = correctness
    details["severity"] = severity

    total_w = cw + sw
    total_reward = cw * correctness + sw * severity

    if fw > 0:
        fmt = compute_full_exercise_format_reward(response)
        details["format"] = fmt
        total_w += fw
        total_reward += fw * fmt

    if total_w == 0:
        details["final_reward"] = 0.0
        return 0.0, details

    reward = max(0.0, min(1.0, total_reward / total_w))
    details["final_reward"] = reward
    return reward, details
