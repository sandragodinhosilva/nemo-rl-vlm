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

"""Visual observations reward — ordinal-distance over categorical options.

Each question has an ordered list of options (from visual_observations_categorical.json).
Reward per question = 1 - |pred_idx - gt_idx| / (n_options - 1), so partial credit
is given for nearby answers. Binary questions (2 options) collapse to exact-match.

Cap at MAX_QUESTIONS_PER_EXERCISE = 7 to stay consistent across exercises.
"""

import json
import os
import re
from typing import Optional

_SCHEMA_PATH = "/home/sgsilva/vlm-post-training/visual_observations_categorical.json"
_schema: dict = {}

MAX_QUESTIONS_PER_EXERCISE = 7


def _load_schema() -> dict:
    global _schema
    if not _schema and os.path.exists(_SCHEMA_PATH):
        with open(_SCHEMA_PATH) as f:
            _schema = json.load(f)
    return _schema


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_items(text: str) -> list[str]:
    """Extract numbered answers from a [VISUAL OBSERVATIONS] block.

    Handles variable counts; always strips the header and returns at most
    MAX_QUESTIONS_PER_EXERCISE items.
    """
    text = _strip_think_tags(text)
    block = re.sub(r"\[VISUAL OBSERVATIONS\]", "", text, flags=re.IGNORECASE).strip()
    items = re.findall(
        r"^\s*\d+\.\s*(.*?)(?=^\s*\d+\.|\Z)",
        block,
        flags=re.DOTALL | re.MULTILINE,
    )
    return [it.strip() for it in items if it.strip()][:MAX_QUESTIONS_PER_EXERCISE]


def compute_visual_obs_reward(
    response: str,
    ground_truth: str,
    config: dict,
) -> tuple[float, dict]:
    """Ordinal-distance reward for [VISUAL OBSERVATIONS] categorical answers.

    For each question (up to MAX_QUESTIONS_PER_EXERCISE):
      score = 1 - |pred_idx - gt_idx| / (n_options - 1)
    Returns mean across all GT questions.

    Args:
        response: Model's generated text.
        ground_truth: Oracle [VISUAL OBSERVATIONS] block (from the dataset).
        config: Must contain "exercise_id" (str) for schema lookup.

    Returns:
        (reward in [0, 1], details_dict)
    """
    exercise_id = str(config.get("exercise_id", ""))
    schema = _load_schema()
    schema_qs = schema.get(exercise_id, {}).get("vlm_observations", [])

    pred_items = _extract_items(response)
    gt_items = _extract_items(ground_truth)

    if not gt_items:
        return 0.0, {"error": "no_gt_items", "exercise_id": exercise_id}

    # Cap to MAX_QUESTIONS_PER_EXERCISE
    gt_items = gt_items[:MAX_QUESTIONS_PER_EXERCISE]

    total, count = 0.0, 0
    per_q = []

    for i, gt_ans in enumerate(gt_items):
        options = (
            [o.lower() for o in schema_qs[i]["options"]]
            if i < len(schema_qs)
            else []
        )
        n = len(options)
        pred_ans = pred_items[i].strip().lower() if i < len(pred_items) else ""
        gt_norm = gt_ans.strip().lower()

        if n >= 2 and gt_norm in options:
            gt_idx = options.index(gt_norm)
            pred_idx = options.index(pred_ans) if pred_ans in options else -1
            if pred_idx >= 0:
                score = 1.0 - abs(pred_idx - gt_idx) / (n - 1)
            else:
                # Pred not a valid option — zero credit
                score = 0.0
        else:
            # No schema or binary — exact match
            score = 1.0 if gt_norm == pred_ans else 0.0

        total += score
        count += 1
        per_q.append({"q": i + 1, "gt": gt_norm, "pred": pred_ans, "score": round(score, 4)})

    reward = total / count if count > 0 else 0.0
    details = {
        "exercise_id": exercise_id,
        "per_question": per_q,
        "num_questions": count,
        "mean": round(reward, 4),
    }
    return max(0.0, min(1.0, reward)), details
