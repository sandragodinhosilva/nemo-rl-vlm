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

"""Comparison task reward functions for Thrive VLM GRPO training."""

import re
from typing import Optional



_VALID_VERDICTS = ("better", "worse", "similar")


def _strip_think(response: str) -> str:
    """Remove the <think>...</think> block from the start of the response."""
    return re.sub(r"^.*?</think>\s*", "", response, flags=re.DOTALL).strip()


def extract_verdict(response: str) -> Optional[str]:
    """Extract verdict from response.

    New 3-way format: response (after any <think> block) is a single word —
    one of "better", "worse", or "similar". Returns the canonical lowercase
    verdict or None if no match.
    """
    cleaned = _strip_think(response).strip()
    # Exact single-word match first
    word = cleaned.lower().strip(".!?,;:\"'`*[]() \t\n")
    if word in _VALID_VERDICTS:
        return word
    # Fallback: first standalone verdict word anywhere in the cleaned response
    m = re.search(r"\b(better|worse|similar)\b", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def compute_comparison_format_reward(response: str) -> float:
    """Compute format adherence reward for comparison responses.

    New 3-way format expects the response (post-<think>) to be exactly one
    word from {better, worse, similar} with no extra content. Returns 1.0 on
    exact match, 0.0 otherwise.
    """
    cleaned = _strip_think(response).strip()
    word = cleaned.lower().strip(".!?,;:\"'`*[]() \t\n")
    if word in _VALID_VERDICTS and word == cleaned.lower().strip():
        return 1.0
    return 0.0


def compute_comparison_reward(response: str, ground_truth: str, config: dict) -> tuple[float, dict]:
    """Top-level comparison reward computation.

    Args:
        config: Dict with keys: comparison_verdict_weight, comparison_format_weight
    """
    gt_verdict = extract_verdict(ground_truth)
    pred_verdict = extract_verdict(response)

    details = {
        "task_type": "comparison",
        "gt_verdict": gt_verdict,
        "pred_verdict": pred_verdict,
    }

    if gt_verdict is None:
        print(f"Could not extract verdict from GT: {ground_truth[:200]}", flush=True)
        details["correctness"] = 0.0
        details["final_reward"] = 0.0
        return 0.0, details

    verdict_correct = (
        pred_verdict is not None
        and gt_verdict is not None
        and pred_verdict.upper() == gt_verdict.upper()
    )
    correctness = 1.0 if verdict_correct else 0.0
    format_score = compute_comparison_format_reward(response)

    details["correctness"] = correctness
    details["format"] = format_score

    vw = config.get("comparison_verdict_weight", 0.8)
    fw = config.get("comparison_format_weight", 0.2)
    total_w = vw + fw
    if total_w == 0:
        reward = correctness
    else:
        reward = (vw * correctness + fw * format_score) / total_w

    details["final_reward"] = reward
    return reward, details
