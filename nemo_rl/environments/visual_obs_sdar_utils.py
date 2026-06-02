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

"""SDAR (Self-Distilled Agentic RL) privileged-context utilities for THRIVE VLM.

The teacher branch in SDAR is the SAME policy run on a longer prompt that
contains a training-only hint derived from the ground-truth label. The hint
gives the teacher a privileged view (movement summary, scores, error names +
severities) so that its per-token logprobs are tighter than the student's;
those logprobs then drive the on-policy self-distillation auxiliary loss.

This module is read-only at runtime: callers pass the GT response text and
get back a plain-text hint block to prepend to the teacher's user message.
"""

from typing import Optional

from nemo_rl.environments.visual_obs_reward_utils import (
    extract_error_scores,
    extract_injury_risk,
    extract_movement_score,
    extract_section,
)

HINT_OPEN = "[PRIVILEGED CONTEXT — DO NOT REFERENCE IN YOUR RESPONSE]"
HINT_INSTRUCTION = (
    "The following are the correct assessments for this repetition. "
    "Use them to guide your reasoning but reason and respond as if you "
    "derived all conclusions from the video alone."
)
HINT_CLOSE = "[END PRIVILEGED CONTEXT]"


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.1f}"


def build_privileged_hint(
    gt_text: str,
    max_movement_chars: int = 240,
) -> str:
    """Build the privileged-context hint string for SDAR teacher prompts.

    The hint contains movement analysis (truncated), the effectiveness and
    injury-risk numbers, and a flat list of `<error_name>=<severity>` pairs.
    It is intentionally compact: the goal is to steer the teacher's
    distribution without giving it the full GT response.

    Args:
        gt_text: Full ground-truth assistant response, in THRIVE's tag format.
        max_movement_chars: Hard cap on the movement-analysis substring (we
            don't want a long hint to dominate the context budget).

    Returns:
        A short multi-line string surrounded by [HINT — training only] /
        [END HINT] markers, suitable for prepending to the user message.
        Returns an empty hint block if the GT text has none of the expected
        sections (caller should treat that as "no privileged signal").
    """
    movement = extract_section(gt_text, "MOVEMENT ANALYSIS") or ""
    eff = extract_movement_score(gt_text)
    risk = extract_injury_risk(gt_text)
    errors = extract_error_scores(gt_text)

    movement = movement.strip()
    if len(movement) > max_movement_chars:
        movement = movement[: max_movement_chars - 1].rstrip() + "…"

    error_pairs = "; ".join(f"{k}={_fmt_score(v)}" for k, v in errors.items())

    lines = [HINT_OPEN, HINT_INSTRUCTION]
    if movement:
        lines.append(f"Movement analysis: {movement}")
    lines.append(f"Effectiveness: {_fmt_score(eff)}; Injury risk: {_fmt_score(risk)}")
    if error_pairs:
        lines.append(f"Errors: {error_pairs}")
    lines.append(HINT_CLOSE)
    return "\n".join(lines) + "\n"


def has_extractable_hint(gt_text: str) -> bool:
    """Quick check whether GT has any of the sections SDAR's hint needs."""
    if extract_section(gt_text, "MOVEMENT ANALYSIS"):
        return True
    if extract_movement_score(gt_text) is not None:
        return True
    if extract_injury_risk(gt_text) is not None:
        return True
    if extract_error_scores(gt_text):
        return True
    return False
