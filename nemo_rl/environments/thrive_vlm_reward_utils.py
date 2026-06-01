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

"""Shared extraction utilities for Thrive VLM reward computation."""

import re
from typing import Optional


def normalize_error_name(name: str) -> str:
    """Normalize error name for matching (case-insensitive, ignore punctuation)."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


# Ordinal index mappings for Q1–Q9 in the Full Exercise Analysis format.
# Keys are lowercase answer strings; values are 0-based ordinal positions (best → worst).
Q_ORDINALS: dict[int, dict[str, int]] = {
    # Q1 — Consistency (ROM, alignment, control)
    0: {
        "highly consistent": 0,
        "mostly consistent": 1,
        "moderate variability": 2,
        "high variability": 3,
        "significant variability": 3,
    },
    # Q2 — Control and Stability
    1: {
        "excellent": 0,
        "good": 1,
        "fair": 2,
        "poor": 3,
    },
    # Q3 — Joint Alignment
    2: {
        "consistently proper": 0,
        "mostly proper": 1,
        "inconsistent": 2,
        "frequently misaligned": 3,
    },
    # Q4 — Range of Motion
    3: {
        "full": 0,
        "full/optimal": 0,
        "slightly reduced": 1,
        "moderately reduced": 2,
        "significantly reduced": 3,
    },
    # Q5 — Errors (frequency/severity)
    4: {
        "no notable errors": 0,
        "none": 0,
        "occasional minor errors": 1,
        "few minor": 1,
        "several minor errors": 2,
        "several minor": 2,
        "frequent or significant errors": 3,
        "frequent and/or severe": 3,
    },
    # Q6 — Pacing
    5: {
        "smooth and controlled": 0,
        "mostly controlled with minor inconsistencies": 1,
        "mostly consistent": 1,
        "uneven or rushed": 2,
        "some inconsistency": 2,
        "erratic": 3,
    },
    # Q7 — Symmetry
    6: {
        "symmetrical": 0,
        "fully symmetrical": 0,
        "minor asymmetry": 1,
        "noticeable asymmetry": 2,
        "significant asymmetry": 3,
    },
    # Q8 — Fatigue Signs
    7: {
        "none": 0,
        "mild in final reps": 1,
        "moderate": 2,
        "moderate, affecting form": 2,
        "significant": 3,
    },
    # Q9 — Trunk Posture
    8: {
        "excellent": 0,
        "good": 1,
        "fair": 2,
        "moderate issues": 2,
        "poor": 3,
    },
}


def extract_section(response: str, section_name: str) -> str:
    """Extract content between [SECTION_NAME] markers.

    Args:
        response: Full model response
        section_name: Name of section (e.g., "ERRORS", "SCORES")

    Returns:
        Content of the section, or empty string if not found
    """
    pattern = rf'\[{section_name}\]\s*\n(.*?)(?=\[|$)'
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_movement_score(response: str) -> Optional[float]:
    """Extract effectiveness/movement score from response.

    Handles both formats:
    - New: [SCORES]\\nEffectiveness: X
    - Old: Movement Score: X or Movement Score: X/Y
    """
    scores_section = extract_section(response, "SCORES")
    if scores_section:
        match = re.search(r'Effectiveness:\s*(\d+)', scores_section, re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Fallback to old format
    match = re.search(r'Movement Score:\s*(\d+)(?:/\d+)?', response, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def extract_injury_risk(response: str) -> Optional[float]:
    """Extract injury risk score from [SCORES] section."""
    scores_section = extract_section(response, "SCORES")
    if scores_section:
        match = re.search(r'Injury Risk:\s*(\d+)', scores_section, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_error_scores(response: str) -> dict[str, float]:
    """Extract error severity scores from response.

    Handles multiple formats:
    - New: [ERRORS]\\nError Name: score
    - Old: "Error Severity Assessment:\\n[Error Name]: score"
    - Plain: "Error Name (description): score"
    """
    errors = {}

    errors_section = extract_section(response, "ERRORS")

    if errors_section:
        text_to_parse = errors_section
    else:
        assessment_match = re.search(
            r'Error Severity Assessment:(.*?)(?=Movement Score:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        text_to_parse = assessment_match.group(1) if assessment_match else response

    pattern = r'^[-•*\s]*\[?([^\]:(]+?)(?:\([^\)]*\))?\]?\s*:\s*(\d+)\s*$'

    for line in text_to_parse.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') or line.startswith('\u201c'):
            continue

        match = re.match(pattern, line)
        if match:
            error_name = match.group(1).strip()
            score = float(match.group(2))
            errors[error_name] = score

    return errors


def extract_qa_answers(response: str) -> list[Optional[str]]:
    """Extract Q1-Q9 answers in order from a full-exercise response.

    Returns a list of exactly 9 elements (None for any missing answer).
    """
    pattern = re.compile(r'A:\s*([^\n]+)', re.IGNORECASE)
    matches = pattern.findall(response)
    answers: list[Optional[str]] = [m.strip() for m in matches]
    while len(answers) < 9:
        answers.append(None)
    return answers[:9]
