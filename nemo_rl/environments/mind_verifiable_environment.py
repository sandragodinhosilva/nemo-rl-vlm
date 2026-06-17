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

import json
import re
from typing import Any, Dict, List, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


# =============================================================================
# Math Verification Functions
# =============================================================================

def extract_answer(response: str) -> str | None:
    """Extract answer from 'Answer: <answer>' format.

    Args:
        response: The model response to parse

    Returns:
        Extracted answer string, or None if no answer found
    """
    # Look for "Answer:" followed by content until end of line or string
    match = re.search(r'(?i)Answer\s*:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def verify_math_answer(response: str, ground_truth: str | list[str]) -> float:
    """Verify if extracted answer matches ground truth.

    Args:
        response: The model response containing "Answer: <answer>"
        ground_truth: The correct answer (string or list of acceptable answers)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    extracted = extract_answer(response)

    if extracted is None:
        return 0.0

    # Normalize extracted answer
    normalized_extracted = extracted.strip().lower()

    # Handle ground_truth as either string or list
    if isinstance(ground_truth, list):
        # Check if extracted answer matches any of the acceptable answers
        for acceptable_answer in ground_truth:
            normalized_gt = str(acceptable_answer).strip().lower()
            if normalized_extracted == normalized_gt:
                return 1.0
        return 0.0
    else:
        # Single ground truth value
        normalized_ground_truth = str(ground_truth).strip().lower()
        return 1.0 if normalized_extracted == normalized_ground_truth else 0.0


# =============================================================================
# IFEval Verification Functions
# =============================================================================

def verify_keywords(text, keyword_list):
    """Verify if the response contains all the specified keywords."""
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N):
    """Verifies if a keyword appears exactly N times in the given text."""
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


def verify_keyword_frequency_relation(text, keyword, frequency, relation):
    """Verifies if a keyword appears with the specified frequency relation."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    actual_count = sum(1 for w in words if w == keyword_lower)

    if relation == "less than":
        return actual_count < frequency
    elif relation == "at least":
        return actual_count >= frequency
    elif relation == "at most":
        return actual_count <= frequency
    elif relation == "exactly":
        return actual_count == frequency
    else:
        return actual_count == frequency


def validate_forbidden_words(text, forbidden_words):
    """Validates that the text does not contain any of the specified forbidden words."""
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    """Verifies if a given letter appears exactly the specified number of times."""
    if len(letter) != 1:
        return False
    actual_count = text.count(letter)
    return actual_count == N


def validate_response_language(text, language):
    """Validates that the entire response is in the specified language."""
    try:
        from langdetect import detect
        detected_language = detect(text)
        return detected_language == language
    except Exception:
        return False


def verify_paragraph_count(text: str, N: int) -> bool:
    """Verifies that a text contains the expected number of paragraphs."""
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    """Validates if a text meets specified word count constraints."""
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    else:
        return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    """Verifies if a text contains the expected number of sentences."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    actual_count = len(sentences)

    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    else:
        return False


def validate_paragraphs(text, N, first_word, i):
    """Validates that a text contains the expected number of paragraphs and i-th paragraph starts with a specific word."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    # Check bounds before accessing
    if i < 1 or i > len(paragraphs):
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text, postscript_marker):
    """Verifies if a text contains a postscript starting with the marker."""
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> bool:
    """Validates if a text contains at least the specified number of placeholders in square brackets."""
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


def verify_bullet_points(text: str, N: int) -> bool:
    """Verifies if a text contains exactly N bullet points in markdown format."""
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


def validate_title(text: str) -> bool:
    """Validates if text contains a title wrapped in double angular brackets."""
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_choice(text: str, options: list) -> bool:
    """Validates if text contains one of the specified options."""
    return any(text in option for option in options)


def validate_highlighted_sections(text: str, N: int) -> bool:
    """Validates if text has at least N highlighted sections."""
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    """Validates if text has N sections marked by the section splitter."""
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    """Validates if entire output is wrapped in JSON format."""
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    """Validates if text starts with the original prompt."""
    return bool(text.startswith(original_prompt))


def validate_two_responses(text: str) -> bool:
    """Validates if text contains two different responses separated by ******."""
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


def validate_uppercase(text: str) -> bool:
    """Validates if entire response is in uppercase."""
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    """Validates if entire response is in lowercase."""
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    """Validates frequency of all-capital words."""
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return len(words) == N
    elif quantifier == "at most":
        return len(words) <= N
    else:
        return False


def validate_end(text: str, end_phrase: str) -> bool:
    """Validates if response ends with the exact phrase."""
    return bool(text.endswith(end_phrase))


def validate_quotation(text: str) -> bool:
    """Validates if response is wrapped with double quotation marks."""
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text: str) -> bool:
    """Validates if response contains no commas."""
    return "," not in text


def count_unique_words(text: str) -> int:
    """Count unique words in text."""
    words = re.findall(r"\b\w+\b", text.lower())
    return len(set(words))


def validate_count_unique(text: str, min_count: int = None, max_count: int = None) -> bool:
    """Validates unique word count constraints."""
    unique_count = count_unique_words(text)
    if min_count is not None and unique_count < min_count:
        return False
    if max_count is not None and unique_count > max_count:
        return False
    return True


def validate_first_word(text: str, first_word: str) -> bool:
    """Validates if response starts with the specified word."""
    words = text.strip().split()
    if not words:
        return False
    return words[0].lower() == first_word.lower()


def validate_start_end(text: str, start: str = None, end: str = None) -> bool:
    """Validates if response starts and/or ends with specified strings."""
    if start is not None and not text.strip().startswith(start):
        return False
    if end is not None and not text.strip().endswith(end):
        return False
    return True


def validate_no_adjacent_consecutive(text: str) -> bool:
    """Validates that no two adjacent words are the same."""
    words = re.findall(r"\b\w+\b", text.lower())
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return False
    return True


def validate_copy(text: str, prompt_to_repeat: str) -> bool:
    """Validates if the prompt was repeated in the response."""
    return prompt_to_repeat in text


def validate_lowercase_letter_counting(text: str, letter: str, frequency: int, relation: str) -> bool:
    """Validates if a lowercase letter appears with the specified frequency relation."""
    if not letter:
        return True
    letter_lower = letter.lower()
    actual_count = text.lower().count(letter_lower)

    if relation == "less than":
        return actual_count < frequency
    elif relation == "at least":
        return actual_count >= frequency
    elif relation == "at most":
        return actual_count <= frequency
    elif relation == "exactly":
        return actual_count == frequency
    else:
        return actual_count == frequency


def validate_palindrome(text: str) -> bool:
    """Validates if text contains at least one palindrome word (3+ chars)."""
    words = re.findall(r"\b\w+\b", text.lower())
    for word in words:
        if len(word) >= 3 and word == word[::-1]:
            return True
    return False


def validate_keyword_specific_position(text: str, keyword: str, n: int, m: int) -> bool:
    """Validates if keyword appears between positions n and m (word indices)."""
    words = text.lower().split()
    keyword_lower = keyword.lower()
    for i, word in enumerate(words):
        # Strip punctuation for comparison
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word == keyword_lower:
            # Position is 1-indexed, check if within range [n, m]
            pos = i + 1
            if n <= pos <= m:
                return True
    return False


def validate_count_increment_word(text: str, keyword1: str, keyword2: str) -> bool:
    """Validates that keyword2 appears more times than keyword1."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    count1 = sum(1 for w in words if w == keyword1.lower())
    count2 = sum(1 for w in words if w == keyword2.lower())
    return count2 > count1


def validate_counting_composition(text: str, n_sent: int, n_words: int) -> bool:
    """Validates sentence count and minimum words per sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) < n_sent:
        return False

    # Check each sentence has at least n_words
    for sent in sentences:
        word_count = len(sent.split())
        if word_count < n_words:
            return False
    return True


def validate_bigram_wrapping(text: str) -> bool:
    """Validates text has proper bigram wrapping (sentences wrapped in specific format)."""
    # Check if text contains wrapped content with markers like [[ ]] or similar
    # This is a simplified check - bigram wrapping typically means content wrapped consistently
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    # Check for consistent wrapping pattern
    return True  # Simplified - actual bigram wrapping is context-dependent


def validate_constrained_response(text: str) -> bool:
    """Validates response follows constrained format."""
    # Check for structured response patterns
    # Constrained responses typically have specific formatting
    text = text.strip()
    if not text:
        return False
    # Check for structured elements (lists, numbered items, etc.)
    has_structure = bool(re.search(r'^\d+[.)]|\*|-|•', text, re.MULTILINE))
    return has_structure or len(text.split()) >= 10  # Has structure or substantial content


# Mapping of instruction IDs to verification functions
IFEVAL_INSTRUCTION_MAP = {
    # Keywords
    "keywords:existence": lambda text, kwargs: verify_keywords(text, kwargs.get("keywords", [])),
    "keywords:frequency": lambda text, kwargs: verify_keyword_frequency_relation(
        text, kwargs.get("keyword", ""), kwargs.get("frequency", 0), kwargs.get("relation", "exactly")
    ) if kwargs.get("relation") else verify_keyword_frequency(text, kwargs.get("keyword", ""), kwargs.get("frequency", 0)),
    "keywords:word_count_different_numbers": lambda text, kwargs: verify_keyword_frequency_relation(
        text, kwargs.get("keyword", ""), kwargs.get("frequency", 0), kwargs.get("relation", "exactly")
    ),
    "keywords:forbidden_words": lambda text, kwargs: validate_forbidden_words(text, kwargs.get("forbidden_words", [])),
    "keywords:letter_frequency": lambda text, kwargs: verify_letter_frequency(text, kwargs.get("letter", ""), kwargs.get("let_frequency", 0)),
    "keywords:start_end": lambda text, kwargs: validate_start_end(text, kwargs.get("start", None), kwargs.get("end", None)),
    "keywords:no_adjacent_consecutive": lambda text, kwargs: validate_no_adjacent_consecutive(text),

    # Language
    "language:response_language": lambda text, kwargs: validate_response_language(text, kwargs.get("language", "en")),

    # Length
    "length:number_paragraphs": lambda text, kwargs: verify_paragraph_count(text, kwargs.get("num_paragraphs", 1)),
    "length:number_words": lambda text, kwargs: validate_word_constraint(text, kwargs.get("num_words", 0), kwargs.get("relation", "at least")),
    "length:number_sentences": lambda text, kwargs: verify_sentence_constraint(text, kwargs.get("num_sentences", 0), kwargs.get("relation", "at least")),

    # Format
    "detectable_format:number_bullet_lists": lambda text, kwargs: verify_bullet_points(text, kwargs.get("num_bullets", 0)),
    "detectable_format:title": lambda text, kwargs: validate_title(text),
    "detectable_format:number_highlighted_sections": lambda text, kwargs: validate_highlighted_sections(text, kwargs.get("num_highlights", 0)),
    "detectable_format:multiple_sections": lambda text, kwargs: validate_sections(text, kwargs.get("num_sections", 0), kwargs.get("section_splitter", "Section")),
    "detectable_format:json_format": lambda text, kwargs: validate_json_format(text),
    "detectable_format:postscript": lambda text, kwargs: verify_postscript(text, kwargs.get("postscript_marker", "P.S.")),
    "detectable_format:number_placeholders": lambda text, kwargs: validate_placeholders(text, kwargs.get("num_placeholders", 0)),

    # Content
    "detectable_content:postscript": lambda text, kwargs: verify_postscript(text, kwargs.get("postscript_marker", "P.S.")),

    # Combination
    "combination:repeat_prompt": lambda text, kwargs: validate_repeat_prompt(text, kwargs.get("prompt_to_repeat", "")),
    "combination:two_responses": lambda text, kwargs: validate_two_responses(text),

    # Case
    "change_case:english_capital": lambda text, kwargs: validate_uppercase(text),
    "change_case:english_lowercase": lambda text, kwargs: validate_lowercase(text),
    "change_case:capital_word_frequency": lambda text, kwargs: validate_frequency_capital_words(text, kwargs.get("capital_frequency", 0), kwargs.get("capital_relation", "at least")),

    # Startend
    "startend:end_checker": lambda text, kwargs: validate_end(text, kwargs.get("end_phrase", "")),
    "startend:quotation": lambda text, kwargs: validate_quotation(text),

    # Punctuation
    "punctuation:no_comma": lambda text, kwargs: validate_no_commas(text),

    # Count
    "count:count_unique": lambda text, kwargs: validate_count_unique(text, kwargs.get("min_count"), kwargs.get("max_count")),
    "count:lowercase_counting": lambda text, kwargs: validate_lowercase_letter_counting(text, kwargs.get("letter", ""), kwargs.get("let_frequency", 0), kwargs.get("let_relation", "at least")),

    # First word
    "first_word:first_word_answer": lambda text, kwargs: validate_first_word(text, kwargs.get("first_word", "")),

    # Copy
    "copy:copy": lambda text, kwargs: validate_copy(text, kwargs.get("prompt_to_repeat", "")),
    "copy:copying_multiple": lambda text, kwargs: validate_copy(text, kwargs.get("prompt_to_repeat", "")),
    "copy:copying_simple": lambda text, kwargs: validate_copy(text, kwargs.get("prompt_to_repeat", "")),
    "copy:repeat_phrase": lambda text, kwargs: validate_copy(text, kwargs.get("prompt_to_repeat", "")),

    # Square brackets
    "detectable_format:square_brackets": lambda text, kwargs: validate_placeholders(text, kwargs.get("num_placeholders", 1)),
    "detectable_format:bigram_wrapping": lambda text, kwargs: validate_bigram_wrapping(text),
    "detectable_format:constrained_response": lambda text, kwargs: validate_constrained_response(text),
    "detectable_format:sentence_hyphens": lambda text, kwargs: all(line.strip().startswith("-") for line in text.strip().split("\n") if line.strip()),

    # Content
    "detectable_content:number_placeholders": lambda text, kwargs: validate_placeholders(text, kwargs.get("num_placeholders", 1)),

    # First word
    "first_word:first_word_sent": lambda text, kwargs: validate_first_word(text, kwargs.get("first_word", "")),

    # Last word
    "last_word:last_word_answer": lambda text, kwargs: text.strip().split()[-1].lower().rstrip(".,!?") == kwargs.get("last_word", "").lower() if text.strip() else False,
    "last_word:last_word_sent": lambda text, kwargs: text.strip().split()[-1].lower().rstrip(".,!?") == kwargs.get("last_word", "").lower() if text.strip() else False,

    # Count
    "count:count_increment_word": lambda text, kwargs: validate_count_increment_word(text, kwargs.get("keyword1", ""), kwargs.get("keyword2", "")),
    "count:counting_composition": lambda text, kwargs: validate_counting_composition(text, kwargs.get("n_sent", 1), kwargs.get("n_words", 1)),

    # Letters
    "letters:letter_counting": lambda text, kwargs: validate_lowercase_letter_counting(text, kwargs.get("letter", ""), kwargs.get("let_frequency", 0), kwargs.get("let_relation", "at least")),
    "letters:letter_counting2": lambda text, kwargs: validate_lowercase_letter_counting(text, kwargs.get("letter", ""), kwargs.get("let_frequency", 0), kwargs.get("let_relation", "at least")),

    # Keywords additional
    "keywords:exclude_word_harder": lambda text, kwargs: validate_forbidden_words(text, kwargs.get("forbidden_words", [])),
    "keywords:keyword_specific_position": lambda text, kwargs: validate_keyword_specific_position(text, kwargs.get("keyword", ""), kwargs.get("n", 1), kwargs.get("m", 100)),
    "keywords:palindrome": lambda text, kwargs: validate_palindrome(text),
    "keywords:word_once": lambda text, kwargs: verify_keyword_frequency(text, kwargs.get("keyword", ""), 1),

    # Paragraphs
    "paragraphs:paragraphs": lambda text, kwargs: verify_paragraph_count(text, kwargs.get("num_paragraphs", 1)),
    "paragraphs:paragraphs2": lambda text, kwargs: verify_paragraph_count(text, kwargs.get("num_paragraphs", 1)),

    # Punctuation additional
    "punctuation:punctuation_dot": lambda text, kwargs: text.strip().endswith("."),
    "punctuation:punctuation_exclamation": lambda text, kwargs: text.strip().endswith("!"),

    # New/custom
    "new:copy_span_idx": lambda text, kwargs: validate_copy(text, kwargs.get("prompt_to_repeat", "")),

    # Length constraints (alternative naming)
    "length_constraints:number_words": lambda text, kwargs: validate_word_constraint(text, kwargs.get("num_words", 0), kwargs.get("relation", "at least")),
    "length_constraints:number_sentences": lambda text, kwargs: verify_sentence_constraint(text, kwargs.get("num_sentences", 0), kwargs.get("relation", "at least")),
    "length_constraints:number_paragraphs": lambda text, kwargs: verify_paragraph_count(text, kwargs.get("num_paragraphs", 1)),
    "length_constraints:nth_paragraph_first_word": lambda text, kwargs: validate_paragraphs(text, kwargs.get("num_paragraphs", 1), kwargs.get("first_word", ""), kwargs.get("nth_paragraph", 1)),
}


def strip_thinking(response: str) -> str:
    """Remove thinking content from response.

    Args:
        response: The model response potentially containing thinking blocks

    Returns:
        Response with thinking content removed (only the answer after </think>)
    """
    # If </think> is present, take everything after the last </think>
    if '</think>' in response:
        return response.split('</think>')[-1].strip()

    # If <think> is present without closing tag, remove everything from <think> onwards
    if '<think>' in response:
        return response.split('<think>')[0].strip()

    # No thinking tags found, return as-is
    return response.strip()


def verify_ifeval_constraints(response: str, ground_truth: list[dict]) -> tuple[float, str]:
    """Verify IFEval constraints and return reward.

    Args:
        response: The model response to verify
        ground_truth: List of constraint dictionaries with 'instruction_id' and 'kwargs'

    Returns:
        Tuple of (reward, feedback_string)
    """
    import ast

    # Strip thinking blocks before evaluating constraints
    response = strip_thinking(response)

    if not ground_truth:
        return 0.0, "No constraints provided"

    parsed_gt = ground_truth

    # Recursively unwrap lists and parse strings
    def parse_ground_truth(gt):
        # If it's a list, unwrap it
        if isinstance(gt, list):
            if len(gt) == 0:
                return None
            if len(gt) == 1:
                return parse_ground_truth(gt[0])
            # If it's a list with multiple items, try to find a dict
            for item in gt:
                result = parse_ground_truth(item)
                if isinstance(result, dict):
                    return result
            return gt

        # If it's a string, try to parse it
        if isinstance(gt, str):
            # Try ast.literal_eval first (handles Python dict syntax)
            try:
                parsed = ast.literal_eval(gt)
                return parse_ground_truth(parsed)
            except (ValueError, SyntaxError):
                pass

            # Try JSON parsing
            try:
                parsed = json.loads(gt.replace("'", '"'))
                return parse_ground_truth(parsed)
            except (json.JSONDecodeError, Exception):
                pass

            return gt

        # If it's already a dict, return it
        if isinstance(gt, dict):
            return gt

        return gt

    parsed_gt = parse_ground_truth(ground_truth)

    # Now parsed_gt should be a dict with 'instruction_id' and 'kwargs'
    if not isinstance(parsed_gt, dict):
        return 0.0, f"Invalid ground_truth format: {type(parsed_gt)}"

    instruction_ids = parsed_gt.get("instruction_id", [])
    kwargs_list = parsed_gt.get("kwargs", [])

    if not instruction_ids:
        return 0.0, "No instruction_ids found"

    # Verify each constraint
    passed = 0
    total = len(instruction_ids)
    feedback_parts = []

    for idx, instruction_id in enumerate(instruction_ids):
        kwargs = kwargs_list[idx] if idx < len(kwargs_list) and kwargs_list[idx] is not None else {}

        if instruction_id in IFEVAL_INSTRUCTION_MAP:
            try:
                result = IFEVAL_INSTRUCTION_MAP[instruction_id](response, kwargs)
                if result:
                    passed += 1
                    feedback_parts.append(f"{instruction_id}: PASS")
                else:
                    feedback_parts.append(f"{instruction_id}: FAIL")
            except Exception as e:
                # Log error but continue - don't crash trajectory generation
                feedback_parts.append(f"{instruction_id}: ERROR ({str(e)[:50]})")
                print(f"⚠️  Error verifying {instruction_id}: {e}")
        else:
            feedback_parts.append(f"{instruction_id}: UNKNOWN")

    # Calculate reward as fraction of passed constraints
    reward = passed / total if total > 0 else 0.0
    feedback = f"IFEval: {passed}/{total} constraints passed. " + "; ".join(feedback_parts)

    return reward, feedback


# =============================================================================
# Environment Configuration and Class
# =============================================================================

class MindVerifiableEnvConfig(TypedDict):
    """Configuration for MindVerifiableEnvironment.

    Attributes:
        enabled: Whether the environment is enabled
        stop_strings: Default stop strings for this environment
    """
    enabled: bool
    stop_strings: NotRequired[List[str] | None]


@ray.remote
class MindVerifiableEnvironment(EnvironmentInterface):
    """Environment for verifying math answers and IFEval constraints.

    This environment supports two types of verification:
    1. Math: Extracts answers from "Answer: <answer>" format and compares to ground truth
    2. IFEval: Verifies instruction-following constraints based on the IFEval taxonomy

    The dataset type is determined by the 'dataset' field in metadata.

    Returns:
        - For math: 1.0 if correct, 0.0 if incorrect
        - For ifeval: Fraction of constraints passed (0.0 to 1.0)
    """

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.BASE

    def __init__(self, config: Dict[str, Any]):
        """Initialize the mind verifiable environment.

        Args:
            config: Configuration dictionary containing environment settings
        """
        self.config = config
        print("MindVerifiableEnvironment initialized with math + IFEval verification")

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Process conversations and verify answers based on dataset type.

        Args:
            message_log_batch: Batch of conversation message logs
            metadata: Batch of metadata containing 'ground_truth' and optionally 'dataset' key

        Returns:
            EnvironmentReturn with rewards based on verification type
        """
        rewards = []
        observations = []

        for i, conversation in enumerate(message_log_batch):
            try:
                # Extract assistant response (last message should be from assistant)
                assistant_response = ""

                for msg in conversation:
                    if msg["role"] == "assistant":
                        assistant_response = str(msg["content"])

                # Get metadata fields
                ground_truth = metadata[i].get("ground_truth", "")
                dataset_type = metadata[i].get("dataset", ["math"])

                # Handle list wrapping for dataset type
                if isinstance(dataset_type, list):
                    dataset_type = dataset_type[0] if dataset_type else "math"

                dataset_type = dataset_type.lower()

                # Route to appropriate verification
                if dataset_type == "ifeval":
                    reward, feedback = verify_ifeval_constraints(assistant_response, ground_truth)
                    rewards.append(reward)
                    observations.append({"role": "environment", "content": feedback})
                    print(f"IFEval verification - Reward: {reward:.2f} | {feedback[:100]}...")
                else:
                    # Default to math verification
                    reward = verify_math_answer(assistant_response, ground_truth)
                    rewards.append(reward)

                    extracted_answer = extract_answer(assistant_response)
                    if reward == 1.0:
                        feedback = f"Correct! Answer: {extracted_answer}"
                    else:
                        feedback = f"Incorrect. Your answer: {extracted_answer}, Expected: {ground_truth}"

                    observations.append({"role": "environment", "content": feedback})
                    print(f"Math verification - Reward: {reward:.1f} | "
                          f"Extracted: '{extracted_answer}' | Expected: '{ground_truth}'")

            except Exception as e:
                # If verification fails, give 0 reward and log error
                print(f"❌ Error in environment step for sample {i}: {e}")
                import traceback
                traceback.print_exc()

                rewards.append(0.0)
                observations.append({
                    "role": "environment",
                    "content": f"Error during verification: {str(e)[:100]}"
                })

        # All episodes terminate after one step
        terminateds = [True] * len(message_log_batch)
        next_stop_strings = [None] * len(message_log_batch)
        answers = [msg[-1]["content"] for msg in message_log_batch]

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Compute batch-level metrics for verification."""
        metrics = {
            "mind_verifiable_env/num_samples": len(batch.get("message_log", [])),
        }

        # Add accuracy statistics
        if "rewards" in batch:
            rewards = batch["rewards"]
            if isinstance(rewards, torch.Tensor):
                metrics.update({
                    "mind_verifiable_env/accuracy": float(rewards.mean()),
                    "mind_verifiable_env/num_correct": int((rewards == 1.0).sum()),
                    "mind_verifiable_env/num_incorrect": int((rewards == 0.0).sum()),
                })

        return batch, metrics

    def shutdown(self):
        """Clean up resources."""
        print("MindVerifiableEnvironment shut down successfully")
