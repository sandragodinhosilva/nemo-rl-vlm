# LOCAL-ONLY (sgsilva) — never push to SWORD origin.
"""Tool-rollout bridge for GRPO over the VObs query_obs tool loop.

Scaffolded 2026-07-13 (report:
~/.claude/reports/post_training/2026-07-13_grpo_vobs_tool_scaffold.md). One GRPO
rollout is a multi-turn tool loop:

    user(video + tool-offered prompt)
      -> assistant(<tool_call>{"name":"query_obs", ...}</tool_call>)
      -> tool(obs answers)                      # injected by ThriveVLMEnvironment.step()
      -> ... (more rounds) ...
      -> assistant(final [MOVEMENT ANALYSIS]/[ERRORS]/[SCORES]/[FEEDBACK])   # scored

This module holds everything the environment's tool branch needs that is NOT
ray/torch: the obs bank backends, tool-call parsing/dispatch, per-rollout
penalty accounting, and the composite reward. It deliberately imports ONLY
stdlib + visual_obs_rep_rewards so the CPU reward canary can exercise it
without a GPU/ray stack.

Provenance: ObsTable / parse_tool_calls / dispatch semantics are a minimal
copy of /home/sgsilva/vlm-post-training/visual_obs/query_obs_tool_executor.py
and run_stage2_tool_loop_probe.py (state as of 2026-07-13). Copied, not
imported: the GRPO env runs inside nemo-rl Ray worker venvs where a cross-repo
sys.path import is fragile. If the source semantics change (question
normalization, dedup, batched-`questions` shape), re-sync BOTH.

THE BANK MODE IS REWARD-DESIGN-CRITICAL (report §2.0b): a `gt` bank makes
"query everything and transcribe" the optimal policy (trained vs an oracle,
deployed vs a ~52% obs model). `bank_mode` therefore defaults to a
deployment-distribution approximation and `gt` exists only for the canary arm.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo_rl.environments.visual_obs_rep_rewards import compute_rep_reward

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Dispatch statuses (structured — never string-match answer text for control flow)
STATUS_OK = "ok"
STATUS_MALFORMED = "malformed"
STATUS_OFFBANK = "offbank"
STATUS_NO_COVERAGE = "no_coverage"
STATUS_DUPLICATE = "duplicate"
STATUS_EMPTY = "empty"
STATUS_UNKNOWN_TOOL = "unknown_tool"


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


class ObsTable:
    """Minimal copy of query_obs_tool_executor.ObsTable (see module docstring).

    Answers query_obs(folder_name, repetition_id, question) against a cached
    obs JSON of shape {exercise_id: {rep_key: {parsed_answers: [...]}}}.
    Question identity = normalized question TEXT (never positional index).
    """

    def __init__(self, path: str):
        self.path = Path(path)
        with open(self.path) as f:
            raw = json.load(f)
        self._index: Dict[Tuple[str, str], List[Dict]] = {}
        for _exercise_id, reps in raw.items():
            if not isinstance(reps, dict):
                continue
            for _rep_key, entry in reps.items():
                if not isinstance(entry, dict):
                    continue
                folder = entry.get("folder_name", "")
                rep_id = entry.get("repetition_id", "")
                answers = entry.get("parsed_answers") or []
                if folder and rep_id and answers:
                    self._index[(folder, rep_id)] = answers

    def has_coverage(self, folder_name: str, repetition_id: str) -> bool:
        return (folder_name, repetition_id) in self._index

    def query_obs(
        self, folder_name: str, repetition_id: str, question: str
    ) -> Tuple[str, str]:
        """Returns (answer_text, status). Statuses: ok | no_coverage | offbank."""
        answers = self._index.get((folder_name, repetition_id))
        if answers is None:
            return (
                f"[query_obs error] no cached observations found for "
                f"{folder_name}/{repetition_id}.",
                STATUS_NO_COVERAGE,
            )
        target = _normalize_question(question)
        for item in answers:
            if _normalize_question(item.get("question", "")) == target:
                return (item.get("answer", "") or "(no answer recorded)", STATUS_OK)
        available = "; ".join(a.get("question", "") for a in answers)
        return (
            f"[query_obs error] question not found in bank for "
            f"{folder_name}/{repetition_id}: {question!r}. "
            f"Available questions: {available}",
            STATUS_OFFBANK,
        )


class CorruptedObsTable(ObsTable):
    """GT bank + deterministic per-(rep, question) corruption at `corruption_rate`.

    Approximates the deployment obs distribution (the 4B stage-1 at ~52.4%
    micro-F1) without a GPU pass. Determinism contract (report §2.0b): the
    corrupt/not decision AND the substituted option depend only on
    (base_seed, folder_name, repetition_id, question) — every rollout in a
    GRPO group sees identical bank answers, so within-group reward variance
    reflects the policy, not env randomness.

    Corruption = a different valid option for that question (uniform over the
    non-GT options). Upgrading to the 4B's measured per-question confusion
    profile is a launch-time refinement (report §7), not a scaffold blocker.
    """

    def __init__(self, path: str, corruption_rate: float = 0.48, base_seed: int = 0):
        super().__init__(path)
        self.corruption_rate = float(corruption_rate)
        self.base_seed = int(base_seed)

    def _u01(self, *keys: str) -> float:
        h = hashlib.sha256(("|".join([str(self.base_seed), *keys])).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    def query_obs(
        self, folder_name: str, repetition_id: str, question: str
    ) -> Tuple[str, str]:
        answer, status = super().query_obs(folder_name, repetition_id, question)
        if status != STATUS_OK:
            return answer, status
        norm_q = _normalize_question(question)
        if self._u01(folder_name, repetition_id, norm_q, "flip") >= self.corruption_rate:
            return answer, status
        # find this question's options to pick a plausible-wrong one
        options: List[str] = []
        for item in self._index.get((folder_name, repetition_id), []):
            if _normalize_question(item.get("question", "")) == norm_q:
                options = [o for o in (item.get("options") or []) if o and o != answer]
                break
        if not options:
            return answer, status  # nothing to corrupt with — serve GT (counted honest)
        pick = int(self._u01(folder_name, repetition_id, norm_q, "pick") * len(options))
        return options[min(pick, len(options) - 1)], status


def make_obs_backend(cfg: Dict[str, Any]):
    """Build the rollout obs backend from the env tool_rollout config block."""
    mode = cfg.get("bank_mode", "corrupted")
    path = cfg["obs_bank_path"]
    if mode == "gt":
        # CANARY-ONLY (report §2.0b): training on a GT bank optimizes
        # trust-and-transcribe against an oracle. Loud, not silent.
        print(
            "⚠️  tool_rollout.bank_mode=gt — GT obs bank is the CANARY arm only; "
            "do NOT train the real run on it (report §2.0b).",
            flush=True,
        )
        return ObsTable(path)
    if mode == "corrupted":
        return CorruptedObsTable(
            path,
            corruption_rate=cfg.get("corruption_rate", 0.48),
            base_seed=cfg.get("corruption_seed", 0),
        )
    if mode == "model_obs":
        # A pregenerated deployment-model obs JSON in the same shape — just a
        # different file; the plain lookup table reads it.
        return ObsTable(cfg.get("model_obs_bank_path") or path)
    raise ValueError(f"Unknown tool_rollout.bank_mode: {mode!r}")


def parse_tool_calls(assistant_text: str) -> List[Dict]:
    """Copy of the probe's tolerant parser — malformed JSON reported, not dropped."""
    calls: List[Dict] = []
    for m in TOOL_CALL_RE.finditer(assistant_text or ""):
        raw = m.group(1).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "name" in parsed:
                parsed.setdefault("arguments", {})
                calls.append(parsed)
            else:
                calls.append(
                    {"name": "__parse_error__", "raw": raw, "error": "not a dict with 'name'"}
                )
        except json.JSONDecodeError as e:
            calls.append({"name": "__parse_error__", "raw": raw, "error": str(e)})
    return calls


def dispatch_tool_calls(
    calls: List[Dict],
    backend: ObsTable,
    folder_name: str,
    repetition_id: str,
    asked: List[str],
) -> Tuple[str, Dict[str, int]]:
    """Execute every call of one assistant turn; returns (tool_message_text, counters).

    Handles both call shapes (`question` str / `questions` list) like the
    probe's dispatch_tool. `asked` is the across-rounds dedup list (normalized
    questions), carried in env metadata; mutated in place.
    Counters: n_malformed, n_offbank, n_questions, n_duplicates.
    """
    counters = {"n_malformed": 0, "n_offbank": 0, "n_questions": 0, "n_duplicates": 0}
    lines: List[str] = []

    def _one(question: str) -> str:
        question = (question or "").strip()
        if not question:
            counters["n_malformed"] += 1
            return "[Tool error] query_obs requires a non-empty question."
        norm = _normalize_question(question)
        if norm in asked:
            counters["n_duplicates"] += 1
            return (
                f"[Tool note] already asked {question!r} earlier this rep — "
                "reusing the same answer above, no re-query."
            )
        asked.append(norm)
        counters["n_questions"] += 1
        answer, status = backend.query_obs(folder_name, repetition_id, question)
        if status == STATUS_OFFBANK:
            counters["n_offbank"] += 1
        return answer

    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}
        if name == "__parse_error__":
            counters["n_malformed"] += 1
            lines.append(
                f"[Tool error] malformed tool_call JSON: {call.get('error')!r}. "
                f"Raw: {call.get('raw')}"
            )
            continue
        if name != "query_obs":
            counters["n_malformed"] += 1
            lines.append(f"[Tool error] unknown tool {name!r}. Available: query_obs.")
            continue
        if "questions" in args:
            questions = args.get("questions") or []
            if not isinstance(questions, list) or not questions:
                counters["n_malformed"] += 1
                lines.append("[Tool error] query_obs requires a non-empty 'questions' list.")
                continue
            for q in questions:
                lines.append(f"Q: {q}\nA: {_one(q)}")
        else:
            q = args.get("question") or ""
            lines.append(f"Q: {q}\nA: {_one(q)}")

    return "\n\n".join(lines), counters


# ---------------------------------------------------------------------------
# Composite reward (report §2.2) — multiplicative bounded penalties.
# ---------------------------------------------------------------------------

PENALTY_TOTAL_CAP = 0.15


def compute_tool_penalty_fraction(meta: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """P in reward = answer * (1 - P); P <= 0.15 so aux terms can only flip
    relative near-ties (report §2.2) and a parseable answer never collides
    with the empty-answer 0.0."""
    p_malformed = min(0.10, cfg.get("malformed_call_penalty", 0.05) * meta.get("tool_n_malformed", 0))
    p_offbank = min(0.09, cfg.get("offbank_question_penalty", 0.03) * meta.get("tool_n_offbank", 0))
    p_rounds = min(0.06, cfg.get("round_penalty", 0.02) * max(0, meta.get("tool_rounds", 0) - 1))
    qp = cfg.get("question_penalty", 0.0)  # default OFF (report §2.1-B)
    p_questions = min(0.05, qp * max(0, meta.get("tool_n_questions", 0) - 3)) if qp else 0.0
    return min(PENALTY_TOTAL_CAP, p_malformed + p_offbank + p_rounds + p_questions)


def compute_tool_final_reward(
    final_answer_text: str,
    ground_truth: str,
    meta: Dict[str, Any],
    tool_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """Score the FINAL assistant turn (never a concatenation of turns) with the
    f1-anchored rep composite, then apply the multiplicative tool penalty."""
    rep_config = {
        "reward_mode": tool_cfg.get("reward_mode", "detection_correctness_severity"),
        "f1_weight": tool_cfg.get("f1_weight", 0.40),
        "detection_weight": tool_cfg.get("detection_weight", 0.15),
        "severity_weight": tool_cfg.get("severity_weight", 0.25),
        "correctness_weight": tool_cfg.get("correctness_weight", 0.10),
        "format_weight": tool_cfg.get("format_weight", 0.10),
        "error_weight": tool_cfg.get("error_weight", 1.0),
        "non_error_weight": tool_cfg.get("non_error_weight", 0.5),
    }
    answer_reward, details = compute_rep_reward(final_answer_text, ground_truth, rep_config)
    # Own display task_type so the grpo-dashboard plots the tool-loop fields
    # (f1/penalty/rounds/questions) — "repetition" is a KNOWN dashboard type
    # with a fixed component list that would hide them. Reward ROUTING is
    # unaffected (the env keys on metadata, not on this display field).
    details["task_type"] = "repetition_tool"
    penalty = compute_tool_penalty_fraction(meta, tool_cfg)
    reward = answer_reward * (1.0 - penalty)
    details["tool_penalty_fraction"] = penalty
    details["answer_reward_prepenalty"] = answer_reward
    for k in ("tool_rounds", "tool_n_malformed", "tool_n_offbank",
              "tool_n_questions", "tool_n_duplicates"):
        details[k] = meta.get(k, 0)
    details["final_reward"] = reward
    return reward, details
