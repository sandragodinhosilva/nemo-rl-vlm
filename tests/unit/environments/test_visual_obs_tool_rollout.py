# LOCAL-ONLY (sgsilva 2026-07-13) — never push to SWORD origin.
"""Unit tests for the VObs tool-loop GRPO bridge (persistent suite).

Companion to the one-shot CPU canary
(/mnt/data/sgsilva/benchmarks/scripts/grpo_reward_canary_vobs_tool.py — that one
runs against the REAL 1806 train bank as a pre-launch gate); this suite is
hermetic (synthetic bank via tmp_path) and encodes the reward INVARIANTS +
final-turn isolation + executor semantics + single-shot regression pins, per
Sandra's 2026-07-13 test-plan review. Report:
~/.claude/reports/post_training/2026-07-13_grpo_vobs_tool_scaffold.md.

NOT covered here (GPU smoke gates, report §7): loss/KL mask golden-array on a
real batch (grpo.py:1717 role rule + ref-logprob context), turn-2 vLLM-prompt
vs message-log token diff, GT-vs-noisy bank behavioral canary arms.

Run:
  PYTHONPATH=/home/sgsilva/nemo-rl-vlm /home/sgsilva/vlm-post-training-home-venv/bin/python \
    -m pytest tests/unit/environments/test_visual_obs_tool_rollout.py -q
"""

import json
import random

import pytest

from nemo_rl.environments.visual_obs_rep_rewards import (
    compute_error_f1,
    compute_rep_reward,
)
from nemo_rl.environments.visual_obs_tool_rollout import (
    PENALTY_TOTAL_CAP,
    CorruptedObsTable,
    ObsTable,
    compute_tool_final_reward,
    compute_tool_penalty_fraction,
    dispatch_tool_calls,
    parse_tool_calls,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GT = """[MOVEMENT ANALYSIS]
Reference analysis.
[ERRORS]
Knee Valgus: 3
Trunk Lean Forward: 1
Limited Squat Depth: 4
Heels Lifting: 1
[SCORES]
Effectiveness: 2
Injury Risk: 2
[FEEDBACK]
Keep knees tracking over toes."""

PERFECT = GT
HALF_WRONG = GT.replace("Knee Valgus: 3", "Knee Valgus: 1").replace(
    "Limited Squat Depth: 4", "Limited Squat Depth: 2"
)
EMPTY = ""

TOOL_CFG = {
    "enabled": True,
    "f1_weight": 0.40, "detection_weight": 0.15, "severity_weight": 0.25,
    "correctness_weight": 0.10, "format_weight": 0.10,
    "malformed_call_penalty": 0.05, "offbank_question_penalty": 0.03,
    "round_penalty": 0.02, "question_penalty": 0.0,
    "max_tool_rounds": 4,
}


@pytest.fixture()
def bank(tmp_path):
    """Tiny synthetic obs bank in the real JSON shape (hermetic)."""
    data = {
        "10001": {
            "rep_a": {
                "folder_name": "sess_a",
                "repetition_id": "repetition_1",
                "parsed_answers": [
                    {"question": "Is the trunk leaning?",
                     "options": ["no lean", "slight lean", "strong lean"],
                     "answer": "slight lean"},
                    {"question": "Which leg lifts?",
                     "options": ["left", "right"],
                     "answer": "left"},
                ],
            }
        }
    }
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(data))
    return ObsTable(str(p))


def _tool_call(payload: str) -> str:
    return f"<tool_call>{payload}</tool_call>"


def _rand_meta(rng):
    return {
        "tool_n_malformed": rng.randint(0, 6),
        "tool_n_offbank": rng.randint(0, 6),
        "tool_rounds": rng.randint(0, 8),
        "tool_n_questions": rng.randint(0, 20),
    }


def _rand_answer(rng):
    """A parseable answer with random severities (some fields dropped)."""
    lines = ["[MOVEMENT ANALYSIS]", "x", "[ERRORS]"]
    for name in ("Knee Valgus", "Trunk Lean Forward", "Limited Squat Depth", "Heels Lifting"):
        if rng.random() > 0.2:
            lines.append(f"{name}: {rng.randint(1, 6)}")
    lines += ["[SCORES]", f"Effectiveness: {rng.randint(1, 3)}",
              f"Injury Risk: {rng.randint(1, 3)}", "[FEEDBACK]", "y"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Reward invariants (executable versions of the report's guarantees)
# ---------------------------------------------------------------------------

class TestRewardInvariants:
    def test_reward_in_unit_interval_fuzz(self):
        rng = random.Random(0)
        for _ in range(300):
            r, _ = compute_tool_final_reward(_rand_answer(rng), GT, _rand_meta(rng), TOOL_CFG)
            assert 0.0 <= r <= 1.0

    def test_penalty_total_bound_fuzz(self):
        rng = random.Random(1)
        for _ in range(300):
            assert 0.0 <= compute_tool_penalty_fraction(_rand_meta(rng), TOOL_CFG) <= PENALTY_TOTAL_CAP
        assert PENALTY_TOTAL_CAP == 0.15  # a cap edit must be a conscious decision

    def test_anchor_messy_perfect_beats_clean_half_wrong(self):
        # §2.2 guarantee: penalties flip ranking only at relative near-ties
        # (<15%); perfect-vs-half gap is ~30%, so worst-path perfect must win.
        worst = {"tool_n_malformed": 9, "tool_n_offbank": 9, "tool_rounds": 9,
                 "tool_n_questions": 99}
        r_messy_perfect, _ = compute_tool_final_reward(PERFECT, GT, worst, TOOL_CFG)
        r_clean_half, _ = compute_tool_final_reward(HALF_WRONG, GT, {}, TOOL_CFG)
        assert r_messy_perfect > r_clean_half

    def test_same_path_ranking_preserved(self):
        # Multiplicative penalty: identical tool path ⇒ answer ordering preserved.
        rng = random.Random(2)
        for _ in range(100):
            meta = _rand_meta(rng)
            a, b = _rand_answer(rng), _rand_answer(rng)
            ra, _ = compute_tool_final_reward(a, GT, dict(meta), TOOL_CFG)
            rb, _ = compute_tool_final_reward(b, GT, dict(meta), TOOL_CFG)
            base_a, _ = compute_rep_reward(a, GT, TOOL_CFG)
            base_b, _ = compute_rep_reward(b, GT, TOOL_CFG)
            if base_a > base_b:
                assert ra >= rb

    def test_empty_strictly_below_any_parseable_fuzz(self):
        # §C guarantee (the old subtractive clamp broke this; multiplicative fixes it).
        rng = random.Random(3)
        worst = {"tool_n_malformed": 9, "tool_n_offbank": 9, "tool_rounds": 9}
        r_empty, _ = compute_tool_final_reward(EMPTY, GT, {}, TOOL_CFG)
        assert r_empty == 0.0
        for _ in range(100):
            ans = _rand_answer(rng)
            base, _ = compute_rep_reward(ans, GT, TOOL_CFG)
            if base > 0:
                r, _ = compute_tool_final_reward(ans, GT, dict(worst), TOOL_CFG)
                assert r > r_empty


# ---------------------------------------------------------------------------
# 2. Final-turn isolation (the highest-risk new code path)
# ---------------------------------------------------------------------------

def _bare_env(tool_cfg, backend):
    """Instantiate the undecorated env class WITHOUT __init__ (no ray workers)."""
    from nemo_rl.environments.visual_obs_environment import ThriveVLMEnvironment
    cls = getattr(ThriveVLMEnvironment, "__ray_actor_class__", None)
    if cls is None:
        cls = ThriveVLMEnvironment.__ray_metadata__.modified_class
    env = object.__new__(cls)
    env.tool_cfg = tool_cfg
    env.tool_backend = backend
    env._step_call_count = 0
    return env


def _meta(rounds=0, **kw):
    m = {"ground_truth": GT, "task_type": "repetition", "tool_rollout": True,
         "folder_name": "sess_a", "repetition_id": "repetition_1",
         "sample_id": "s", "tool_rounds": rounds}
    m.update(kw)
    return m


class TestFinalTurnIsolation:
    def test_multiturn_transcript_scores_final_turn_only(self, bank):
        env = _bare_env(TOOL_CFG, bank)
        decoy = ('<think>maybe [SCORES] Effectiveness: 1</think>\n'
                 + _tool_call('{"name": "query_obs", "arguments": {"question": "Is the trunk leaning?"}}'))
        conv = [
            {"role": "user", "content": "analyze the rep"},
            {"role": "assistant", "content": decoy},
            {"role": "tool", "content": "Q: Is the trunk leaning?\nA: slight lean"},
            {"role": "assistant", "content": PERFECT},
        ]
        meta = _meta(rounds=2)
        out = env._step_tool_rollout([conv], [meta])
        direct, _ = compute_tool_final_reward(PERFECT, GT, dict(meta), TOOL_CFG)
        assert out.rewards[0].item() == pytest.approx(direct)
        assert out.terminateds[0].item() == 1.0
        # golden: the decoy turn's "Effectiveness: 1" must not have leaked; round
        # penalty is per round BEYOND the 1st → rounds=2 gives P=0.02 exactly.
        assert out.rewards[0].item() == pytest.approx(1.0 * (1 - 0.02))

    def test_intermediate_turn_reward_zero_and_tool_obs(self, bank):
        env = _bare_env(TOOL_CFG, bank)
        conv = [
            {"role": "user", "content": "analyze"},
            {"role": "assistant", "content": _tool_call(
                '{"name": "query_obs", "arguments": {"questions": ["Is the trunk leaning?", "Which leg lifts?"]}}')},
        ]
        meta = _meta()
        out = env._step_tool_rollout([conv], [meta])
        assert out.rewards[0].item() == 0.0
        assert out.terminateds[0].item() == 0.0
        assert out.observations[0]["role"] == "tool"
        assert "slight lean" in out.observations[0]["content"]
        assert meta["tool_rounds"] == 1 and meta["tool_n_questions"] == 2

    def test_round_cap_forces_terminal_scoring(self, bank):
        env = _bare_env(TOOL_CFG, bank)
        conv = [
            {"role": "user", "content": "analyze"},
            {"role": "assistant", "content": _tool_call(
                '{"name": "query_obs", "arguments": {"question": "Which leg lifts?"}}')},
        ]
        meta = _meta(rounds=TOOL_CFG["max_tool_rounds"])
        out = env._step_tool_rollout([conv], [meta])
        assert out.terminateds[0].item() == 1.0
        # an unanswered tool-call turn parses as no answer → ~0 (implicit penalty)
        assert out.rewards[0].item() == 0.0

    def test_mixed_batch_raises(self, bank):
        env = _bare_env(TOOL_CFG, bank)
        conv = [{"role": "user", "content": "x"}, {"role": "assistant", "content": PERFECT}]
        import re as _re
        with pytest.raises(ValueError, match="mixes tool rows"):
            # exercise the routing guard in step() itself
            env.step([conv, conv], [_meta(), {"ground_truth": GT, "task_type": "repetition"}])

    def test_missing_bank_key_raises(self, bank):
        env = _bare_env(TOOL_CFG, bank)
        conv = [{"role": "user", "content": "x"}, {"role": "assistant", "content": PERFECT}]
        with pytest.raises(ValueError, match="folder_name/repetition_id"):
            env._step_tool_rollout([conv], [_meta(folder_name="")])


# ---------------------------------------------------------------------------
# 4. Executor semantics (back the shaping penalties)
# ---------------------------------------------------------------------------

class TestExecutorSemantics:
    def test_offbank_structured_status_maps_to_penalty(self, bank):
        calls = parse_tool_calls(_tool_call(
            '{"name": "query_obs", "arguments": {"question": "Is the moon full?"}}'))
        _, counters = dispatch_tool_calls(calls, bank, "sess_a", "repetition_1", [])
        assert counters["n_offbank"] == 1
        p = compute_tool_penalty_fraction({"tool_n_offbank": 1}, TOOL_CFG)
        assert p == pytest.approx(0.03)

    def test_duplicate_does_not_requery_but_the_round_still_costs(self, bank):
        # Resolves the open question: dedup consumes NO question budget, but a
        # dup-only assistant TURN still increments tool_rounds in the env (the
        # env counts turns, not questions) → the round penalty applies. So
        # "repeats waste a round" is a fact, verified end-to-end here.
        asked = []
        calls = parse_tool_calls(_tool_call(
            '{"name": "query_obs", "arguments": {"question": "Which leg lifts?"}}'))
        _, c1 = dispatch_tool_calls(calls, bank, "sess_a", "repetition_1", asked)
        _, c2 = dispatch_tool_calls(calls, bank, "sess_a", "repetition_1", asked)
        assert c1["n_questions"] == 1 and c2["n_questions"] == 0 and c2["n_duplicates"] == 1

        env = _bare_env(TOOL_CFG, bank)
        conv = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": _tool_call(
                    '{"name": "query_obs", "arguments": {"question": "Which leg lifts?"}}')}]
        meta = _meta(rounds=1, tool_asked=["which leg lifts?"])
        env._step_tool_rollout([conv], [meta])
        assert meta["tool_rounds"] == 2  # the dup-only round still counted

    def test_malformed_penalty_caps_at_0_10(self, bank):
        text = _tool_call('{"name": broken}') * 3
        calls = parse_tool_calls(text)
        _, counters = dispatch_tool_calls(calls, bank, "sess_a", "repetition_1", [])
        assert counters["n_malformed"] == 3
        p = compute_tool_penalty_fraction({"tool_n_malformed": 3}, TOOL_CFG)
        assert p == pytest.approx(0.10)  # 3×0.05 capped

    def test_braceless_tool_call_is_not_a_call(self):
        # Probe-inherited regex edge (documented, report §6): brace-less text
        # between tags parses as ZERO calls → the turn scores as a final answer.
        assert parse_tool_calls("<tool_call>not json at all</tool_call>") == []

    def test_corrupted_bank_deterministic_and_never_gt(self, tmp_path, bank):
        cb1 = CorruptedObsTable(str(bank.path), corruption_rate=1.0, base_seed=7)
        cb2 = CorruptedObsTable(str(bank.path), corruption_rate=1.0, base_seed=7)
        for q, gt_a in [("Is the trunk leaning?", "slight lean"), ("Which leg lifts?", "left")]:
            a1, s1 = cb1.query_obs("sess_a", "repetition_1", q)
            a2, _ = cb2.query_obs("sess_a", "repetition_1", q)
            assert a1 == a2 and s1 == "ok" and a1 != gt_a


# ---------------------------------------------------------------------------
# 5. Regression pins — the new math must not move existing semantics
# ---------------------------------------------------------------------------

class TestSingleShotRegression:
    def test_default_config_reward_unchanged_golden(self):
        # Pre-change compute_rep_reward defaults (no f1_weight): pin exact values.
        default_cfg = {"reward_mode": "detection_correctness_severity"}
        r_perfect, _ = compute_rep_reward(PERFECT, GT, default_cfg)
        r_half, _ = compute_rep_reward(HALF_WRONG, GT, default_cfg)
        assert r_perfect == pytest.approx(1.0)
        assert r_half == pytest.approx(0.6208333333, abs=1e-6)  # golden, pinned 2026-07-13

    def test_f1_weight_zero_is_identity(self):
        cfg0 = {"reward_mode": "detection_correctness_severity"}
        cfg_zero = dict(cfg0, f1_weight=0.0)
        for ans in (PERFECT, HALF_WRONG, EMPTY):
            a, _ = compute_rep_reward(ans, GT, cfg0)
            b, _ = compute_rep_reward(ans, GT, cfg_zero)
            assert a == b

    def test_non_tool_metadata_routes_past_tool_branch(self, bank):
        # A batch with NO tool_rollout flags must fall through the tool guard
        # (it would then hit the verify workers, which this bare env lacks —
        # AttributeError, NOT the tool branch's ValueError/EnvironmentReturn).
        env = _bare_env(TOOL_CFG, bank)
        conv = [{"role": "user", "content": "x"}, {"role": "assistant", "content": PERFECT}]
        with pytest.raises(AttributeError):
            env.step([conv], [{"ground_truth": GT, "task_type": "repetition"}])


# ---------------------------------------------------------------------------
# 6. F1 sub-component degeneracies (the only genuinely new math)
# ---------------------------------------------------------------------------

class TestErrorF1:
    def test_perfect(self):
        assert compute_error_f1({"A": 3, "B": 1}, {"A": 3, "B": 1}) == 1.0

    def test_omitted_present_error_is_fn(self):
        # audit-flaw-#5 guard: omission must not pay
        assert compute_error_f1({"A": 3, "B": 1}, {"B": 1}) == 0.0

    def test_false_flag_is_fp(self):
        assert compute_error_f1({"A": 3, "B": 1}, {"A": 3, "B": 4}) == pytest.approx(2 / 3)

    def test_hallucinated_name_ignored_like_eval(self):
        assert compute_error_f1({"A": 3}, {"A": 3, "Invented": 5}) == 1.0

    def test_no_error_rep_clean_vs_flagged(self):
        assert compute_error_f1({"A": 1, "B": 1}, {"A": 1, "B": 1}) == 1.0
        assert compute_error_f1({"A": 1, "B": 1}, {"A": 3, "B": 1}) == 0.0

    def test_why_f1_is_the_anchor_not_accuracy(self):
        # 9 absent fields + 1 present, prediction says "all absent":
        # detection ACCURACY pays 90%; F1 pays 0. This asymmetry is the reason
        # f1_weight anchors the composite (report §2.1-A).
        gt = {f"E{i}": 1 for i in range(9)} | {"E9": 4}
        pred = {f"E{i}": 1 for i in range(10)}
        acc = sum(1 for k in gt if (gt[k] > 1) == (pred[k] > 1)) / len(gt)
        assert acc == pytest.approx(0.9)
        assert compute_error_f1(gt, pred) == 0.0
