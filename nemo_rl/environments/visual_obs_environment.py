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
import logging
import re
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.judge_config import JudgeConfig
from nemo_rl.environments.visual_obs_aux_rewards import (
    AUX_TASK_TYPES,
    compute_aux_reward,
)
from nemo_rl.environments.visual_obs_comparison_rewards import (
    compute_comparison_reward,
)
from nemo_rl.environments.visual_obs_full_exercise_rewards import (
    compute_full_exercise_reward,
)
from nemo_rl.environments.visual_obs_rep_rewards import (
    compute_rep_reward,
)
from nemo_rl.environments.visual_obs_rewards import (
    compute_visual_obs_reward,
)
from nemo_rl.environments.visual_obs_tool_rollout import (
    compute_tool_final_reward,
    dispatch_tool_calls,
    make_obs_backend,
    parse_tool_calls,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.models.generation.vllm import VllmGeneration


class ThriveVLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    reward_mode: Optional[str]  # "detection_correctness_severity", "detection_correctness_severity_quadratic", "detection_correctness_severity_sqrt", "detection_correctness_severity_multiplicative" (default: "detection_correctness_severity")
    # Detection-Correctness-Severity mode parameters
    error_weight: Optional[float]  # Weight for errors with GT severity > 1 in severity component (default: 1.0)
    non_error_weight: Optional[float]  # Weight for errors with GT severity = 1 in severity component (default: 0.5)
    detection_weight: Optional[float]  # Weight for per-field error detection (default: 0.25)
    correctness_weight: Optional[float]  # Weight for exact match across all fields (default: 0.25)
    severity_weight: Optional[float]  # Weight for distance-based severity accuracy (default: 0.5)
    format_weight: Optional[float]  # Weight for response format adherence (default: 0.0)
    # Full-exercise analysis reward mode — controls distance type for severity component
    # ("vanilla", "vanilla_quadratic", "vanilla_sqrt"; default: "vanilla")
    # Uses correctness + severity + format components. Effectiveness (1-3) and Injury Risk (1-3).
    full_exercise_reward_mode: Optional[str]
    # Full-exercise weight overrides (separate from per-rep weights)
    fe_correctness_weight: Optional[float]  # Weight for exact match across all fields (default: 0.25)
    fe_severity_weight: Optional[float]  # Weight for distance-based severity accuracy (default: 0.5)
    fe_format_weight: Optional[float]  # Weight for response format adherence (default: 0.0)
    # Comparison task reward weights
    comparison_verdict_weight: Optional[float]  # Weight for correct verdict (default: 0.8)
    comparison_format_weight: Optional[float]  # Weight for proper answer format (default: 0.2)
    # Auxiliary task reward weights (MCQ, exercise name, keypoint prediction/labeling)
    mcq_correctness_weight: Optional[float]  # default: 1.0
    mcq_format_weight: Optional[float]  # default: 0.0
    exercise_name_fuzzy_weight: Optional[float]  # default: 0.8
    exercise_name_exact_weight: Optional[float]  # default: 0.2
    kp_oks_weight: Optional[float]  # default: 0.7
    kp_detection_weight: Optional[float]  # default: 0.3
    kl_f1_weight: Optional[float]  # default: 0.8
    kl_exact_weight: Optional[float]  # default: 0.2
    # Judge configuration - when enabled, LLM judge is used instead of rule-based verification
    # The judge can work with any reward_mode to determine how to evaluate and score responses
    judge: Optional[JudgeConfig]
    # Multi-turn query_obs tool-loop rollouts (LOCAL-ONLY, sgsilva 2026-07-13 —
    # report 2026-07-13_grpo_vobs_tool_scaffold.md). Keys: enabled, obs_bank_path,
    # bank_mode (model_obs|corrupted|gt — gt is canary-only), corruption_rate,
    # corruption_seed, model_obs_bank_path, max_tool_rounds, f1_weight,
    # detection_weight, severity_weight, correctness_weight, format_weight,
    # malformed_call_penalty, offbank_question_penalty, round_penalty,
    # question_penalty. Requires grpo.max_rollout_turns > 1 in the recipe.
    tool_rollout: Optional[dict]






@ray.remote
class ThriveVLMVerifyWorker:
    def __init__(self, cfg: ThriveVLMEnvConfig) -> None:
        logging.getLogger("thrive_vlm_worker").setLevel(logging.CRITICAL)

        # Reward mode configuration (per-rep task)
        self.reward_mode = cfg.get("reward_mode", "detection_correctness_severity")

        # Detection-Correctness-Severity mode parameters
        self.error_weight = cfg.get("error_weight", 1.0)
        self.non_error_weight = cfg.get("non_error_weight", 0.5)
        self.detection_weight = cfg.get("detection_weight", 0.25)
        self.correctness_weight = cfg.get("correctness_weight", 0.25)
        self.severity_weight = cfg.get("severity_weight", 0.5)
        self.format_weight = cfg.get("format_weight", 0.0)

        # Full-exercise analysis reward mode
        self.full_exercise_reward_mode = cfg.get("full_exercise_reward_mode", "vanilla")

        # Full-exercise weight overrides (fall back to per-rep weights if not set)
        self.fe_correctness_weight = cfg.get("fe_correctness_weight", self.correctness_weight)
        self.fe_severity_weight = cfg.get("fe_severity_weight", self.severity_weight)
        self.fe_format_weight = cfg.get("fe_format_weight", self.format_weight)

        # Comparison task reward weights
        self.comparison_verdict_weight = cfg.get("comparison_verdict_weight", 0.8)
        self.comparison_format_weight = cfg.get("comparison_format_weight", 0.2)

        # Build config dicts for each task type's reward module
        self.rep_config = {
            "reward_mode": self.reward_mode,
            "detection_weight": self.detection_weight,
            "correctness_weight": self.correctness_weight,
            "severity_weight": self.severity_weight,
            "format_weight": self.format_weight,
            "error_weight": self.error_weight,
            "non_error_weight": self.non_error_weight,
        }
        self.fe_config = {
            "fe_correctness_weight": self.fe_correctness_weight,
            "fe_severity_weight": self.fe_severity_weight,
            "fe_format_weight": self.fe_format_weight,
            "full_exercise_reward_mode": self.full_exercise_reward_mode,
        }
        self.comparison_config = {
            "comparison_verdict_weight": self.comparison_verdict_weight,
            "comparison_format_weight": self.comparison_format_weight,
        }
        self.aux_config = {
            "mcq_correctness_weight": cfg.get("mcq_correctness_weight", 1.0),
            "mcq_format_weight": cfg.get("mcq_format_weight", 0.0),
            "exercise_name_fuzzy_weight": cfg.get("exercise_name_fuzzy_weight", 0.8),
            "exercise_name_exact_weight": cfg.get("exercise_name_exact_weight", 0.2),
            "kp_oks_weight": cfg.get("kp_oks_weight", 0.7),
            "kp_detection_weight": cfg.get("kp_detection_weight", 0.3),
            "kl_f1_weight": cfg.get("kl_f1_weight", 0.8),
            "kl_exact_weight": cfg.get("kl_exact_weight", 0.2),
        }
        # visual_obs_config is populated per-call with exercise_id from the dataset
        self.visual_obs_config: dict = {}

    def extract_debug_info(
        self,
        responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
        exercise_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Extract parsed scores and component rewards for debug printing."""
        if task_types is None:
            task_types = ["repetition"] * len(responses)
        if exercise_ids is None:
            exercise_ids = [""] * len(responses)

        results = []
        for response, gt, task_type, exercise_id in zip(
            responses, ground_truths, task_types, exercise_ids
        ):
            if task_type == "visual_obs":
                _, info = compute_visual_obs_reward(
                    response, gt, {"exercise_id": exercise_id}
                )
                info["task_type"] = task_type
            elif task_type in AUX_TASK_TYPES:
                _, info = compute_aux_reward(task_type, response, gt, self.aux_config)
                info["task_type"] = task_type
            elif task_type == "comparison":
                _, info = compute_comparison_reward(response, gt, self.comparison_config)
            elif task_type == "full_exercise":
                _, info = compute_full_exercise_reward(response, gt, self.fe_config)
            else:
                _, info = compute_rep_reward(response, gt, self.rep_config)
            results.append(info)
        return results

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
        exercise_ids: Optional[list[str]] = None,
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Computes distance-based rewards for severity scores and movement scores.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. Ground truth text in the same format as responses.
            task_types: Optional per-sample task type.
            exercise_ids: Optional per-sample exercise IDs (needed for visual_obs reward).

        Returns:
            list[float]. The rewards for each predicted response.
        """
        if task_types is None:
            task_types = ["repetition"] * len(pred_responses)
        if exercise_ids is None:
            exercise_ids = [""] * len(pred_responses)

        results = []
        self._last_reward_details = []
        for idx, (response, ground_truth_str, task_type, exercise_id) in enumerate(
            zip(pred_responses, ground_truths, task_types, exercise_ids)
        ):
            try:
                self.visual_obs_config = {"exercise_id": exercise_id}
                reward, details = self._compute_reward_with_details(response, ground_truth_str, task_type=task_type, sample_idx=idx)
                results.append(float(reward))
                self._last_reward_details.append(details)
            except Exception:
                results.append(0.0)
                self._last_reward_details.append({})
        return results

    def get_last_reward_details(self) -> list[dict]:
        """Return intermediate reward component scores from the last verify() call."""
        return getattr(self, "_last_reward_details", [])

    def _compute_reward(
        self, response: str, ground_truth_str: str, task_type: str = "repetition", sample_idx: int = -1
    ) -> float:
        """Compute reward by dispatching to task-specific reward module."""
        if task_type == "visual_obs":
            reward, _ = compute_visual_obs_reward(response, ground_truth_str, self.visual_obs_config)
            return reward
        if task_type in AUX_TASK_TYPES:
            reward, _ = compute_aux_reward(task_type, response, ground_truth_str, self.aux_config)
            return reward
        if task_type == "comparison":
            reward, _ = compute_comparison_reward(response, ground_truth_str, self.comparison_config)
            return reward
        if task_type == "full_exercise":
            reward, _ = compute_full_exercise_reward(response, ground_truth_str, self.fe_config)
            return reward
        reward, _ = compute_rep_reward(response, ground_truth_str, self.rep_config)
        return reward

    def _compute_reward_with_details(
        self, response: str, ground_truth_str: str, task_type: str = "repetition", sample_idx: int = -1
    ) -> tuple[float, dict]:
        """Compute reward and return intermediate component scores."""
        if task_type == "visual_obs":
            return compute_visual_obs_reward(response, ground_truth_str, self.visual_obs_config)
        if task_type in AUX_TASK_TYPES:
            return compute_aux_reward(task_type, response, ground_truth_str, self.aux_config)
        if task_type == "comparison":
            return compute_comparison_reward(response, ground_truth_str, self.comparison_config)
        if task_type == "full_exercise":
            return compute_full_exercise_reward(response, ground_truth_str, self.fe_config)
        return compute_rep_reward(response, ground_truth_str, self.rep_config)


class ThriveVLMJudgeWorker:
    """Worker that uses an LLM judge (via vLLM) to score Thrive VLM responses.

    This class creates a VllmGeneration instance using the provided cluster.
    It's designed to be created inside a Ray actor environment where the cluster
    is created during the actor's __init__.
    """

    def __init__(
        self,
        judge_config: JudgeConfig,
        cluster: Optional[RayVirtualCluster],
    ) -> None:
        """Initialize the judge worker and create vLLM generation instance.

        Args:
            judge_config: Configuration for the judge model
            cluster: Virtual cluster for GPU allocation (None if colocated - not yet supported)
        """
        import os

        from nemo_rl.prompts.thrive_judge_rubrics import (
            format_judge_prompt,
            get_rubric,
            parse_judge_output,
        )

        logging.getLogger("thrive_vlm_judge_worker").setLevel(logging.CRITICAL)

        self.judge_config = judge_config
        self.rubric_type = judge_config["rubric_type"]
        self.custom_rubric_path = judge_config.get("custom_rubric_path", None)
        self.output_format = judge_config.get("output_format", "json")
        self.include_ground_truth = judge_config.get("include_ground_truth", True)
        self.batch_size = judge_config.get("batch_size", 4)

        # Load rubric template
        self.rubric = get_rubric(self.rubric_type, self.custom_rubric_path)

        # Store parse function
        self.parse_judge_output = parse_judge_output
        self.format_judge_prompt = format_judge_prompt

        # Create vLLM generation for judge
        if cluster is None:
            raise NotImplementedError(
                "Judge colocation mode is not yet supported in self-contained environment architecture. "
                "Please set judge.colocated.enabled: false and provide judge.resources config."
            )

        print(f"🔨 Initializing judge vLLM with model: {judge_config['model_name']}")

        # Load tokenizer for judge model to configure generation properly
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.models.generation import configure_generation_config

        judge_tokenizer = get_tokenizer(
            judge_config["generation"].get("tokenizer", {"name": judge_config["model_name"]}),
            get_processor=False,
        )

        # Build vLLM config from judge config
        vllm_config = {
            "backend": "vllm",
            "model_name": judge_config["model_name"],
            "vllm_cfg": judge_config["generation"]["vllm_cfg"],
            "max_new_tokens": judge_config["generation"].get("max_new_tokens", 2048),
            "temperature": judge_config["generation"].get("temperature", 0.0),
            "top_p": judge_config["generation"].get("top_p", 1.0),
            "top_k": judge_config["generation"].get("top_k", None),
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": judge_config["colocated"],
        }

        # Add tokenizer config if present
        if "tokenizer" in judge_config["generation"]:
            vllm_config["tokenizer"] = judge_config["generation"]["tokenizer"]

        # Add vllm_kwargs if present (from generation config)
        if "vllm_kwargs" in judge_config["generation"]:
            vllm_config["vllm_kwargs"] = judge_config["generation"]["vllm_kwargs"]

        # Configure generation config to set internal fields like _pad_token_id
        vllm_config = configure_generation_config(vllm_config, judge_tokenizer, is_eval=True)

        # Remove CUDA_VISIBLE_DEVICES to let ray control GPU allocation
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.judge_vllm = VllmGeneration(
            cluster=cluster,
            config=vllm_config,
            name_prefix="thrive_judge",
        )

        print("✅ Judge vLLM initialized successfully")

    def judge_batch(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
        video_contexts: Optional[list[str]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Judge a batch of responses using the LLM judge.

        Args:
            pred_responses: Predicted responses from the model
            ground_truths: Ground truth responses
            task_types: Task type for each response ("repetition" or "full_exercise")
            video_contexts: Optional video context descriptions

        Returns:
            Tuple of (rewards, parsed_details) where:
                - rewards: List of total reward scores (floats between 0 and 1)
                - parsed_details: List of dicts with detailed scores and reasoning
        """
        if task_types is None:
            task_types = ["repetition"] * len(pred_responses)
        if video_contexts is None:
            video_contexts = ["Video of exercise performance"] * len(pred_responses)

        # Format judge prompts
        judge_prompts = []
        for pred, gt, task_type, context in zip(
            pred_responses, ground_truths, task_types, video_contexts
        ):
            prompt = self.format_judge_prompt(
                rubric=self.rubric,
                video_context=context,
                model_response=pred,
                ground_truth=gt if self.include_ground_truth else None,
                task_type=task_type,
            )
            judge_prompts.append(prompt)

        # Call judge vLLM to get scores
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        # Create input data for vLLM
        judge_data = BatchedDataDict(prompts=judge_prompts)

        # Generate judge evaluations (greedy decoding for consistency)
        judge_outputs = self.judge_vllm.generate_text(judge_data, greedy=True)

        # Parse judge outputs to extract scores
        rewards = []
        parsed_details = []
        for i, output_text in enumerate(judge_outputs["texts"]):
            try:
                parsed = self.parse_judge_output(output_text)
                reward = parsed["total_reward"]
                parsed_details.append(parsed)

            except Exception as e:
                print(f"⚠️  Failed to parse judge output for sample {i}: {e}")
                print(f"Judge output: {output_text[:200]}")
                reward = 0.5  # Fallback to neutral score
                parsed_details.append({"total_reward": reward})
            rewards.append(reward)

        return rewards, parsed_details

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Verify responses using the judge (alias for judge_batch for compatibility).

        This method maintains the same interface as ThriveVLMVerifyWorker.

        Args:
            pred_responses: Predicted responses from the model
            ground_truths: Ground truth responses
            task_types: Task type for each response ("repetition" or "full_exercise")

        Returns:
            Tuple of (rewards, parsed_details)
        """
        return self.judge_batch(pred_responses, ground_truths, task_types)


class ThriveVLMEnvironmentMetadata(TypedDict):
    ground_truth: str
    task_type: str  # "repetition" or "full_exercise"


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ThriveVLMEnvironment(EnvironmentInterface):
    def __init__(
        self,
        cfg: ThriveVLMEnvConfig,
    ):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self._step_call_count = 0

        # Tool-rollout bridge (LOCAL-ONLY, sgsilva 2026-07-13). Bank loads once
        # per env actor; the bank MODE is reward-design-critical (§2.0b of the
        # scaffold report) — make_obs_backend prints loudly on the gt canary mode.
        self.tool_cfg = cfg.get("tool_rollout") or {}
        self.tool_backend = (
            make_obs_backend(self.tool_cfg) if self.tool_cfg.get("enabled") else None
        )

        # Always create rule-based verify workers
        self.verify_workers = [
            ThriveVLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

        # Initialize judge if enabled (self-contained architecture)
        self.use_judge = cfg.get("judge", {}).get("enabled", False) if "judge" in cfg else False
        self.judge_worker = None
        self.judge_weight = 0.5  # Default

        if self.use_judge:
            judge_config = cfg["judge"]
            print(f"🔨 Initializing ThriveVLM judge (self-contained)...")

            # Create judge virtual cluster (if not colocated)
            if not judge_config["colocated"]["enabled"]:
                judge_resources = judge_config.get("resources", {"gpus_per_node": 2, "num_nodes": 1})
                self.judge_virtual_cluster = RayVirtualCluster(
                    name="thrive_vlm_judge_cluster",
                    bundle_ct_per_node_list=[judge_resources["gpus_per_node"]] * judge_resources["num_nodes"],
                    use_gpus=True,
                    num_gpus_per_node=judge_resources["gpus_per_node"],
                    max_colocated_worker_groups=1,
                )
                print(f"  ✓ Created dedicated judge cluster: {judge_resources['num_nodes']} nodes × {judge_resources['gpus_per_node']} GPUs")
            else:
                # For colocated mode, the cluster will be shared with generation/policy
                # The cluster must be created externally and passed via colocation mechanism
                self.judge_virtual_cluster = None
                print(f"  ⚠ Judge colocation enabled - cluster management not supported in self-contained mode")
                print(f"     Colocation target: {judge_config['colocated'].get('colocation_target', 'generation')}")

            # Create judge worker (which will create VllmGeneration internally)
            self.judge_worker = ThriveVLMJudgeWorker(judge_config, self.judge_virtual_cluster)
            self.judge_weight = judge_config.get("judge_weight", 0.5)
            print(f"  ✓ Judge initialized with weight={self.judge_weight}")
        else:
            self.judge_virtual_cluster = None

    def shutdown(self) -> None:
        # shutdown verify workers
        for worker in self.verify_workers:
            ray.kill(worker)

        # shutdown judge worker if it exists
        # Note: judge_worker is not a Ray actor, so we don't need to kill it
        # The VllmGeneration workers inside it will be cleaned up when the cluster is destroyed

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[ThriveVLMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the thrive-vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: list[ThriveVLMEnvironmentMetadata]. Ground truth scores in JSON format.

        Returns:
            EnvironmentReturn: A tuple containing observations, metadata, stop strings, rewards, and done flags.
        """
        # Tool-rollout branch (LOCAL-ONLY, sgsilva 2026-07-13): rows flagged by
        # the loader take the multi-turn path. Batches must be homogeneous —
        # a mixed batch would need two interleaved return paths; refuse loudly
        # rather than silently mis-scoring either side.
        if self.tool_backend is not None:
            is_tool_row = [bool(m.get("tool_rollout")) for m in metadata]
            if any(is_tool_row):
                if not all(is_tool_row):
                    raise ValueError(
                        "tool_rollout is enabled and this batch mixes tool rows with "
                        f"non-tool rows ({sum(is_tool_row)}/{len(is_tool_row)} tool). "
                        "Use a homogeneous vobs_tool dataset for tool-rollout GRPO."
                    )
                return self._step_tool_rollout(message_log_batch, metadata)

        # Extract the assistant's responses from the message history
        assistant_response_batch = []
        full_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            full_response = "".join(assistant_responses)
            full_response_batch.append(full_response)

            # Strip reasoning trace for reasoning models
            # Remove everything from start up to and including </think> tag
            cleaned_response = re.sub(r'^.*?</think>\s*', '', full_response, flags=re.DOTALL)

            assistant_response_batch.append(cleaned_response.strip())

        ground_truths = [g["ground_truth"] for g in metadata]
        task_types = [g.get("task_type", "repetition") for g in metadata]
        exercise_ids = [g.get("exercise_id", "") for g in metadata]
        sample_ids = [g.get("sample_id", "") for g in metadata]

        # Print debug info for 2 samples, once every 64 step() calls
        self._step_call_count += 1
        should_print = (self._step_call_count % 64 == 1)
        if should_print:
            for i in range(min(2, len(assistant_response_batch))):
                full = full_response_batch[i]
                # Qwen3-VL: chat template injects "<think>\n" into the prompt,
                # so the generation starts inside the reasoning block and only
                # emits the closing </think>. Split on the close tag.
                think_match = re.match(r"^(.*?)</think>\s*(.*)$", full, flags=re.DOTALL)
                print(f"\n[Sample {i}] (id={sample_ids[i]}) Generated Response:")
                if think_match:
                    print("[REASONING]")
                    print(think_match.group(1).strip())
                    print("[ANSWER]")
                    print(think_match.group(2).strip())
                else:
                    print(full)

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        chunked_task_types = chunk_list_to_workers(task_types, self.num_workers)
        chunked_exercise_ids = chunk_list_to_workers(exercise_ids, self.num_workers)

        # Process each chunk in parallel with verify workers
        verify_futures = [
            self.verify_workers[i].verify.remote(chunk, ground_truth_chunk, task_type_chunk, exercise_id_chunk)
            for i, (chunk, ground_truth_chunk, task_type_chunk, exercise_id_chunk) in enumerate(
                zip(
                    chunked_assistant_response_batch,
                    chunked_ground_truths,
                    chunked_task_types,
                    chunked_exercise_ids,
                )
            )
        ]

        verify_results = ray.get(verify_futures)
        # flatten the verify results
        verify_results = [item for sublist in verify_results for item in sublist]

        # Collect intermediate reward details from workers
        detail_futures = [w.get_last_reward_details.remote() for w in self.verify_workers]
        all_details = ray.get(detail_futures)
        reward_details = [item for sublist in all_details for item in sublist]

        # Store reward details in metadata for downstream logging
        for i, m in enumerate(metadata):
            if i < len(reward_details) and isinstance(m, dict):
                m["reward_details"] = reward_details[i]

        # Debug: print reward breakdown for first 2 samples (gated by step count)
        if should_print:
            debug_results = ray.get(
                self.verify_workers[0].extract_debug_info.remote(
                    assistant_response_batch[:2], ground_truths[:2], task_types[:2],
                    exercise_ids[:2],
                )
            )
            for i, info in enumerate(debug_results):
                sid = sample_ids[i] if i < len(sample_ids) else ""
                task_label = info.get("task_type", "repetition")
                print(f"\n[Sample {i}] (id={sid}) [{task_label}] Reward Breakdown:")
                if task_label in AUX_TASK_TYPES:
                    # Print aux task details compactly
                    detail_items = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in info.items() if k != "task_type"]
                    print(f"  {' | '.join(detail_items)}")
                elif task_label == "comparison":
                    print(f"  GT verdict: {info.get('gt_verdict')} | Pred verdict: {info.get('pred_verdict')} | Correct: {info.get('correctness')}")
                elif task_label == "full_exercise":
                    print(f"  GT:   effectiveness={info.get('gt_effectiveness')} injury_risk={info.get('gt_injury_risk')}")
                    print(f"  Pred: effectiveness={info.get('pred_effectiveness')} injury_risk={info.get('pred_injury_risk')}")
                    gt_ans = info.get("gt_answers", [])
                    pred_ans = info.get("pred_answers", [])
                    for q_idx in range(9):
                        g = gt_ans[q_idx] if q_idx < len(gt_ans) else None
                        p = pred_ans[q_idx] if q_idx < len(pred_ans) else None
                        print(f"  Q{q_idx+1}: gt={g} pred={p}")
                else:
                    print(f"  GT:   effectiveness={info.get('gt_effectiveness')} injury_risk={info.get('gt_injury_risk')}")
                    print(f"  Pred: effectiveness={info.get('pred_effectiveness')} injury_risk={info.get('pred_injury_risk')}")
                    print(f"  GT errors:   {info.get('gt_errors', {})}")
                    print(f"  Pred errors: {info.get('pred_errors', {})}")
                components = []
                if "detection" in info:
                    components.append(f"detection={info['detection']:.4f}")
                if "correctness" in info:
                    components.append(f"correctness={info['correctness']:.4f}")
                if "severity" in info:
                    components.append(f"severity={info['severity']:.4f}")
                if "format" in info:
                    components.append(f"format={info['format']:.4f}")
                print(f"  Components: {' | '.join(components)}")
                print(f"  Final Reward: {verify_results[i]:.4f}")

        # If judge is enabled, also get judge rewards
        if self.use_judge and self.judge_worker:
            # Call the single judge worker with the full batch
            # (vLLM handles parallelism internally via DP)
            # Note: judge_worker is not a Ray actor, so we call it directly
            judge_results, judge_details = self.judge_worker.verify(
                assistant_response_batch, ground_truths, task_types
            )

            # Combine verify and judge rewards using judge_weight
            results = [
                (1 - self.judge_weight) * v_reward + self.judge_weight * j_reward
                for v_reward, j_reward in zip(verify_results, judge_results)
            ]

            # Print reward combination for first 2 samples (gated by step count)
            if should_print:
                for idx in range(min(2, len(results))):
                    print(f"[Sample {idx}] verify={verify_results[idx]:.4f} judge={judge_results[idx]:.4f} final={results[idx]:.4f}")
        else:
            # Use only verify rewards
            results = verify_results

        observations = [
            {
                "role": "environment",
                "content": f"Environment: reward={result:.3f}",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def _step_tool_rollout(  # LOCAL-ONLY (sgsilva 2026-07-13) — tool-loop GRPO bridge
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[ThriveVLMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Multi-turn step: execute query_obs mid-rollout or score the final turn.

        Per row, inspects the LAST assistant turn only (a <tool_call> turn must
        never reach the answer parser — report §4.2):
        - tool call(s) present and round budget left → dispatch ALL calls in the
          turn (multibatch shape), return the tool text as a role:"tool"
          observation with reward 0.0 / terminateds 0. The rollout loop injects
          it (loss-masked by role, grpo.py:1717) and resumes generation.
        - otherwise the turn is the final answer → f1-anchored composite with
          the multiplicative tool penalty (report §2.2), terminateds 1.
        Round/penalty counters ride in metadata, which the rollout loop
        round-trips into extra_env_info each turn — the env stays stateless.
        next_stop_strings stays None on purpose: the probe's stop-seq-in-<think>
        bug (2026-07-12) showed why "</tool_call>" as a stop string is unsafe.
        """
        max_rounds = int(self.tool_cfg.get("max_tool_rounds", 4))
        observations: list[dict[str, str]] = []
        rewards_list: list[float] = []
        terminateds_list: list[float] = []

        self._step_call_count += 1
        should_print = (self._step_call_count % 64 == 1)

        for idx, (conversation, meta) in enumerate(zip(message_log_batch, metadata)):
            last_assistant = ""
            for interaction in reversed(conversation):
                if interaction["role"] == "assistant":
                    last_assistant = interaction["content"]
                    break

            folder_name = meta.get("folder_name", "")
            repetition_id = meta.get("repetition_id", "")
            if not folder_name or not repetition_id:
                raise ValueError(
                    "tool_rollout row is missing folder_name/repetition_id in "
                    f"extra_env_info (sample_id={meta.get('sample_id', '')!r}) — "
                    "the loader must hard-fail before this; refusing to guess."
                )

            calls = parse_tool_calls(last_assistant)
            rounds = int(meta.get("tool_rounds", 0))

            if calls and rounds < max_rounds:
                asked = list(meta.get("tool_asked") or [])
                tool_text, counters = dispatch_tool_calls(
                    calls, self.tool_backend, folder_name, repetition_id, asked
                )
                meta["tool_asked"] = asked
                meta["tool_rounds"] = rounds + 1
                for key, value in counters.items():
                    meta["tool_" + key] = int(meta.get("tool_" + key, 0)) + int(value)
                observations.append({"role": "tool", "content": tool_text})
                # Intermediate turns MUST return exactly 0.0 — the rollout loop
                # ACCUMULATES per-turn rewards (rollouts.py:464).
                rewards_list.append(0.0)
                terminateds_list.append(0.0)
                if should_print and idx < 2:
                    print(
                        f"\n[Tool rollout {idx}] (id={meta.get('sample_id','')}) "
                        f"round {rounds + 1}/{max_rounds}: {counters}",
                        flush=True,
                    )
            else:
                # Final answer (or round budget exhausted — an unanswered
                # tool-call turn parses as no answer and scores ~0, the
                # implicit strong penalty).
                cleaned = re.sub(
                    r'^.*?</think>\s*', '', last_assistant, flags=re.DOTALL
                ).strip()
                reward, details = compute_tool_final_reward(
                    cleaned, meta["ground_truth"], meta, self.tool_cfg
                )
                meta["reward_details"] = details
                observations.append(
                    {"role": "environment", "content": f"Environment: reward={reward:.3f}"}
                )
                rewards_list.append(float(reward))
                terminateds_list.append(1.0)
                if should_print and idx < 2:
                    print(
                        f"\n[Tool rollout {idx}] (id={meta.get('sample_id','')}) FINAL: "
                        f"reward={reward:.4f} pre-penalty={details.get('answer_reward_prepenalty', 0):.4f} "
                        f"P={details.get('tool_penalty_fraction', 0):.3f} "
                        f"rounds={meta.get('tool_rounds', 0)} "
                        f"f1={details.get('f1', 'n/a')}",
                        flush=True,
                    )

        rewards = torch.tensor(rewards_list).cpu()
        terminateds = torch.tensor(terminateds_list).cpu()
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    @staticmethod
    def _mean_per_group_std(prompts: "torch.Tensor", rewards: "torch.Tensor") -> float:
        """Reward-collapse canary: mean over prompt-groups of the within-group reward std.

        GRPO's advantage is (r - group_mean)/group_std; when within-group reward std -> 0 there
        is no learning signal and the policy can collapse onto a low-variance attractor (see
        ~/.claude/reports/2026-06-26_grpo_ordinal_distance_reward_collapse.md). Watching this
        trend to 0 is the leading indicator — it shows before val accuracy moves. Groups by
        unique prompt rows, mirroring calculate_pass_rate_per_prompt. Local-only metric.
        """
        unique_prompts = torch.unique(prompts, dim=0)
        stds = []
        for i in range(len(unique_prompts)):
            mask = (prompts == unique_prompts[i]).all(1)
            g = rewards[mask].float()
            if g.numel() > 1:
                stds.append(g.std(unbiased=False).item())
        return float(sum(stds) / len(stds)) if stds else 0.0

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch."""
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences

        # Compute average reward for correctly ended sequences
        if (batch["is_end"] == 1).float().sum() > 0:
            avg_reward_correct = (
                batch["rewards"][batch["is_end"] == 1].float().mean().item()
            )
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["is_end"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            avg_reward_correct = 0.0
            correct_solution_generation_lengths = 0

        metrics = {
            "avg_reward": batch["rewards"].mean().item(),
            "avg_reward_correct_endings": avg_reward_correct,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            # reward-collapse canary (local-only): within-group reward std, mean over prompts.
            # -> 0 means GRPO has no signal to learn from (report 2026-06-26).
            "avg_reward_std": self._mean_per_group_std(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
