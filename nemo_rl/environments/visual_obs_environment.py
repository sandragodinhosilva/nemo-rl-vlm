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
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
