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

import argparse
import os
import pprint
from collections import defaultdict
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.datasets.response_datasets.mind_grpo import MindGRPODataset
from nemo_rl.data.interfaces import (
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.data.processors import math_hf_data_processor, grpo_mind_preprocessor
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.llm_judge_server_environment import LLMJudgeServerEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments.

    Returns:
        Tuple of (parsed_args, overrides) where:
        - parsed_args: Namespace object containing parsed arguments
        - overrides: List of remaining unparsed arguments (Hydra overrides)
    """
    parser = argparse.ArgumentParser(description="Run GRPO training with HTTP LLM judge")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides

TokenizerType = PreTrainedTokenizerBase

def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\n▶ Setting up data with HTTP LLM judge...")

    data_cls = data_config.get("data_cls", data_config["dataset_name"])

    if data_cls == "mind":
        task_name = "mind"
        task_spec = TaskDataSpec(
            task_name=task_name,
        )

        data = MindGRPODataset(dataset_name=data_config["dataset_name"])
        train_data = data.formatted_ds["train"]
        val_data = data.formatted_ds["validation"]

        # data processor
        task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
            defaultdict(lambda: (task_spec, grpo_mind_preprocessor))
        )
        task_data_processors[task_name] = (task_spec, grpo_mind_preprocessor)

    else:
        task_name = "math"
        task_spec = TaskDataSpec(
            task_name=task_name,
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )

        # load dataset
        data: Any = load_response_dataset(data_config, seed)
        train_data = data.formatted_ds["train"]
        val_data = data.formatted_ds["validation"]

        # data processor
        task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
            defaultdict(lambda: (task_spec, math_hf_data_processor))
        )
        task_data_processors[task_name] = (task_spec, math_hf_data_processor)

    # Setup server LLM judge environment (no Ray virtual cluster needed!)
    llm_judge_env = LLMJudgeServerEnvironment.options(  # type: ignore
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.llm_judge_server_environment.LLMJudgeServerEnvironment"
            ),
            "env_vars": dict(os.environ),
        }
    ).remote(env_configs["llm_judge"])

    dataset = AllTaskProcessedDataset(
        train_data,
        tokenizer,
        task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if val_data:
        val_dataset = AllTaskProcessedDataset(
            val_data,
            tokenizer,
            task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: llm_judge_env)
    task_to_env[task_name] = llm_judge_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    print("🤖 Using HTTP LLM Judge for reward evaluation")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    print("🔧 Setting up tokenizer...")
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    print("✅ Tokenizer setup complete")
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    print("✅ Generation config setup complete")

    # setup data
    print("🔧 Setting up data with HTTP LLM judge...")
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])
    print("✅ Data setup complete")


    # Print first 10 examples from training dataset for debugging
    print("\n" + "="*80)
    print("🔍 DEBUGGING: First 10 examples from training dataset:")
    print("="*80)
    for i in range(min(10, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i} ---")
        for key, value in example.items():
            if key == "message_log":
                print(f"  {key}: {len(value)} messages")
                for j, msg in enumerate(value):
                    content = msg.get('content', '')
                    print(f"    Message {j}: role={msg.get('role', 'N/A')}")
                    print(f"      content: {content[:100]}...")  # Truncate for readability
            elif key == "extra_env_info":
                print(f"  {key}:")
                if value:
                    for env_key, env_val in value.items():
                        if env_key == "rubric":
                            print(f"    {env_key}: {env_val.split(chr(10))[0] if env_val else None}")  # First line only
                        else:
                            print(f"    {env_key}: {env_val}")
                else:
                    print(f"    None")
            else:
                print(f"  {key}: {value}")
    print("="*80 + "\n")


    print("🔧 Running main setup() function...")
    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    print("✅ Main setup() complete")

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )

    for task_name in val_task_to_env.keys():
        env = val_task_to_env[task_name]
        env.shutdown.remote()


if __name__ == "__main__":
    main()