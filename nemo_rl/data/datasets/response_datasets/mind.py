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


from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from nemo_rl.data.interfaces import TaskDataSpec
import os


def format(
    data: dict[str, str | float | int], output_key: str = "messages"
) -> dict[str, list[Any] | str]:
    result = {"messages": data[output_key]}
    # Pass through rubric field if present (for per-prompt custom judge rubrics)
    if "rubric" in data:
        result["rubric"] = data["rubric"]
    # Pass through user_system_prompt for per-prompt user LLM persona
    # Extract from patient_messages if present (first message contains patient persona)
    if "patient_messages" in data and len(data["patient_messages"]) > 0:
        patient_msg = data["patient_messages"][0]
        if isinstance(patient_msg, dict) and "content" in patient_msg:
            result["user_system_prompt"] = patient_msg["content"]
    elif "user_system_prompt" in data:
        result["user_system_prompt"] = data["user_system_prompt"]
    # Pass through env_info and max_turns for per-prompt environment config
    if "env_info" in data:
        result["env_info"] = data["env_info"]
        # Also extract max_turns at top level for easy access
        if "max_turns" in data["env_info"]:
            result["max_turns"] = data["env_info"]["max_turns"]
    elif "max_turns" in data:
        result["max_turns"] = data["max_turns"]
    return result


def prepare_mind_dataset(
    seed: int = 42,
    test_size: float = 0.01,
    dataset_name: str = ''
) -> dict[str, Dataset | None]:


    # Load the original dataset
    if 'swordhealth' in dataset_name:
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        original_ds = load_dataset(dataset_name, split='train', token=token)
    else:
        original_ds = load_from_disk(dataset_name)
        if 'train' in original_ds:
            original_ds = original_ds['train']

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    # Format the examples, removing original columns except those we want to keep
    # We preserve rubric, user_system_prompt, env_info, max_turns for per-prompt configs
    cols_to_preserve = ["rubric", "user_system_prompt", "env_info", "max_turns", "patient_messages"]
    train_cols_to_remove = [c for c in split_ds["train"].column_names if c not in cols_to_preserve]
    val_cols_to_remove = [c for c in split_ds["test"].column_names if c not in cols_to_preserve]

    train_formatted = split_ds["train"].map(
        format,
        remove_columns=train_cols_to_remove,
    )
    val_formatted = split_ds["test"].map(
        format,
        remove_columns=val_cols_to_remove,
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class MindDataset:
    def __init__(
        self,
        dataset_name,
        seed: int = 42,
        test_size: float = 0.001,
    ):
        """Initialize the Mind dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        

        self.formatted_ds = prepare_mind_dataset(dataset_name=dataset_name,
                                                 seed=seed, test_size=test_size,)

        self.task_spec = TaskDataSpec(
            task_name="Mind",
        ) 
