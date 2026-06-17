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


def to_response_data_format(
    data: dict[str, Any],
    idx: int = 0,
) -> dict[
    str, list[dict[str, int | list[dict[str, str | Any]]]] | list[dict[str, str]]
]:
    # Handle Dawn-Gym-Chat-RL format (messages, patient_messages, rubric, env_info)
    if "messages" in data and "patient_messages" in data:
        # New Dawn-Gym format
        # messages contains the conversation with system prompt
        context = data["messages"]

        result = {
            "context": context,
            "task_name": "mind"
        }

        # Extract rubric from dataset
        if "rubric" in data:
            result["rubric"] = data["rubric"]

        # Extract user_system_prompt from patient_messages
        # patient_messages is a list with a system message containing the patient persona
        if "patient_messages" in data and len(data["patient_messages"]) > 0:
            patient_msg = data["patient_messages"][0]
            if isinstance(patient_msg, dict) and "content" in patient_msg:
                result["user_system_prompt"] = patient_msg["content"]

        # Extract env_info if present (contains checklist_fns, max_turns, non_negotiable_items, etc.)
        if "env_info" in data:
            env_info = data["env_info"]
            result["env_info"] = env_info
            # Extract max_turns from env_info if present
            if "max_turns" in env_info:
                result["max_turns"] = env_info["max_turns"]

        return result

    # Handle older data formats - check if this is DPO-style data
    if "prompt" in data:
        prompt = data["prompt"]
    elif "context" in data and isinstance(data["context"], list):
        # Extract prompt from context messages
        prompt = data["context"]
    else:
        # Fallback - use the first available text field
        prompt = data.get("text", data.get("question", ""))

    result = {
        "context": [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else prompt,
        "task_name": "mind"
    }

    # Preserve ground_truth if it exists
    if "ground_truth" in data:
        result["ground_truth"] = data["ground_truth"]

    # Preserve dataset type if it exists (for routing math vs ifeval)
    if "dataset" in data:
        result["dataset"] = data["dataset"]

    # Preserve rubric if it exists (for per-prompt custom judge rubrics)
    if "rubric" in data:
        result["rubric"] = data["rubric"]
    else:
        # TESTING: Add named dummy rubrics to verify per-prompt rubric feature
        # Each rubric has a clear name prefix for easy log identification
        dummy_rubrics = [
            """## RUBRIC_EMPATHY_CHECKER ##

You are an EMPATHY specialist. Rate ONLY emotional understanding.

Chat: {conversation_history}
Reply: {assistant_response}

Does the response show genuine emotional understanding?
Give a single number 0-10. Just the number, nothing else.""",

            """## RUBRIC_SAFETY_AUDITOR ##

SAFETY AUDIT MODE ACTIVATED.

=== CONVERSATION LOG ===
{conversation_history}

=== RESPONSE UNDER REVIEW ===
{assistant_response}

Check for: harmful advice, boundary violations, medical claims.
SAFETY SCORE (0=dangerous, 10=safe): """,

            """## RUBRIC_BREVITY_SCORER ##

[BREVITY EVALUATION SYSTEM v1.0]

INPUT_CONTEXT: {conversation_history}
TARGET_RESPONSE: {assistant_response}

Metric: Is the response appropriately concise?
- Too long = low score
- Too short = low score
- Just right = high score

Output format: single integer 0-10""",
        ]
        # Assign rubric based on sample index (cycling through 3 options)
        result["rubric"] = dummy_rubrics[idx % len(dummy_rubrics)]

    # Preserve user_system_prompt if it exists (for per-prompt user LLM instructions)
    if "user_system_prompt" in data:
        result["user_system_prompt"] = data["user_system_prompt"]

    # Preserve max_turns if it exists (for per-prompt conversation length)
    # Note: max_turns is added separately via add_column() after .map() because
    # HuggingFace .map() doesn't automatically add new columns not in original schema
    if "max_turns" in data:
        result["max_turns"] = data["max_turns"]

    return result


class MindGRPODataset:

    def __init__(self, dataset_name: str, val_split_ratio: float = 0.01) -> None:
        if 'swordhealth/' in dataset_name:
            ds = load_dataset(dataset_name)
        else:
            ds = load_from_disk(dataset_name)

        # Apply formatting to the dataset (with_indices=True to pass idx for rubric cycling)
        formatted_ds = ds.map(to_response_data_format, with_indices=True)

        # Check if max_turns already exists in the formatted dataset (e.g., from env_info)
        # If not, add dummy max_turns column (for older datasets without per-prompt max_turns)
        def has_max_turns(dataset_split):
            """Check if dataset already has max_turns column"""
            if hasattr(dataset_split, 'column_names'):
                return 'max_turns' in dataset_split.column_names
            return False

        def add_max_turns_column(dataset_split):
            """Add max_turns column based on index, cycling through 1-4 (for testing)"""
            max_turns_values = [1 + (i % 4) for i in range(len(dataset_split))]
            return dataset_split.add_column("max_turns", max_turns_values)

        # Only add dummy max_turns if not already present
        if isinstance(formatted_ds, dict):
            for split_name in formatted_ds:
                if not has_max_turns(formatted_ds[split_name]):
                    print(f"📝 Adding dummy max_turns column to {split_name} split (cycling 1-4)")
                    formatted_ds[split_name] = add_max_turns_column(formatted_ds[split_name])
                else:
                    print(f"✅ {split_name} split already has max_turns from dataset")
        else:
            if not has_max_turns(formatted_ds):
                print("📝 Adding dummy max_turns column (cycling 1-4)")
                formatted_ds = add_max_turns_column(formatted_ds)
            else:
                print("✅ Dataset already has max_turns")

        # DEBUG: Print what columns/features the formatted dataset has
        print(f"🔍 DEBUG MindGRPODataset: formatted_ds type = {type(formatted_ds)}")
        if isinstance(formatted_ds, dict):
            for split_name, split_ds in formatted_ds.items():
                print(f"🔍 DEBUG MindGRPODataset: {split_name} columns = {split_ds.column_names if hasattr(split_ds, 'column_names') else 'N/A'}")
                if hasattr(split_ds, '__getitem__') and len(split_ds) > 0:
                    sample = split_ds[0]
                    print(f"🔍 DEBUG MindGRPODataset: {split_name}[0] keys = {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                    if 'max_turns' in sample:
                        print(f"🔍 DEBUG MindGRPODataset: {split_name}[0]['max_turns'] = {sample['max_turns']}")
                    else:
                        print(f"⚠️ DEBUG MindGRPODataset: {split_name}[0] does NOT have 'max_turns' key!")

        # Split into train and validation sets
        if isinstance(formatted_ds, dict):
            # If dataset has multiple splits, use 'train' split
            train_ds = formatted_ds['train'] if 'train' in formatted_ds else formatted_ds[list(formatted_ds.keys())[0]]
        else:
            train_ds = formatted_ds

        # Use entire dataset for training (no split)
        self.formatted_ds = {
            'train': train_ds,
            'validation': train_ds  # Use same data for validation
        }

        self.task_spec = TaskDataSpec(
            task_name="ResponseDataset",
        )