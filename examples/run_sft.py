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
from datetime import datetime
import os
import pprint
import sys
from bisect import bisect_right
from functools import partial

from datasets import concatenate_datasets
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    load_response_dataset,
    update_single_dataset_config,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def _configure_streaming_output():
    """Make stdout/stderr flush promptly, especially when redirected to a file."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--ignore-previous",
        action="store_true",
        help=(
            "Start from scratch by writing checkpoints to a fresh timestamped "
            "directory instead of resuming from the latest checkpoint in the "
            "configured checkpoint_dir."
        ),
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


class CombinedDataset:
    """Indexable concatenation for mixed dataset backends.

    Unlike `datasets.concatenate_datasets`, this also supports lightweight
    custom dataset wrappers such as `PreservingDataset`.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        prev_cum_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = idx - prev_cum_size
        return self.datasets[dataset_idx][sample_idx]


def _merge_raw_datasets(datasets):
    """Merge raw datasets while supporting non-HF dataset wrappers."""
    if len(datasets) == 1:
        return datasets[0]

    if all(hasattr(dataset, "features") for dataset in datasets):
        return concatenate_datasets(datasets)

    return CombinedDataset(datasets)


def _summarize_message_content(message):
    """Return a concise multimodal summary for a processed message."""
    content = message.get("content")
    token_ids = message["token_ids"]

    num_content_items = 0
    num_image_items = 0
    num_video_items = 0
    num_text_items = 0
    if isinstance(content, list):
        num_content_items = len(content)
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"image", "image_url"}:
                num_image_items += 1
            elif item_type == "video":
                num_video_items += 1
            elif item_type == "text":
                num_text_items += 1

    return {
        "content": content,
        "token_ids": token_ids,
        "content_type": type(content).__name__,
        "num_content_items": num_content_items,
        "num_image_items": num_image_items,
        "num_video_items": num_video_items,
        "num_text_items": num_text_items,
    }


def _classify_sample_modality(sample):
    """Classify a processed sample as video, image, or text."""
    has_image = False
    has_video = False

    for message in sample["message_log"]:
        summary = _summarize_message_content(message)
        has_image = has_image or summary["num_image_items"] > 0
        has_video = has_video or summary["num_video_items"] > 0

    if has_video:
        return "video"
    if has_image:
        return "image"
    return "text"


def _print_debug_sample(sample_idx, sample, tokenizer):
    """Pretty-print one processed sample for debugging."""
    sample_modality = _classify_sample_modality(sample)
    print(f"\n[Sample {sample_idx + 1}] modality={sample_modality}")
    extra_env_info = sample.get("extra_env_info") or {}
    for key in ("source_dataset", "source_modality", "source_split", "question_category"):
        if key in extra_env_info:
            print(f"  {key}={extra_env_info[key]}")
    if "reasoning" in extra_env_info:
        print(f"  reasoning={extra_env_info['reasoning']}")
    if sample_modality == "video":
        if "need_to_flip" in extra_env_info:
            print(f"  need_to_flip={extra_env_info['need_to_flip']}")
    for message_idx, message in enumerate(sample["message_log"], start=1):
        role = message.get("role", "unknown")
        summary = _summarize_message_content(message)
        token_ids = summary["token_ids"]
        decoded = tokenizer.decode(token_ids)

        num_video_tokens = decoded.count("<|video_pad|>")
        num_image_tokens = decoded.count("<|image_pad|>")
        num_vision_starts = decoded.count("<|vision_start|>")
        num_vision_ends = decoded.count("<|vision_end|>")
        assistant_target_tokens = len(token_ids) if role == "assistant" else 0

        print(f"\n  [Message {message_idx}] role={role}")
        print(
            "    content="
            f"{summary['content_type']}"
            f" items={summary['num_content_items']}"
            f" image_items={summary['num_image_items']}"
            f" video_items={summary['num_video_items']}"
            f" text_items={summary['num_text_items']}"
        )
        print(
            "    tokens="
            f"{len(token_ids)}"
            f" image_pad={num_image_tokens}"
            f" video_pad={num_video_tokens}"
            f" vision_start={num_vision_starts}"
            f" vision_end={num_vision_ends}"
        )
        if role == "assistant":
            print(
                "    supervision="
                f"assistant_target_tokens={assistant_target_tokens}"
            )

        debug_decode_chars = getattr(_print_debug_sample, "debug_decode_chars", 0)
        if debug_decode_chars != 0:
            preview = decoded if debug_decode_chars < 0 else decoded[:debug_decode_chars]
            if debug_decode_chars > 0 and len(decoded) > debug_decode_chars:
                preview += "...<truncated>"
            print("    decoded_preview:")
            print(f"    {preview}")


# =======================================================
# Data Processing
# =======================================================


# TODO @yukih: move to nemo_rl/data/utils.py after data processor refactored
def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    assert "train" in data_config, (
        "The dataset config structure is updated. Please refer to https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md#datasets "
        "and the Migrate Guide in https://github.com/NVIDIA-NeMo/RL/pull/1649 to update the dataset config."
    )

    print("\n▶ Setting up data...")
    # setup train dataset
    task_data_processors = {}
    data_list = []

    if isinstance(data_config["train"], dict):
        data_config["train"] = [data_config["train"]]

    for cfg in data_config["train"]:
        # load dataset
        if "default" in data_config and data_config["default"] is not None:
            update_single_dataset_config(cfg, data_config["default"])
        data = load_response_dataset(cfg)
        data_list.append(data)
        # bind task_name to task_data_processors
        # For thrive-vlm, add datum_preprocessor to handle video data
        if data.task_name == "thrive-vlm":
            from nemo_rl.data.datasets.response_datasets.thrive_vlm import (
                format_thrive_vlm_dataset,
            )

            data_processor = partial(
                data.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
                datum_preprocessor=partial(format_thrive_vlm_dataset, return_pil=True),
            )
        else:
            data_processor = partial(
                data.processor,
                add_bos=data_config["add_bos"],
                add_eos=data_config["add_eos"],
                add_generation_prompt=data_config["add_generation_prompt"],
            )
        task_data_processors[data.task_name] = (data.task_spec, data_processor)

    merged_data = _merge_raw_datasets([data.dataset for data in data_list])
    dataset = AllTaskProcessedDataset(
        merged_data,
        tokenizer,
        None,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    print(f"  ✓ Training dataset loaded with {len(dataset)} samples.")

    # Debug: inspect the first processed sample with concise multimodal logging.
    debug_max_samples = int(data_config.get("debug_max_samples", 0) or 0)
    debug_decode_chars = int(data_config.get("debug_decode_chars", 400) or 0)
    debug_one_sample_per_modality = bool(
        data_config.get("debug_one_sample_per_modality", False)
    )
    debug_one_sample_per_task_type = bool(
        data_config.get("debug_one_sample_per_task_type", False)
    )
    did_debug_print = False
    debug_completed = False
    if debug_max_samples > 0 or debug_one_sample_per_modality or debug_one_sample_per_task_type:
        print("\n" + "=" * 100)
        print("DEBUG: Tokenized sample inspection")
        print("=" * 100)
        _print_debug_sample.debug_decode_chars = debug_decode_chars

        if debug_one_sample_per_task_type:
            seen_task_types = set()
            for sample_idx, sample in enumerate(dataset):
                if sample_idx > 0 and sample_idx % 250 == 0:
                    print(
                        "DEBUG: scanned "
                        f"{sample_idx} samples for task types; found={len(seen_task_types)}",
                        flush=True,
                    )
                extra_env_info = sample.get("extra_env_info") or {}
                task_type = extra_env_info.get("source_dataset")
                if not task_type or task_type in seen_task_types:
                    continue
                _print_debug_sample(sample_idx, sample, tokenizer)
                did_debug_print = True
                seen_task_types.add(task_type)
            debug_completed = did_debug_print
            print(
                "DEBUG: task-type scan complete; "
                f"found {len(seen_task_types)} unique source_dataset values."
            )
        elif debug_one_sample_per_modality:
            remaining_modalities = {"text", "image", "video"}
            for sample_idx, sample in enumerate(dataset):
                if sample_idx > 0 and sample_idx % 250 == 0:
                    found_modalities = sorted({"text", "image", "video"} - remaining_modalities)
                    missing_modalities = sorted(remaining_modalities)
                    print(
                        "DEBUG: scanned "
                        f"{sample_idx} samples; "
                        f"found={found_modalities}; missing={missing_modalities}",
                        flush=True,
                    )
                sample_modality = _classify_sample_modality(sample)
                if sample_modality not in remaining_modalities:
                    continue

                _print_debug_sample(sample_idx, sample, tokenizer)
                did_debug_print = True
                remaining_modalities.remove(sample_modality)
                if not remaining_modalities:
                    debug_completed = True
                    break

            if remaining_modalities:
                print(
                    "\nMissing modalities from debug scan: "
                    f"{sorted(remaining_modalities)}"
                )
        else:
            for sample_idx, sample in enumerate(dataset):
                if sample_idx >= debug_max_samples:
                    break
                _print_debug_sample(sample_idx, sample, tokenizer)
                did_debug_print = True
            debug_completed = did_debug_print

        print("\n" + "=" * 100 + "\n")

    if (
        did_debug_print
        and debug_completed
        and bool(data_config.get("debug_stop_after_samples", False))
    ):
        print("DEBUG: Stopping after sample inspection as requested.")
        sys.exit(0)


    # setup validation dataset
    val_task_data_processors = {}
    val_data_list = []

    # validation dataset from train dataset (when train dataset's split_validation_size > 0)
    for data in data_list:
        if hasattr(data, "val_dataset") and data.val_dataset is not None:
            val_data_list.append(data.val_dataset)
            # bind task_name to task_data_processors
            task_name = data.task_name
            val_task_data_processors[task_name] = task_data_processors[task_name]

    # validation dataset from config
    if "validation" in data_config and data_config["validation"] is not None:
        if isinstance(data_config["validation"], dict):
            data_config["validation"] = [data_config["validation"]]

        for cfg in data_config["validation"]:
            # load dataset
            if "default" in data_config and data_config["default"] is not None:
                update_single_dataset_config(cfg, data_config["default"])
            val_data = load_response_dataset(cfg)
            val_data_list.append(val_data.dataset)
            # bind task_name to task_data_processors
            # For thrive-vlm, add datum_preprocessor to handle video data
            if val_data.task_name == "thrive-vlm":
                from nemo_rl.data.datasets.response_datasets.thrive_vlm import (
                    format_thrive_vlm_dataset,
                )

                val_data_processor = partial(
                    val_data.processor,
                    add_bos=data_config["add_bos"],
                    add_eos=data_config["add_eos"],
                    add_generation_prompt=data_config["add_generation_prompt"],
                    datum_preprocessor=partial(format_thrive_vlm_dataset, return_pil=True),
                )
            else:
                val_data_processor = partial(
                    val_data.processor,
                    add_bos=data_config["add_bos"],
                    add_eos=data_config["add_eos"],
                    add_generation_prompt=data_config["add_generation_prompt"],
                )
            val_task_data_processors[val_data.task_name] = (
                val_data.task_spec,
                val_data_processor,
            )

    val_dataset = None
    if len(val_data_list) > 0:
        merged_val_data = _merge_raw_datasets(val_data_list)
        val_dataset = AllTaskProcessedDataset(
            merged_val_data,
            tokenizer,
            None,
            val_task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
        print(f"  ✓ Validation dataset loaded with {len(val_dataset)} samples.")

    return dataset, val_dataset


def main(is_vlm: bool = False):
    """Main entry point."""
    _configure_streaming_output()
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    tokenizer_name = config["policy"]["tokenizer"]["name"]
    model_name = config["policy"]["model_name"]
    if not is_vlm:
        lower_names = f"{tokenizer_name} {model_name}".lower()
        if "vl" in lower_names or "conditionalgeneration" in lower_names:
            print(
                "⚠️ Detected a likely VLM model from the config; enabling processor-based tokenization automatically. "
                "You can also run this config via examples/run_vlm_sft.py."
            )
            is_vlm = True

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        if args.ignore_previous:
            base_checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fresh_checkpoint_dir = f"{base_checkpoint_dir}__fresh_{timestamp}"
            config["checkpointing"]["checkpoint_dir"] = fresh_checkpoint_dir
            print(
                "⏭️ Ignoring previous checkpoints. "
                f"Using fresh checkpoint directory: {fresh_checkpoint_dir}"
            )
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer (or processor)
    tokenizer = get_tokenizer(config["policy"]["tokenizer"], get_processor=is_vlm)

    # setup data
    dataset, val_dataset = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()
