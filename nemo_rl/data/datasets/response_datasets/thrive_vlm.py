# Dataset loader for THRIVE-VLM multimodal SFT datasets.
#
# Supports three modalities:
#   - Video: samples with 'video_frames' (list of frame paths) or 'video' field
#   - Image: samples with 'image' or 'images' field
#   - Text-only: no video/image fields
#
# Frame flipping:
#   Many video samples have need_to_flip=True (camera was mounted mirror-side).
#   Horizontal flip is applied via ImageOps.mirror() at load time for all three
#   modalities (video_frames branch: line ~50, video branch: line ~75,
#   image branch: line ~130). Verified 2026-03-30: all 7205 mcqa_video_2403
#   train samples have need_to_flip=True and are correctly mirrored before
#   being passed to the model processor.
#
# Message format:
#   Multimodal content (image/video items) is prepended to the user message
#   content list. If 'messages' is already in the example (standard HF format),
#   it is used directly with multimodal content injected into the user turn.

from typing import Any, Optional
import re

from datasets import load_from_disk
from PIL import Image, ImageOps
from transformers.video_utils import VideoMetadata

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.response_datasets.thrive_vlm_common import (
    filter_dataset_missing_local_media,
    load_rgb_image,
    normalize_thrive_media_paths,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY


def _normalize_video_frame_geometry(frames: list[Image.Image]) -> list[Image.Image]:
    """Resize variable-size PIL frames to a common geometry.

    Some promoted video rows contain frame lists with small within-video size
    drift (for example one frame cropped a few pixels taller than the rest).
    Qwen's video processor stacks frames into a single array, so mixed shapes
    crash before tokenization. Keep frame order and timing unchanged and only
    resize when necessary.
    """
    if not frames:
        return frames

    base_size = frames[0].size
    if all(frame.size == base_size for frame in frames):
        return frames

    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    return [
        frame if frame.size == base_size else frame.resize(base_size, resample=resampling)
        for frame in frames
    ]


def format_thrive_vlm_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the THRIVE-VLM dataset into an OpenAI-API-like message log.

    Supports datasets with different modalities:
    - Video samples (with 'video' or 'video_frames' fields)
    - Image samples (with 'image' or 'images' fields)
    - Text-only samples (no video or image fields)

    Args:
        example: Dataset example containing video/image/text data
        return_pil: If True, return raw video frames/images (PIL Images)
                   If False, use video/image path/URL directly

    Returns:
        Formatted message log dictionary
    """

    example = normalize_thrive_media_paths(example)
    extra_env_info = dict(example.get("extra_env_info") or {})
    for key in (
        "source_dataset",
        "source_modality",
        "source_split",
        "question_category",
        "exercise_code",
        "exercise_name",
        "body_region",
        "variant",
        "session_mode",
        "pain_bucket",
    ):
        value = example.get(key)
        if value not in (None, "", [], {}):
            extra_env_info[key] = value

    # Collect multimodal content
    multimodal_content = []

    video_frames_value = example.get("video_frames")
    if video_frames_value in (None, "", [], {}):
        video_frames_value = None

    video_value = example.get("video")
    if video_value in (None, "", [], {}):
        video_value = None

    if video_frames_value is not None:
        # Keep frame lists grouped as a single video item. This matches the
        # Qwen3.5-VL processor/model expectation that video tensors and video
        # token slots stay aligned per sample instead of being expanded into a
        # sequence of unrelated image items.
        video_value = video_frames_value

    # Handle video content (native path/URL or preloaded frames stored in "video")
    if video_value is not None:
        extra_env_info["need_to_flip"] = bool(example.get("need_to_flip", False))
        # If video_value is a list of strings (frame paths), load them as PIL Images
        if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
            if isinstance(video_value[0], str) and return_pil:
                # Load frame paths into PIL Images
                video_value = [
                    load_rgb_image(
                        frame_path,
                        example,
                        "video_frames" if video_frames_value is not None else "video",
                    )
                    for frame_path in video_value
                ]

                # Apply horizontal flip if requested in dataset
                if example.get("need_to_flip", False):
                    video_value = [ImageOps.mirror(frame) for frame in video_value]

                # Qwen video preprocessing expects all frames in a sample to
                # share the same spatial size.
                video_value = _normalize_video_frame_geometry(video_value)

        video_content = {
            "type": "video",
            "video": video_value,  # Path, URL, or PIL frames - no encoding needed
        }

        # Add video metadata
        if example.get('fps') is None and 'dataset_type' in example:
            print(example['dataset_type'])
        if "fps" in example or "sample_fps" in example:
            fps_value = example.get("fps", example.get("sample_fps", 10.0))
            if isinstance(fps_value, str):
                fps_value = float(fps_value)
        else:
            fps_value = 10.0

        if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
            # Get frame dimensions from first frame
            first_frame = video_value[0]
            if isinstance(first_frame, str):
                temp_frame = load_rgb_image(
                    first_frame,
                    example,
                    "video_frames" if video_frames_value is not None else "video",
                )
                width, height = temp_frame.size
            elif hasattr(first_frame, 'size'):  # PIL Image
                width, height = first_frame.size
            else:
                height, width = first_frame.shape[-2:]

            frame_indices = example.get("frame_indices")
            if not isinstance(frame_indices, list) or len(frame_indices) != len(video_value):
                frame_indices = list(range(len(video_value)))
            total_num_frames = example.get("num_frames")
            if total_num_frames in (None, "", [], {}):
                total_num_frames = len(video_value)

            video_metadata = VideoMetadata(
                total_num_frames=int(total_num_frames),
                fps=fps_value,
                width=width,
                height=height,
                frames_indices=[int(idx) for idx in frame_indices],
            )
            video_content["video_metadata"] = video_metadata

        if "max_pixels" in example:
            video_content["max_pixels"] = int(example["max_pixels"])
        if "min_pixels" in example:
            video_content["min_pixels"] = int(example["min_pixels"])

        multimodal_content.append(video_content)
    # Handle image content (check both "image" and "images" field names) if no video
    else:
        image_value = example.get("image")
        if image_value in (None, "", [], {}):
            image_value = example.get("images")
        if image_value in (None, "", [], {}):
            image_value = None

        if image_value is not None:
            # Ensure images is always a list
            if not isinstance(image_value, (list, tuple)):
                image_value = [image_value]

            # If image_value is a list of strings (image paths), load them as PIL Images
            if len(image_value) > 0 and isinstance(image_value[0], str) and return_pil:
                image_value = [
                    load_rgb_image(img_path, example, "image")
                    for img_path in image_value
                ]

                # Apply horizontal flip if requested in dataset
                if example.get("need_to_flip", False):
                    image_value = [ImageOps.mirror(img) for img in image_value]

            # Add each image as separate content item
            for img in image_value:
                image_content = {
                    "type": "image",
                    "image": img,  # PIL Image or path - no encoding needed
                }

                # Add pixel constraints if present
                if "max_pixels" in example:
                    image_content["max_pixels"] = int(example["max_pixels"])
                if "min_pixels" in example:
                    image_content["min_pixels"] = int(example["min_pixels"])

                multimodal_content.append(image_content)

    # If messages are already formatted in the dataset, use them directly
    if "messages" in example:
        # Find the user message and inject the multimodal content (if any)
        messages = example["messages"]
        for msg in messages:
            if msg["role"] == "user":
                # Some upstream THRIVE rows carry an empty image_url alongside a
                # text item. Strip it before injecting media so the chat template
                # sees only the actual multimodal payload we provide here.
                if isinstance(msg.get("content"), list):
                    cleaned_content = []
                    for item in msg["content"]:
                        if not isinstance(item, dict):
                            cleaned_content.append(item)
                            continue
                        cleaned_item = item.copy()
                        if cleaned_item.get("type") == "text":
                            image_url = cleaned_item.get("image_url")
                            if image_url == {"url": ""} or image_url == "":
                                cleaned_item.pop("image_url", None)
                        cleaned_content.append(cleaned_item)
                    msg["content"] = cleaned_content
                # If content is a string, convert to list format
                if isinstance(msg["content"], str):
                    # Prepend multimodal content (videos/images) before text
                    msg["content"] = multimodal_content + [
                        {"type": "text", "text": msg["content"]}
                    ]
                # If content is already a list, prepend multimodal content
                elif isinstance(msg["content"], list):
                    msg["content"] = multimodal_content + msg["content"]
                break

        ret = {
            "messages": messages,
            "task_name": example.get("task_name", "thrive-vlm"),
            "extra_env_info": extra_env_info or None,
        }
    else:
        # Original format: build messages from question/answer fields
        # Add multimodal content first, then text
        user_content = multimodal_content + [
            {
                "type": "text",
                "text": str(example["question"]),
            },
        ]

        assistant_content = str(example["answer"]).strip()

        ret = {
            "messages": [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ],
            "task_name": "thrive-vlm",
            "extra_env_info": extra_env_info or None,
        }

    assistant_has_reasoning = False
    for message in ret["messages"]:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            assistant_has_reasoning = bool(
                re.search(r"<(?:think|thinking)>", content, re.IGNORECASE)
            )
        elif isinstance(content, list):
            text_parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            ]
            assistant_has_reasoning = bool(
                re.search(r"<(?:think|thinking)>", "".join(text_parts), re.IGNORECASE)
            )
        if assistant_has_reasoning:
            break

    merged_extra_env_info = dict(ret.get("extra_env_info") or {})
    merged_extra_env_info["reasoning"] = assistant_has_reasoning
    ret["extra_env_info"] = merged_extra_env_info

    return ret


def prepare_thrive_vlm_dataset(
    split: str = "train",
    dataset_name: str = "thrive-vlm",
    task_name: Optional[str] = None,
    split_validation_size: float = 0.05,
    seed: int = 42,
    skip_missing_local_media_filter: bool = False,
):
    """Prepare THRIVE-VLM dataset for training.

    Args:
        split: Dataset split to load (train or validation)
        dataset_name: Local path to dataset directory (saved with save_to_disk)
        task_name: Optional task name override
        split_validation_size: Validation fraction to derive from train when
            the dataset root does not provide a native validation split
        seed: Seed for deterministic train/validation splitting

    Returns:
        Dictionary with train and validation splits
    """
    if task_name is None:
        task_name = "thrive-vlm"

    # Load dataset from disk using load_from_disk
    raw = load_from_disk(dataset_name)

    # GUARD: a VIDEO dataset without a need_to_flip column would silently train
    # on MIRRORED frames (format_thrive_vlm_dataset's `.get("need_to_flip",
    # False)` skips the un-mirroring flip) while eval flips — inverting every
    # left/right judgment the model learns. All Thrive/SWORD video is stored
    # mirrored, so the column is REQUIRED on every video dataset; fail at
    # setup, before any GPU time. Root-caused 2026-07-07 on the EXP-B stage-2
    # family (61% which-leg inversion at eval; audit report
    # 2026-07-07_expb_stage2_lr_inversion_audit.md).
    _splits_to_check = (
        raw.values() if hasattr(raw, "keys") and callable(raw.keys) else [raw]
    )
    for _ds in _splits_to_check:
        cols = set(_ds.column_names)
        if ({"video_frames", "video"} & cols) and "need_to_flip" not in cols:
            raise ValueError(
                f"Video dataset '{dataset_name}' has {sorted({'video_frames', 'video'} & cols)} "
                f"but NO 'need_to_flip' column — refusing to train: the loader would "
                f"silently skip the un-mirroring flip and the model would learn "
                f"inverted left/right. Backfill the column (all Thrive video needs "
                f"need_to_flip=True unless frames were pre-flipped) or fix the "
                f"builder to carry it from the source split."
            )

    # Check if raw is a DatasetDict or a single Dataset
    if hasattr(raw, 'keys') and callable(raw.keys):
        # It's a DatasetDict with splits
        native_val_dataset = raw.get("validation", raw.get("val"))
        if native_val_dataset is not None:
            if split == "train":
                train_dataset = raw["train"]
                val_dataset = native_val_dataset
            else:
                train_dataset = native_val_dataset
                val_dataset = native_val_dataset
        else:
            if "train" not in raw:
                raise ValueError(f"Split '{split}' not found. Available: {list(raw.keys())}")
            if split_validation_size <= 0:
                derived_train = raw["train"]
                derived_val = raw["train"]
            else:
                split_dataset = raw["train"].train_test_split(
                    test_size=split_validation_size, seed=seed
                )
                derived_train = split_dataset["train"]
                derived_val = split_dataset["test"]

            if split == "train":
                train_dataset = derived_train
                val_dataset = derived_val
            else:
                train_dataset = derived_val
                val_dataset = derived_val
    else:
        # It's a single Dataset, use it for both train and validation
        if split_validation_size > 0:
            split_dataset = raw.train_test_split(test_size=split_validation_size, seed=seed)
            derived_train = split_dataset["train"]
            derived_val = split_dataset["test"]
            if split == "train":
                train_dataset = derived_train
                val_dataset = derived_val
            else:
                train_dataset = derived_val
                val_dataset = derived_val
        else:
            train_dataset = raw
            val_dataset = raw

    train_dataset = filter_dataset_missing_local_media(
        train_dataset,
        "train",
        "THRIVE-VLM SFT",
        skip_missing_local_media_filter=skip_missing_local_media_filter,
    )
    val_split_name = "validation" if split == "train" else split
    val_dataset = filter_dataset_missing_local_media(
        val_dataset,
        val_split_name,
        "THRIVE-VLM SFT",
        skip_missing_local_media_filter=skip_missing_local_media_filter,
    )

    # Format - disable features to avoid schema conflicts
    train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class ThriveVLMDataset:
    """Dataset class for THRIVE-VLM video question answering.

    This dataset handles video inputs with question-answer pairs for VLM training.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        data_path: Optional[str] = None,
        split_validation_size: float = 0.05,
        seed: int = 42,
        skip_missing_local_media_filter: bool = False,
        **kwargs,  # Accept other params from config
    ):
        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'validation'."
            )
        self.task_name = "thrive-vlm"

        # Use data_path if provided, otherwise use dataset_name as the path
        path = data_path if data_path is not None else dataset_name

        self.formatted_ds = prepare_thrive_vlm_dataset(
            split=split,
            task_name=self.task_name,
            dataset_name=path,
            split_validation_size=split_validation_size,
            seed=seed,
            skip_missing_local_media_filter=skip_missing_local_media_filter,
        )

        self.task_spec = TaskDataSpec(
            task_name="thrive-vlm",
        )

        # Initialize these to None, will be set by set_task_spec and set_processor
        self.data_config = None
        self.processor = None
        self.val_dataset = self.formatted_ds.get("validation")
        # split_validation_size=0 means "no derived validation": prepare_thrive_vlm_dataset
        # then returns the FULL train set as the "validation" view. Exposing that as
        # val_dataset makes run_sft.py concatenate the whole train set into validation
        # alongside any explicit data.validation entry, contaminating val_loss (which
        # checkpointing keep_top_k selects on). Configs that set 0 always provide an
        # explicit validation dataset, so drop the train-copy here.
        if split == "train" and float(split_validation_size or 0) <= 0:
            self.val_dataset = None

    @property
    def dataset(self):
        """Expose the training dataset."""
        return self.formatted_ds["train"]

    def set_task_spec(
        self, data_config: ResponseDatasetConfig
    ):
        """Set task specification from data config.

        Args:
            data_config: Configuration dict containing system_prompt_file and prompt_file
        """
        self.data_config = data_config
        system_prompt_file = self.data_config.get("system_prompt_file", None)
        prompt_file = self.data_config.get("prompt_file", None)
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

    def set_processor(self):
        """Set the data processor to sft_processor.

        For Thrive VLM, we use sft_processor which will be configured
        with format_thrive_vlm_dataset as datum_preprocessor in run_sft.py.
        """
        from nemo_rl.data.processors import sft_processor

        self.processor = sft_processor
