#!/usr/bin/env python3
"""
Verify and optionally fix SFT config schedule parameters based on actual dataset size.

Supports two config styles:
1. Legacy HF dataset roots via data.dataset_name with train/validation subdirs
2. Multi-dataset configs via data.train / data.validation, including JSONL datasets

Usage:
    python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_4b_4epoch_task1_original.yaml
    python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml
    python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml --fix
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml_preserving_structure(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def deep_merge_dicts(base: Any, override: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    merged = dict(base)
    for key, value in override.items():
        if key in merged:
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_default_config_path(config_path: Path, entry: str) -> Path | None:
    if entry == "_self_":
        return None

    candidate = Path(entry)
    if not candidate.suffix:
        candidate = candidate.with_suffix('.yaml')
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate


def resolve_cli_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    search_roots = [
        Path.cwd(),
        repo_root,
        repo_root / "examples" / "configs",
    ]

    attempted: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)

        direct_candidate = resolved_root / candidate
        attempted.append(direct_candidate)
        if direct_candidate.exists():
            return direct_candidate.resolve()

        if candidate.name != str(candidate):
            continue

        matched = sorted(resolved_root.rglob(candidate.name))
        if len(matched) == 1:
            return matched[0].resolve()
        if len(matched) > 1:
            match_list = "\n".join(f"  - {path}" for path in matched[:10])
            raise FileNotFoundError(
                f"Config name '{config_arg}' is ambiguous. Matching files:\n{match_list}"
            )

    attempted_list = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        f"Config file not found: {config_arg}\nSearched:\n{attempted_list}"
    )


def load_composed_config(config_path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    seen = seen or set()
    if config_path in seen:
        raise ValueError(f"Recursive config defaults detected at {config_path}")
    seen = set(seen)
    seen.add(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    defaults = raw_config.get("defaults") or []
    if isinstance(defaults, (str, dict)):
        defaults = [defaults]
    composed: dict[str, Any] = {}

    for entry in defaults:
        if isinstance(entry, str):
            default_path = resolve_default_config_path(config_path, entry)
            if default_path is None:
                continue
            composed = deep_merge_dicts(composed, load_composed_config(default_path, seen))
        elif isinstance(entry, dict):
            for value in entry.values():
                if isinstance(value, str):
                    default_path = resolve_default_config_path(config_path, value)
                    if default_path is None:
                        continue
                    composed = deep_merge_dicts(composed, load_composed_config(default_path, seen))

    raw_config.pop("defaults", None)
    return deep_merge_dicts(composed, raw_config)


def _load_from_disk(path: str):
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' package is required to inspect HuggingFace disk datasets. "
            "Install it in this environment or use a JSONL-based config."
        ) from e
    return load_from_disk(path)


def count_jsonl_rows(path: str) -> int:
    row_count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row_count += 1
    return row_count


def get_dataset_size(path: str) -> int:
    ds = _load_from_disk(path)
    return len(ds)


def get_hf_split_size(dataset_root: str, split: str) -> int:
    split_path = Path(dataset_root) / split
    ds = _load_from_disk(str(split_path))
    return len(ds)


def verify_hf_split_ratio(dataset_root: str, expected_ratio: float = 0.9):
    try:
        train_size = get_hf_split_size(dataset_root, "train")
        val_size = get_hf_split_size(dataset_root, "validation")
        total_size = train_size + val_size
        actual_ratio = train_size / total_size
        if abs(actual_ratio - expected_ratio) > 0.01:
            return (
                False,
                train_size,
                val_size,
                actual_ratio,
                f"❌ CRITICAL: Dataset split ratio is {actual_ratio:.3f}, expected {expected_ratio:.2f}",
            )
        return (
            True,
            train_size,
            val_size,
            actual_ratio,
            f"✅ Dataset split ratio is correct: {actual_ratio:.3f}",
        )
    except Exception as e:
        return (False, None, None, None, f"ERROR: Could not verify split ratio: {e}")


def calculate_steps(num_samples: int, batch_size: int, num_epochs: int):
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = (total_steps + 9) // 10
    return steps_per_epoch, total_steps, warmup_steps


def as_dataset_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError(f"Unsupported dataset config container: {type(value)!r}")


def resolve_config_reference(config: dict[str, Any], value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([^}]+)\}", value.strip())
    if not match:
        return value

    current: Any = config
    for part in match.group(1).split("."):
        if not isinstance(current, dict) or part not in current:
            return value
        current = current[part]
    return current


def inspect_thrive_vlm_entry(entry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = entry.get("dataset_name")
    data_path = entry.get("data_path")
    split = entry.get("split", "train")
    split_validation_size = float(resolve_config_reference(config, entry.get("split_validation_size", 0.05)) or 0)
    seed = int(resolve_config_reference(config, entry.get("seed", 42)) or 42)

    path = str(data_path or dataset_name or "unknown")
    raw = _load_from_disk(path)

    if hasattr(raw, "keys") and callable(raw.keys):
        native_val = raw.get("validation", raw.get("val"))
        if native_val is not None:
            val_name = "validation" if "validation" in raw else "val"
            if split == "train":
                return {
                    "size": len(raw["train"]),
                    "label": f"{path}/train",
                    "implicit_validation": (f"{path}/{val_name}", len(native_val)),
                    "kind": "hf_dataset_dict",
                }
            if split == "validation":
                return {
                    "size": len(native_val),
                    "label": f"{path}/{val_name}",
                    "implicit_validation": None,
                    "kind": "hf_dataset_dict",
                }
            raise ValueError(f"Unsupported thrive-vlm-sft split: {split}")

        if "train" not in raw:
            raise ValueError(f"Split '{split}' not found. Available: {list(raw.keys())}")
        base_dataset = raw["train"]
        base_label = f"{path}/train"
    else:
        base_dataset = raw
        base_label = path

    if split_validation_size > 0:
        derived = base_dataset.train_test_split(test_size=split_validation_size, seed=seed)
        derived_train = derived["train"]
        derived_val = derived["test"]
    else:
        derived_train = base_dataset
        derived_val = base_dataset

    if split == "train":
        return {
            "size": len(derived_train),
            "label": base_label,
            "implicit_validation": (f"{base_label}#derived_validation", len(derived_val)),
            "kind": "hf_dataset_or_split",
        }
    if split == "validation":
        return {
            "size": len(derived_val),
            "label": f"{base_label}#derived_validation",
            "implicit_validation": None,
            "kind": "hf_dataset_or_split",
        }
    raise ValueError(f"Unsupported thrive-vlm-sft split: {split}")


def resolve_dataset_entry_size(entry: dict[str, Any]) -> tuple[int | None, str, str | None]:
    dataset_name = entry.get("dataset_name")
    data_path = entry.get("data_path")
    split = entry.get("split", "train")

    if data_path and str(data_path).endswith(".jsonl"):
        return count_jsonl_rows(str(data_path)), str(data_path), "jsonl"

    if dataset_name == "openai_format" and data_path:
        return count_jsonl_rows(str(data_path)), str(data_path), "jsonl"

    if dataset_name == "thrive-vlm-sft":
        # thrive-vlm-sft needs config context (split_validation_size, seed); callers must route
        # these through inspect_thrive_vlm_entry(entry, config) before reaching this resolver.
        raise ValueError("thrive-vlm-sft entries must be inspected with config context")

    if data_path and Path(str(data_path)).is_dir():
        if (Path(str(data_path)) / "dataset_info.json").exists():
            return get_dataset_size(str(data_path)), str(data_path), "hf_split_dir"
        return get_hf_split_size(str(data_path), str(split)), f"{data_path}/{split}", "hf_split"

    if dataset_name and Path(str(dataset_name)).is_dir():
        return get_hf_split_size(str(dataset_name), str(split)), f"{dataset_name}/{split}", "hf_split"

    return None, str(data_path or dataset_name or "unknown"), None


def inspect_config_datasets(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})

    if data_cfg.get("dataset_name"):
        dataset_root = str(data_cfg["dataset_name"])
        train_size = get_hf_split_size(dataset_root, "train")
        split_ok, split_train_size, val_size, actual_ratio, split_msg = verify_hf_split_ratio(dataset_root)
        return {
            "mode": "hf_root",
            "display_dataset": dataset_root,
            "train_size": train_size,
            "validation_size": val_size,
            "train_entries": [(f"{dataset_root}/train", train_size)],
            "validation_entries": [(f"{dataset_root}/validation", val_size)] if val_size is not None else [],
            "split_ok": split_ok,
            "split_ratio": actual_ratio,
            "split_message": split_msg,
        }

    train_entries_cfg = as_dataset_list(data_cfg.get("train"))
    if not train_entries_cfg:
        raise ValueError("Could not find either data.dataset_name or data.train in config")

    validation_entries_cfg = as_dataset_list(data_cfg.get("validation"))
    train_entries: list[tuple[str, int]] = []
    validation_entries: list[tuple[str, int]] = []

    for entry in train_entries_cfg:
        if entry.get("dataset_name") == "thrive-vlm-sft":
            inspected = inspect_thrive_vlm_entry(entry, config)
            train_entries.append((inspected["label"], inspected["size"]))
            implicit_validation = inspected["implicit_validation"]
            if implicit_validation is not None:
                validation_entries.append(implicit_validation)
            continue
        size, label, kind = resolve_dataset_entry_size(entry)
        if size is None:
            raise ValueError(f"Unsupported train dataset entry: {entry}")
        train_entries.append((label, size))

    for entry in validation_entries_cfg:
        if entry.get("dataset_name") == "thrive-vlm-sft":
            inspected = inspect_thrive_vlm_entry(entry, config)
            validation_entries.append((inspected["label"], inspected["size"]))
            continue
        size, label, kind = resolve_dataset_entry_size(entry)
        if size is None:
            raise ValueError(f"Unsupported validation dataset entry: {entry}")
        validation_entries.append((label, size))

    train_size = sum(size for _, size in train_entries)
    validation_size = sum(size for _, size in validation_entries) if validation_entries else None

    split_msg = (
        f"✅ Explicit dataset lists detected: {len(train_entries)} train source(s), "
        f"{len(validation_entries)} validation source(s)."
    )

    return {
        "mode": "multi_dataset",
        "display_dataset": "multiple datasets via data.train/data.validation",
        "train_size": train_size,
        "validation_size": validation_size,
        "train_entries": train_entries,
        "validation_entries": validation_entries,
        "split_ok": True,
        "split_ratio": None,
        "split_message": split_msg,
    }


def extract_current_values(content: str) -> tuple[int, int, int]:
    save_period_match = re.search(r"^\s*save_period:\s*(\d+)", content, flags=re.MULTILINE)
    lr_decay_match = re.search(r"^\s*lr_decay_iters:\s*(\d+)", content, flags=re.MULTILINE)
    lr_warmup_match = re.search(r"^\s*lr_warmup_iters:\s*(\d+)", content, flags=re.MULTILINE)

    if not (save_period_match and lr_decay_match and lr_warmup_match):
        raise ValueError("Could not find required parameters in config")

    return (
        int(save_period_match.group(1)),
        int(lr_decay_match.group(1)),
        int(lr_warmup_match.group(1)),
    )


def extract_val_period(content: str) -> int | None:
    """val_period lives under `sft:` and is OPTIONAL — some configs (e.g. mcqa_video_*) set a
    small fixed validation cadence (val_period: 25) deliberately decoupled from the epoch length,
    while multimodal-mix configs set val_period == save_period == steps/epoch. Return it if present
    so the caller can WARN (not hard-fail) when it drifted from save_period within an epoch-aligned
    config; None when absent."""
    m = re.search(r"^\s*val_period:\s*(\d+)", content, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def update_numeric_field(content: str, field_name: str, new_value: int) -> str:
    pattern = rf'^(\s*{re.escape(field_name)}:\s*)(\d+)(\s*(#.*)?)$'
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    if not matches:
        raise ValueError(f"Field '{field_name}' not found for update")
    if len(matches) > 1:
        # Refuse to blindly rewrite every occurrence — that would corrupt sibling blocks
        # (e.g. a save_period under both `checkpointing` and some other section).
        lines = [content[:m.start()].count("\n") + 1 for m in matches]
        raise ValueError(
            f"Field '{field_name}' is ambiguous: {len(matches)} occurrences at lines {lines}. "
            "Refusing to auto-fix; resolve manually."
        )
    return re.sub(pattern, rf'\g<1>{new_value}\g<3>', content, count=1, flags=re.MULTILINE)


def verify_config(config_path: str, fix: bool = False, fix_val_period: bool = False) -> bool:
    try:
        config_path = resolve_cli_config_path(config_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return False

    config = load_composed_config(config_path)

    try:
        dataset_info = inspect_config_datasets(config)
    except Exception as e:
        print(f"ERROR: Could not inspect datasets for config: {e}")
        return False

    batch_size = int(config.get("policy", {}).get("train_global_batch_size") or 16)
    num_epochs = int(config.get("sft", {}).get("max_num_epochs") or 4)
    num_samples = int(dataset_info["train_size"])
    steps_per_epoch, total_steps, warmup_steps = calculate_steps(num_samples, batch_size, num_epochs)

    content = load_yaml_preserving_structure(config_path)
    try:
        current_save_period, current_lr_decay, current_lr_warmup = extract_current_values(content)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False

    # save_period validity is a RANGE check, not equality. A periodic cadence (e.g. 500 within a
    # 2556-step epoch) that saves several times mid-run is BETTER than one-save-at-epoch-end, because
    # a mid-run node eviction can otherwise lose the whole run (war story: job 76407 had
    # save_period=2557 > total_steps=2556 → zero checkpoints ever written, ~14h lost). So:
    #   - 0 < save_period <= total_steps  → ✅ valid (any periodic cadence within the run is fine)
    #   - save_period > total_steps (or <= 0) → ❌ the only real error: it can never fire mid-run
    # steps_per_epoch is still shown as the RECOMMENDED value, but it is not the only correct one.
    save_period_ok = 0 < current_save_period <= total_steps

    current_val_period = extract_val_period(content)
    # When is val_period STALE vs DELIBERATELY DECOUPLED?
    #  - A small fixed cadence (e.g. 25 while save_period is 220) is intentional frequent validation
    #    — NOT stale. Don't flag it.
    #  - A val_period that nearly tracks save_period but is off by a little (e.g. 737 vs 736 after a
    #    row-count change) is the stale/drift case we want to catch.
    # Heuristic: flag only when val_period is CLOSE to save_period (within 5% or 5 steps) but not equal
    # AND the schedule itself is epoch-aligned. This catches the off-by-N drift without touching the
    # deliberately-small mcqa_video cadences.
    epoch_aligned = current_save_period == steps_per_epoch
    if current_val_period is None or not epoch_aligned or current_val_period == steps_per_epoch:
        val_period_drifted = False
    else:
        tol = max(5, round(0.05 * steps_per_epoch))
        val_period_drifted = abs(current_val_period - steps_per_epoch) <= tol

    all_correct = (
        save_period_ok
        and current_lr_decay == total_steps
        and current_lr_warmup == warmup_steps
    )

    print(f"\n{'=' * 80}")
    print(f"Config: {config_path.name}")
    print(f"{'=' * 80}")
    print(f"Dataset mode: {dataset_info['mode']}")
    print(f"Dataset source: {dataset_info['display_dataset']}")
    print(f"Train samples: {num_samples}")
    if dataset_info["validation_size"] is not None:
        print(f"Validation samples: {dataset_info['validation_size']}")
    print(f"Train global batch size: {batch_size}")
    print(f"Max epochs: {num_epochs}")

    print("\nTrain dataset breakdown:")
    for label, size in dataset_info["train_entries"]:
        print(f"  - {label}: {size}")
    if dataset_info["validation_entries"]:
        print("\nValidation dataset breakdown:")
        for label, size in dataset_info["validation_entries"]:
            print(f"  - {label}: {size}")

    print("\nDataset Verification:")
    print(f"  {dataset_info['split_message']}")
    if dataset_info["split_ratio"] is not None:
        print(f"  Actual ratio: {dataset_info['split_ratio']:.3f} (expected: 0.90)")

    print(f"\n{'Parameter':<30} | {'Current':<10} | {'Correct':<10} | {'Status'}")
    print(f"{'-' * 30}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    def status_icon(current: int, correct: int) -> str:
        return "✅" if current == correct else "❌"

    # save_period: ✅ for any cadence in (0, total_steps]; the "Correct" column shows the recommended
    # value (steps_per_epoch) as a hint. Only > total_steps (never fires mid-run) is ❌.
    if save_period_ok:
        sp_status = "✅" if current_save_period == steps_per_epoch else f"✅ periodic (rec: {steps_per_epoch})"
        sp_correct = str(steps_per_epoch)
    else:
        sp_status = f"❌ > total_steps={total_steps}: never saves mid-run"
        sp_correct = str(steps_per_epoch)
    print(f"{'save_period':<30} | {current_save_period:<10} | {sp_correct:<10} | {sp_status}")
    print(f"{'lr_decay_iters':<30} | {current_lr_decay:<10} | {total_steps:<10} | {status_icon(current_lr_decay, total_steps)}")
    print(f"{'lr_warmup_iters':<30} | {current_lr_warmup:<10} | {warmup_steps:<10} | {status_icon(current_lr_warmup, warmup_steps)}")
    if current_val_period is not None:
        if val_period_drifted:
            vp_status = "⚠️  drifted"
            vp_correct = str(steps_per_epoch)
        elif current_val_period == steps_per_epoch:
            vp_status = "✅"
            vp_correct = str(steps_per_epoch)
        else:
            # present, not near save_period — a deliberate cadence (e.g. mcqa_video val_period:25)
            vp_status = "ℹ️  intentional cadence"
            vp_correct = "(n/a)"
        print(f"{'val_period':<30} | {current_val_period:<10} | {vp_correct:<10} | {vp_status}")

    def warn_val_period():
        if val_period_drifted:
            print(
                f"\n⚠️  WARNING: val_period={current_val_period} but this is an epoch-aligned config "
                f"(save_period==steps/epoch=={steps_per_epoch}). val_period is likely stale and should "
                f"be {steps_per_epoch}. Re-run with --fix-val-period to update it (it is NOT auto-fixed "
                "by --fix because some configs decouple validation cadence on purpose)."
            )

    if not dataset_info["split_ok"]:
        print("\n❌ CRITICAL ERROR: Dataset split verification failed.")
        return False

    # Optionally repair val_period drift on epoch-aligned configs (opt-in, independent of --fix).
    if val_period_drifted and fix_val_period:
        content = update_numeric_field(content, "val_period", steps_per_epoch)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n🔧 val_period {current_val_period} -> {steps_per_epoch} (epoch-aligned). Saved.")
        current_val_period = steps_per_epoch
        val_period_drifted = False

    if all_correct and dataset_info["split_ok"]:
        print("\n✅ Config is CORRECT!")
        warn_val_period()   # non-blocking: schedule params are right; val_period may still warn
        return True

    if not fix:
        print("\n❌ Config has errors. Run with --fix to correct them.")
        warn_val_period()
        return False

    print("\n🔧 Fixing config...")
    # Only repair save_period when it is INVALID (> total_steps / <= 0). A valid periodic cadence
    # (0 < save_period <= total_steps) is left untouched — clobbering it back to steps_per_epoch
    # would re-introduce the one-save-at-end fragility (job 76407).
    if not save_period_ok:
        content = update_numeric_field(content, "save_period", steps_per_epoch)
        print(f"   save_period {current_save_period} -> {steps_per_epoch} (was > total_steps={total_steps})")
    else:
        print(f"   save_period {current_save_period} left as-is (valid periodic cadence)")
    content = update_numeric_field(content, "lr_decay_iters", total_steps)
    content = update_numeric_field(content, "lr_warmup_iters", warmup_steps)
    if val_period_drifted and fix_val_period:
        content = update_numeric_field(content, "val_period", steps_per_epoch)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Config fixed and saved to {config_path}")
    warn_val_period()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify and fix training config parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify legacy HF-dataset config
  python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_4b_4epoch_task1_original.yaml

  # Verify transcript multi-JSONL config
  python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml

  # Verify and fix config
  python scripts/verify_and_fix_config.py --config examples/configs/sft_vlm_qwen35_4b_patient_qa_postsession_transcripts_megatron.yaml --fix
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--fix", action="store_true", help="Fix save_period/lr_decay_iters/lr_warmup_iters if wrong")
    parser.add_argument(
        "--fix-val-period",
        action="store_true",
        help="Also fix val_period when it drifted on an epoch-aligned config (not done by --fix alone, "
        "since some configs decouple validation cadence intentionally)",
    )

    args = parser.parse_args()
    success = verify_config(args.config, fix=args.fix, fix_val_period=args.fix_val_period)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
