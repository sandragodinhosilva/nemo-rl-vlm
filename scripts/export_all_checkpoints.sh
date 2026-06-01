#!/bin/bash
# Export all Megatron SFT checkpoints in a directory to Hugging Face format.
#
# Usage:
#   bash scripts/export_all_checkpoints.sh <checkpoint_dir> [output_base_dir] [model_name_prefix]
#
# Optional env vars:
#   PYTHON_BIN=<python>                Override Python executable
#   EXPORT_OVERWRITE=1                 Re-export even when a valid HF dir exists
#   EXPORT_CLEAN_PARTIAL=1             Delete incomplete HF dirs and retry export
#
# Example:
#   bash scripts/export_all_checkpoints.sh #     /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts #     /mnt/data/sgsilva/models #     qwen35-4b-patient-qa-postsession-transcripts

set -euo pipefail

CHECKPOINT_DIR="${1:-}"
OUTPUT_BASE_DIR="${2:-/mnt/data/sgsilva/models}"
MODEL_NAME_PREFIX="${3:-}"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Missing required arguments"
    echo ""
    echo "Usage:"
    echo "  $0 <checkpoint_dir> <output_base_dir> [model_name_prefix]"
    echo ""
    echo "Example:"
    echo "  $0 /mnt/data/sgsilva/checkpoints/sft_qwen35_4b_patient_qa_postsession_transcripts /mnt/data/sgsilva/models qwen35-4b-patient-qa-postsession-transcripts"
    exit 1
fi

resolve_checkpoint_dir() {
    local requested="$1"
    local -a roots=(
        "/mnt/data/sgsilva/checkpoints"
        "/home/sgsilva/checkpoints"
    )
    local requested_base
    requested_base="$(basename "$requested")"
    local -a candidates=()

    for root in "${roots[@]}"; do
        [ -d "$root" ] || continue

        while IFS= read -r path; do
            [ -n "$path" ] || continue
            candidates+=("$path")
        done < <(find "$root" -maxdepth 1 -mindepth 1 -type d \
            \( -name "$requested_base" -o -name "${requested_base}__fresh_*" \) 2>/dev/null | sort -V)
    done

    if [ "${#candidates[@]}" -eq 1 ]; then
        echo "${candidates[0]}"
        return 0
    fi

    if [ "${#candidates[@]}" -gt 1 ]; then
        echo "MULTIPLE"
        printf '%s\n' "${candidates[@]}"
        return 0
    fi

    return 1
}

if [ ! -d "$CHECKPOINT_DIR" ]; then
    RESOLVED_RESULT="$(resolve_checkpoint_dir "$CHECKPOINT_DIR" || true)"
    if [ -n "$RESOLVED_RESULT" ]; then
        if [[ "$RESOLVED_RESULT" == MULTIPLE$'\n'* ]] || [ "$RESOLVED_RESULT" = "MULTIPLE" ]; then
            echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
            echo "Found multiple likely checkpoint roots under common results directories:"
            echo "$RESOLVED_RESULT" | tail -n +2 | sed 's/^/  - /'
            echo "Pass one of the resolved paths explicitly."
            exit 1
        fi

        echo "INFO: Checkpoint directory not found at requested path."
        echo "INFO: Resolved to likely materialized run directory: $RESOLVED_RESULT"
        CHECKPOINT_DIR="$RESOLVED_RESULT"
    else
        echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
        echo "Hint: many NeMo SFT configs write checkpoints under:"
        echo "  /mnt/data/sgsilva/checkpoints/<checkpoint_dir>__fresh_<timestamp>"
        exit 1
    fi
fi

mkdir -p "$OUTPUT_BASE_DIR"

CHECKPOINT_DIR="$(cd "$CHECKPOINT_DIR" && pwd)"
OUTPUT_BASE_DIR="$(cd "$OUTPUT_BASE_DIR" && pwd)"

if [ -z "$MODEL_NAME_PREFIX" ]; then
    MODEL_NAME_PREFIX="$(basename "$CHECKPOINT_DIR")"
fi

STEP_DIRS=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'step_*' | sort -V)

if [ -z "$STEP_DIRS" ]; then
    echo "ERROR: No checkpoint directories found in $CHECKPOINT_DIR"
    echo "Looking for directories matching: step_*"
    exit 1
fi

TOTAL_CKPTS=$(echo "$STEP_DIRS" | wc -l)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_HOME_PYTHON="/home/sgsilva/nemo-rl-vlm-home-venv/bin/python"
DEFAULT_HOME_PROJECT_PYTHON="/home/sgsilva/nemo-rl-vlm/.venv/bin/python"
DEFAULT_PROJECT_PYTHON="$PROJECT_DIR/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
EXPORT_OVERWRITE="${EXPORT_OVERWRITE:-0}"
EXPORT_CLEAN_PARTIAL="${EXPORT_CLEAN_PARTIAL:-0}"

if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$DEFAULT_HOME_PROJECT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_HOME_PROJECT_PYTHON"
    elif [ -x "$DEFAULT_HOME_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_HOME_PYTHON"
    else
        PYTHON_BIN="$DEFAULT_PROJECT_PYTHON"
    fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Expected python executable not found: $PYTHON_BIN"
    echo "Set PYTHON_BIN explicitly or ensure the /home serving env exists."
    exit 1
fi

echo "========================================"
echo "AUTOMATIC CHECKPOINT EXPORT"
echo "========================================"
echo "  Checkpoint dir:     $CHECKPOINT_DIR"
echo "  Output base dir:    $OUTPUT_BASE_DIR"
echo "  Model name prefix:  $MODEL_NAME_PREFIX"
echo "  Python:             $PYTHON_BIN"
echo "  Overwrite exports:  $EXPORT_OVERWRITE"
echo "  Clean partial dirs: $EXPORT_CLEAN_PARTIAL"
echo "========================================"
echo ""
echo "Found $TOTAL_CKPTS checkpoint(s) to export:"
echo "$STEP_DIRS" | while read -r dir; do
    [ -n "$dir" ] || continue
    echo "  - $(basename "$dir")"
done
echo ""

CURRENT=0
FAILED=0
SUCCESS=0
SKIPPED=0

for STEP_DIR in $STEP_DIRS; do
    CURRENT=$((CURRENT + 1))
    STEP_NAME=$(basename "$STEP_DIR")
    STEP_NUM=${STEP_NAME#step_}

    CONFIG_PATH="$STEP_DIR/config.yaml"
    MEGATRON_CKPT_PATH="$STEP_DIR/policy/weights/iter_0000000"
    HF_OUTPUT_PATH="$OUTPUT_BASE_DIR/${MODEL_NAME_PREFIX}-step${STEP_NUM}"

    echo "========================================"
    echo "[$CURRENT/$TOTAL_CKPTS] Exporting $STEP_NAME"
    echo "========================================"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "  ERROR: Config not found: $CONFIG_PATH"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    if [ ! -d "$MEGATRON_CKPT_PATH" ]; then
        echo "  ERROR: Megatron checkpoint not found: $MEGATRON_CKPT_PATH"
        FAILED=$((FAILED + 1))
        echo ""
        continue
    fi

    if [ -d "$HF_OUTPUT_PATH" ]; then
        if [ -f "$HF_OUTPUT_PATH/config.json" ] && [ "$EXPORT_OVERWRITE" != "1" ]; then
            echo "  Already exported: $HF_OUTPUT_PATH"
            SKIPPED=$((SKIPPED + 1))
            echo ""
            continue
        fi

        if [ ! -f "$HF_OUTPUT_PATH/config.json" ] && [ "$EXPORT_CLEAN_PARTIAL" = "1" ]; then
            echo "  Removing incomplete export directory: $HF_OUTPUT_PATH"
            rm -rf "$HF_OUTPUT_PATH"
        elif [ "$EXPORT_OVERWRITE" = "1" ]; then
            echo "  Removing existing export directory due to EXPORT_OVERWRITE=1: $HF_OUTPUT_PATH"
            rm -rf "$HF_OUTPUT_PATH"
        else
            echo "  Existing export directory blocks retry: $HF_OUTPUT_PATH"
            echo "  Set EXPORT_CLEAN_PARTIAL=1 to auto-remove incomplete exports"
            echo "  or EXPORT_OVERWRITE=1 to force a full re-export."
            FAILED=$((FAILED + 1))
            echo ""
            continue
        fi
    fi

    echo "  Config:           $CONFIG_PATH"
    echo "  Megatron ckpt:    $MEGATRON_CKPT_PATH"
    echo "  HF output:        $HF_OUTPUT_PATH"
    echo ""
    echo "  Starting export..."

    cd "$PROJECT_DIR"
    export_log="/tmp/export_${STEP_NAME}.log"
    if \
        PYTHONPATH="3rdparty/Megatron-LM-workspace/Megatron-LM:${PYTHONPATH:-}" \
        "$PYTHON_BIN" \
            examples/converters/convert_megatron_to_hf.py \
            --config "$CONFIG_PATH" \
            --megatron-ckpt-path "$MEGATRON_CKPT_PATH" \
            --hf-ckpt-path "$HF_OUTPUT_PATH" \
            2>&1 | tee "$export_log"
    then

        echo ""
        if [ -f "$HF_OUTPUT_PATH/config.json" ]; then
            MODEL_SIZE=$(du -sh "$HF_OUTPUT_PATH" | cut -f1)
            echo "  Export successful"
            echo "  Verification: config.json found"
            echo "  Model size: $MODEL_SIZE"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  WARNING: export completed but config.json not found"
            FAILED=$((FAILED + 1))
        fi
    else
        echo ""
        echo "  Export FAILED"
        echo "  Log saved to: $export_log"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "========================================"
echo "EXPORT SUMMARY"
echo "========================================"
echo "  Total checkpoints:    $TOTAL_CKPTS"
echo "  Successfully exported: $SUCCESS"
echo "  Failed:               $FAILED"
echo "  Skipped existing:     $SKIPPED"
echo "========================================"
echo ""

if [ $SUCCESS -gt 0 ]; then
    echo "Exported models:"
    find "$OUTPUT_BASE_DIR" -maxdepth 1 -type d -name "${MODEL_NAME_PREFIX}-step*" | sort -V
    echo ""
fi

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED checkpoint(s) failed to export"
    exit 1
fi

echo "All available checkpoints exported successfully."
