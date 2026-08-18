#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SPEC_TRAIN_DIR="$(cd -- "${WORKSPACE_DIR}/.." && pwd)"
readonly SPECULATORS_DIR="${SPEC_TRAIN_DIR}/speculators"
readonly PREPARE_DATA="${SPECULATORS_DIR}/scripts/prepare_data.py"
readonly PYTHON_BIN="${SPEC_TRAIN_DIR}/speculators_venv/bin/python"
readonly MODEL_DIR="/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1"
readonly DATASET_DIR="/mnt/sfs_turbo/dataset/hugging-face/long_context_24k"
readonly RAW_SOURCE_FILE="${DATASET_DIR}/long_sft_24k_64k.jsonl"
readonly INPUT_FILE="${DATASET_DIR}/long_sft_24k_32k.jsonl"
readonly OUTPUT_DIR="${DATASET_DIR}/hf_dataset_glm52_24k_32k"
readonly ASSISTANT_PATTERN='<\|assistant\|>((?:(?!<\|user\|>|<\|assistant\|>).)*)'

# Override these without editing the script, for example:
#   WORKERS=16 OUTPUT_DIR=/path/to/your/output ./prepare_long_sft_24k_32k_dataset.sh --overwrite
readonly SEQ_LENGTH="${SEQ_LENGTH:-32768}"
readonly WORKERS="${WORKERS:-8}"
readonly SEED="${SEED:-42}"
readonly MINIMUM_VALID_TOKENS="${MINIMUM_VALID_TOKENS:-1}"
readonly EXTRACT_MIN_LENGTH="${EXTRACT_MIN_LENGTH:-24576}"
readonly EXTRACT_MAX_LENGTH="${EXTRACT_MAX_LENGTH:-32768}"
readonly EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-8}"
readonly OVERWRITE_EXTRACT="${OVERWRITE_EXTRACT:-0}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python is unavailable: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -f "${PREPARE_DATA}" ]] || {
  echo "Preprocessing script not found: ${PREPARE_DATA}" >&2
  exit 1
}
[[ -d "${MODEL_DIR}" ]] || {
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 1
}
[[ -f "${RAW_SOURCE_FILE}" ]] || {
  echo "Raw source dataset not found: ${RAW_SOURCE_FILE}" >&2
  exit 1
}

if [[ -f "${INPUT_FILE}" && "${OVERWRITE_EXTRACT}" != "1" ]]; then
  echo "Using existing extracted dataset: ${INPUT_FILE}"
else
  if [[ "${OVERWRITE_EXTRACT}" == "1" ]]; then
    rm -f "${INPUT_FILE}"
  fi
  echo "Extracting 24k-32k samples from: ${RAW_SOURCE_FILE}"
  "${PYTHON_BIN}" "${WORKSPACE_DIR}/extract_24k_32k_jsonl.py" \
    --input "${RAW_SOURCE_FILE}" \
    --output "${INPUT_FILE}" \
    --tokenizer "${MODEL_DIR}" \
    --min-length "${EXTRACT_MIN_LENGTH}" \
    --max-length "${EXTRACT_MAX_LENGTH}" \
    --batch-size "${EXTRACT_BATCH_SIZE}"
fi

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SPECULATORS_DIR}/src:${SPECULATORS_DIR}/hs_connectors/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

echo "Input:                ${INPUT_FILE}"
echo "Output:               ${OUTPUT_DIR}"
echo "Model:                ${MODEL_DIR}"
echo "Sequence length:      ${SEQ_LENGTH}"
echo "Workers:              ${WORKERS}"
echo "Shuffle seed:         ${SEED}"
echo "Minimum valid tokens: ${MINIMUM_VALID_TOKENS}"
echo "Extract min length:   ${EXTRACT_MIN_LENGTH}"
echo "Extract max length:   ${EXTRACT_MAX_LENGTH}"
echo "Extract batch size:   ${EXTRACT_BATCH_SIZE}"
echo "Overwrite extract:    ${OVERWRITE_EXTRACT}"

cd "${SPECULATORS_DIR}"
exec "${PYTHON_BIN}" "${PREPARE_DATA}" \
  --model "${MODEL_DIR}" \
  --data "${INPUT_FILE}" \
  --output "${OUTPUT_DIR}" \
  --seq-length "${SEQ_LENGTH}" \
  --num-preprocessing-workers "${WORKERS}" \
  --seed "${SEED}" \
  --minimum-valid-tokens "${MINIMUM_VALID_TOKENS}" \
  --assistant-pattern "${ASSISTANT_PATTERN}" \
  "$@"
