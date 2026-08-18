#!/usr/bin/env bash
set -euo pipefail

readonly PORT="${PORT:-8000}"
readonly ENDPOINT="http://127.0.0.1:${PORT}/v1"
readonly EXPECTED_MODEL="glm-5.2"
readonly DATASET_PATH="/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/hf_dataset_glm52_24k_32k"
readonly OUTPUT_PATH="${DATASET_PATH}/hidden_states_native_fp4"
readonly EXPECTED_TRANSPORT_PATH="${DATASET_PATH}/.vllm_hidden_states_tmp"
readonly EXPECTED_MAX_MODEL_LEN="40000"
readonly SPECULATORS_PATH="/mnt/paas/spec_train/speculators"
readonly GENERATOR="${SPECULATORS_PATH}/scripts/data_generation_offline.py"
readonly PYTHON_BIN="/mnt/paas/spec_train/speculators_venv/bin/python"
readonly SPECULATORS_SRC="${SPECULATORS_PATH}/src"
readonly CONNECTORS_SRC="${SPECULATORS_PATH}/hs_connectors/src"
readonly CONCURRENCY="${CONCURRENCY:-4}"
readonly REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
readonly MAX_RETRIES="${MAX_RETRIES:-2}"
readonly MAX_CONSECUTIVE_ERRORS="${MAX_CONSECUTIVE_ERRORS:-8}"
readonly VALIDATE_OUTPUTS="${VALIDATE_OUTPUTS:-true}"

# Never send local vLLM traffic through proxies.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

usage() {
  echo "Usage: $0 [--dry-run] [MAX_SAMPLES]" >&2
}

dry_run=false
sample_arg_seen=false
target_max_samples="${MAX_SAMPLES:-29938}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      if [[ "${dry_run}" == true ]]; then
        echo "--dry-run may only be specified once" >&2
        usage
        exit 2
      fi
      dry_run=true
      ;;
    -*)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 2
      ;;
    *)
      if [[ "${sample_arg_seen}" == true ]]; then
        echo "Only one MAX_SAMPLES value may be specified" >&2
        usage
        exit 2
      fi
      target_max_samples="${arg}"
      sample_arg_seen=true
      ;;
  esac
done

if [[ ! "${target_max_samples}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be a positive integer: ${target_max_samples}" >&2
  exit 2
fi
readonly TARGET_MAX_SAMPLES="${target_max_samples}"
max_samples_args=(--max-samples "${TARGET_MAX_SAMPLES}")
generator_mode_args=()
if [[ "${dry_run}" == true ]]; then
  generator_mode_args+=(--dry-run)
fi
generator_option_args=()
case "${VALIDATE_OUTPUTS,,}" in
  true|1|yes)
    generator_option_args+=(--validate-outputs)
    ;;
  false|0|no)
    ;;
  *)
    echo "VALIDATE_OUTPUTS must be one of true/false/1/0/yes/no: ${VALIDATE_OUTPUTS}" >&2
    exit 1
    ;;
esac

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Python is unavailable: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -f "${GENERATOR}" ]] || {
  echo "Hidden-state generator not found: ${GENERATOR}" >&2
  exit 1
}
[[ -d "${DATASET_PATH}" ]] || {
  echo "Preprocessed 24k-32k dataset not found: ${DATASET_PATH}" >&2
  echo "Run /mnt/paas/spec_train/collect_hidden_states_24k_32k_dataset/prepare_long_sft_24k_32k_dataset.sh first." >&2
  exit 1
}
if ! compgen -G "${DATASET_PATH}/*.arrow" >/dev/null; then
  echo "No Arrow shards found in preprocessed dataset: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ "${dry_run}" == false ]]; then
  if ! models_json="$(
    curl --silent --show-error --fail --max-time 5 "${ENDPOINT}/models"
  )"; then
    echo "GLM-5.2 hidden-state service is unavailable at ${ENDPOINT}" >&2
    exit 1
  fi

  if ! "${PYTHON_BIN}" -c \
    'import json, sys; data=json.loads(sys.argv[1]); raise SystemExit(0 if any(item.get("id") == sys.argv[2] for item in data.get("data", [])) else 1)' \
    "${models_json}" "${EXPECTED_MODEL}"; then
    echo "Endpoint ${ENDPOINT} is not serving ${EXPECTED_MODEL}" >&2
    exit 1
  fi

  matching_vllm_process=false
  while IFS= read -r process_line; do
    if [[ "${process_line}" == *"${EXPECTED_TRANSPORT_PATH}"* && \
          "${process_line}" == *"--max-model-len ${EXPECTED_MAX_MODEL_LEN}"* ]]; then
      matching_vllm_process=true
      break
    fi
  done < <(pgrep -af '[v]llm.entrypoints.cli.main.*serve' || true)

  if [[ "${matching_vllm_process}" == false ]]; then
    echo "The running vLLM process does not match the 24k-32k collector configuration." >&2
    echo "Expected transport: ${EXPECTED_TRANSPORT_PATH}" >&2
    echo "Expected model limit: --max-model-len ${EXPECTED_MAX_MODEL_LEN}" >&2
    echo "Restart it with /mnt/paas/spec_train/collect_hidden_states_24k_32k_dataset/start_glm5.2_hidden_states_24_32k.sh before collection." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_PATH}"

if [[ "${dry_run}" == true ]]; then
  echo "Collecting GLM-5.2 24-32K hidden states (dry run)"
else
  echo "Collecting GLM-5.2 24-32K hidden states"
fi
echo "  endpoint:        ${ENDPOINT}"
echo "  dataset:         ${DATASET_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  max samples:     ${TARGET_MAX_SAMPLES}"
echo "  concurrency:     ${CONCURRENCY}"
echo "  validate outputs: ${VALIDATE_OUTPUTS}"
echo "  request timeout: ${REQUEST_TIMEOUT}s"
echo "  max retries:     ${MAX_RETRIES}"
echo "  max consecutive errors: ${MAX_CONSECUTIVE_ERRORS}"
echo "Existing hs_<index>.safetensors files are retained and skipped."

exec env \
  PYTHONPATH="${SPECULATORS_SRC}:${CONNECTORS_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${GENERATOR}" \
  --model "${EXPECTED_MODEL}" \
  --endpoint "${ENDPOINT}" \
  --preprocessed-data "${DATASET_PATH}" \
  --output "${OUTPUT_PATH}" \
  "${max_samples_args[@]}" \
  --concurrency "${CONCURRENCY}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --max-retries "${MAX_RETRIES}" \
  --max-consecutive-errors "${MAX_CONSECUTIVE_ERRORS}" \
  "${generator_mode_args[@]}" \
  "${generator_option_args[@]}"
