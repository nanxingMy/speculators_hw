#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_PATH="/mnt/paas/GLM-5.2-NVFP4-W4A4-MG39-BNT3/v1"
readonly DATASET_PATH="/mnt/sfs_turbo/dataset/hugging-face/long_context_24k/hf_dataset_glm52_24k_32k"
readonly HIDDEN_STATES_PATH="${DATASET_PATH}/.vllm_hidden_states_tmp"
readonly SPECULATORS_PATH="/mnt/paas/spec_train/speculators"
readonly VLLM_ENV="/mnt/paas/dspark_test/miniconda3"
readonly PYTHON_BIN="${VLLM_ENV}/bin/python"
readonly CUDA_TOOLKIT="${VLLM_ENV}/lib/python3.14/site-packages/nvidia/cu13"
readonly PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_TOOLKIT}}"
export PATH="${VLLM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="/usr/local/nvidia/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="${SPECULATORS_PATH}/src:${SPECULATORS_PATH}/hs_connectors/src${PYTHONPATH:+:${PYTHONPATH}}"
export HTTP_PROXY="${HTTP_PROXY:-http://192.168.10.154:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://192.168.10.154:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

readonly MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
readonly MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-40000}"
readonly MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
readonly TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
readonly GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
readonly MOE_BACKEND="${MOE_BACKEND:-cutlass}"
readonly ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHMLA_SPARSE}"
readonly KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
readonly ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
readonly EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "vLLM python is unavailable: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -f "${SPECULATORS_PATH}/scripts/launch_vllm.py" ]] || {
  echo "vLLM launcher not found: ${SPECULATORS_PATH}/scripts/launch_vllm.py" >&2
  exit 1
}
[[ -d "${MODEL_PATH}" ]] || {
  echo "Model directory not found: ${MODEL_PATH}" >&2
  exit 1
}
[[ -d "${DATASET_PATH}" ]] || {
  echo "Dataset directory not found: ${DATASET_PATH}" >&2
  exit 1
}
mkdir -p "${HIDDEN_STATES_PATH}"

echo "Model:                  ${MODEL_PATH}"
echo "Dataset:                ${DATASET_PATH}"
echo "Hidden-state transport: ${HIDDEN_STATES_PATH}"
echo "Context length:         ${MAX_MODEL_LEN}"
echo "Max batched tokens:     ${MAX_NUM_BATCHED_TOKENS}"
echo "Max concurrent seqs:    ${MAX_NUM_SEQS}"
echo "Attention backend:      ${ATTENTION_BACKEND}"
echo "Tensor parallel size:   ${TENSOR_PARALLEL_SIZE}"
echo "MoE backend:            ${MOE_BACKEND}"
echo "Enforce eager:          ${ENFORCE_EAGER}"
echo "Extra vLLM args:        ${EXTRA_VLLM_ARGS:-}"
echo "Port:                   ${PORT}"

vllm_extra_args=()
if [[ "${ENFORCE_EAGER,,}" == "1" || "${ENFORCE_EAGER,,}" == "true" || "${ENFORCE_EAGER,,}" == "yes" ]]; then
  vllm_extra_args+=(--enforce-eager)
fi
if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  # shell-safe split: callers should provide args as a quoted string
  read -r -a parsed_extra_args <<< "${EXTRA_VLLM_ARGS}"
  vllm_extra_args+=("${parsed_extra_args[@]}")
fi

exec "${PYTHON_BIN}" "${SPECULATORS_PATH}/scripts/launch_vllm.py" "${MODEL_PATH}" \
  --hidden-states-path "${HIDDEN_STATES_PATH}" \
  --target-layer-ids 8 23 39 55 70 \
  -- \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name glm-5.2 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --quantization compressed-tensors \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --moe-backend "${MOE_BACKEND}" \
  "${vllm_extra_args[@]}"
