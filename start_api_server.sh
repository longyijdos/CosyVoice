#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50001}"
MODEL_DIR="${MODEL_DIR:-pretrained_models/CosyVoice2-0.5B}"
STT_MODEL="${STT_MODEL:-pretrained_models/stt/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f}"
STT_DEVICE="${STT_DEVICE:-auto}"
STT_COMPUTE_TYPE="${STT_COMPUTE_TYPE:-int8_float16}"
STT_DOWNLOAD_ROOT="${STT_DOWNLOAD_ROOT:-pretrained_models/stt}"

args=(
  api_server.py
  --host "$HOST"
  --port "$PORT"
  --model_dir "$MODEL_DIR"
  --stt_model "$STT_MODEL"
  --stt_device "$STT_DEVICE"
  --stt_compute_type "$STT_COMPUTE_TYPE"
  --stt_download_root "$STT_DOWNLOAD_ROOT"
)

if [[ "${FP16:-0}" == "1" ]]; then
  args+=(--fp16)
fi

if [[ "${LOAD_VLLM:-0}" == "1" ]]; then
  args+=(--load_vllm)
fi

if [[ "${LOAD_JIT:-0}" == "1" ]]; then
  args+=(--load_jit)
fi

if [[ "${LOAD_TRT:-0}" == "1" ]]; then
  args+=(--load_trt)
fi

if [[ -n "${TRT_CONCURRENT:-}" ]]; then
  args+=(--trt_concurrent "$TRT_CONCURRENT")
fi

exec uv run "${args[@]}"
