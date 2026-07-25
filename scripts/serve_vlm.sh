#!/usr/bin/env bash
# VLM 서버 (문서 이미지 파싱용, vLLM OpenAI 호환 API)
#
# --limit-mm-per-prompt : 요청당 이미지 수 제한. 문서 파싱은 페이지당 1장이면 충분.
# --max-model-len       : 문서 1페이지의 마크다운 출력이 4K 토큰을 넘는 경우가
#                         드물어 12K로 넉넉히 잡았다. 이미지 토큰도 여기 포함된다.
set -euo pipefail

MODEL="${KODOC_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${1:-8001}"

vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 1}'
