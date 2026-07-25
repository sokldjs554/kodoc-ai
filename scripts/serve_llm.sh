#!/usr/bin/env bash
# LLM 서버 (vLLM, OpenAI 호환 API)
#
# 플래그 설명:
# --max-model-len         : 시퀀스당 컨텍스트 상한. PagedAttention은 온디맨드 할당이라
#                           이 값을 줄여도 짧은 트래픽의 메모리 사용은 거의 그대로다.
#                           역할은 (a) 초장문 요청이 KV 풀을 독식하는 것을 막는 상한,
#                           (b) 기동 검증 — 풀에 max-len 1개가 안 들어가면 기동 거부.
# --gpu-memory-utilization: (VRAM × 이 값) − 가중치 − 활성값이 KV 캐시 풀이 된다.
#                           풀을 키우는 가장 큰 손잡이는 양자화(가중치 축소)와 이 값.
#                           0.90 초과는 OOM 마진이 얇아지므로 주의.
# --enable-prefix-caching : RAG처럼 시스템 프롬프트가 반복되는 워크로드에서
#                           동일 prefix 구간의 prefill을 건너뛰어 TTFT를 줄인다.
set -euo pipefail

MODEL="${KODOC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${1:-8000}"

vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
