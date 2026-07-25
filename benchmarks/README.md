# 서빙 벤치마크

`bench_serving.py`는 OpenAI 호환 엔드포인트라면 무엇이든(vLLM, TGI, SGLang, 상용 API)
같은 방법으로 측정한다. 따라서 **엔진 간 / 양자화 간 비교**는 서버를 바꿔 띄우고
같은 명령을 다시 실행하는 것만으로 끝난다.

## 측정 지표

| 지표 | 의미 | 왜 중요한가 |
|---|---|---|
| TTFT | 첫 토큰까지의 시간 | 채팅 UX의 체감 반응성. prefill 성능 + 큐 대기의 함수 |
| TPOT | 토큰당 생성 시간 | 스트리밍 타이핑 속도. decode 성능의 함수 |
| tok/s | 시스템 전체 처리량 | GPU 1장으로 감당 가능한 동시 사용자 수 결정 → 비용 |
| p95 latency | 꼬리 지연 | SLA. 평균만 보면 batching 큐 대기를 놓친다 |

## 실험 시나리오 예시

### 1. 동시성 스케일링 (continuous batching 효과 확인)

```bash
python benchmarks/bench_serving.py \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --num-requests 64 --concurrency 1 4 16 64 --max-tokens 256 \
  --output benchmarks/results/vllm_fp16_scaling.json
```

기대: 동시성이 1→64로 늘 때 tok/s는 크게 오르고, TTFT p95는 완만하게 늘어난다.
정적 배칭 서버라면 TTFT가 배치 경계에서 계단식으로 튄다.

### 2. 양자화 비교 (FP16 vs AWQ)

```bash
# 서버 A
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
# 서버 B (다른 GPU 또는 순차 실행)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --port 8000

# 각각에 대해 동일 명령 실행 후 결과 JSON 비교
```

확인 포인트: AWQ는 선형층 가중치를 4bit로 줄인다 (선형층 기준 FP16 대비 ~1/4,
임베딩 등 비양자화 텐서를 포함한 전체 체크포인트 기준으로는 ~1/3). 줄어든 만큼
같은 GPU에서 KV 캐시 풀이 커져 동시성 여유가 생긴다. 단, 품질 저하 여부는
`eval/run_eval.py`와 실제 태스크로 별도 검증해야 한다.

### 3. gpu-memory-utilization / 스케줄링 튜닝

KV 캐시 풀 크기는 (VRAM × `--gpu-memory-utilization`) − 가중치 − 활성값으로
정해진다. 풀을 키우는 실질 손잡이는 양자화와 `--gpu-memory-utilization`이고,
`--max-model-len`은 워스트케이스 상한·기동 검증에 관여한다 — 원리는
`docs/inference-optimization.md` 2장, 플래그별 요약은 `scripts/serve_llm.sh` 주석 참고.
`--max-num-seqs`를 함께 스윕하면 지연-처리량 트레이드오프 곡선을 그릴 수 있다.

## 결과 기록 규칙

- 측정 결과 JSON은 `benchmarks/results/`에 커밋하지 않는다(.gitignore).
- 보고서에 옮길 때는 **GPU 모델, 드라이버, vLLM 버전, 서버 실행 플래그**를 반드시
  함께 기록한다. 이 네 가지가 다르면 숫자는 비교 불가능하다.
