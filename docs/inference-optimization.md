# LLM 추론·서빙 최적화 노트

프로젝트를 진행하며 정리한 추론 최적화의 원리. 벤치마크(`benchmarks/`)가 측정하는
지표들이 왜 그렇게 움직이는지에 대한 배경이다.

## 1. 추론의 두 국면: Prefill vs Decode

| | Prefill (프롬프트 처리) | Decode (토큰 생성) |
|---|---|---|
| 연산 형태 | 전체 프롬프트를 한 번에 어텐션 | 토큰 1개씩 자기회귀 생성 |
| 병목 | **연산(compute)-bound** | **메모리 대역폭-bound** |
| 좌우하는 지표 | TTFT | TPOT |

Decode가 메모리-bound인 이유: 토큰 1개를 만들 때마다 모델 가중치 전체를 HBM에서
읽어야 하는데, 연산량은 그에 비해 작다. 그래서 decode 처리량을 올리는 거의 모든
기법은 "가중치를 한 번 읽을 때 더 많은 토큰을 처리"(배칭) 아니면 "읽을 바이트를
줄이기"(양자화)로 수렴한다.

## 2. KV 캐시 — 왜 동시성의 상한을 결정하는가

시퀀스마다 과거 토큰의 Key/Value를 저장해야 한다. 크기는:

```
KV bytes = 2(K,V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

예: Qwen2.5-7B(28층, KV 헤드 4, head_dim 128, FP16)에서 시퀀스 1개(8K 토큰) ≈
2 × 28 × 4 × 128 × 8192 × 2B ≈ **470MB**. GQA(KV 헤드 4개) 덕에 MHA 대비 1/7인데도
이 정도다. 이 풀이 동시에 붙들 수 있는 토큰 총량이 실효 동시성을 결정한다.

여기서 흔한 오해 하나를 짚어야 한다. vLLM에서 KV 캐시 풀의 크기는
**(VRAM × `--gpu-memory-utilization`) − 가중치 − 프로파일링된 활성값**으로 정해지고,
`--max-model-len`은 이 식에 직접 들어가지 않는다. PagedAttention이 블록을
온디맨드로 할당하므로(3장), 실제 트래픽이 짧다면 max-model-len을 32K→8K로 줄여도
그 트래픽의 메모리 사용량과 동시성은 거의 변하지 않는다. `--max-model-len`의 실제
역할은 (a) 시퀀스당 워스트케이스 상한 — 소수의 초장문 요청이 풀을 독식하는 것을
막는 보호 장치, (b) 기동 가능성 — 풀에 max-len 시퀀스 1개가 안 들어가면 vLLM이
기동을 거부한다. 따라서 처리량을 올리는 실질적 순서는: **양자화로 가중치를 줄여
풀 자체를 키우고 → `--gpu-memory-utilization`을 OOM 마진 안에서 올리고 →
`--max-num-seqs`·청크드 prefill 등 스케줄링을 조정**하는 쪽이다.

## 3. PagedAttention (vLLM)

전통 방식은 시퀀스마다 max_len짜리 연속 버퍼를 예약해 내부 단편화가 심했다
(실제 길이가 짧으면 나머지는 낭비). PagedAttention은 OS 가상 메모리처럼 KV 캐시를
고정 크기 블록(기본 16토큰)으로 쪼개 비연속 할당한다.

- 단편화 낭비가 사라져 같은 메모리로 **동시 시퀀스 수가 수 배** 늘어난다.
- 블록 공유가 가능해져 prefix caching, beam search 메모리 공유가 공짜로 따라온다.

## 4. Continuous Batching

정적 배칭은 배치 내 가장 긴 시퀀스가 끝날 때까지 전체가 대기한다(convoy effect).
Continuous batching은 **iteration 단위**로 스케줄링한다 — 매 스텝마다 끝난 시퀀스를
내보내고 대기 중인 요청을 즉시 끼워 넣는다.

벤치마크에서 보이는 신호: 동시성을 1→64로 올릴 때 tok/s가 준선형으로 오르다
GPU 포화와 함께 평탄해지고, TTFT p95가 완만히 증가하면 continuous batching이
제대로 동작하는 것. TTFT가 계단식으로 튀면 배치 경계 대기가 있다는 뜻이다.

## 5. 양자화

| 방식 | 대상 | 특징 |
|---|---|---|
| AWQ | 가중치 4bit | activation 분포를 보고 중요 채널을 보호. 추론 커널 성숙 |
| GPTQ | 가중치 4bit | 헤시안 기반 오차 보정. AWQ와 품질 비슷, 커널에 따라 속도 차이 |
| FP8 (W8A8) | 가중치+활성값 | H100대에서 하드웨어 지원. 품질 손실 거의 없음 |
| KV 캐시 양자화 | KV 캐시 FP8 | 동시성 2배 확보. 긴 컨텍스트에서 품질 확인 필요 |

가중치 4bit의 효과는 이중이다: (a) decode가 메모리-bound라 **가중치 읽기 자체가
빨라지고**, (b) 남는 VRAM이 KV 캐시로 가서 **동시성이 커진다**. 단 품질은 태스크
의존적이므로 — 특히 한국어처럼 캘리브레이션 데이터에 적게 들어간 언어 —
반드시 자체 평가셋으로 검증해야 한다(`eval/run_eval.py`를 만든 이유).

## 6. Prefix Caching

RAG 서비스는 모든 요청이 같은 시스템 프롬프트로 시작한다. vLLM의
`--enable-prefix-caching`은 동일 prefix 블록의 KV를 재사용해 해당 구간 prefill을
건너뛴다. 시스템 프롬프트가 길수록, 트래픽이 몰릴수록 TTFT 개선이 커진다.
KoDoc처럼 "고정 시스템 프롬프트 + 가변 컨텍스트" 구조에서 비용 0의 최적화.

## 7. Speculative Decoding

작은 draft 모델이 토큰 여러 개를 제안하고 큰 모델이 한 번의 forward로 검증한다.
수락률이 높으면 TPOT가 크게 줄지만, 배치가 큰 상황에서는 검증 오버헤드가
배칭 이득을 깎을 수 있다 — 저동시성·저지연 시나리오에 적합한 기법.

## 8. 서비스 계층에서의 최적화 (이 프로젝트에서 실제 적용한 것)

- **SSE 스트리밍 + sources 선전송**: 시스템 TTFT와 별개로 "체감" 첫 응답을
  검색 완료 시점(수십 ms)까지 당겼다.
- **비동기 파이프라인**: httpx AsyncClient로 LLM 대기 중에도 이벤트 루프가
  다른 요청을 처리한다. 서비스 워커 수 ≪ 동시 사용자 수가 가능해진다.
- **엔진 독립성**: 최적화 실험(엔진/양자화/플래그)이 서비스 코드 변경 없이
  가능하도록 경계를 그었다. 최적화는 일회성 작업이 아니라 반복 실험이기 때문.

## 참고 자료

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — continuous batching의 원류
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (MLSys 2024)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (ICML 2023)
