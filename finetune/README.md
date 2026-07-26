# finetune — 작은 모델에게 RAG 형식 계약 가르치기

KoDoc이 LLM에게 요구하는 것은 자유로운 생성이 아니라 **계약 준수**다.

1. 제공된 청크만 근거로 답한다
2. `[n]` 형식으로 인용한다
3. 근거가 없으면 정해진 문장으로 거절한다

7B는 프롬프트만으로 이걸 지키지만 0.5B급은 못 지킨다. 서빙 비용 때문에 작은 모델을
쓰고 싶다면 프롬프트로 안 되는 부분을 학습으로 메워야 한다 — 그게 이 디렉터리다.

## 구성

| 파일 | 역할 |
|---|---|
| `build_dataset.py` | 교사 모델로 SFT 데이터 생성 + **규칙 검증으로 불량 샘플 제거** |
| `train_lora.py` | LoRA 학습 — **답변 토큰에만 손실**을 건다 |
| `gen_local.py` | 평가셋 답변 생성 (base / base+LoRA), 서버 없이 로컬 추론 |
| `eval_format.py` | 형식 준수를 규칙으로 채점 (LLM-as-a-judge 없음) |
| `data/` | 생성된 학습·평가 데이터 (`sft.jsonl`, `eval.jsonl`) |
| `results/` | 파인튜닝 전후 평가 결과 |

학습은 GPU가 필요해 Kaggle 무료 T4에서 수행한다 — 절차는 [`docs/finetune-runbook.md`](../docs/finetune-runbook.md).

**실험 결과와 해석은 [`docs/finetune-result.md`](../docs/finetune-result.md).**
인용률 0.167 → 1.000, 거절 재현율 0.444 → 1.000, 오거절률 0.053 → 0.026.

## 설계에서 신경 쓴 것

**교사 출력을 믿지 않는다.** `validate()`가 모든 생성 샘플을 검사한다 — 인용이 있는가,
인용 번호가 실재하는가, 답변의 수치가 근거 청크에 실제로 존재하는가. 교사 모델도
형식을 틀리기 때문에, 거르지 않으면 그 오류를 그대로 학습시키게 된다. 탈락 사유별
집계를 출력하므로 "교사가 얼마나 틀렸는지"가 기록으로 남는다.

**학습/평가를 청크 단위로 나눈다.** 질문 단위로 나누면 같은 청크에서 나온 Q/A가 양쪽에
걸쳐, 사실상 답을 외운 상태로 평가하게 된다.

**거절 샘플은 교사 없이 만든다.** 정답이 고정 문장이므로 생성할 이유가 없다. 다른
도메인 청크를 근거로 주고 답할 수 없는 질문을 던지는 방식으로 규칙 생성한다.
(README 로드맵의 "미근거 질문(unanswerable) 평가셋"이 여기서 채워진다.)

**채점을 규칙으로만 한다.** 요구사항이 전부 기계 검증 가능한 계약이라 모델 심판이
필요 없고, 채점자가 모델이면 채점 자체를 의심받는다.

**답변 토큰에만 손실을 건다.** 프롬프트(시스템 규칙 + 검색된 청크)까지 학습시키면
모델이 문서 내용을 외우는 쪽으로 샌다. 가르치려는 것은 지식이 아니라 "주어진 근거로
형식을 지켜 답하는 행동"이다. `max_len`을 넘으면 프롬프트 앞을 자르고 답변을 남긴다 —
반대로 자르면 학습 신호 자체가 사라진다.

**생성과 채점을 분리한다.** `gen_local.py`가 답변을 만들고 `eval_format.py`가 채점한다.
채점 규칙이 한 곳에만 있어야 base와 LoRA가 같은 잣대로 재진다. 평가는 그리디 디코딩
고정 — 샘플링하면 같은 모델도 매번 다른 점수가 나와 비교가 성립하지 않는다.

## 지표

| 지표 | 의미 |
|---|---|
| `cite_rate` | 답변에 `[n]` 인용이 있는 비율 (거절 케이스 제외) |
| `cite_valid_rate` | 인용 번호가 제공된 청크 범위 안인 비율 — 환각 인용 탐지 |
| `grounded_rate` | 근거의 핵심 수치를 그대로 옮긴 비율 (**수치가 있는 문항만**) |
| `n_grounded_scored` | 위 비율의 분모 — 수치 없는 문항은 채점 대상이 아니다 |
| `refusal_recall` | 근거 없는 질문에 정확히 거절한 비율 |
| `false_refusal_rate` | 답할 수 있는데 거절한 비율 (낮을수록 좋음) |

**지표는 상한을 먼저 확인하고 쓴다.** 정답을 그대로 예측으로 넣어 채점하면 전 항목이
1.0이어야 한다. 초기 구현은 `grounded_rate`가 0.63에서 막혔는데, 수치 없는 답변
14건(38건 중)을 실패로 세고 있었기 때문이다. 상한이 1이 아닌 지표는 모델이 아니라
평가셋 구성을 재고 있는 것이다.

**마지막 두 개를 함께 봐야 한다.** 거절을 가르치면 모델이 과하게 거절하는 쪽으로
기울 수 있다. 한 지표만 올리고 다른 지표를 망가뜨리는 것은 최적화가 아니다.

## 사용

```bash
# 1) 데이터 생성 (교사 모델 필요 — KODOC_LLM_* 설정 사용)
python finetune/build_dataset.py --per-chunk 6

# 2) 파인튜닝 전 baseline
python finetune/gen_local.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --out finetune/results/preds_base.jsonl
python finetune/eval_format.py --pred-file finetune/results/preds_base.jsonl \
    --label base --out finetune/results/base.json

# 3) 학습 (GPU 필요)
python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out finetune/out/lora

# 4) 파인튜닝 후 — 같은 평가셋, 같은 채점기
python finetune/gen_local.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter finetune/out/lora --out finetune/results/preds_lora.jsonl
python finetune/eval_format.py --pred-file finetune/results/preds_lora.jsonl \
    --label lora --out finetune/results/lora.json
```

서빙 중인 OpenAI 호환 엔드포인트를 그대로 평가하려면 `--pred-file` 대신
`--base-url`/`--model`을 준다.
