# LoRA 파인튜닝 실행 절차 (Kaggle 무료 T4)

작은 모델이 RAG 형식 계약을 못 지키는 문제를 학습으로 메울 수 있는지 확인하는 실험.
데이터 생성은 로컬, 학습·평가는 GPU가 필요해 Kaggle에서 한다.

## 0. 전제

- `finetune/data/sft.jsonl`, `finetune/data/eval.jsonl`이 생성되어 있을 것
  (`python finetune/build_dataset.py --per-chunk 6`)
- Kaggle 노트북 설정: **Accelerator = GPU T4 x2**, **Internet = On**
  (T4 한 장만 써도 0.5B LoRA는 충분히 들어간다)

## 1. 왜 이 구성인가

| 결정 | 이유 |
|---|---|
| 0.5B 베이스 | 7B는 프롬프트만으로 계약을 지킨다. 학습이 필요한 크기에서 해야 실험에 의미가 있다 |
| LoRA | 무료 T4 16GB에서 전체 파인튜닝은 안 들어간다. 어댑터만 학습하면 여유롭다 |
| fp16 | T4는 Turing 세대라 bf16이 없다. bf16로 두면 런타임에서 죽는다 |
| 답변 토큰만 학습 | 프롬프트까지 학습시키면 문서 내용을 외운다. 배워야 할 것은 *형식을 지키는 행동* |
| 그리디 디코딩 평가 | 샘플링하면 같은 모델도 매번 다른 점수가 나온다 |

## 2. 노트북 셀

```python
# [1] 설치 — Kaggle 기본 이미지에 torch는 이미 있다
!pip install -q peft transformers accelerate

# [2] 저장소 + 데이터
!git clone -q https://github.com/sokldjs554/kodoc-ai.git
%cd kodoc-ai
# sft.jsonl / eval.jsonl이 커밋되어 있어야 한다 (없으면 Kaggle Dataset으로 올려서 복사)
!ls finetune/data/
```

```python
# [3] 파인튜닝 전 baseline — 비교 대상이 없으면 숫자가 의미를 못 가진다
!python finetune/gen_local.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --out finetune/results/preds_base.jsonl

!python finetune/eval_format.py \
    --pred-file finetune/results/preds_base.jsonl \
    --label base --out finetune/results/base.json
```

```python
# [4] 학습 — 187건 x 3에폭이면 T4에서 10분 안쪽
!python finetune/train_lora.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --out finetune/out/lora
```

```python
# [5] 파인튜닝 후 — 같은 평가셋, 같은 채점기
!python finetune/gen_local.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter finetune/out/lora \
    --out finetune/results/preds_lora.jsonl

!python finetune/eval_format.py \
    --pred-file finetune/results/preds_lora.jsonl \
    --label lora --out finetune/results/lora.json
```

```python
# [6] 비교표
import json
base = json.load(open("finetune/results/base.json"))["summary"]
lora = json.load(open("finetune/results/lora.json"))["summary"]
for k in base:
    if isinstance(base[k], float):
        print(f"{k:22s} {base[k]:.3f} -> {lora[k]:.3f}")
```

## 3. 결과를 읽는 법

**`refusal_recall`과 `false_refusal_rate`를 같이 본다.** 거절을 가르치면 모델이 과하게
거절하는 쪽으로 기운다. 거절 재현율만 올라가고 오거절률도 같이 올랐다면 그건 개선이
아니라 편향 이동이다.

**`cite_valid_rate`가 `cite_rate`보다 중요하다.** 인용을 다는 것은 흉내 내기 쉽고,
있는 번호만 인용하는 것은 실제로 근거를 보고 있다는 뜻에 더 가깝다.

**안 좋게 나와도 그대로 기록한다.** 학습 데이터 187건은 적고, 0.5B는 작다. 개선이
없었다면 "이 데이터 규모에서는 안 됐다"가 결과다 — 실험을 했다는 사실과 판단 근거가
남는 것이 목적이지, 좋은 숫자를 만드는 것이 목적이 아니다.

## 4. 막히는 지점

| 증상 | 원인 / 대응 |
|---|---|
| `bf16 is not supported` | T4에는 bf16이 없다. `--fp16` 경로를 쓰고 있는지 확인 |
| CUDA OOM | `--batch-size 2 --grad-accum 8`로 낮춘다 (유효 배치는 유지) |
| 답변이 계속 잘림 | `--max-new-tokens`를 올린다. 다만 계약상 답변은 두 문장이라 256이면 보통 충분 |
| `n건이 max_len을 넘어` 경고 | 청크가 길어 프롬프트 앞부분이 잘린 것. 답변은 보존되므로 소수면 무시 가능 |
| 예측 n건 != 평가 n건 | 서로 다른 eval 파일로 생성했다. `--eval-file`을 맞춘다 |
