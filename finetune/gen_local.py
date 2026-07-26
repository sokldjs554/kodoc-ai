"""평가셋에 대한 답변을 로컬 HF 모델로 생성한다 (base 또는 base+LoRA).

`eval_format.py`는 OpenAI 호환 엔드포인트를 부르는데, 노트북 한 개 안에서
파인튜닝 전후를 비교하려고 vLLM 서버를 띄우는 건 과하다. 여기서 답변만 미리
만들어 두고 채점은 `eval_format.py --pred-file`에 맡긴다 — 채점 규칙은 한 곳에만
있어야 base와 LoRA가 같은 잣대로 재진다.

사용:
    python finetune/gen_local.py --model Qwen/Qwen2.5-0.5B-Instruct --out preds_base.jsonl
    python finetune/gen_local.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter finetune/out/lora --out preds_lora.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, adapter: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # 추론 속도를 위해 어댑터를 가중치에 흡수
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


@torch.no_grad()
def generate(model, tok, rows: list[dict], max_new_tokens: int, batch_size: int) -> list[str]:
    answers: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tok.apply_chat_template(
                [m for m in r["messages"] if m["role"] != "assistant"],
                add_generation_prompt=True,
                tokenize=False,
            )
            for r in batch
        ]
        # 생성이므로 왼쪽 패딩이어야 한다 — 오른쪽 패딩이면 패드 토큰 뒤에서 이어쓴다.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 평가는 그리디로 고정 — 재현되지 않는 수치는 의미가 없다
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        answers.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(start + batch_size, len(rows))}/{len(rows)}")
    return [a.strip() for a in answers]


def main() -> None:
    ap = argparse.ArgumentParser(description="로컬 모델로 평가셋 답변 생성")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA 어댑터 경로 (없으면 base)")
    ap.add_argument("--eval-file", default="finetune/data/eval.jsonl")
    ap.add_argument("--out", required=True, help="예측 저장 경로 (jsonl)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [
        json.loads(x)
        for x in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    label = "base+LoRA" if args.adapter else "base"
    print(f"생성 {len(rows)}건 / {args.model} ({label})")

    model = load_model(args.model, args.adapter)
    answers = generate(model, tok, rows, args.max_new_tokens, args.batch_size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps({"answer": a}, ensure_ascii=False) + "\n")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
