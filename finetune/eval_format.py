"""RAG 생성 태스크의 형식 준수를 규칙으로 채점한다.

LLM-as-a-judge를 쓰지 않는다. 이 태스크에서 요구하는 것이 전부 기계 검증 가능한
계약이기 때문이다 — 인용을 달았는가, 그 인용이 실재하는가, 근거가 없을 때 거절했는가.
채점자가 모델이면 채점 자체를 의심받는다.

측정 항목:
- cite_rate       : 답변에 [n] 인용이 하나라도 있는 비율 (거절 케이스 제외)
- cite_valid_rate : 인용 번호가 실제 제공된 청크 범위 안인 비율 (환각 인용 탐지)
- grounded_rate   : 정답 문자열이 답변에 포함된 비율 (수치 정확도의 대리 지표)
- refusal_recall  : 근거 없는 질문에 정확히 거절한 비율
- false_refusal   : 답할 수 있는데 거절해버린 비율 (낮을수록 좋음)

사용:
    python finetune/eval_format.py --base-url http://localhost:8000/v1 --model <모델명>
    python finetune/eval_format.py ... --eval-file finetune/data/eval.jsonl --out result.json

    # 이미 생성된 답변을 채점만 할 때 (gen_local.py 출력) — 엔드포인트 불필요
    python finetune/eval_format.py --pred-file preds_lora.jsonl --label lora --out lora.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from kodoc.llm.client import LLMClient, LLMError

REFUSAL_MARKERS = ("찾을 수 없습니다", "찾을수 없습니다", "확인할 수 없습니다")
CITE = re.compile(r"\[(\d+)\]")


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


def score_one(answer: str, n_chunks: int, gold: list[str] | None, unanswerable: bool) -> dict:
    cites = [int(c) for c in CITE.findall(answer)]
    refused = is_refusal(answer)
    out = {
        "refused": refused,
        "has_cite": bool(cites),
        "cite_valid": bool(cites) and all(1 <= c <= n_chunks for c in cites),
    }
    if unanswerable:
        out["refusal_correct"] = refused
    else:
        out["false_refusal"] = refused
        # 근거의 핵심 수치를 그대로 옮겼는가. gold는 교사 답변에 있던 수치이고,
        # 데이터 생성 시 "답변의 모든 수치는 청크에 존재"를 검증했으므로 실재값이다.
        #
        # gold가 비었으면(= 답변에 수치가 없는 질문) 이 항목은 **채점하지 않는다**.
        # 실패로 세면 완벽한 모델도 상한에 막힌다 — 실제로 이 평가셋은 답변가능
        # 38건 중 14건이 수치 없는 답변이라 상한이 0.63이었다. 물을 수 없는 것을
        # 틀렸다고 하면 지표가 모델이 아니라 평가셋 구성을 재게 된다.
        if gold:
            out["grounded"] = all(g in answer for g in gold)
    return out


async def run(
    client: LLMClient, rows: list[dict], max_tokens: int, temperature: float | None
) -> list[str]:
    answers = []
    for i, row in enumerate(rows, start=1):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        try:
            answer = await client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        except LLMError as e:
            print(f"  [{i}/{len(rows)}] 호출 실패: {e}")
            answer = ""
        answers.append(answer)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")
    return answers


def score_all(rows: list[dict], answers: list[str]) -> list[dict]:
    """답변 목록을 평가 행에 맞춰 채점한다. 생성 경로(엔드포인트/로컬)와 무관하게 공용."""
    out = []
    for row, answer in zip(rows, answers, strict=True):
        prompt = [m for m in row["messages"] if m["role"] != "assistant"][-1]["content"]
        n_chunks = len(CITE.findall(prompt.split("질문:")[0]))
        out.append(
            {
                "answer": answer,
                **score_one(
                    answer,
                    n_chunks=max(n_chunks, 1),
                    gold=row.get("gold"),
                    unanswerable=row.get("kind") == "unanswerable",
                ),
            }
        )
    return out


def summarize(rows: list[dict], results: list[dict]) -> dict:
    ans = [r for r, s in zip(results, rows, strict=True) if s.get("kind") != "unanswerable"]
    una = [r for r, s in zip(results, rows, strict=True) if s.get("kind") == "unanswerable"]

    def rate(items: list[dict], key: str) -> float | None:
        vals = [i[key] for i in items if key in i]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        # grounded는 수치가 있는 문항에서만 잰다. 분모를 함께 내보내지 않으면
        # 몇 건으로 낸 비율인지 알 수 없어 조용한 제외가 된다.
        "n_grounded_scored": sum(1 for a in ans if "grounded" in a),
        "cite_rate": rate([a for a in ans if not a["refused"]], "has_cite"),
        "cite_valid_rate": rate([a for a in ans if a["has_cite"]], "cite_valid"),
        "grounded_rate": rate(ans, "grounded"),
        "false_refusal_rate": rate(ans, "false_refusal"),
        "refusal_recall": rate(una, "refusal_correct"),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 형식 준수 평가 (규칙 기반)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", help="평가할 모델명 (--pred-file 사용 시 생략 가능)")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--eval-file", default="finetune/data/eval.jsonl")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument(
        "--pred-file",
        help="이미 생성된 답변 jsonl (gen_local.py 출력). 주면 모델을 호출하지 않고 채점만 한다",
    )
    ap.add_argument("--out", help="결과 JSON 저장 경로")
    ap.add_argument("--label", default="", help="결과에 남길 구성 이름 (base / lora 등)")
    args = ap.parse_args()

    if not args.pred_file and not args.model:
        ap.error("--model 또는 --pred-file 중 하나는 필요하다")

    lines = Path(args.eval_file).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    if args.pred_file:
        pred_lines = Path(args.pred_file).read_text(encoding="utf-8").splitlines()
        answers = [json.loads(line)["answer"] for line in pred_lines if line.strip()]
        if len(answers) != len(rows):
            raise SystemExit(
                f"예측 {len(answers)}건 != 평가 {len(rows)}건 — 같은 eval 파일로 생성한 것인지 확인"
            )
        print(f"채점 {len(rows)}건 / 예측 파일 {args.pred_file}")
    else:
        print(f"평가 {len(rows)}건 / 모델 {args.model}")
        client = LLMClient(base_url=args.base_url, model=args.model, api_key=args.api_key)
        try:
            temp = None if args.no_temperature else args.temperature
            answers = await run(client, rows, args.max_tokens, temp)
        finally:
            await client.aclose()

    results = score_all(rows, answers)

    summary = summarize(rows, results)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "label": args.label or args.model or args.pred_file,
                    "model": args.model,
                    "summary": summary,
                    "samples": [
                        {
                            "question": r["messages"][1]["content"].split("질문:")[-1].strip(),
                            "kind": r.get("kind", "answerable"),
                            **s,
                        }
                        for r, s in zip(rows, results, strict=True)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"저장: {out}")


if __name__ == "__main__":
    asyncio.run(main())
