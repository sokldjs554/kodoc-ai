"""RAG 생성 태스크의 SFT 데이터셋을 만든다.

목적: 작은 모델(0.5B급)이 KoDoc의 프롬프트 계약 — "검색된 청크만으로 답하고
[n]으로 인용하며, 근거가 없으면 정해진 문장으로 거절" — 을 지키도록 가르치기 위한
학습 데이터.

만드는 방식:
1. data/samples 문서를 KoDoc과 동일한 청커로 자른다.
2. 상위 모델(교사)에게 각 청크에서 질문·답변을 만들게 한다 (distillation).
3. **생성 결과를 규칙으로 검증**해 통과한 것만 남긴다. 교사 모델도 형식을 틀리기
   때문에, 걸러내지 않으면 그 오류를 그대로 학습시키게 된다.
4. 근거 없는 질문(unanswerable)을 섞는다 — 거절을 배우려면 거절 예시가 필요하다.
   이 샘플은 교사 없이 규칙으로 만든다(정답이 고정 문장이라 생성할 이유가 없다).

교사 모델은 OpenAI 호환 엔드포인트면 무엇이든 된다(KODOC_LLM_* 설정 사용).

사용:
    python finetune/build_dataset.py --out finetune/data/sft.jsonl
    python finetune/build_dataset.py --out finetune/data/sft.jsonl --per-chunk 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from kodoc.config import get_settings
from kodoc.llm.client import LLMClient, LLMError
from kodoc.llm.prompts import RAG_SYSTEM_PROMPT
from kodoc.rag.chunker import chunk_text

REFUSAL = "제공된 문서에서 해당 내용을 찾을 수 없습니다."
CITE = re.compile(r"\[(\d+)\]")

TEACHER_PROMPT = """다음은 한국어 업무 문서의 발췌입니다.

---
{chunk}
---

이 발췌만으로 답할 수 있는 질문과 답변을 {n}쌍 만드세요.

규칙:
1. 질문은 실제 사용자가 물어볼 법한 자연스러운 한국어여야 합니다.
2. 답변은 발췌에 적힌 수치·단위·조건을 그대로 옮겨야 합니다. 추측 금지.
3. 답변 끝에는 반드시 `[1]`을 붙이세요.
4. 답변은 두 문장을 넘기지 마세요.
5. 아래 JSON 배열 형식으로만 출력하세요. 설명 금지.

[{{"question": "...", "answer": "... [1]"}}]"""


@dataclass
class Sample:
    question: str
    context_chunks: list[str]
    answer: str
    kind: str  # "answerable" | "unanswerable"
    chunk_idx: int = -1  # 학습/평가 분리를 청크 단위로 하기 위한 출처 표시
    gold: list[str] | None = None  # 평가용: 답변에 반드시 나와야 하는 수치들


def load_chunks(samples_dir: Path, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    """(문서명, 청크 텍스트) 목록. parsed/ 하위의 VLM 산출물은 제외한다."""
    out: list[tuple[str, str]] = []
    for path in sorted(samples_dir.glob("*.md")):
        chunks = chunk_text(
            path.read_text(encoding="utf-8"),
            doc_id=path.stem,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        out.extend((path.stem, c.text) for c in chunks)
    return out


def _extract_json_array(text: str) -> list[dict]:
    """교사 응답에서 JSON 배열만 뽑는다. 코드펜스나 잡담이 붙어도 견디게."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def validate(question: str, answer: str, chunk: str) -> str | None:
    """통과하면 None, 걸러내야 하면 사유 문자열.

    교사 모델의 출력도 믿지 않는다 — 형식 오류를 학습시키면 안 된다.
    """
    if not question or not answer:
        return "빈 질문/답변"
    if len(question) < 5 or len(question) > 120:
        return "질문 길이 이상"
    if len(answer) > 400:
        return "답변이 너무 김"
    cites = CITE.findall(answer)
    if not cites:
        return "인용 없음"
    if any(c != "1" for c in cites):
        return "존재하지 않는 청크 인용"
    if REFUSAL in answer:
        return "답할 수 있는데 거절"
    # 답변의 숫자는 반드시 청크에 있어야 한다 — 지어낸 수치 차단
    for num in re.findall(r"\d+(?:[.,]\d+)?", CITE.sub("", answer)):
        if num not in chunk:
            return f"청크에 없는 수치({num})"
    return None


def answer_numbers(answer: str) -> list[str]:
    """답변에 등장하는 수치. 평가에서 '핵심 값을 그대로 옮겼는가'의 판정 기준이 된다.

    validate()가 이미 "답변의 모든 수치는 청크에 존재"를 보장하므로, 이 값들은
    근거에 실재하는 수치다.
    """
    return re.findall(r"\d+(?:[.,]\d+)?", CITE.sub("", answer))


def build_unanswerable(
    chunks: list[tuple[str, str]], count: int, rng: random.Random
) -> list[Sample]:
    """다른 도메인 청크를 근거로 주고 답할 수 없는 질문을 던진다.

    질문 자체는 다른 문서에서 가져오므로 '자연스럽지만 이 근거로는 답할 수 없는'
    상태가 만들어진다. 정답은 고정 거절 문장이라 교사 모델이 필요 없다.
    """
    by_doc: dict[str, list[str]] = {}
    for doc, text in chunks:
        by_doc.setdefault(doc, []).append(text)
    docs = list(by_doc)
    if len(docs) < 2:
        return []

    # 각 문서의 첫 문장을 질문 소재로 쓴다 (해당 문서 밖에서는 답이 없다)
    topics: dict[str, list[str]] = {}
    for doc, texts in by_doc.items():
        heads = []
        for t in texts:
            first = t.strip().splitlines()[0][:60].strip()
            if first:
                heads.append(first)
        topics[doc] = heads

    out: list[Sample] = []
    for _ in range(count):
        qdoc, cdoc = rng.sample(docs, 2)
        if not topics[qdoc]:
            continue
        topic = rng.choice(topics[qdoc])
        question = f"{topic}에 대해 알려주세요."
        context = rng.sample(by_doc[cdoc], min(2, len(by_doc[cdoc])))
        out.append(
            Sample(question=question, context_chunks=context, answer=REFUSAL, kind="unanswerable")
        )
    return out


def to_chat(sample: Sample, with_gold: bool = False) -> dict:
    """학습용 chat 형식. 추론 시 build_rag_messages가 만드는 것과 같은 모양이어야 한다."""
    blocks = [f"[{i}] {t}" for i, t in enumerate(sample.context_chunks, start=1)]
    context = "\n\n".join(blocks)
    user = f"다음은 검색된 문서 발췌입니다.\n\n{context}\n\n질문: {sample.question}"
    row = {
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": sample.answer},
        ],
        "kind": sample.kind,
    }
    if with_gold and sample.gold:
        row["gold"] = sample.gold
    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 생성 태스크 SFT 데이터셋 생성")
    ap.add_argument("--samples-dir", default="data/samples")
    ap.add_argument("--out", default="finetune/data/sft.jsonl")
    ap.add_argument("--eval-out", default="finetune/data/eval.jsonl")
    ap.add_argument("--eval-chunk-ratio", type=float, default=0.2, help="평가로 뺄 청크 비율")
    ap.add_argument("--per-chunk", type=int, default=5, help="청크당 생성할 Q/A 수")
    ap.add_argument("--unanswerable-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None, help="교사 모델 (기본: KODOC_LLM_MODEL)")
    args = ap.parse_args()

    settings = get_settings()
    rng = random.Random(args.seed)

    chunks = load_chunks(Path(args.samples_dir), settings.chunk_size, settings.chunk_overlap)
    print(f"문서 청크 {len(chunks)}개")

    client = LLMClient(
        base_url=settings.llm_base_url,
        model=args.model or settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )

    samples: list[Sample] = []
    rejected: dict[str, int] = {}
    try:
        for idx, (doc, chunk) in enumerate(chunks, start=1):
            prompt = TEACHER_PROMPT.format(chunk=chunk, n=args.per_chunk)
            try:
                raw = await client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=settings.llm_temperature,
                    max_tokens=1500,
                )
            except LLMError as e:
                print(f"  [{idx}/{len(chunks)}] {doc} — 호출 실패: {e}")
                continue

            kept = 0
            for item in _extract_json_array(raw):
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                reason = validate(q, a, chunk)
                if reason:
                    rejected[reason.split("(")[0]] = rejected.get(reason.split("(")[0], 0) + 1
                    continue
                samples.append(
                    Sample(
                        question=q,
                        context_chunks=[chunk],
                        answer=a,
                        kind="answerable",
                        chunk_idx=idx - 1,
                        gold=answer_numbers(a),
                    )
                )
                kept += 1
            print(f"  [{idx}/{len(chunks)}] {doc} — 채택 {kept}")
    finally:
        await client.aclose()

    # 학습/평가 분리는 **청크 단위**로 한다. 같은 청크에서 나온 Q/A가 양쪽에 걸치면
    # 사실상 답을 외운 상태로 평가하게 되어 수치가 부풀려진다.
    eval_chunks = set(
        rng.sample(range(len(chunks)), max(1, int(len(chunks) * args.eval_chunk_ratio)))
    )
    train_s = [s for s in samples if s.chunk_idx not in eval_chunks]
    eval_s = [s for s in samples if s.chunk_idx in eval_chunks]

    def add_unanswerable(target: list[Sample], pool: list[tuple[str, str]]) -> None:
        n = int(len(target) * args.unanswerable_ratio / max(1e-9, 1 - args.unanswerable_ratio))
        target.extend(build_unanswerable(pool, n, rng))

    add_unanswerable(train_s, [c for i, c in enumerate(chunks) if i not in eval_chunks])
    add_unanswerable(eval_s, [c for i, c in enumerate(chunks) if i in eval_chunks])
    rng.shuffle(train_s)
    rng.shuffle(eval_s)

    write_jsonl(Path(args.out), [to_chat(s) for s in train_s])
    write_jsonl(Path(args.eval_out), [to_chat(s, with_gold=True) for s in eval_s])

    def counts(items: list[Sample]) -> str:
        a = sum(1 for s in items if s.kind == "answerable")
        return f"{len(items)}건 (답변가능 {a} / 거절 {len(items) - a})"

    print(f"\n학습 {counts(train_s)} → {args.out}")
    print(f"평가 {counts(eval_s)} → {args.eval_out}")
    print(f"평가용으로 뺀 청크 {len(eval_chunks)}/{len(chunks)}개 — 학습과 겹치지 않음")
    if rejected:
        print("검증 탈락:", json.dumps(rejected, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
