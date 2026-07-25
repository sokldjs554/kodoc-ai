"""검색 품질 평가: 어절 BM25 vs 형태소 BM25 vs dense vs hybrid.

평가 방식:
- eval_set.jsonl의 각 항목은 (질문, gold 스니펫) 쌍이다.
- gold 스니펫을 포함하는 청크가 상위 k 안에 검색되면 hit로 간주한다.
  (청크 경계와 무관하게 정답 텍스트 기준으로 판정하므로 청킹 설정이
  바뀌어도 평가가 깨지지 않는다)
- 지표: Hit@1, Hit@3, Hit@5, MRR@10
- bm25(어절)은 형태소 분석 없이 공백/기호 단위 토큰으로 색인한 baseline이다.
  한국어에서 조사·어미가 검색을 얼마나 무너뜨리는지 정량화하기 위해 포함했다.

주의: 이 평가셋은 단일 문서(8청크)·18문항의 스모크 테스트 규모다. 상대 비교
(토크나이저/랭커 간)에는 유효하지만 절대 수치를 일반화하면 안 된다.

사용:
  python eval/run_eval.py                     # hash 임베더 (오프라인)
  KODOC_EMBEDDER=bge-m3 python eval/run_eval.py   # bge-m3 (권장, GPU/CPU)
"""

from __future__ import annotations

import json
from pathlib import Path

from kodoc.config import get_settings
from kodoc.rag.chunker import chunk_text
from kodoc.rag.embedder import create_embedder
from kodoc.rag.retriever import HybridRetriever
from kodoc.rag.tokenizer import RegexTokenizer

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "data" / "samples" / "smartfarm_manual.md"
EVAL_PATH = ROOT / "eval" / "data" / "eval_set.jsonl"

KS = [1, 3, 5]
MRR_DEPTH = 10


def evaluate(retriever: HybridRetriever, cases: list[dict], mode: str) -> dict:
    hits = dict.fromkeys(KS, 0)
    reciprocal_ranks: list[float] = []
    for case in cases:
        results = retriever.search(case["question"], k=MRR_DEPTH, mode=mode)
        rank = next(
            (i for i, r in enumerate(results, start=1) if case["gold"] in r.chunk.text),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in KS:
            if rank is not None and rank <= k:
                hits[k] += 1
    n = len(cases)
    return {
        **{f"hit@{k}": hits[k] / n for k in KS},
        f"mrr@{MRR_DEPTH}": sum(reciprocal_ranks) / n,
    }


def main() -> None:
    settings = get_settings()
    print(f"임베더: {settings.embedder}")

    cases = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = chunk_text(
        DOC_PATH.read_text(encoding="utf-8"),
        doc_id=DOC_PATH.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    retriever = HybridRetriever(embedder=create_embedder(settings.embedder))
    retriever.add_chunks(chunks)

    # 어절 baseline: 같은 청크를 형태소 분석 없이(RegexTokenizer) 색인
    eojeol_retriever = HybridRetriever(
        embedder=create_embedder(settings.embedder), tokenizer=RegexTokenizer()
    )
    eojeol_retriever.add_chunks(chunks)

    print(f"문서: {DOC_PATH.name} ({len(chunks)}개 청크), 질문 {len(cases)}개")
    print("(소규모 스모크 평가 — 랭커 간 상대 비교용)\n")

    rows = [
        ("bm25(어절)", eojeol_retriever, "bm25"),
        ("bm25(형태소)", retriever, "bm25"),
        ("dense", retriever, "dense"),
        ("hybrid", retriever, "hybrid"),
    ]
    metric_names = [f"hit@{k}" for k in KS] + [f"mrr@{MRR_DEPTH}"]
    header = "| mode         | " + " | ".join(f"{m:>7}" for m in metric_names) + " |"
    print(header)
    print("|--------------|" + "|".join(["---------"] * len(metric_names)) + "|")
    for label, r, mode in rows:
        metrics = evaluate(r, cases, mode)
        row = " | ".join(f"{metrics[m]:>7.3f}" for m in metric_names)
        print(f"| {label:<12} | {row} |")


if __name__ == "__main__":
    main()
