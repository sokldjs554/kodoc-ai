"""하이브리드 검색기: 임베딩(dense) + 형태소 BM25(sparse), RRF 융합.

왜 하이브리드인가:
- dense는 패러프레이즈("실내 온도가 너무 높으면?" -> "고온 대응 절차")에 강하지만
  고유명사·수치·코드 같은 정확 매칭에 약하다.
- BM25는 정확 매칭에 강하지만 표현이 달라지면 무너진다.
- 두 랭킹을 RRF(Reciprocal Rank Fusion)로 합치면 점수 스케일을 맞출 필요 없이
  (rank만 사용) 안정적으로 양쪽 장점을 얻는다.

RRF: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet
and Individual Rank Learning Methods" (SIGIR 2009). rrf_k=60은 논문의 권장값.

동시성: 색인(add_chunks)은 store/토큰 리스트/BM25 세 자료구조를 함께 갱신하므로
락 없이 동시 호출되면 순서가 어긋나 검색이 조용히 오염된다 (FastAPI의 sync
엔드포인트는 스레드풀에서 동시 실행된다). RLock으로 읽기·쓰기를 직렬화한다 —
검색은 ms 단위라 직렬화 비용이 무시 가능하고, 이 규모에서는 단순함이 정답이다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from kodoc.rag.bm25 import BM25
from kodoc.rag.chunker import Chunk
from kodoc.rag.embedder import Embedder, HashEmbedder
from kodoc.rag.store import VectorStore
from kodoc.rag.tokenizer import Tokenizer, get_tokenizer

# 각 랭커에서 최종 k의 몇 배수까지 후보를 볼지. RRF는 랭킹 깊이에 관대하지만
# 너무 얕으면 한쪽 랭커에만 잡힌 문서가 유실된다.
CANDIDATE_MULTIPLIER = 4


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion. score(d) = Σ 1 / (k + rank_i(d)), rank는 1부터."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        tokenizer: Tokenizer | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.store = VectorStore(embedder or HashEmbedder())
        self.tokenizer = tokenizer or get_tokenizer()
        self.rrf_k = rrf_k
        self._token_lists: list[list[str]] = []
        self._bm25: BM25 | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.store)

    @property
    def chunks(self) -> list[Chunk]:
        return self.store.chunks

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # 토큰화·임베딩(비싼 부분)은 락 밖에서 수행하고, 자료구조 갱신만 직렬화한다.
        token_lists = [self.tokenizer.tokenize(c.text) for c in chunks]
        with self._lock:
            self.store.add(chunks)
            self._token_lists.extend(token_lists)
            # BM25는 IDF/평균길이가 코퍼스 전체에 걸리므로 증분 갱신 대신 재구축한다.
            # (수만 청크 규모에서 재구축은 밀리초~수십 밀리초 수준)
            self._bm25 = BM25(self._token_lists)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchResult]:
        """mode: "hybrid" | "dense" | "bm25" — 평가/디버깅을 위해 단일 랭커도 노출."""
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 합니다: {k}")
        with self._lock:
            return self._search(query, k, mode)

    def _search(self, query: str, k: int, mode: str) -> list[SearchResult]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        depth = max(k * CANDIDATE_MULTIPLIER, k)

        dense_ids = [i for i, _ in self.store.search(query, depth)]
        bm25_ids = (
            [i for i, _ in self._bm25.top_n(self.tokenizer.tokenize(query), depth)]
            if self._bm25
            else []
        )

        if mode == "dense":
            fused = rrf_fuse([dense_ids], k=self.rrf_k)
        elif mode == "bm25":
            # 순수 BM25 랭킹만 사용한다. 질의 토큰이 코퍼스에 없으면 빈 결과 —
            # 평가/디버깅용 모드에 다른 랭커를 섞으면 측정이 오염된다.
            fused = rrf_fuse([bm25_ids], k=self.rrf_k)
        else:
            fused = rrf_fuse([dense_ids, bm25_ids], k=self.rrf_k)

        dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_ids, start=1)}
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_ids, start=1)}

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            SearchResult(
                chunk=self.chunks[doc_id],
                score=score,
                dense_rank=dense_rank.get(doc_id),
                bm25_rank=bm25_rank.get(doc_id),
            )
            for doc_id, score in top
        ]

    # --- 영속화 ---------------------------------------------------------

    def save(self, index_dir: str) -> None:
        self.store.save(index_dir)

    @classmethod
    def load(
        cls,
        index_dir: str,
        embedder: Embedder | None = None,
        tokenizer: Tokenizer | None = None,
        rrf_k: int = 60,
    ) -> HybridRetriever:
        retriever = cls(embedder=embedder, tokenizer=tokenizer, rrf_k=rrf_k)
        retriever.store = VectorStore.load(index_dir, retriever.store.embedder)
        retriever._token_lists = [
            retriever.tokenizer.tokenize(c.text) for c in retriever.store.chunks
        ]
        if retriever._token_lists:
            retriever._bm25 = BM25(retriever._token_lists)
        return retriever
