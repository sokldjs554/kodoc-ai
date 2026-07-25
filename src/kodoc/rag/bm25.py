"""BM25 (Okapi) 구현.

외부 라이브러리 대신 직접 구현했다. 역색인(posting list)을 사용해
질의 토큰이 등장하는 문서만 순회하므로 코퍼스가 커져도 질의 비용은
질의 토큰의 문서 빈도에 비례한다.

IDF는 음수를 피하기 위해 log(1 + (N - df + 0.5) / (df + 0.5)) 변형을 쓴다
(Lucene과 동일). 흔한 토큰이 점수를 깎아버리는 고전 Okapi IDF의 문제를 피한다.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(
        self,
        corpus_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_lens = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0

        # 역색인: token -> [(doc_id, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(corpus_tokens):
            for token, tf in Counter(tokens).items():
                self.postings.setdefault(token, []).append((doc_id, tf))

        self.idf = {
            token: math.log(1.0 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for token, plist in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """전체 문서에 대한 BM25 점수 벡터."""
        out = [0.0] * self.n_docs
        if not self.n_docs:
            return out
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for doc_id, tf in self.postings[token]:
                norm_len = 1.0 - self.b + self.b * self.doc_lens[doc_id] / self.avgdl
                out[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm_len)
        return out

    def top_n(self, query_tokens: list[str], n: int) -> list[tuple[int, float]]:
        """점수 내림차순 상위 n개의 (doc_id, score). 점수 0은 제외한다."""
        scores = self.scores(query_tokens)
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0.0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]
