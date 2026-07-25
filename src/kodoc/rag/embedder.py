"""임베더 추상화.

- HashEmbedder: 문자 3-gram 해싱 기반의 결정적 임베더. 외부 의존성·모델 다운로드
  없이 동작하므로 테스트/CI/오프라인 데모에 쓴다. 표면형이 겹치는 텍스트에는
  의미있는 유사도를 주지만 패러프레이즈에는 약하다 — 실서비스 품질용이 아니다.
- SentenceTransformerEmbedder: BAAI/bge-m3 등 실제 모델. 다국어(한국어 포함)
  검색 벤치마크에서 검증된 모델로, 설치 시(-E embeddings) 이걸 쓴다.

모든 임베더는 L2 정규화된 벡터를 돌려주므로 내적 == 코사인 유사도다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HashEmbedder:
    """문자 3-gram 해싱 BoW 임베더 (결정적, 의존성 없음)."""

    def __init__(self, dim: int = 512) -> None:
        self.name = "hash"
        self.dim = dim

    def _hash(self, gram: str) -> int:
        # 파이썬 내장 hash()는 프로세스마다 시드가 달라지므로 FNV-1a를 쓴다.
        h = 0x811C9DC5
        for ch in gram:
            h ^= ord(ch)
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            t = text.lower()
            if len(t) < 3:
                t = t + "  "
            for i in range(len(t) - 2):
                matrix[row, self._hash(t[i : i + 3])] += 1.0
        return _l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """sentence-transformers 기반 임베더 (기본: BAAI/bge-m3)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def create_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name in ("bge-m3", "BAAI/bge-m3"):
        return SentenceTransformerEmbedder("BAAI/bge-m3")
    # 그 외 문자열은 sentence-transformers 모델 이름으로 간주
    return SentenceTransformerEmbedder(name)
