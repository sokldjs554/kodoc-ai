"""벡터 스토어 (numpy 기반, 파일 영속화).

포트폴리오 규모(수만 청크 이하)에서는 브루트포스 내적이 FAISS보다 단순하고
충분히 빠르다. 인터페이스를 좁게 유지했으므로 규모가 커지면 FAISS/Milvus로
교체하기 쉽다.

주의: 이 클래스 자체는 스레드 안전하지 않다. 동시 접근 직렬화는 상위의
HybridRetriever(락 보유)가 담당한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kodoc.rag.chunker import Chunk
from kodoc.rag.embedder import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2 정규화됨

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.encode([c.text for c in chunks])
        if self._matrix is None:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """코사인 유사도 상위 k개의 (chunk 인덱스, 유사도)."""
        if self._matrix is None or not self.chunks:
            return []
        query_vec = self.embedder.encode([query])[0]
        sims = self._matrix @ query_vec
        k = min(k, len(self.chunks))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        if self._matrix is not None:
            np.save(path / "embeddings.npy", self._matrix)
        meta = {"embedder": self.embedder.name, "dim": self.embedder.dim, "n": len(self.chunks)}
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path, embedder: Embedder) -> VectorStore:
        path = Path(index_dir)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"인덱스는 '{meta['embedder']}'로 만들어졌는데 현재 임베더는 "
                f"'{embedder.name}'입니다. 같은 임베더로 다시 인덱싱하세요."
            )
        store = cls(embedder)
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            store.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        embeddings_path = path / "embeddings.npy"
        if embeddings_path.exists():
            store._matrix = np.load(embeddings_path)
        return store
