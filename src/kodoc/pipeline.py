"""RAG 파이프라인 오케스트레이션: 인제스트 → 검색 → 생성.

동시성 주의: 검색(임베딩 인코딩 + BM25)은 동기 CPU 작업이다. async 경로(ask,
ask_stream)에서는 asyncio.to_thread로 이벤트 루프 밖에서 실행한다 — bge-m3처럼
질의당 수십~수백 ms 걸리는 임베더를 쓰면 루프 블로킹이 진행 중인 모든 SSE
스트림을 함께 멈추기 때문이다.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from kodoc.config import Settings
from kodoc.llm.client import LLMClient
from kodoc.llm.prompts import build_rag_messages
from kodoc.rag.chunker import chunk_text
from kodoc.rag.retriever import HybridRetriever, SearchResult


@dataclass
class Answer:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # --- 인제스트 --------------------------------------------------------

    def ingest_text(self, text: str, doc_id: str) -> int:
        """텍스트/마크다운 문서를 청킹·색인한다. 생성된 청크 수를 돌려준다.

        settings.index_dir이 설정되어 있으면 색인 후 디스크에 저장한다
        (서버 재시작 시 build_pipeline이 같은 경로에서 복원한다).
        """
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.add_chunks(chunks)
        if chunks and self.settings.index_dir:
            # 전체 재저장(O(n))이지만 이 규모에서는 수십 ms — 단순함을 택한다.
            self.retriever.save(self.settings.index_dir)
        return len(chunks)

    # --- 질의 ------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None, mode: str = "hybrid"):
        return self.retriever.search(query, k=top_k or self.settings.top_k, mode=mode)

    async def ask(self, question: str, top_k: int | None = None) -> Answer:
        results = await asyncio.to_thread(self.search, question, top_k)
        if not results:
            return Answer(answer="색인된 문서가 없습니다. 먼저 문서를 등록해 주세요.", sources=[])
        answer = await self.llm.chat(
            build_rag_messages(question, results),
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        return Answer(answer=answer, sources=results)

    async def ask_stream(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[SearchResult], AsyncIterator[str]]:
        """(검색 결과, 토큰 스트림)을 돌려준다.

        검색 결과를 먼저 확정해서 돌려주므로, 서비스 계층은 본문 생성이 시작되기
        전에 근거(sources)를 즉시 클라이언트로 내려보낼 수 있다.
        """
        results = await asyncio.to_thread(self.search, question, top_k)

        async def stream() -> AsyncIterator[str]:
            if not results:
                yield "색인된 문서가 없습니다. 먼저 문서를 등록해 주세요."
                return
            async for delta in self.llm.chat_stream(
                build_rag_messages(question, results),
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ):
                yield delta

        return results, stream()
