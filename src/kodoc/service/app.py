"""FastAPI 서비스 계층.

스트리밍 설계:
- POST /ask/stream 은 SSE로 응답한다.
- 첫 이벤트(sources)로 검색 근거를 먼저 내려보내고, 이후 token 이벤트로
  LLM 델타를 흘려보낸다. 클라이언트는 근거를 즉시 렌더링한 뒤 답변이
  타이핑되듯 채워지는 UX를 만들 수 있다 (체감 TTFT 개선).
- 스트림 시작 후에는 HTTP 상태코드를 바꿀 수 없으므로, 생성 중 오류는
  error 이벤트로 명시한다. 정상 종료는 항상 done 이벤트 — 클라이언트가
  "완료 / 오류 / 네트워크 절단"을 구분할 수 있어야 한다.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from kodoc import __version__
from kodoc.llm.client import LLMError
from kodoc.pipeline import RAGPipeline
from kodoc.service.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SourceOut,
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    if pipeline is None:
        from kodoc.service.deps import build_pipeline

        pipeline = build_pipeline()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await app.state.pipeline.llm.aclose()

    app = FastAPI(
        title="KoDoc AI",
        version=__version__,
        description="한국어 문서 이해 RAG 서비스 — VLM 파싱 · 하이브리드 검색 · 스트리밍 응답",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline

    def get_pipeline() -> RAGPipeline:
        return app.state.pipeline

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok", version=__version__, num_chunks=len(get_pipeline().retriever)
        )

    @app.post("/documents", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        p = get_pipeline()
        num_chunks = p.ingest_text(request.text, doc_id=request.doc_id)
        if num_chunks == 0:
            raise HTTPException(status_code=422, detail="문서에서 청크를 만들지 못했습니다.")
        return IngestResponse(
            doc_id=request.doc_id, num_chunks=num_chunks, total_chunks=len(p.retriever)
        )

    @app.post("/documents/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile) -> IngestResponse:
        if not file.filename or not file.filename.lower().endswith((".md", ".txt")):
            raise HTTPException(
                status_code=415,
                detail="텍스트(.txt)/마크다운(.md)만 지원합니다. "
                "이미지·PDF는 VLM 파서(kodoc.parsing)로 먼저 변환하세요.",
            )
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(status_code=422, detail="UTF-8 텍스트가 아닙니다.") from e
        p = get_pipeline()
        num_chunks = p.ingest_text(text, doc_id=file.filename)
        if num_chunks == 0:
            raise HTTPException(status_code=422, detail="문서에서 청크를 만들지 못했습니다.")
        return IngestResponse(
            doc_id=file.filename, num_chunks=num_chunks, total_chunks=len(p.retriever)
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        results = get_pipeline().search(request.query, top_k=request.top_k, mode=request.mode)
        return SearchResponse(results=[SourceOut.from_result(r) for r in results])

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        answer = await get_pipeline().ask(request.question, top_k=request.top_k)
        return AskResponse(
            answer=answer.answer,
            sources=[SourceOut.from_result(r) for r in answer.sources],
        )

    @app.post("/ask/stream")
    async def ask_stream(request: AskRequest) -> StreamingResponse:
        p = get_pipeline()
        sources, token_stream = await p.ask_stream(request.question, top_k=request.top_k)

        async def event_stream() -> AsyncIterator[str]:
            yield _sse(
                "sources",
                {"sources": [SourceOut.from_result(r).model_dump() for r in sources]},
            )
            try:
                async for delta in token_stream:
                    yield _sse("token", {"delta": delta})
            except (LLMError, httpx.HTTPError) as e:
                yield _sse("error", {"message": f"LLM 호출 실패: {e}"})
                return
            yield _sse("done", {})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


# 실행: uvicorn --factory kodoc.service.app:create_app  (또는 `kodoc serve`)
# 모듈 레벨에서 create_app()을 호출하지 않는다 — 임포트만으로 설정을 읽고
# 임베더/HTTP 클라이언트가 생성되는 부작용을 피하기 위해 팩토리로만 노출한다.
