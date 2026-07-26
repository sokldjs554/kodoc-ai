"""파이프라인 조립. 설정 → 구성요소 생성을 한곳에 모은다."""

from __future__ import annotations

from pathlib import Path

from kodoc.config import Settings, get_settings
from kodoc.llm.client import LLMClient
from kodoc.parsing.vlm_parser import VLMDocParser
from kodoc.pipeline import RAGPipeline
from kodoc.rag.embedder import create_embedder
from kodoc.rag.retriever import HybridRetriever


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or get_settings()
    embedder = create_embedder(settings.embedder)

    # index_dir에 저장된 인덱스가 있으면 복원한다 — `kodoc ingest`로 만든 색인과
    # 서버 재시작 전 API로 등록한 문서가 여기서 이어진다.
    index_dir = Path(settings.index_dir) if settings.index_dir else None
    if index_dir and (index_dir / "meta.json").exists():
        retriever = HybridRetriever.load(
            str(index_dir), embedder=embedder, rrf_k=settings.rrf_k
        )
    else:
        retriever = HybridRetriever(embedder=embedder, rrf_k=settings.rrf_k)

    llm = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
    return RAGPipeline(retriever=retriever, llm=llm, settings=settings)


def build_vlm_parser(settings: Settings | None = None) -> VLMDocParser:
    """설정에서 VLM 파서를 만든다. temperature 지원 여부가 모델마다 달라 설정에서 받는다."""
    settings = settings or get_settings()
    return VLMDocParser(
        base_url=settings.vlm_base_url,
        model=settings.vlm_model,
        api_key=settings.vlm_api_key,
        temperature=settings.vlm_temperature,
    )
