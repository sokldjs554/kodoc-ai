from kodoc.service.deps import build_pipeline
from tests.conftest import SAMPLE_DOC


def test_ingest_returns_chunk_count(pipeline):
    n = pipeline.ingest_text(SAMPLE_DOC, doc_id="manual.md")
    assert n > 0
    assert len(pipeline.retriever) == n


async def test_ask_returns_answer_with_sources(loaded_pipeline, fake_llm):
    result = await loaded_pipeline.ask("주간 적정 온도는?")
    assert result.answer == fake_llm.answer
    assert result.sources
    # LLM에 전달된 컨텍스트에 검색된 청크가 들어가야 한다
    user_message = fake_llm.last_messages[-1]["content"]
    assert "온도" in user_message
    assert "[1]" in user_message


async def test_ask_without_documents(pipeline):
    result = await pipeline.ask("아무 질문")
    assert result.sources == []
    assert "문서" in result.answer


async def test_ask_stream_yields_sources_then_tokens(loaded_pipeline, fake_llm):
    sources, stream = await loaded_pipeline.ask_stream("양액 EC 기준은?")
    assert sources  # 스트림 소비 전에 근거가 먼저 확정된다
    collected = "".join([delta async for delta in stream])
    assert collected == fake_llm.answer


def test_search_modes_via_pipeline(loaded_pipeline):
    hybrid = loaded_pipeline.search("온도 관리", mode="hybrid")
    dense = loaded_pipeline.search("온도 관리", mode="dense")
    bm25 = loaded_pipeline.search("온도 관리", mode="bm25")
    assert hybrid and dense and bm25


async def test_ingest_persists_and_new_pipeline_reloads(settings):
    """kodoc ingest → serve, 또는 서버 재시작 시 인덱스가 이어지는지 검증."""
    p1 = build_pipeline(settings)
    n = p1.ingest_text(SAMPLE_DOC, doc_id="manual.md")
    assert n > 0

    p2 = build_pipeline(settings)  # 같은 index_dir에서 복원
    assert len(p2.retriever) == n
    results = p2.search("주간 적정 온도")
    assert results
    assert results[0].chunk.section == "온도 관리"

    await p1.llm.aclose()
    await p2.llm.aclose()
