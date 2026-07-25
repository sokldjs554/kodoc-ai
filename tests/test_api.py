import pytest
from fastapi.testclient import TestClient

from kodoc.llm.client import LLMError
from kodoc.pipeline import RAGPipeline
from kodoc.rag.embedder import HashEmbedder
from kodoc.rag.retriever import HybridRetriever
from kodoc.service.app import create_app
from tests.conftest import SAMPLE_DOC, FakeLLM


@pytest.fixture
def client(pipeline) -> TestClient:
    return TestClient(create_app(pipeline))


@pytest.fixture
def loaded_client(client) -> TestClient:
    response = client.post("/documents", json={"doc_id": "manual.md", "text": SAMPLE_DOC})
    assert response.status_code == 200
    return client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_chunks"] == 0


def test_ingest_document(client):
    response = client.post("/documents", json={"doc_id": "manual.md", "text": SAMPLE_DOC})
    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] > 0
    assert body["total_chunks"] == body["num_chunks"]


def test_ingest_rejects_empty_text(client):
    response = client.post("/documents", json={"doc_id": "d", "text": ""})
    assert response.status_code == 422


def test_ingest_file_upload(client):
    response = client.post(
        "/documents/file",
        files={"file": ("manual.md", SAMPLE_DOC.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    assert response.json()["doc_id"] == "manual.md"


def test_ingest_file_rejects_unsupported_type(client):
    response = client.post(
        "/documents/file",
        files={"file": ("scan.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 415


def test_search(loaded_client):
    response = loaded_client.post("/search", json={"query": "양액 EC 범위", "top_k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["section"] == "양액 관리"
    assert "score" in results[0]


def test_search_invalid_mode(loaded_client):
    response = loaded_client.post("/search", json={"query": "온도", "mode": "elastic"})
    assert response.status_code == 422


def test_ask(loaded_client, fake_llm):
    response = loaded_client.post("/ask", json={"question": "주간 적정 온도는?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_llm.answer
    assert body["sources"]


def test_ask_stream_sse_format(loaded_client):
    response = loaded_client.post("/ask/stream", json={"question": "양액 pH 기준은?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    text = response.text
    assert "event: sources" in text
    assert "event: token" in text
    assert text.rstrip().endswith("data: {}")
    assert "event: done" in text


class FailingStreamLLM(FakeLLM):
    """생성 도중 실패하는 LLM — 스트림 절단 시나리오 재현용."""

    async def chat_stream(self, messages, temperature=None, max_tokens=None):
        yield "부분 응답"
        raise LLMError("LLM 서버 오류 503")


def test_ask_stream_emits_error_event_on_llm_failure(settings):
    pipeline = RAGPipeline(
        retriever=HybridRetriever(embedder=HashEmbedder()),
        llm=FailingStreamLLM(),
        settings=settings,
    )
    pipeline.ingest_text(SAMPLE_DOC, doc_id="manual.md")
    client = TestClient(create_app(pipeline))

    response = client.post("/ask/stream", json={"question": "온도 기준은?"})
    # 스트림 시작 후라 상태코드는 200이지만, error 이벤트로 실패를 명시해야 한다
    assert response.status_code == 200
    text = response.text
    assert "event: sources" in text
    assert "event: error" in text
    assert "LLM 호출 실패" in text
    assert "event: done" not in text
