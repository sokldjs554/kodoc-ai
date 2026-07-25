import pytest

from kodoc.config import Settings
from kodoc.pipeline import RAGPipeline
from kodoc.rag.embedder import HashEmbedder
from kodoc.rag.retriever import HybridRetriever

SAMPLE_DOC = """# 스마트팜 운영 매뉴얼

## 온도 관리
주간 적정 온도는 24도에서 27도 사이로 유지한다. \
야간에는 18도 이하로 떨어지지 않도록 보온 커튼을 사용한다. \
온도가 30도를 넘으면 차광 커튼을 닫고 환기팬을 최대 출력으로 가동한다.

## 양액 관리
양액의 EC는 1.8에서 2.2 dS/m 범위를 유지한다. pH는 5.8에서 6.2 사이가 적정이다. \
양액 탱크는 매주 월요일 오전에 세척하고 세척 기록을 남긴다.

## 병해충 대응
흰가루병이 발견되면 즉시 해당 구역을 격리하고 유황 훈증을 실시한다. \
진딧물은 천적인 무당벌레를 방사하여 방제할 수 있다.
"""


class FakeLLM:
    """LLM 서버 없이 파이프라인/API를 테스트하기 위한 대역."""

    def __init__(self, answer: str = "주간 적정 온도는 24도에서 27도입니다. [1]") -> None:
        self.answer = answer
        self.last_messages: list[dict] | None = None

    async def chat(self, messages, temperature=None, max_tokens=None) -> str:
        self.last_messages = messages
        return self.answer

    async def chat_stream(self, messages, temperature=None, max_tokens=None):
        self.last_messages = messages
        for i in range(0, len(self.answer), 5):
            yield self.answer[i : i + 5]

    async def aclose(self) -> None:
        pass


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        embedder="hash",
        chunk_size=200,
        chunk_overlap=50,
        top_k=3,
        index_dir=str(tmp_path / "index"),
        _env_file=None,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(settings, fake_llm) -> RAGPipeline:
    retriever = HybridRetriever(embedder=HashEmbedder(), rrf_k=settings.rrf_k)
    return RAGPipeline(retriever=retriever, llm=fake_llm, settings=settings)


@pytest.fixture
def loaded_pipeline(pipeline) -> RAGPipeline:
    pipeline.ingest_text(SAMPLE_DOC, doc_id="manual.md")
    return pipeline
