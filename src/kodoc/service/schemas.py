from pydantic import BaseModel, Field

from kodoc.rag.retriever import SearchResult


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="문서 식별자")
    text: str = Field(..., min_length=1, description="문서 본문 (텍스트/마크다운)")


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int
    total_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")


class SourceOut(BaseModel):
    key: str
    doc_id: str
    section: str
    text: str
    score: float
    dense_rank: int | None
    bm25_rank: int | None

    @classmethod
    def from_result(cls, result: SearchResult) -> "SourceOut":
        return cls(
            key=result.chunk.key,
            doc_id=result.chunk.doc_id,
            section=result.chunk.section,
            text=result.chunk.text,
            score=result.score,
            dense_rank=result.dense_rank,
            bm25_rank=result.bm25_rank,
        )


class SearchResponse(BaseModel):
    results: list[SourceOut]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    num_chunks: int
