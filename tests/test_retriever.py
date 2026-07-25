import threading

import pytest

from kodoc.rag.chunker import Chunk, chunk_text
from kodoc.rag.embedder import HashEmbedder
from kodoc.rag.retriever import HybridRetriever, rrf_fuse
from kodoc.rag.tokenizer import RegexTokenizer
from tests.conftest import SAMPLE_DOC


def _build() -> HybridRetriever:
    retriever = HybridRetriever(embedder=HashEmbedder())
    retriever.add_chunks(chunk_text(SAMPLE_DOC, doc_id="manual.md", chunk_size=200))
    return retriever


def test_rrf_fuse_math():
    fused = rrf_fuse([[10, 20], [20, 10]], k=60)
    # 두 랭킹에서 1위+2위를 나눠 가진 문서는 점수가 같아야 한다
    assert abs(fused[10] - fused[20]) < 1e-12
    assert abs(fused[10] - (1 / 61 + 1 / 62)) < 1e-12


def test_rrf_fuse_prefers_doc_in_both_rankings():
    fused = rrf_fuse([[1, 2], [1, 3]], k=60)
    assert fused[1] > fused[2]
    assert fused[1] > fused[3]


def test_hybrid_search_finds_relevant_section():
    retriever = _build()
    results = retriever.search("양액 EC 적정 범위", k=2)
    assert results
    assert results[0].chunk.section == "양액 관리"


def test_search_modes():
    retriever = _build()
    for mode in ("hybrid", "dense", "bm25"):
        results = retriever.search("주간 적정 온도", k=2, mode=mode)
        assert results, f"mode={mode}에서 결과가 없다"
        assert results[0].chunk.section == "온도 관리", f"mode={mode}"


def test_bm25_mode_returns_empty_on_oov_query():
    # 순수 평가 모드: 질의 토큰이 코퍼스에 없으면 다른 랭커를 섞지 않고 빈 결과
    retriever = _build()
    assert retriever.search("blockchain kubernetes", k=2, mode="bm25") == []
    # hybrid에서는 dense가 받쳐주므로 결과가 나온다
    assert retriever.search("blockchain kubernetes", k=2, mode="hybrid")


def test_invalid_k_raises():
    retriever = _build()
    with pytest.raises(ValueError, match="k는 1 이상"):
        retriever.search("온도", k=0)


def test_concurrent_add_chunks_keeps_alignment():
    """동시 색인 회귀 테스트: 락 없이는 chunks/_token_lists/_matrix 정렬이 깨진다."""
    retriever = HybridRetriever(embedder=HashEmbedder(), tokenizer=RegexTokenizer())

    def worker(tag: int) -> None:
        for batch in range(5):
            chunks = [
                Chunk(
                    doc_id=f"d{tag}",
                    chunk_id=batch * 3 + j,
                    text=f"문서{tag} 배치{batch} 청크{j} 내용",
                )
                for j in range(3)
            ]
            retriever.add_chunks(chunks)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(retriever) == 4 * 5 * 3
    # BM25 토큰 리스트가 청크 순서와 정확히 정렬되어야 한다
    for chunk, tokens in zip(retriever.chunks, retriever._token_lists, strict=True):
        assert retriever.tokenizer.tokenize(chunk.text) == tokens
    # dense 행렬 행 수도 일치해야 한다 (lost-update 검출)
    assert retriever.store._matrix.shape[0] == len(retriever.chunks)


def test_empty_retriever_returns_empty():
    retriever = HybridRetriever(embedder=HashEmbedder())
    assert retriever.search("질문", k=5) == []


def test_k_larger_than_corpus():
    retriever = _build()
    results = retriever.search("온도", k=100)
    assert len(results) == len(retriever)


def test_save_load_preserves_search(tmp_path):
    retriever = _build()
    before = retriever.search("흰가루병 격리", k=1)
    retriever.save(str(tmp_path / "index"))

    loaded = HybridRetriever.load(str(tmp_path / "index"), embedder=HashEmbedder())
    after = loaded.search("흰가루병 격리", k=1)
    assert before[0].chunk.key == after[0].chunk.key


def test_ranks_populated_in_hybrid():
    retriever = _build()
    results = retriever.search("양액 pH", k=3)
    assert any(r.dense_rank is not None for r in results)
    assert any(r.bm25_rank is not None for r in results)
