import pytest

from kodoc.rag.chunker import Chunk
from kodoc.rag.embedder import HashEmbedder
from kodoc.rag.store import VectorStore


def _chunks(texts: list[str]) -> list[Chunk]:
    return [Chunk(doc_id="d", chunk_id=i, text=t) for i, t in enumerate(texts)]


def test_search_returns_most_similar_first():
    store = VectorStore(HashEmbedder())
    store.add(
        _chunks(
            ["양액 탱크는 매주 세척한다", "온도는 24도로 유지한다", "무당벌레로 진딧물을 방제한다"]
        )
    )
    results = store.search("양액 탱크 세척 주기", k=3)
    assert results[0][0] == 0
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_empty_store_search():
    store = VectorStore(HashEmbedder())
    assert store.search("아무거나", k=5) == []


def test_save_load_roundtrip(tmp_path):
    store = VectorStore(HashEmbedder())
    store.add(_chunks(["첫 번째 문서 내용", "두 번째 문서 내용"]))
    store.save(tmp_path / "index")

    loaded = VectorStore.load(tmp_path / "index", HashEmbedder())
    assert len(loaded) == 2
    assert loaded.chunks[0].text == "첫 번째 문서 내용"
    assert loaded.search("첫 번째 문서", k=1)[0][0] == 0


def test_load_with_different_embedder_raises(tmp_path):
    store = VectorStore(HashEmbedder())
    store.add(_chunks(["내용"]))
    store.save(tmp_path / "index")

    other = HashEmbedder()
    other.name = "다른임베더"
    with pytest.raises(ValueError, match="같은 임베더"):
        VectorStore.load(tmp_path / "index", other)


def test_incremental_add():
    store = VectorStore(HashEmbedder())
    store.add(_chunks(["하나"]))
    store.add([Chunk(doc_id="d2", chunk_id=0, text="둘")])
    assert len(store) == 2
    assert len(store.search("둘", k=2)) == 2
