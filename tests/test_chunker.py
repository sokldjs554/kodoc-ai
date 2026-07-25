from kodoc.rag.chunker import Chunk, chunk_text, split_sentences
from tests.conftest import SAMPLE_DOC


def test_split_sentences_korean():
    sentences = split_sentences("온도를 유지한다. 야간에는 커튼을 닫는다! 기록은 필수다.")
    assert len(sentences) == 3
    assert sentences[0] == "온도를 유지한다."


def test_split_sentences_handles_newlines():
    assert split_sentences("첫 줄\n둘째 줄") == ["첫 줄", "둘째 줄"]


def test_chunk_respects_size_limit():
    text = " ".join(f"{i}번째 문장입니다." for i in range(100))
    chunks = chunk_text(text, doc_id="doc", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_chunk_overlap_repeats_tail_sentence():
    sentences = [f"문장 번호 {i} 입니다." for i in range(30)]
    chunks = chunk_text(" ".join(sentences), doc_id="doc", chunk_size=120, chunk_overlap=40)
    assert len(chunks) > 1
    # overlap 때문에 어떤 문장은 두 개 이상의 청크에 등장해야 한다
    appearance = {s: sum(s in chunk.text for chunk in chunks) for s in sentences}
    assert any(count >= 2 for count in appearance.values())


def test_section_title_prefixed():
    chunks = chunk_text(SAMPLE_DOC, doc_id="manual.md", chunk_size=300, chunk_overlap=50)
    sections = {chunk.section for chunk in chunks}
    assert "온도 관리" in sections
    for chunk in chunks:
        if chunk.section:
            assert chunk.text.startswith(f"[{chunk.section}]")


def test_section_prefix_counts_toward_chunk_size():
    # "[섹션] " 프리픽스를 포함한 최종 텍스트가 chunk_size를 넘으면 안 된다
    text = "## 아주 길고 긴 섹션 제목입니다\n" + " ".join(f"{i}번 문장입니다." for i in range(50))
    chunks = chunk_text(text, doc_id="d", chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_long_sentence_force_split():
    text = "가" * 500
    chunks = chunk_text(text, doc_id="doc", chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 5
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_chunk_ids_sequential_and_key():
    chunks = chunk_text(SAMPLE_DOC, doc_id="manual.md", chunk_size=200, chunk_overlap=40)
    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].key == "manual.md#0"


def test_chunk_roundtrip_dict():
    chunk = Chunk(doc_id="d", chunk_id=3, text="본문", section="섹션")
    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_empty_text_returns_no_chunks():
    assert chunk_text("", doc_id="doc") == []
    assert chunk_text("\n\n\n", doc_id="doc") == []
