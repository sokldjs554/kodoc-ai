from kodoc.rag.bm25 import BM25

CORPUS = [
    ["온도", "관리", "커튼", "환기"],
    ["양액", "ec", "ph", "탱크", "세척"],
    ["병해충", "흰가루병", "격리", "진딧물"],
    ["온도", "야간", "보온"],
]


def test_exact_term_ranks_matching_doc_first():
    bm25 = BM25(CORPUS)
    top = bm25.top_n(["양액", "세척"], n=2)
    assert top[0][0] == 1


def test_term_in_multiple_docs():
    bm25 = BM25(CORPUS)
    top_ids = [doc_id for doc_id, _ in bm25.top_n(["온도"], n=4)]
    assert set(top_ids) == {0, 3}


def test_unknown_token_scores_zero():
    bm25 = BM25(CORPUS)
    assert bm25.top_n(["없는토큰"], n=3) == []
    assert bm25.scores(["없는토큰"]) == [0.0] * len(CORPUS)


def test_idf_positive_even_for_common_terms():
    # 모든 문서에 등장하는 토큰도 음수 IDF가 되면 안 된다 (Lucene 변형 IDF)
    corpus = [["공통", "a"], ["공통", "b"], ["공통", "c"]]
    bm25 = BM25(corpus)
    assert bm25.idf["공통"] > 0.0


def test_empty_corpus():
    bm25 = BM25([])
    assert bm25.scores(["아무거나"]) == []
    assert bm25.top_n(["아무거나"], n=5) == []
