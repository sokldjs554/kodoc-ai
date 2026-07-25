import pytest

from kodoc.rag.tokenizer import KiwiTokenizer, RegexTokenizer, get_tokenizer


def test_regex_tokenizer_splits_words():
    tokens = RegexTokenizer().tokenize("온도는 24도, pH 5.8 유지!")
    assert "온도는" in tokens  # 어절 단위 (조사 미분리 — 폴백의 한계)
    assert "24도" in tokens
    assert "ph" in tokens  # 소문자화


def test_regex_tokenizer_empty():
    assert RegexTokenizer().tokenize("...!!!") == []


def test_get_tokenizer_returns_working_tokenizer():
    tokenizer = get_tokenizer()
    assert tokenizer.tokenize("스마트팜 온도 관리") != []


kiwi_installed = True
try:
    import kiwipiepy  # noqa: F401
except ImportError:
    kiwi_installed = False


@pytest.mark.skipif(not kiwi_installed, reason="kiwipiepy 미설치")
def test_kiwi_tokenizer_strips_josa():
    tokenizer = KiwiTokenizer()
    tokens = tokenizer.tokenize("온도를 관리한다")
    # 조사 "를", 어미가 제거되고 내용어만 남아야 한다
    assert "온도" in tokens
    assert "를" not in tokens


@pytest.mark.skipif(not kiwi_installed, reason="kiwipiepy 미설치")
def test_kiwi_tokenizer_matches_inflected_forms():
    tokenizer = KiwiTokenizer()
    # 표면형이 달라도("유지한다" vs "유지") 같은 내용어 토큰을 공유해야 한다
    a = set(tokenizer.tokenize("온도를 유지한다"))
    b = set(tokenizer.tokenize("온도 유지 기준"))
    assert "온도" in a and "온도" in b
    assert "유지" in a and "유지" in b
