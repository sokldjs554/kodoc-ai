"""BM25용 한국어 토크나이저.

한국어는 조사·어미가 어절에 붙기 때문에 공백 단위 토큰으로는 BM25가 제대로
동작하지 않는다 ("온도를" != "온도"). Kiwi 형태소 분석으로 내용어만 남기면
어휘 매칭 기반 검색의 재현율이 크게 오른다. Kiwi가 설치되지 않은 환경에서는
정규식 폴백으로 동작을 보장한다.
"""

from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


_WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")


class RegexTokenizer:
    """형태소 분석기가 없을 때의 최소 폴백. 어절 단위 분리만 수행한다."""

    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD_RE.findall(text)]


class KiwiTokenizer:
    """Kiwi 형태소 분석 기반 토크나이저.

    조사(J*)·어미(E*)·기호를 걸러내고 내용어만 남긴다:
    체언(NN*, NR, NP), 용언 어간(VV, VA), 어근(XR), 외국어/숫자(SL, SN, SH).
    """

    _KEEP_PREFIX = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "SH")
    _shared_kiwi = None  # Kiwi 모델 로딩(~1초)은 프로세스당 1회면 충분하다

    def __init__(self) -> None:
        from kiwipiepy import Kiwi

        if KiwiTokenizer._shared_kiwi is None:
            KiwiTokenizer._shared_kiwi = Kiwi()
        self._kiwi = KiwiTokenizer._shared_kiwi

    def tokenize(self, text: str) -> list[str]:
        return [
            token.form.lower()
            for token in self._kiwi.tokenize(text)
            if token.tag.startswith(self._KEEP_PREFIX)
        ]


def get_tokenizer() -> Tokenizer:
    """가능하면 Kiwi, 아니면 정규식 폴백을 돌려준다."""
    try:
        return KiwiTokenizer()
    except ImportError:
        return RegexTokenizer()
