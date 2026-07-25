"""한국어 문서 청킹.

전략:
1. 마크다운 헤딩 기준으로 섹션을 나눈다. 섹션 제목은 청크 앞에 "[제목]"으로
   붙여, 검색 시 문맥 단서로 쓰이게 한다 (표제어가 본문에 반복되지 않는
   매뉴얼류 문서에서 특히 효과가 크다).
2. 섹션 내부는 문장 경계에서 자른다. 문장을 중간에 끊으면 임베딩 품질과
   LLM 인용 품질이 모두 떨어지기 때문이다.
3. 인접 청크는 문장 단위 overlap을 둬서 경계에 걸친 내용의 유실을 줄인다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…。])\s+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    section: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


def split_sentences(text: str) -> list[str]:
    """줄바꿈과 문장부호([.!?…。] + 공백) 기준의 가벼운 문장 분리."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for piece in _SENT_SPLIT_RE.split(line):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """(섹션 제목, 본문) 목록. 헤딩이 없으면 제목 없는 단일 섹션."""
    sections: list[tuple[str, str]] = []
    title = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            title = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return sections


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """문서를 섹션·문장 경계 기반 청크로 나눈다.

    chunk_size / chunk_overlap 은 문자 수 기준이다 (한국어는 토큰-문자 비율이
    모델마다 크게 달라 문자 수가 더 예측 가능한 기준이다). chunk_size는 섹션
    프리픽스("[제목] ")를 포함한 최종 청크 텍스트 길이의 상한이다 — 임베딩·컨텍스트
    예산을 chunk_size로 잡아도 조용히 초과되는 일이 없어야 하기 때문.
    """
    chunks: list[Chunk] = []
    chunk_id = 0

    def emit(sentences: list[str], section: str, prefix: str) -> None:
        nonlocal chunk_id
        body = " ".join(sentences).strip()
        if not body:
            return
        chunks.append(Chunk(doc_id=doc_id, chunk_id=chunk_id, text=prefix + body, section=section))
        chunk_id += 1

    for section, body in _split_markdown_sections(text):
        prefix = f"[{section}] " if section else ""
        # 프리픽스를 포함해도 chunk_size를 넘지 않도록 본문 예산을 줄인다.
        # (병적으로 긴 제목이 예산을 다 먹지 않게 최소 16자는 본문에 보장)
        budget = max(chunk_size - len(prefix), 16)

        sentences = split_sentences(body)
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            # budget을 넘는 단일 문장은 그대로 강제 분할한다.
            if len(sentence) > budget:
                if current:
                    emit(current, section, prefix)
                    current, current_len = [], 0
                for i in range(0, len(sentence), budget):
                    emit([sentence[i : i + budget]], section, prefix)
                continue

            if current and current_len + len(sentence) + 1 > budget:
                emit(current, section, prefix)
                # 직전 청크 꼬리 문장들을 overlap 만큼 유지
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) + 1 > chunk_overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                # overlap을 붙이면 새 문장이 안 들어가는 경우엔 overlap 포기
                if tail_len + len(sentence) + 1 > budget:
                    tail, tail_len = [], 0
                current, current_len = tail, tail_len

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            emit(current, section, prefix)

    return chunks
