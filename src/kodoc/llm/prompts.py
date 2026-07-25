"""RAG / 문서 파싱 프롬프트.

프롬프트 원칙:
- 근거 없는 생성(할루시네이션)을 막기 위해 "문서에 없으면 없다고 답하라"를
  명시하고, 답변에 근거 청크 번호를 인용하게 한다.
- 컨텍스트는 [1], [2] 번호로 감싸 인용을 기계적으로 검증할 수 있게 한다.
"""

from __future__ import annotations

from kodoc.rag.retriever import SearchResult

RAG_SYSTEM_PROMPT = """당신은 문서 기반 질의응답 어시스턴트입니다.

규칙:
1. 반드시 아래 제공된 문서 발췌 내용만 근거로 답하세요. 사전 지식으로 추측하지 마세요.
2. 답변에 사용한 발췌의 번호를 [1], [2] 형식으로 문장 끝에 인용하세요.
3. 제공된 발췌에 답이 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다."라고만 답하세요.
4. 수치·단위·조건은 문서에 적힌 그대로 정확히 옮기세요.
5. 한국어로 간결하게 답하세요."""


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(f"[{i}] (출처: {result.chunk.key})\n{result.chunk.text}")
    return "\n\n".join(blocks)


def build_rag_messages(question: str, results: list[SearchResult]) -> list[dict]:
    context = format_context(results)
    user = f"""다음은 검색된 문서 발췌입니다.

{context}

질문: {question}"""
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# VLM 문서 파싱 프롬프트 — 표/구조 보존이 핵심이다.
VLM_PARSE_PROMPT = """이 문서 이미지를 마크다운으로 정확하게 변환하세요.

규칙:
1. 모든 텍스트를 빠짐없이 옮기되, 이미지에 없는 내용을 추가하지 마세요.
2. 제목/소제목은 마크다운 헤딩(#, ##)으로, 표는 마크다운 표로 변환하세요.
3. 표의 병합 셀은 값을 각 행에 반복해서 채우세요.
4. 도장·서명·로고는 [서명], [로고]처럼 대괄호로 표시만 하세요.
5. 읽을 수 없는 부분은 [판독불가]로 표시하세요.
6. 설명이나 주석 없이 변환 결과만 출력하세요."""
