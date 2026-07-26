"""VLM 기반 문서 파싱: 문서 이미지/PDF → 마크다운.

OCR 파이프라인 대신 VLM(Qwen2.5-VL 등)을 쓰는 이유:
- 표·병합셀·2단 레이아웃 등 구조를 마크다운으로 한 번에 복원할 수 있다.
- 전처리(레이아웃 분석 → OCR → 후처리) 단계가 사라져 파이프라인이 단순해진다.
- 트레이드오프: GPU 비용이 크고, 저해상도·손글씨에서 환각 위험이 있다.
  그래서 프롬프트에서 "없는 내용 추가 금지 / 판독불가 표기"를 강제한다.

VLM 서버도 OpenAI 호환 API(vLLM)로 띄우므로 별도 torch 의존성이 없다.
"""

from __future__ import annotations

import base64
from pathlib import Path

from kodoc.llm.client import LLMClient
from kodoc.llm.prompts import VLM_PARSE_PROMPT

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

PAGE_SEPARATOR = "\n\n---\n\n"


class VLMDocParser:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
    ) -> None:
        self._llm = LLMClient(base_url=base_url, model=model, api_key=api_key)
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def aclose(self) -> None:
        await self._llm.aclose()

    async def parse_image_bytes(self, data: bytes, mime: str = "image/png") -> str:
        b64 = base64.b64encode(data).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PARSE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ]
        # 파싱은 창의성이 필요 없으므로 temperature=0이 기본이다. 다만 이 값을 지원하지
        # 않는 모델이 있어(설정에서 비우면 None) 그때는 요청에서 아예 빠진다.
        return await self._llm.chat(
            messages, temperature=self._temperature, max_tokens=self._max_tokens
        )

    async def parse_image_file(self, path: str | Path) -> str:
        path = Path(path)
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"지원하지 않는 이미지 형식: {path.suffix}")
        return await self.parse_image_bytes(path.read_bytes(), mime=mime)

    async def parse_pdf(self, path: str | Path, scale: float = 2.0) -> str:
        """PDF 각 페이지를 이미지로 렌더링해 순서대로 파싱한다.

        pypdfium2/pillow가 필요하다: pip install -e ".[pdf]"
        """
        import io

        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            raise ImportError('PDF 파싱에는 pip install -e ".[pdf]" 가 필요합니다.') from e

        pdf = pdfium.PdfDocument(str(path))
        pages_md: list[str] = []
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                pages_md.append(await self.parse_image_bytes(buf.getvalue(), mime="image/png"))
        finally:
            pdf.close()
        return PAGE_SEPARATOR.join(pages_md)
