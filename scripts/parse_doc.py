"""문서 이미지를 VLM으로 파싱해 마크다운으로 저장한다.

설정(KODOC_VLM_*)에서 엔드포인트·모델·temperature를 읽으므로, 모델을 바꿔가며
비교할 때 환경변수만 갈아끼우면 된다. 파싱 모델 비교 기록은 docs/vlm-comparison.md.

사용 예:
  # vLLM으로 띄운 Qwen2.5-VL
  KODOC_VLM_BASE_URL=http://localhost:8000/v1 \
  KODOC_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
  python scripts/parse_doc.py scan.png -o scan_parsed.md

  # temperature를 거부하는 모델은 값을 비운다 (요청에서 아예 제외된다)
  KODOC_VLM_TEMPERATURE= python scripts/parse_doc.py scan.png
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kodoc.config import get_settings
from kodoc.service.deps import build_vlm_parser


async def main() -> None:
    ap = argparse.ArgumentParser(description="VLM 문서 파싱 (이미지/PDF → 마크다운)")
    ap.add_argument("path", help="입력 파일 (.png/.jpg/.jpeg/.webp/.pdf)")
    ap.add_argument("-o", "--output", help="저장 경로 (기본: 입력파일명_parsed.md)")
    args = ap.parse_args()

    src = Path(args.path)
    settings = get_settings()
    print(f"모델: {settings.vlm_model} @ {settings.vlm_base_url}")
    print(f"temperature: {settings.vlm_temperature!r} (None이면 요청에서 제외)")

    parser = build_vlm_parser(settings)
    try:
        if src.suffix.lower() == ".pdf":
            markdown = await parser.parse_pdf(src)
        else:
            markdown = await parser.parse_image_file(src)
    finally:
        await parser.aclose()

    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_parsed.md")
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n---\n저장 완료: {out}")


if __name__ == "__main__":
    asyncio.run(main())
