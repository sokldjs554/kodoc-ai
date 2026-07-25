from pathlib import Path

import pytest

from kodoc.llm.prompts import VLM_PARSE_PROMPT
from kodoc.parsing.vlm_parser import VLMDocParser


class RecordingLLM:
    """VLM 서버 없이 요청 구성만 검증하기 위한 대역."""

    def __init__(self) -> None:
        self.last_messages: list[dict] | None = None
        self.last_temperature: float | None = None

    async def chat(self, messages, temperature=None, max_tokens=None) -> str:
        self.last_messages = messages
        self.last_temperature = temperature
        return "# 파싱된 문서\n\n| 항목 | 값 |\n|---|---|\n| 온도 | 24도 |"


@pytest.fixture
async def parser():
    p = VLMDocParser(base_url="http://unused:8001/v1", model="test-vlm")
    real_client = p._llm
    p._llm = RecordingLLM()
    yield p
    await real_client.aclose()


async def test_parse_image_bytes_builds_openai_vision_message(parser):
    result = await parser.parse_image_bytes(b"\x89PNG fake image", mime="image/png")
    assert result.startswith("# 파싱된 문서")

    fake = parser._llm
    assert fake.last_messages is not None
    content = fake.last_messages[0]["content"]
    text_parts = [p for p in content if p["type"] == "text"]
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert text_parts[0]["text"] == VLM_PARSE_PROMPT
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_parse_is_deterministic_temperature_zero(parser):
    await parser.parse_image_bytes(b"img", mime="image/jpeg")
    assert parser._llm.last_temperature == 0.0


async def test_parse_image_file_maps_mime_by_suffix(parser, tmp_path: Path):
    path = tmp_path / "scan.jpg"
    path.write_bytes(b"fake jpeg")
    await parser.parse_image_file(path)
    url = parser._llm.last_messages[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


async def test_parse_image_file_rejects_unknown_suffix(parser, tmp_path: Path):
    path = tmp_path / "scan.bmp"
    path.write_bytes(b"bmp")
    with pytest.raises(ValueError, match="지원하지 않는 이미지 형식"):
        await parser.parse_image_file(path)
