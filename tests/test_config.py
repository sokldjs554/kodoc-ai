from kodoc.config import Settings


def test_blank_temperature_becomes_none():
    """빈 문자열 → None: temperature를 거부하는 모델을 위해 요청에서 제외한다."""
    s = Settings(llm_temperature="", vlm_temperature="  ", _env_file=None)
    assert s.llm_temperature is None
    assert s.vlm_temperature is None


def test_numeric_temperature_still_parsed():
    s = Settings(llm_temperature="0.7", vlm_temperature="0", _env_file=None)
    assert s.llm_temperature == 0.7
    assert s.vlm_temperature == 0.0


def test_defaults_unchanged():
    s = Settings(_env_file=None)
    assert s.llm_temperature == 0.2
    assert s.vlm_temperature == 0.0
