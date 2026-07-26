from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """서비스 전역 설정. 환경변수(KODOC_*) 또는 .env 파일로 재정의한다."""

    model_config = SettingsConfigDict(env_prefix="KODOC_", env_file=".env", extra="ignore")

    # LLM — OpenAI 호환 서버 (vLLM serve 등)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    llm_max_tokens: int = 1024
    llm_temperature: float | None = 0.2
    llm_timeout: float = 120.0

    # VLM — 문서 이미지 파싱용
    vlm_base_url: str = "http://localhost:8001/v1"
    vlm_api_key: str = "EMPTY"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # 파싱은 창의성이 필요 없어 0이 기본이지만, 최신 모델(Claude Sonnet 5 등)은
    # temperature 자체를 거부하므로 비워서 끌 수 있어야 한다. 아래 검증기 참고.
    vlm_temperature: float | None = 0.0

    @field_validator("llm_temperature", "vlm_temperature", mode="before")
    @classmethod
    def _blank_means_omit(cls, v: object) -> object:
        """빈 문자열은 "이 파라미터를 보내지 않음"으로 해석한다.

        OpenAI 호환 API라고 해서 파라미터 지원까지 같지는 않다. 최신 Claude 모델은
        temperature/top_p/top_k를 받으면 400을 돌려주므로, `KODOC_LLM_TEMPERATURE=`
        처럼 빈 값을 주면 요청에서 아예 빠지도록 한다 (LLMClient가 None을 제외한다).
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # 검색/인덱싱
    embedder: str = "hash"  # "hash" | "bge-m3"
    chunk_size: int = 600
    chunk_overlap: int = 120
    top_k: int = 5
    rrf_k: int = 60
    index_dir: str = "./index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
