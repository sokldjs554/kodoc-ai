from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """서비스 전역 설정. 환경변수(KODOC_*) 또는 .env 파일로 재정의한다."""

    model_config = SettingsConfigDict(env_prefix="KODOC_", env_file=".env", extra="ignore")

    # LLM — OpenAI 호환 서버 (vLLM serve 등)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_timeout: float = 120.0

    # VLM — 문서 이미지 파싱용
    vlm_base_url: str = "http://localhost:8001/v1"
    vlm_api_key: str = "EMPTY"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

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
