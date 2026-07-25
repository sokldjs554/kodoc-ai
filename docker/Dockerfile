FROM python:3.11-slim

WORKDIR /app

# 프로젝트 규모상 단일 설치 레이어로 충분하다 (빌드 수 초).
# 규모가 커지면 requirements 분리로 의존성 레이어 캐싱을 고려할 것.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[korean]"

COPY data ./data

EXPOSE 9000
CMD ["uvicorn", "--factory", "kodoc.service.app:create_app", "--host", "0.0.0.0", "--port", "9000"]
