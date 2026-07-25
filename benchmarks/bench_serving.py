"""LLM 서빙 벤치마크 — OpenAI 호환 엔드포인트 부하 측정.

측정 지표:
- TTFT (Time To First Token): 요청 전송 → 첫 토큰 수신까지. 체감 반응성을 결정.
- TPOT (Time Per Output Token): 첫 토큰 이후 토큰당 평균 생성 시간.
- E2E latency: 요청 전송 → 스트림 종료까지.
- Throughput: 전체 출력 토큰 수 / 벽시계 시간 (tok/s), 그리고 req/s.

동시성 수준을 바꿔가며 실행하면 continuous batching의 효과(동시성이 올라가도
TTFT/TPOT가 완만하게 늘어나는 것)를 직접 확인할 수 있다. vLLM vs 다른 엔진,
FP16 vs AWQ 같은 비교는 서버만 바꿔 띄우고 같은 명령을 다시 실행하면 된다.

사용 예:
  # vLLM 서버 측정 (동시성 1, 8, 32를 순서대로)
  python benchmarks/bench_serving.py \\
      --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
      --num-requests 64 --concurrency 1 8 32 --max-tokens 256 \\
      --output benchmarks/results/vllm_awq.json

주의: 토큰 수는 SSE 델타 청크 수로 근사한다(vLLM은 기본적으로 토큰 단위로
스트리밍). 정확한 토큰 수가 필요하면 응답의 usage 필드를 지원하는 서버에서
--use-usage 옵션을 켠다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# 실제 서비스 트래픽을 흉내낸 한국어 프롬프트 (길이 다양화)
DEFAULT_PROMPTS = [
    "스마트팜에서 온도 관리가 중요한 이유를 세 가지로 요약해 주세요.",
    "양액의 EC와 pH가 작물 생육에 미치는 영향을 초보자도 이해할 수 있게 설명해 주세요.",
    "다음 문장을 존댓말로 바꿔 주세요: 내일까지 보고서 제출해라.",
    "RAG 시스템에서 검색 품질이 답변 품질에 미치는 영향을 설명해 주세요.",
    "한국의 사계절 중 농업에 가장 큰 영향을 주는 계절과 그 이유를 설명해 주세요.",
    "온실 환기 시스템의 종류를 나열하고 각각의 장단점을 표로 정리해 주세요.",
    "병해충 통합 관리(IPM)의 개념을 설명하고 화학적 방제와 비교해 주세요.",
    "데이터 기반 농업 의사결정의 사례를 두 가지 들어 주세요.",
]


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    num_tokens: int = 0
    error: str = ""

    @property
    def tpot(self) -> float:
        if self.num_tokens <= 1:
            return 0.0
        return (self.latency - self.ttft) / (self.num_tokens - 1)


@dataclass
class BenchConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    num_requests: int = 32
    max_tokens: int = 256
    temperature: float = 0.7
    timeout: float = 300.0
    use_usage: bool = False
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))


async def run_single(client: httpx.AsyncClient, config: BenchConfig, prompt: str) -> RequestResult:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": True,
    }
    if config.use_usage:
        payload["stream_options"] = {"include_usage": True}

    start = time.perf_counter()
    ttft = 0.0
    num_tokens = 0
    usage_tokens: int | None = None
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                return RequestResult(ok=False, error=f"HTTP {response.status_code}: {body[:200]!r}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = usage["completion_tokens"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    if num_tokens == 0:
                        ttft = time.perf_counter() - start
                    num_tokens += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")

    latency = time.perf_counter() - start
    if usage_tokens is not None:
        num_tokens = usage_tokens
    return RequestResult(ok=True, ttft=ttft, latency=latency, num_tokens=num_tokens)


async def run_load(config: BenchConfig, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [config.prompts[i % len(config.prompts)] for i in range(config.num_requests)]

    async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=config.timeout,
    ) as client:
        # 워밍업 (모델 로드/컴파일 지연이 측정에 섞이지 않게)
        await run_single(client, config, prompts[0])

        async def bounded(prompt: str) -> RequestResult:
            async with semaphore:
                return await run_single(client, config, prompt)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(bounded(p) for p in prompts))
        wall_time = time.perf_counter() - wall_start

    ok_results = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"  경고: {len(failed)}건 실패 (첫 오류: {failed[0].error})")
    if not ok_results:
        return {"concurrency": concurrency, "error": "모든 요청 실패"}

    def pct(values: list[float], p: float) -> float:
        if len(values) == 1:
            return values[0]
        qs = statistics.quantiles(values, n=100, method="inclusive")
        return qs[min(int(p) - 1, 98)]

    ttfts = [r.ttft for r in ok_results]
    latencies = [r.latency for r in ok_results]
    tpots = [r.tpot for r in ok_results if r.tpot > 0]
    total_tokens = sum(r.num_tokens for r in ok_results)

    return {
        "concurrency": concurrency,
        "num_ok": len(ok_results),
        "num_failed": len(failed),
        "wall_time_s": round(wall_time, 3),
        "requests_per_s": round(len(ok_results) / wall_time, 3),
        "output_tok_per_s": round(total_tokens / wall_time, 1),
        "ttft_p50_ms": round(pct(ttfts, 50) * 1000, 1),
        "ttft_p95_ms": round(pct(ttfts, 95) * 1000, 1),
        "tpot_p50_ms": round(pct(tpots, 50) * 1000, 2) if tpots else None,
        "latency_p50_s": round(pct(latencies, 50), 3),
        "latency_p95_s": round(pct(latencies, 95), 3),
    }


def print_table(rows: list[dict]) -> None:
    columns = [
        ("concurrency", "동시성"),
        ("requests_per_s", "req/s"),
        ("output_tok_per_s", "tok/s"),
        ("ttft_p50_ms", "TTFT p50(ms)"),
        ("ttft_p95_ms", "TTFT p95(ms)"),
        ("tpot_p50_ms", "TPOT p50(ms)"),
        ("latency_p95_s", "지연 p95(s)"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    print("\n" + header)
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row.get(key, "-")) for key, _ in columns) + " |")


async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 호환 서버 서빙 벤치마크")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use-usage", action="store_true", help="usage 필드로 정확한 토큰 수 집계")
    parser.add_argument("--output", help="결과 JSON 저장 경로")
    args = parser.parse_args()

    config = BenchConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_usage=args.use_usage,
    )

    print(f"대상: {config.base_url} / {config.model}")
    print(f"요청 {config.num_requests}개, max_tokens={config.max_tokens}")

    rows = []
    for concurrency in args.concurrency:
        print(f"\n동시성 {concurrency} 측정 중...")
        row = await run_load(config, concurrency)
        rows.append(row)
        print(f"  완료: {json.dumps(row, ensure_ascii=False)}")

    print_table(rows)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"config": {k: v for k, v in vars(args).items()}, "results": rows},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
