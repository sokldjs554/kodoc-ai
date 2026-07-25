"""간단한 CLI: 문서 색인, 검색, 서버 실행.

사용 예:
  kodoc ingest data/samples/smartfarm_manual.md --index-dir ./index
  kodoc search "양액 EC 기준" --index-dir ./index
  kodoc serve --port 9000

serve는 KODOC_INDEX_DIR(기본 ./index)에 저장된 인덱스를 기동 시 복원하므로,
ingest → serve 순서로 실행하면 색인한 문서가 그대로 서비스된다. --index-dir을
기본값이 아닌 경로로 썼다면 서버에도 KODOC_INDEX_DIR로 같은 경로를 알려줘야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kodoc.config import get_settings
from kodoc.rag.chunker import chunk_text
from kodoc.rag.embedder import create_embedder
from kodoc.rag.retriever import HybridRetriever


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = Path(args.file)
    if not path.exists():
        print(f"파일이 없습니다: {path}", file=sys.stderr)
        return 1
    embedder = create_embedder(settings.embedder)
    index_dir = Path(args.index_dir)
    if (index_dir / "meta.json").exists():
        retriever = HybridRetriever.load(str(index_dir), embedder=embedder, rrf_k=settings.rrf_k)
    else:
        retriever = HybridRetriever(embedder=embedder, rrf_k=settings.rrf_k)
    chunks = chunk_text(
        path.read_text(encoding="utf-8"),
        doc_id=path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retriever.add_chunks(chunks)
    retriever.save(str(index_dir))
    print(f"색인 완료: {path.name} → {len(chunks)}개 청크 (전체 {len(retriever)}개)")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedder = create_embedder(settings.embedder)
    retriever = HybridRetriever.load(args.index_dir, embedder=embedder, rrf_k=settings.rrf_k)
    results = retriever.search(args.query, k=args.top_k, mode=args.mode)
    for i, result in enumerate(results, start=1):
        preview = result.chunk.text.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        print(
            f"{i}. [{result.chunk.key}] rrf={result.score:.4f} "
            f"(dense#{result.dense_rank}, bm25#{result.bm25_rank})\n   {preview}"
        )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("kodoc.service.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kodoc", description="KoDoc — 한국어 문서 이해 RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="문서를 색인한다 (.md/.txt)")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--index-dir", default="./index")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_search = sub.add_parser("search", help="색인에서 검색한다")
    p_search.add_argument("query")
    p_search.add_argument("--index-dir", default="./index")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p_search.set_defaults(func=_cmd_search)

    p_serve = sub.add_parser("serve", help="API 서버를 띄운다")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=9000)
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
