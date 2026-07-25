.PHONY: install install-full lint test serve eval bench

install:
	pip install -e ".[dev,korean]"

install-full:
	pip install -e ".[dev,korean,embeddings,pdf]"

lint:
	ruff check src tests

test:
	pytest -q

serve:
	uvicorn --factory kodoc.service.app:create_app --host 0.0.0.0 --port 9000 --reload

eval:
	python eval/run_eval.py

bench:
	python benchmarks/bench_serving.py \
		--base-url $${KODOC_LLM_BASE_URL:-http://localhost:8000/v1} \
		--model $${KODOC_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
