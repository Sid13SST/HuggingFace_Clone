.PHONY: help setup up down logs migrate test lint fmt eval eval-baseline clean

help:
	@echo "setup          install deps into .venv"
	@echo "up / down      start / stop postgres+minio"
	@echo "migrate        apply schema for both projects"
	@echo "test           run the test suite"
	@echo "lint / fmt     ruff check / ruff format"
	@echo "eval           run both eval suites against the current code"
	@echo "eval-baseline  run and write the result as the new baseline"

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[ledgerline,sightline,dev]" || \
	  .venv/Scripts/pip install -e ".[ledgerline,sightline,dev]"

up:
	docker compose up -d --wait

down:
	docker compose down

logs:
	docker compose logs -f db

migrate:
	python -m shared.db migrate ledgerline/schema.sql sightline/schema.sql

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

eval:
	evalctl run ledgerline sightline

eval-baseline:
	evalctl run ledgerline sightline --write-baseline

clean:
	rm -rf .pytest_cache .ruff_cache data/cache evals/runs
