.PHONY: ingest rebuild-kb verify-m1 install run test lint fmt up down migrate revision

install:
	cd backend && pip install -e ".[dev,rag]"

run:
	cd backend && uvicorn app.main:app --reload --port 8000

test:
	cd backend && pytest

lint:
	cd backend && ruff check . && ruff format --check . && mypy app tests

fmt:
	cd backend && ruff check --fix . && ruff format .

up:
	docker compose up --build

down:
	docker compose down

migrate:
	cd backend && alembic upgrade head

revision:
	cd backend && alembic revision --autogenerate -m "$(m)"

verify-m1:
	python scripts/verify_m1.py

ingest:
	python scripts/ingest_docs.py

rebuild-kb:
	python scripts/ingest_docs.py --rebuild
