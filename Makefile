.PHONY: up stop down logs test lint fmt

up:
	docker compose up -d --build

# Stops the containers and KEEPS the database. This is the one to use between sessions.
stop:
	docker compose stop

# `-v` deletes the database volume: every ingested document and embedding goes with it,
# and re-ingesting the corpus takes minutes. Use only for a deliberate clean slate.
down:
	docker compose down -v

logs:
	docker compose logs -f api

test:
	.venv/bin/python -m pytest -v

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

fmt:
	.venv/bin/ruff format .
