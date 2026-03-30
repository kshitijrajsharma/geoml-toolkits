default:
    @just --list

install:
    uv sync

lint:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/

format:
    uv run ruff check --fix src/ tests/
    uv run ruff format src/ tests/

test:
    uv run pytest

typecheck:
    uv run ty check src/

check: lint typecheck test
