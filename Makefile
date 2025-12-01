.PHONY: help install install-dev test test-cov format lint type-check sort-imports clean build

help:
	@echo "SophoSet Makefile commands:"
	@echo "  make install       - Install the package"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run flake8 linter"
	@echo "  make type-check    - Run mypy type checker"
	@echo "  make sort-imports  - Sort imports with isort"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make build         - Build distribution packages"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=sophoset --cov-report=html --cov-report=term

format:
	black sophoset tests

lint:
	flake8 sophoset tests

type-check:
	mypy sophoset

sort-imports:
	isort sophoset tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build
