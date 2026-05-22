.PHONY: install test lint clean

install:
	pip install -e ".[dev]"

test:
	pytest evals/tests/ -v --tb=short

lint:
	black evals/ --check
	mypy evals/

format:
	black evals/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ __pycache__/
	find . -name "*.pyc" -delete

run:
	oaieval

status:
	python3 -m evals.cli.oaieval status
