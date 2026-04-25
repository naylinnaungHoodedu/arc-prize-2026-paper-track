.PHONY: test-all test-unit audit

test-all: test-unit audit

test-unit:
	python -m pytest -m "not integration"

audit:
	python scripts/run_audit.py
