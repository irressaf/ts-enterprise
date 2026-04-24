.PHONY: process-data train-forecast fit_all test test-data test-all lint

PYTHON = ./.venv/bin/python
PYTEST = ./.venv/bin/pytest
RUFF = ./.venv/bin/ruff

process-data:
	$(PYTHON) scripts/process_raw_data.py

train-forecast:
	$(PYTHON) scripts/train_and_forecast.py

fit-all:
	$(MAKE) process-data
	$(MAKE) train-forecast

test:
	$(PYTEST) -m "unit or integration"

test-data:
	$(PYTEST) -m "data_quality" --no-cov

test-all:
	$(MAKE) test
	$(MAKE) test-data

lint:
	$(RUFF) check .
