VENV ?= .venv
UV ?= uv

.PHONY: install test clean

install:
	$(UV) venv $(VENV)
	$(UV) pip install --python $(VENV)/bin/python --editable .

test: install
	$(VENV)/bin/python -m unittest

clean:
	rm -rf $(VENV)

.PHONY: lint
lint: install
	$(UV) run --python $(VENV)/bin/python pylint $(shell git ls-files '*.py')
