VENV ?= .venv
UV ?= uv
FILE ?= examples/open_window.py

.PHONY: install test clean

install:
	$(UV) venv $(VENV)
	$(UV) sync --python $(VENV)/bin/python --locked

test: install
	$(VENV)/bin/pytest

clean:
	rm -rf $(VENV)

.PHONY: lint
lint: install
	$(UV) run --python $(VENV)/bin/python pylint $(shell git ls-files '*.py')

.PHONY: check-shortcuts
check-shortcuts: install
	$(VENV)/bin/python -m yek.shortcuts $(FILE)
