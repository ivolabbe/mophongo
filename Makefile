PY := .venv/bin/python

.PHONY: test test-fast test-scene test-fit

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -x --ff

test-scene:
	$(PY) -m pytest tests/test_scene_fitter.py -v

test-fit:
	$(PY) -m pytest tests/test_scene_fitter.py tests/test_nnls.py -v
