# Development tasks for the whole suite. Run from the repository root, inside the
# environment built from environment.yml.

PACKAGES := neu-lib blockrun neu-vol neu-morpho neu-mark neu-glance neu-draw neu-proc neu-eval
PYTHON ?= python

.PHONY: install test docs check-versions $(addprefix test-,$(PACKAGES))

# --no-deps: the environment already provides every dependency, and letting pip resolve
# them again would replace conda's compiled packages with PyPI builds.
install:
	$(PYTHON) -m pip install --no-deps $(foreach p,$(PACKAGES),-e ./$(p))

test: $(addprefix test-,$(PACKAGES))

$(addprefix test-,$(PACKAGES)): test-%:
	cd $* && $(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m sphinx -b html -W --keep-going docs docs/_build/html

check-versions:
	$(PYTHON) tools/release.py --check
