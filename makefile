LOCDIR          := .
PYHESIVE_DIR     = $(addprefix $(LOCDIR)/, pyhesive)
TESTDIR          = $(addprefix $(PYHESIVE_DIR)/, test)
PYTHON3          = python3
PIP3             = $(PYTHON3) -m pip
PYVENV_DIRS     := $(addprefix $(LOCDIR)/, venv)

.PHONY: test clean clean-build clean-venv clean-pyc clean-pytest $(PYVENV_DIRS) profile

help:
	@printf "Usage: make [MAKE_OPTIONS] [target] (see 'make --help' for MAKE_OPTIONS)\n"
	@printf ""
	@awk '								\
	{								\
	  if ($$0 ~ /^.PHONY: [a-zA-Z\-\0-9]+$$/) {			\
	    helpCommand = substr($$0, index($$0, ":") + 2);		\
	    if (helpMessage) {						\
	      printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
	      helpMessage = "";						\
	    }								\
	  } else if ($$0 ~ /^[a-zA-Z\-\0-9.]+:/) {			\
	    helpCommand = substr($$0, 0, index($$0, ":"));		\
	    if (helpMessage) {						\
	      printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
	      helpMessage = "";						\
	    }								\
	  } else if ($$0 ~ /^##/) {					\
	    if (helpMessage) {						\
	      helpMessage = helpMessage"\n                     "substr($$0, 3); \
	    } else {							\
	      helpMessage = substr($$0, 3);				\
	    }								\
	  } else {							\
	    if (helpMessage) {						\
	      print "\n                     "helpMessage"\n";		\
	    }								\
	    helpMessage = "";						\
	  }								\
	}'								\
	$(MAKEFILE_LIST)

## -- commonly used --

## install the library from src
install:
	-@$(PIP3) install .

## remove generated local files
clean: clean-build clean-venv clean-pyc clean-pytest

## uninstall the package (due to limitations with pip
## does not uninstall dependencies)
uninstall:
	-@$(PIP3) uninstall pyhesive

## -- testing --

package: clean
	@$(PYTHON3) setup.py sdist bdist_wheel

## upload package to testpypi
test-upload: package
	-@$(PYTHON3) -m twine upload --repository testpypi dist/*

## upload package to pypi
upload: package
	-@$(PYTHON3) -m twine upload dist/*

create-venv:
	-@test -d venv || virtualenv -p $(PYTHON3) venv

## test that package installs cleanly from testpypi and pypi
test-install: create-venv
	@echo "==================================================================="
	@echo "                 Installing From TestPyPi"
	@echo "==================================================================="
	@. $(LOCDIR)/venv/bin/activate && \
	$(PIP3) install --upgrade pip setuptools && \
	$(PIP3) install --index-url https://test.pypi.org/simple/ --no-deps --upgrade pyhesive && \
	$(PIP3) install pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@. $(LOCDIR)/venv/bin/activate && \
	cd $(LOCDIR)/bin && pyhesive-insert --help > /dev/null && cd - >/dev/null
	@echo "==================================================================="
	@echo "                   Installing From PyPi"
	@echo "==================================================================="
	@. $(LOCDIR)/venv/bin/activate && $(PIP3) install --upgrade pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@echo "==================================================================="
	@echo "          All Install Tests Completed Successfully"
	@echo "==================================================================="

vermin:
	@vermin ./pyhesive ./bin

## run full test-suite
test: create-venv vermin
	@. $(LOCDIR)/venv/bin/activate && \
	$(PIP3) install --upgrade pip setuptools && \
	$(PIP3) install -e .[test] && \
	vermin ./pyhesive ./bin && \
	$(PYTHON3) -m pytest $(PYTEST_ARGS)
	@echo "==================================================================="
	@echo "               All Tests Completed Successfully"
	@echo "==================================================================="

## -- misc --

clean-pyc:
	-find . -name '*.pyc' -exec rm -f {} +
	-find . -name '*.pyo' -exec rm -f {} +
	-find . -name '*~' -exec rm -f {} +
	-find . -name '__pycache__' -exec rm -rf {} +

clean-build:
	-rm -rf build/
	-rm -rf dist/
	-rm -rf .eggs/
	-find . -name '*.egg-info' -exec rm -rf {} +
	-find . -name '*.egg' -exec rm -f {} +

clean-pytest:
	-rm -rf .pytest_cache/
	-rm -f .coverage

clean-venv: $(PYVENV_DIRS)
	-rm -f ./pyvenv.cfg

$(PYVENV_DIRS):
	-${RM} -r $@

## install the library in development mode
install-dev:
	-@$(PIP3) install -e .[test]

## profile the code
profile:
	-@cd $(LOCDIR)/bin && \
	$(PYTHON3) -m cProfile -o pyhesive.prof ./pyhesive-insert $(PROFILE_ARGS) && \
	snakeviz pyhesive.prof
