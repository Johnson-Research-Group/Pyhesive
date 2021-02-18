LOCDIR           = ./
PACKAGE_ROOT     = $(addprefix $(LOCDIR), pyhesive)
TESTDIR          = $(addprefix $(PACKAGE_ROOT), test)
PYTHON3          = python3
VENV_DIRS        = venv
PYVENV_DIRS     := $(addprefix $(LOCDIR), $(VENV_DIRS))

.PHONY: test clean clean-build clean-venv clean-pyc clean-pytest $(PYVENV_DIRS) profile

style:
	@black --line-length=100 ./pyhesive/
	@black --line-length=100 ./bin/pyhesive-insert

profile:
	-@cd $(LOCDIR)bin && \
	$(PYTHON3) -m cProfile -o pyhesive.prof ./pyhesive-insert $(PROFILE_ARGS) && \
	snakeviz pyhesive.prof

package: clean
	@$(PYTHON3) setup.py sdist bdist_wheel

test-upload: package
	-@$(PYTHON3) -m twine upload --repository testpypi dist/*

upload: package
	-@$(PYTHON3) -m twine upload dist/*

create-venv:
	-@test -d venv || virtualenv -p $(PYTHON3) venv

test-install: create-venv
	@echo "==================================================================="
	@echo "                 Installing From TestPyPi"
	@echo "==================================================================="
	@. $(LOCDIR)venv/bin/activate && \
	pip3 install --upgrade pip setuptools && \
	$(PYTHON3) -m pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade pyhesive && \
	$(PYTHON3) -m pip install pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@. $(LOCDIR)venv/bin/activate && \
	cd $(LOCDIR)bin && pyhesive-insert --help > /dev/null && cd - >/dev/null
	@echo "==================================================================="
	@echo "                   Installing From PyPi"
	@echo "==================================================================="
	@. $(LOCDIR)venv/bin/activate && \
	$(PYTHON3) -m pip install --upgrade pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@echo "==================================================================="
	@echo "          All Install Tests Completed Successfully"
	@echo "==================================================================="

test: create-venv
	@. $(LOCDIR)venv/bin/activate && \
	pip3 install --upgrade pytest pytest-xdist pyhesive && \
	$(PYTHON3) -m pytest $(PYTEST_ARGS)
	@echo "==================================================================="
	@echo "               All Tests Completed Successfully"
	@echo "==================================================================="

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

clean-venv: $(PYVENV_DIRS)

$(PYVENV_DIRS):
	-${RM} -r $@

clean: clean-build clean-venv clean-pyc clean-pytest
