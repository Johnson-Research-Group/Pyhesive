LOCDIR           = ./
PACKAGE_ROOT     = $(addprefix $(LOCDIR), pyhesive)
TESTDIR          = $(addprefix $(PACKAGE_ROOT), test)
PACKAGE_DIRS     = build dist pyhesive.egg-info
PYTHON3          = python3
VENV_DIRS        = venv
PYPACKAGE_DIRS  := $(addprefix $(LOCDIR), $(PACKAGE_DIRS))
PYVENV_DIRS     := $(addprefix $(LOCDIR), $(VENV_DIRS))

.PHONY: test clean clean-package clean-venv $(PYPACKAGE_DIRS) $(PYVENV_DIRS) profile

profile:
	-@cd $(LOCDIR)bin && \
	$(PYTHON3) -m cProfile -o pyhesive.prof ./pyhesive-insert $(PROFILE_ARGS) && \
	snakeviz pyhesive.prof

package:
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

clean-package: $(PYPACKAGE_DIRS)

$(PYPACKAGE_DIRS):
	-${RM} -r $@

clean-venv: $(PYVENV_DIRS)

$(PYVENV_DIRS):
	-${RM} -r $@

clean: clean-package clean-venv
	-${RM} -r $(LOCDIR)pyhesive/__pycache__
