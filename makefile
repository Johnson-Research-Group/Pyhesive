LOCDIR           = ./
DIRS             = $(addprefix $(LOCDIR), pyhesive)
TESTDIR          = $(addprefix $(LOCDIR), test)
PACKAGE_DIRS     = build dist pyhesive.egg-info
PYTHON3          = python3
VENV_DIRS        = venv
PYPACKAGE_DIRS  := $(addprefix $(LOCDIR), $(PACKAGE_DIRS))
PYVENV_DIRS     := $(addprefix $(LOCDIR), $(VENV_DIRS))

.PHONY: clean clean-package clean-venv $(PYPACKAGE_DIRS) $(PYVENV_DIRS)

package:
	-@$(PYTHON3) setup.py sdist bdist_wheel

test-upload: package
	-@$(PYTHON3) -m twine upload --repository testpypi dist/*

upload: package
	-@$(PYTHON3) -m twine upload dist/*

test-install:
	-@test -d venv || virtualenv -p $(PYTHON3) venv
	@echo "==================================================================="
	@echo "                 Installing From TestPyPi"
	@echo "==================================================================="
	@. $(LOCDIR)venv/bin/activate && \
	pip3 install --upgrade pip setuptools && \
	$(PYTHON3) -m pip install --index-url https://test.pypi.org/simple/ --no-deps --no-cache-dir pyhesive && \
	$(PYTHON3) -m pip install --no-cache-dir pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@. $(LOCDIR)venv/bin/activate && \
	cd $(LOCDIR)bin && pyhesive-insert --help && cd -
	@echo "==================================================================="
	@echo "                   Installing From PyPi"
	@echo "==================================================================="
	@. $(LOCDIR)venv/bin/activate && \
	$(PYTHON3) -m pip install --no-cache-dir pyhesive && \
	$(PYTHON3) $(TESTDIR)/testPackage.py
	@echo "==================================================================="
	@echo "          All Install Tests Completed Successfully"
	@echo "==================================================================="


clean-package: $(PYPACKAGE_DIRS)

$(PYPACKAGE_DIRS):
	-${RM} -r $@

clean-venv: $(PYVENV_DIRS)

$(PYVENV_DIRS):
	-${RM} -r $@

clean: clean-package clean-venv
	-${RM} -r __pycache__
