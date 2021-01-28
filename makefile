LOCDIR           = ./
DIRS             = $(addprefix $(LOCDIR), pyhesive)
TESTDIR          = $(addprefix $(LOCDIR), test)
PYTHON           = python3
PACKAGE_DIRS     = build dist pyhesive.egg-info
VENV_DIRS        = venv
PYPACKAGE_DIRS  := $(addprefix $(LOCDIR), $(PACKAGE_DIRS))
PYVENV_DIRS     := $(addprefix $(LOCDIR), $(VENV_DIRS))

.PHONY: clean clean-package clean-venv $(PYPACKAGE_DIRS) $(PYVENV_DIRS)

package:
	-@$(PYTHON) setup.py sdist bdist_wheel

test-upload: package
	-@$(PYTHON) -m twine upload --repository testpypi dist/*

test-install: venv
	-@test -d venv || virtualenv -p $(PYTHON) venv
	@. $(LOCDIR)venv/bin/activate && \
	$(PYTHON) -m pip install --index-url https://test.pypi.org/simple/ --no-deps pyhesive && \
	$(PYTHON) -m pip install pyhesive && \
	$(PYTHON) $(TESTDIR)/testPackage.py && echo "Install tests successfully passed"

clean-package: $(PYPACKAGE_DIRS)

$(PYPACKAGE_DIRS):
	-${RM} -r $@

clean-venv: $(PYVENV_DIRS)

$(PYVENV_DIRS):
	-${RM} -r $@

clean: clean-package clean-venv
	-${RM} -r __pycache__
