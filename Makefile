SHELL = /bin/bash
PYTHON ?= python3


.PHONY: all
all: install


# autodetect CUDA version if possible
CUDA ?= $(shell (which nvcc && nvcc --version) | grep -oP "(?<=release )[0-9.]+")


# Determine correct torch package to install
TORCH_CUDA_8.0 = cu80
TORCH_CUDA_9.0 = cu90
TORCH_CUDA_10.0 = cu100
TORCH_PLATFORM ?= $(if $(TORCH_CUDA_$(CUDA)),$(TORCH_CUDA_$(CUDA)),cpu)
PY3_MINOR = $(shell $(PYTHON) -c "import sys; print(sys.version_info.minor)")
TORCH_Linux = http://download.pytorch.org/whl/${TORCH_PLATFORM}/torch-1.0.0-cp3${PY3_MINOR}-cp3${PY3_MINOR}m-linux_x86_64.whl
TORCH_Darwin = torch
TORCH ?= $(TORCH_$(shell uname -s))


# determine correct cupy package to install
CUPY_8.0 = cupy-cuda80
CUPY_9.0 = cupy-cuda90
CUPY_10.0 = cupy-cuda100
CUPY ?= $(CUPY_$(CUDA))


.PHONY: show_cuda_version
show_cuda_version:
	@echo Found CUDA version: $(if $(CUDA), $(CUDA), None)
	@echo Will install torch with: $(if $(TORCH), pip install $(TORCH), **not installing torch**)
	@echo 'Will install cupy with: ' $(if $(CUPY), pip install $(CUPY), **not installing cupy**)


envDir = venv
envPrompt ?= "(taiyaki) "
pyTestArgs ?=
override pyTestArgs += --durations=20 -v

buildDir = build


.PHONY: install
install:
	rm -rf ${envDir}
	virtualenv --python=${PYTHON} --prompt=${envPrompt} ${envDir}
	source ${envDir}/bin/activate && \
	    pip install pip --upgrade && \
	    mkdir -p ${buildDir}/wheelhouse/${CUDA} && \
	    pip download --dest ${buildDir}/wheelhouse/${CUDA} ${TORCH} && \
	    pip install --find-links ${buildDir}/wheelhouse/${CUDA} --no-index torch && \
	    pip install -r requirements.txt ${CUPY} && \
	    pip install -r develop_requirements.txt && \
	    ${PYTHON} setup.py develop
	@echo "To activate your new environment:  source ${envDir}/bin/activate"


.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python3-virtualenv python3-pip python3-setuptools git \
	    libblas3 libblas-dev python3-dev lsb-release virtualenv


.PHONY: sdist
sdist:
	${PYTHON} setup.py sdist


.PHONY: bdist_wheel
bdist_wheel:
	${PYTHON} setup.py bdist_wheel
	ls -l dist/*.whl


.PHONY: test
test: unittest


.PHONY: unittest
unittest:
	mkdir -p ${buildDir}/unittest
	cd ${buildDir}/unittest && ${PYTHON} -m pytest ${pyTestArgs} ../../test/unit


.PHONY: acctest
accset ?=
acctest:
	mkdir -p ${buildDir}/acctest
	pip install -r test/acceptance/requirements.txt
	cd ${buildDir}/acctest && ${PYTHON} -m pytest ${pyTestArgs} ../../test/acceptance/${accset}


.PHONY: clean
clean:
	rm -rf ${buildDir}/ dist/ deb_dist/ *.egg-info/ ${envDir}/
	rm taiyaki/ctc/ctc.c \
		taiyaki/squiggle_match/squiggle_match.c taiyaki/version.py
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete


.PHONY: autopep8 pep8
pyDirs := taiyaki test bin models misc
pyFiles := $(shell find *.py ${pyDirs} -type f -name "*.py")
autopep8:
	autopep8 --ignore E203 -i --max-line-length=120 ${pyFiles}
pep8:
	pep8 --ignore E203,E402 --max-line-length=120 ${pyFiles}


.PHONY: workflow
workflow:
	./workflow/remap_from_samrefs_then_train_test_workflow.sh
	./workflow/remap_from_samrefs_then_train_multireadf5_test_workflow.sh
	./workflow/remap_from_samrefs_then_train_squiggle_test_workflow.sh
#(The scripts each check to see if the training log file and chunk log file exist and contain data)
