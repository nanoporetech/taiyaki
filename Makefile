SHELL = /bin/bash
PYTHON ?= python3.6


.PHONY: all
all: install


# autodetect CUDA version if possible
CUDA ?= $(shell (which nvcc && nvcc --version) | grep -oP "(?<=release )[0-9.]+")

PY3_MINOR = $(shell $(PYTHON) -c "import sys; print(sys.version_info.minor)")

# Determine correct torch package to install
TORCH_CUDA_ = https://download.pytorch.org/whl/cpu/torch-1.5.1%2Bcpu-cp3${PY3_MINOR}-cp3${PY3_MINOR}m-linux_x86_64.whl
TORCH_CUDA_9.2 = https://download.pytorch.org/whl/cu92/torch-1.5.1%2Bcu92-cp3${PY3_MINOR}-cp3${PY3_MINOR}m-linux_x86_64.whl
TORCH_CUDA_10.1 = https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp3${PY3_MINOR}-cp3${PY3_MINOR}m-linux_x86_64.whl
TORCH_CUDA_10.2 = https://download.pytorch.org/whl/cu102/torch-1.5.1-cp3${PY3_MINOR}-cp3${PY3_MINOR}m-linux_x86_64.whl
TORCH_Linux ?= $(TORCH_CUDA_$(CUDA))
TORCH_Darwin = torch
TORCH ?= $(TORCH_$(shell uname -s))


# determine correct cupy package to install
CUPY_9.2 = cupy-cuda92
CUPY_10.0 = cupy-cuda100
CUPY_10.1 = cupy-cuda101
CUPY_10.2 = cupy-cuda102
CUPY ?= $(CUPY_$(CUDA))


.PHONY: show_cuda_version
show_cuda_version:
	@echo Found CUDA version: $(if $(CUDA), $(CUDA), None)
	@echo Will install torch with: $(if $(TORCH), pip install $(TORCH), **not installing torch**)
	@echo 'Will install cupy with: ' $(if $(CUPY), pip install $(CUPY), **not installing cupy**)


envPrompt ?= "(taiyaki) "
pyTestArgs ?=
override pyTestArgs += --durations=20 -v

buildDir = build
cacheDir = $(HOME)/.cache/taiyaki

venv:
	virtualenv --python=${PYTHON} --prompt=${envPrompt} venv || (rm -rf venv; false)
	@if [ -z "${TORCH}" ]; then echo "Torch URL not specified for cuda=${CUDA}. Please check supported cuda versions"; exit 1; fi
	(source venv/bin/activate && \
	    ${PYTHON} venv/bin/pip install pip --upgrade && \
	    mkdir -p ${cacheDir}/wheelhouse/${CUDA} && \
	    ${PYTHON} venv/bin/pip download --dest ${cacheDir}/wheelhouse/${CUDA} ${TORCH} && \
	    ${PYTHON} venv/bin/pip install --find-links ${cacheDir}/wheelhouse/${CUDA} --no-index torch && \
	    ${PYTHON} venv/bin/pip install -r requirements.txt ${CUPY} && \
	    ${PYTHON} venv/bin/pip install -r develop_requirements.txt && \
	    ${PYTHON} setup.py develop) || (rm -rf venv ; false)
	@echo "To activate your new environment:  source venv/bin/activate"


.PHONY: install
install: venv


.PHONY: rebuild
rebuild: venv
	source venv/bin/activate && \
	    python setup.py build develop


.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python3-virtualenv python3-pip python3-setuptools git \
	    libblas3 libblas-dev python3-dev lsb-release virtualenv


.PHONY: test
test: unittest acctest


.PHONY: unittest
unittest: rebuild clean_unittest
	source venv/bin/activate && \
		mkdir -p ${buildDir}/unittest && \
		cd ${buildDir}/unittest && ${PYTHON} -m pytest ${pyTestArgs} ../../test/unit


.PHONY: acctest
acctest: rebuild clean_acctest
	source venv/bin/activate && \
		mkdir -p ${buildDir}/acctest && \
		${PYTHON} venv/bin/pip install -r test/acceptance/requirements.txt && \
		cd ${buildDir}/acctest && ${PYTHON} -m pytest ${pyTestArgs} ../../test/acceptance/


.PHONY: clean
clean:
	rm -rf ${buildDir}/ dist/ deb_dist/ *.egg-info/ venv/
	rm -f taiyaki/ctc/ctc.c taiyaki/squiggle_match/squiggle_match.c
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete

.PHONY: clean_acctest
clean_acctest:
	rm -rf ${buildDir}/acctest

.PHONY: clean_unittest
clean_unittest:
	rm -rf ${buildDir}/unittest


.PHONY: autopep8 pep8
pyDirs := taiyaki test bin models misc
pyFiles := $(shell find *.py ${pyDirs} -type f -name "*.py")
autopep8:
	autopep8 -i ${pyFiles} --max-line-length=79

pep8:
	pep8 --ignore E203,E402 --max-line-length=79 ${pyFiles}


.PHONY: workflow
workflow: rebuild
	source venv/bin/activate && \
		envDir=venv ./workflow/remap_from_samrefs_then_train_test_workflow.sh && \
		envDir=venv ./workflow/remap_from_samrefs_then_train_multireadf5_test_workflow.sh && \
		envDir=venv ./workflow/remap_from_samrefs_then_train_squiggle_test_workflow.sh && \
		envDir=venv ./workflow/remap_from_mod_fasta_then_train_test_mod_workflow.sh
#  Needs to be in venv, despite apparently being set in script -- fix?
#(The scripts each check to see if the training log file and chunk log file exist and contain data)


# By default, test_multiGPU.sh uses GPUs 0,1.
# If a different combination is required then set the environment variable CUDA_VISIBLE_DEVICES before running.
# E.g.
# export CUDA_VISIBLE_DEVICES="2,6"
# make multiGPU_test
.PHONY: multiGPU_test
multiGPU_test:
	./workflow/test_multiGPU.sh

