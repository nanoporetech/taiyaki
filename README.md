<p align="center">
  <img src="ONT_logo.png">
</p>

# Taiyaki

Taiyaki is research software for training models for basecalling Oxford Nanopore reads. 

Oxford Nanopore's devices measure the flow of ions through a nanopore, and detect changes
in that flow as molecules pass through the pore.
These signals can be highly complex and exhibit long-range dependencies, much like spoken 
or written language. Taiyaki can be used to train neural networks to understand the 
complex signal from a nanopore device, using techniques inspired by state-of-the-art
language processing.

Taiyaki is used to train the models used to basecall DNA and RNA found in Oxford Nanopore's 
Guppy basecaller (version 2.2 at time of writing). This includes the flip-flop models,
which are trained using a technique inspired by Connectionist Temporal Classification
(Graves et al 2006).

Main features:
*  Prepare data for training basecallers by remapping signal to reference sequence
*  Train neural networks for flip-flop basecalling and squiggle prediction
*  Export basecaller models for use in Guppy

Taiyaki is built on top of pytorch and is compatible with Python 3.5 or later.
It is aimed at advanced users, and it is an actively evolving research project, so
expect to get your hands dirty.


# Contents

1. [Installing system prerequisites](#installing-system-prerequisites)
2. [Installing Taiyaki](#installation)
3. [Tests](#tests)
4. [Walk through](#walk-through)
5. [Workflows](#workflows)<br>
        * [Using the workflow Makefile](#using-the-workflow-makefile)<br>
        * [Steps from fast5 files to basecalling](#steps-from-fast5-files-to-basecalling)<br>
        * [Preparing a training set](#preparing-a-training-set)<br>
        * [Basecalling](#basecalling)<br>
        * [Modified bases](#modified-bases)<br>
        * [Abinitio training](#abinitio-training)<br>
6. [Guppy compatibility](#guppy-compatibility)<br>
        * [Standard model parameters](#standard-model-parameters)<br>
7. [Environment variables](#environment-variables)
8. [CUDA](#cuda)<br>
        * [Troubleshooting](#troubleshooting)<br>
9. [Using multiple GPUs](#using-multiple-gpus)<br>
        * [How to launch training with multiple GPUs](#how-to-launch-training-with-multiple-gpus)<br>
        * [Choice of learning rates for multi-GPU training](#choice-of-learning-rates-for-multi-gpu-training)<br>
        * [Selection of GPUs](#selection-of-gpus-for-multi-gpu-training)<br>
        * [More than one multi-GPU training group on a single machine](#more-than-one-multi-gpu-training-group-on-a-single-machine)<br>
10. [Running on SGE](#running-on-an-sge-cluster)<br>
        * [Installation](#SGEinstall)<br>
        * [Execution](#SGEexec)<br>
        * [Selection of multiple GPUs in SGE](#SGEmultiGPU)<br>
11. [Diagnostics](#diagnostics)


# Installing system prerequisites

To install required system packages on ubuntu 16.04:

    sudo make deps

Other linux platforms may be compatible, but are untested.

In order to accelerate model training with a GPU you will need to install CUDA (which should install nvcc and add it to your path.)
See instructions from NVIDIA and the [CUDA](#cuda) section below.

Taiyaki also makes use of the OpenMP extensions for multi-processing.  These are supported
by the system installed compiler on most modern Linux systems but require a more modern version
of the clang/llvm compiler than that installed on MacOS machines.  Support for OpenMP was
adding in clang/llvm in version 3.7 (see http://llvm.org or use brew). Alternatively you
can install GCC on MacOS using homebrew.

Some analysis scripts require a recent version of the [BWA aligner](https://github.com/lh3/bwa).

Windows is not supported.

# Installing Taiyaki

---
**NOTE**
If you intend to use Taiyaki with a GPU, make sure you have installed and set up [CUDA](#cuda) before proceeding.
---

## Install Taiyaki in a new virtual environment

We recommend installing Taiyaki in a self-contained [virtual environment](https://docs.python.org/3/tutorial/venv.html).

The following command creates a complete environment for developing and testing Taiyaki, in the directory **venv**:

    make install

Taiyaki will be installed in [development mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode) so that you can easily test your changes.
You will need to run `source venv/bin/activate` at the start of each session when you want to use this virtual environment.

## Install Taiyaki system-wide or into activated Python environment

Taiyaki can be installed from source using either:

    python3 setup.py install
    python3 setup.py develop #[development mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode)

Alternatively, you can use pip with either:

    pip install path/to/taiyaki/repo
    pip install -e path/to/taiyaki/repo #[development mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode)

# Tests

Tests can be run as follows:

    make workflow           #runs scripts which carry out the workflow for basecall-network training and for squiggle-predictor training
    make acctest            #runs acceptance tests
    make unittest           #runs unit tests
    make multiGPU_test      #runs multi-GPU test (GPUs 0 and 1 must be available, and CUDA must be installed - see below)

If Taiyaki has been installed in a virtual environment, it will have to activated before running tests: `source venv/bin/activate`.  To deactivate, run `deactivate`.

# Walk throughs and further documentation
For a walk-through of Taiyaki model training, including how to obtain sample training data, see [docs/walkthrough.rst](docs/walkthrough.rst).

For an example of training a modifed base model, see [docs/modbase.rst](docs/modbase.rst).

# Workflows

## Using the workflow Makefile 

The file at **workflow/Makefile** can be used to direct the process of generating ingredients for training and then running the training itself.

For example, if we have a directory **read_dir** containing fast5 files, and a fasta file **refs.fa** containing a ground-truth reference sequence for each read, we can (from the Taiyaki root directory) use the command line

    make -f workflow/Makefile MAXREADS=1000 \
        READDIR=read_dir USER_PER_READ_REFERENCE_FILE=refs.fa \
        DEVICE=3 train_remapuser_ref

This will place the training ingredients in a directory **RESULTS/training_ingredients** and the training output (including logs and trained models)
in **RESULTS/remap_training**, using GPU 3 and only reading the first 1000 reads in the directory. The fast5 files may be single or multi-read.

Using command line options to **make**, it is possible to change various other options, including the directory where the results go. Read the Makefile to find out about these options.
The Makefile can also be used to follow a squiggle-mapping workflow.

The paragraph below describes the steps in the workflow in more detail.

## Steps from fast5 files to basecalling

The script **bin/prepare_mapped_reads.py** prepares a file containing mapped signals. This file is the main ingredient used to train a basecalling model.

The simplest workflow looks like this. The flow runs from top to bottom and lines show the inputs required for each stage.
The scripts in the Taiyaki package are shown, as are the files they work with.

                       fast5 files
                      /          \
                     /            \
                    /              \
                   /   generate_per_read_params.py
                   |                |
                   |                |               fasta with reference
                   |   per-read-params file         sequence for each read
                   |   (tsv, contains shift,        (produced with get_refs_from_sam.py
                   |   scale, trim for each read)   or some other method)
                    \               |               /
                     \              |              /
                      \             |             /
                       \            |            /
                        \           |           /
                         \          |          /
                         prepare_mapped_reads.py
                         (also uses remapping flip-flop
                         model from models/)
                                    |
                                    |
                         mapped-signal-file (hdf5)
                                    |
                                    |
                         train_flipflop.py
                         (also uses definition
                         of model to be trained)
                                    |
                                    |
                         trained flip-flop model
                                    |
                                    |
                              dump_json.py
                                    |
                                    |
                         json model definition
                         (suitable for use by Guppy)

Each script in bin/ has lots of options, which you can find out about by reading the scripts.
Basic usage is as follows:

    bin/generate_per_read_params.py <directory containing fast5 files> --output <name of output per_read_tsv file>

    bin/get_refs_from_sam.py <genomic references fasta> <one or more SAM/BAM files> --output <name of output reference_fasta>

    bin/prepare_mapped_reads.py <directory containing fast5 files> <per_read_tsv> <output mapped_signal_file>  <file containing model for remapping>  <reference_fasta>

    bin/train_flipflop.py --device <digit specifying GPU> <pytorch model definition> <mapped-signal files to train with>

Some scripts mentioned also have a useful option **--limit** which limits the number of reads to be used. This allows a quick test of a workflow.


## Preparing a training set

The `prepare_mapped_reads.py` script prepares a data set to use to train a new basecaller. Each member of this data set contains:

  * The raw signal for a complete nanopore read (lifted from a fast5 file)
  * A reference sequence that is the "ground truth" for the that read
  * An alignment between the signal and the reference

As input to this script, we need a directory containing fast5 files (either single-read or multi-read) and a fasta file that contains the ground-truth reference for each read. In order to match the raw signal to the correct ground-truth sequence, the IDs in the fasta file should be the unique read ID assigned by MinKnow (these are the same IDs that Guppy uses in its fastq output). For example, a record in the fasta file might look like:

    >17296436-f2f1-4713-adaf-169ed9cf6aa6
    TATGATGTGAGCTTATATTATTAATTTTGTATCAATCTTATTTTCTAATGTATGCATTTTAATGCTATAAATTTCCTTCTAAGCACTAC...

The recommended way to produce this fasta file is as follows:

  1. Align Guppy fastq basecalls to a reference genome using Guppy Aligner or Minimap. This will produce one or more SAM files.
  2. Use the `get_refs_from_sam.py` script to extract a snippet of the reference for each mapped read. You can filter reads by coverage.

The final input required by `prepare_mapped_signal.py` is a pre-trained basecaller model, which is used to determine the alignment between raw signal and reference sequence.
An example of such a model (for DNA sequenced with pore r9) is provided at `models/mGru256_flipflop_remapping_model_r9_DNA.checkpoint`.
This does make the entire training process somewhat circular: you need a model to train a model.
However, the new training set can be somewhat different from the data that the remapping model was trained on and things still work out.
So, for example, if your samples are a bit weird and whacky, you may be able to improve basecall accuracy by retraining a model with Taiyaki.
Internally, we use Taiyaki to train basecallers after incremental pore updates, and as a research tool into better basecalling methods.
Taiyaki is not intended to enable training basecallers from scratch for novel nanopores.
If it seems like remapping will not work for your data set, then you can use alternative methods
so long as they produce data conformant with [this format](docs/FILE_FORMATS.md).


## Basecalling

Taiyaki comes with a script to perform flip-flop basecalling using a GPU.
This script requires CUDA and cupy to be installed.

Example usage:

    bin/basecall.py <directory containing fast5s> <model checkpoint>  >  <output fasta>

A limited range of models can also be used with Guppy, which will provide better performance and stability.
See the section on [Guppy compatibility](#guppy-compatibility) for more details.

Note: due to the RNA motor processing along the strand from 3' to 5', the base caller sees the read reversed relative to the natural orientation.  Use `bin/basecall.py --reverse` to output the basecall of the read in its natural direction.


## Modified Bases

Taiyaki enables the training of models to predict the presence of modified bases (a.k.a. non-canonical or alternative bases) alongside the standard flip-flop canonical base probabilities via an alteration to the model architecture (model architecture referred to as categorical modifications, or `cat_mod` for short).
This alteration results in a second stream of data from the neural network which represents the probability that any base is canonical or modified (potentially including any number of modifications).

A number of adjustments to the training workflow are required to train a modified base model.
These adjustments begin with the “FASTA with reference sequence for each read” which is input to the `prepare_mapped_reads.py` command.
This FASTA file should contain ground truth per-read references annotated with modified base locations.

The single letter codes used to represent modified bases can be arbitrary and are defined using the `--mod` command line argument to the `prepare_mapped_reads.py` command.
The `--mod` argument takes three parameters: the letter representing the modified base, the letter representing its canonical representation, and a long name.
The `--mod` argument should be repeated once for each modification.
For example, to encode 5-methyl-cytosine and 6-methyl-adenosine with the single letter codes `Z` and `Y` respectively, the following commandline arguments would be added `--mod Z C 5mC --mod Y A 6mA`.
These values will be stored in the prepared signal mapped HDF5 output file for use in training downstream.

Next the `mapped-signal-file` is passed into the `train_flipflop.py` command.
This script requires a `cat_mod` model to be provided (e.g. `taiyaki/models/mGru_cat_mod_flipflop.py`).
The `--mod_factor` argument controls the proportion of the training loss attributed to the modified base output stream in comparison to the canonical base output stream.
When training a model from scratch it is generally recommended to set this factor to a lower value (`0.01` for example) to train the model to call canonical bases and then restart training with a larger value, `0.1 -- 1`, value in order to train the model to identify modified bases.

Modified base models can be used in `guppy` to call modified base anchored to the basecalls or `megalodon` to call modified bases anchored to a reference.

## Abinitio training

'Ab initio' is an alternative entry point for Taiyaki that obtains acceptable models with fewer input requirements,
particularly it does not require a previously trained model.

The input for ab initio training is a set of signal-sequence pairs:

- Fixed length chunks from reads
- A reference sequence trimmed for each chunk.

The models produced are not as accurate as those produced by the normal training process but can be used to bootstrap it.


The process is described in the [abinitio](docs/abinitio.rst) walk-through.

## RNA

During DNA sequencing, the strands of DNA go through the pore starting at the 5' end of the molecule. In contrast, during direct RNA sequencing the strands go through the pore starting at the 3' end. As a consequence, the per-read reference sequences used for RNA training must be **reversed** with respect to the genome/exome reference sequence (there is no need to **complement** the sequences). Basecalls produced with RNA models will then need to be reversed again in order to align them to a reference.

In terms of the workflow described above, the following steps need to be changed:

- If using `get_refs_from_sam.py` to produce per-read references, then add the `--reverse` option.
- If using the `basecall.py` script in taiyaki, then add the `--reverse` option.
- If basecalling with Guppy then use an RNA-specific config file (see the Guppy docs for more info).

# Guppy compatibility

In order to train a model that is compatible with Guppy (version 2.2 at time of writing), we recommend that you
use the model defined in `models/mGru_flipflop.py` and that you call `train_flipflop.py` with:

    train_flipflop.py --size 256 --stride 2 --winlen 19 mGru_flipflop.py <other options...>

You should then be able to export your checkpoint to json (using bin/dump_json.py) that can be used to basecall with Guppy.

See Guppy documentation for more information on how to do this.

Key options include selecting the Guppy config file to be appropriate for your application, and passing the complete path of your .json file.

For example:

    guppy_basecaller --input_path /path/to/input_reads --save_path /path/to/save_dir --config dna_r9.4.1_450bps_flipflop.cfg --model path/to/model.json --device cuda:1

Certain other model architectures may also be Guppy-compatible, but it is hard to give an exhaustive list
and so we recommend you contact us to get confirmation.


## Standard model parameters

Because of differences in the chemistry, particularly sequencing speed, and sample rate, the models used in Guppy are trained with different parameters depending on condition.
The default parameters for Taiyaki are generally those appropriate for a high accuracy DNA model and should be changed depending on what sample is being trained.
The table below describes the parameters currently used to train the production models released as part of Guppy:


| Condition                | chunk\_len\_min | chunk\_len\_max | size | stride | winlen |
|--------------------------|-----------------|-----------------|------|--------|--------|
| DNA, high accuracy       |   2000          |   4000          | 256  | 2 or 3 | 19     |
| DNA, fast                |   2000          |   4000          | 96   | 4      | 19     |
| RNA, high accuracy       |   2000          |   4000          | 256  | 10     | 31     |
| RNA, fast                |   2000          |   4000          | 96   | 12     | 31     |


# Environment variables

The environment variables `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` can have an impact on performance.
The optimal value will depend on your system and on the jobs you are running, so experiment.
As a starting point, we recommend:

    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=8


# CUDA

In order to use a GPU to accelerate model training, you will need to ensure that CUDA is installed (specifically nvcc) and that CUDA-related environment variables are set.
This should be done before running `make install` described above. If you forgot to do this, just run `make install` again once everything is set up.
The Makefile will try to detect which version of CUDA is present on your system, and install matching versions of pytorch and cupy. 

To see what version of CUDA will be detected and which torch and cupy packages will be installed you can run:

    make show_cuda_version

Expert users can override the detected versions on the command line. For example, you might want to do this if you are building Taiyaki on one machine to run on another.

    # Force CUDA version 8.0
    CUDA=8.0 make install

    # Override torch package, and don't install cupy at all
    TORCH=my-special-torch-package CUPY= make install

Users who install Taiyaki system-wide or into an existing activated Python environment will need to make sure CUDA and a corresponding version of PyTorch have been installed.

## Troubleshooting

During training, if this error occurs:

    AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'

or any other error related to the device, it suggests that you are trying to use pytorch's CUDA functionality but that CUDA (specifically nvcc) is either not installed or not correctly set up. 

If:

    nvcc --version

returns

    -bash: nvcc: command not found

nvcc is not installed or it is not on your path.

Ensure that you have installed CUDA (check NVIDIA's intructions) and that the CUDA compiler `nvcc` is on your path.

To place cuda on your path enter the following:

    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

Once CUDA is correctly configured and you are installing Taiyaki in a new virtual environment (as recommended), you may need to run `make install` again to ensure that you have the correct pytorch package to match your CUDA version.


# Using multiple GPUs

The script **bin/train_flipflop.py** can be used in multi-GPU mode with Pytorch's DistributedDataParallel class. With N GPUs available on a single machine, we can run N processes, each
using one of the GPUs and processing different random selections from the same training data. The gradients are synchronised by averaging across the processes. The outcome is that the batch
size is larger by a factor N than the batch size in single-GPU mode.

## How to launch training with multiple GPUs

Multi-GPU training runs can be launched using the Pytorch **distributed.launch** module. For example, in a Taiyaki environment:

    python -m torch.distributed.launch --nproc_per_node=4 train_flipflop.py --lr_max 0.004 --lr_min 0.0002 taiyaki/models/mGru_flipflop.py mapped_reads.hdf5
    
This command line launches four processes, each using a GPU. Four GPUs numbered 0,1,2,3 must be available.

Note that all command-line options for **train_flipflop.py** are used in the same way as normal, apart from **device**.

The script **workflow/test_multiGPU.sh** provides an example. Note that the line choosing GPUs (**export CUDA_VISIBLE_DEVICES...**) may need to be edited to specify the GPUs to be used on your system.

## Choice of learning rates for multi-GPU training

A higher learning rate can be used for large-batch or multi-GPU training. As a starting point, with N GPUs we recommend using a learning rate sqrt(N) times higher than used for a single GPU. With these
settings we expect to make roughly the same training progress as a single-GPU training run but in N times fewer batches. This will not always be true: as always, experiments are necessary to find the
best choice of hyperparameters. In particular, a lower learning rate than suggested by the square-root rule may be necessary in the early stages of training. One way to achieve this is by using the
command-line arguments **lr_warmup** and **warmup_batches**. Also bear in mind that the timescale for the learning rate schedule, **lr_cosine_iters** should be changed to take into account the faster progress of training.

## Selection of GPUs for multi-GPU training

The settings above use the first **nproc_per_node** GPUs available on the machine. For example, with 8 GPUs and **nproc_per_node** = 4, we will use the GPUs numbered 0,1,2,3. This selection can be altered
using the environment variable **CUDA_VISIBLE_DEVICES**. For example,

    export CUDA_VISIBLE_DEVICES="2,4,6,7"
    
will make the GPUs numbered 2,4,6,7 available to CUDA as if they were numbers 0,1,2,3.
If we then launch using the command line above (**python -m torch.distributed.launch...**), GPUs 2,4,6,7 will be used.

See below for how this applies in a SGE system.

## More than one multi-GPU training group on a single machine

Suppose that there are 8 GPUs on your machine and you want to train two models, each using 4 GPUs. Setting **CUDA_VISIBLE_DEVICES** to **"4,5,6,7"** for the second training job, you set things off,
but find that the second job fails with an error message like this

    File "./bin/train_flipflop.py", line 178, in main
        torch.distributed.init_process_group(backend='nccl')
    File "XXXXXX/taiyaki/venv/lib/python3.5/site-packages/torch/distributed/distributed_c10d.py", line 354, in init_process_group
        store, rank, world_size = next(rendezvous(url))
    File "XXXXXX/taiyaki/venv/lib/python3.5/site-packages/torch/distributed/rendezvous.py", line 143, in _env_rendezvous_handler
        store = TCPStore(master_addr, master_port, start_daemon)
    RuntimeError: Address already in use

The reason is that **torch.distributed.launch** sets up the process group with a fixed default IP address and port for communication between
processes (**master_addr** 127.0.0.1, **master_port** 29500). The two process groups are trying to use the same port.
To fix this, set off your second process group with a different address and port:

    python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.2 --master_port 29501 train_flipflop.py <command-line-options>


# Running on an SGE cluster

There are two things to get right: installing with the correct CUDA version, and executing with the correct choice of GPU.

## Installation
It is important that **when the package is installed**, it knows which version of the CUDA compiler is available on the machine where it will be executed.
When running on an SGE cluster we might want to do installation on a different machine from execution.
There are two ways of getting around this. You can qlogin to a node which has the same resources
as the execution node, and then install using that machine:

    qlogin -l h=<nodename>
    cd <taiyaki_directory>
    make install

...or you can tell Taiyaki at the installation stage which version of CUDA to use. For example

    CUDA=8.0 make install

## Execution
When **executing** on an SGE cluster you need to make sure you run on a node which has GPUs available, and then tell Taiyaki to use the correct GPU.

You tell the system to wait for a node which has an available GPU by adding the option **-l gpu=1** to your qsub command.
To find out which GPU has been allocated to your job, you need to look at the environment variable **SGE_HGR_gpu**. If it has the value **cuda0**, then
use GPU number 0, and if it has the value **cuda1**, then use GPU 1. The command line option **--device** (used by **train_flipflop.py**
accepts inputs such as 'cuda0' or 'cuda1' or integers 0 or 1, so SGE_HGR_gpu can be passed straight into the **--device** option.

The easy way to achieve this is with a Makefile like the one in the directory **workflow**. This Makefile contains comments which will help users run the package on a UGE system.

## Selection of multiple GPUs in SGE

When multiple GPUs are available to a SGE job (for example, if we use the command line option **-l gpu=4** in **qsub** to request 4 GPUS), the GPUs allocated are passed to the process
as a space-separated list in **SGE_HGR_gpu**. Unfortunately, **CUDA_VISIBLE_DEVICES** requires a comma-separated list. To get around this we execute

    export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu// /,}

Also note that on nodes with many GPUs, port clashes may occur (see 'More than one multi-GPU training group on a single machine' above). They can be avoided with clever use of
the command-line arguments of **torch.distributed.launch**.

# Diagnostics

The **misc** directory contains several scripts that are useful for working out where things went wrong (or understanding why they went right).

Graphs showing the information in mapped read files can be plotted using the script **plot_mapped_signals.py**
A graph showing the progress of training can be plotted using the script **plot_training.py**


---

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence. 
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.

© 2019 Oxford Nanopore Technologies Ltd.
Taiyaki is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.
