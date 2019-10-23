Walk-through
============

This walk-through trains a new basecalling model for Oxford Nanopore devices using the Taiyaki software.

A new Guppy-compatible model will be trained, from a set of raw reads, using the Taiyaki remapping workflow.

1. Basecall reads
2. Map basecalls to reference
3. Create input data files for read mapping

   a. Per-read reference sequence
   b. Read scaling and trimming parameters

4. Create mapped read file
5. Train model


For the impatient, the following commands will be run.  These will be explained, step-by-step, in following sections.

.. code-block:: bash

    # Download and unpack training data (approx. 10GB)
    wget  https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_walkthrough.tar.gz
    tar zxvf taiyaki_walkthrough.tar.gz
    cd taiyaki_walkthrough

    # Obtain and install Taiyaki
    git clone https://github.com/nanoporetech/taiyaki
    cd taiyaki && make install && cd ..

    # Basecall Reads -- replace config file with one appropriate for your data (e.g. dna_r9.4.1_450bps_hac_prom.cfg)
    guppy_basecaller -i reads -s basecalls -c /opt/ont/guppy/data/dna_r9.4.1_450bps_hac.cfg --device cuda:0

    # BAM of Mapped Basecalls
    minimap2 -I 16G -x map-ont -t 32 -a --secondary=no reference.fasta basecalls/*.fastq | samtools view -bST reference.fasta - > basecalls.bam

    # Extract Per-read References
    get_refs_from_sam.py reference.fasta basecalls.bam --min_coverage 0.8 > read_references.fasta

    # Create Per-read Scaling Parameters
    generate_per_read_params.py --jobs 32 reads > read_params.tsv

    # Create Mapped Read File
    prepare_mapped_reads.py  --jobs 32 reads read_params.tsv mapped_reads.hdf5  pretrained/r941_dna_minion.checkpoint read_references.fasta

    # Train a Model
    train_flipflop.py --device 0 taiyaki/models/mGru_flipflop.py mapped_reads.hdf5

    # Export to Guppy
    dump_json.py training/model_final.checkpoint > model.json

    # Basecall test set of reads with new model -- replace config file with one appropriate for your data
    guppy_basecaller -i reads -s basecalls_new -c /opt/ont/guppy/data/dna_r9.4.1_450bps_hac.cfg -m `pwd`/model.json --device cuda:0


Prerequisites
-------------
Taiyaki is developed on Ubuntu Linux 16.04 LTS and assumes a modern Linux environment.

- Linux environment with CUDA development enivornment
- GPU supported by PyTorch v1.0 
- Guppy basecaller
- Minimap2, see https://lh3.github.io/minimap2/ 


Download and unpack training data
---------------------------------
Download the training data, consisting of 50k R9.4.1 reads from a DNA sample of the following organisms:

- \E. coli (SCS110)
- \H. sapiens (NA12878)
- \S. cerevisiae (NCYC1052)

The total download is about 10GB.

.. code-block:: bash

    wget  https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_walkthrough.tar.gz
    # On some platforms, curl may be installed instead of wget
    # curl -O https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_walkthrough.tar.gz
    tar zxvf taiyaki_walkthrough.tar.gz
    cd taiyaki_walkthrough

Unpacking the ``taiyaki_walkthrough.tar.gz`` archive creates a directory ``taiyaki_walkthrough`` containing the files needed for this walk through. 
An additional directory ``taiyaki_walkthrough/intermediate_files`` contains examples of the outputs that will be created.

Contents
........

- ``intermediate_files`` (examples of file created during walk-through)

  - ``basecalls`` (directory produced by Guppy)
  - ``basecalls.bam``
  - ``mapped_reads.hdf5``
  - ``read_params.tsv``
  - ``read_references.fasta``

- ``pretrained`` (pre-trained models used for mapping reads)

  - ``r941_dna_minion.checkpoint``
  - ``r941_rna_minion.checkpoint``

- ``reads``

  - 50k single-read *fast5* files for training

- ``references.fasta``



Obtain and install Taiyaki
--------------------------
Download the *Taiyaki* software and install into a Python virtual environment.
For further information, see https://github.com/nanoporetech/taiyaki

.. code-block:: bash

    git clone https://github.com/nanoporetech/taiyaki
    (cd taiyaki && make install)
    source taiyaki/venv/activate

The remainder of this walk-through assumes that the working directory is ``taiyaki_walkthrough``, containing the data to train from, and that the *taiyaki* virtual environment is activated.



Basecall Reads
--------------
Here we are going to use the Guppy software, supported by by Oxford Nanopore, but other basecallers could be used instead.
The basecalls are used by *Taiyaki* to associate each read with a fragment of sequence.

Guppy will read the raw reads from the directory ``reads`` and write *fastq* format basecalls into a directory called ``basecalls``.

.. code-block:: bash

    guppy_basecaller -i reads -s basecalls -c /opt/ont/guppy/data/dna_r9.4.1_450bps_hac.cfg --device cuda:0



+-------------------------------------------------------+-----------------------------------------------------------------+
| -i reads                                              | Read raw *fast5* files from directory ``reads``                 |
+-------------------------------------------------------+-----------------------------------------------------------------+
| -s basecalls                                          | Write output into ``basecalls`` directory, *fastq* format.      |
|                                                       | Directory created when ``guppy_basecaller`` is run              |
+-------------------------------------------------------+-----------------------------------------------------------------+
| -c /opt/ont/guppy/data/dna_r9.4.1_450bps_hac.cfg      | Configuration file for model.                                   |
|                                                       | Here we use the flip-flop basecaller                            |
+-------------------------------------------------------+-----------------------------------------------------------------+
| --device cuda:0                                       | Run the basecalling on CUDA device ``cuda:0``.                  |
|                                                       | If you have more than one GPU, you may need to change this value|
+-------------------------------------------------------+-----------------------------------------------------------------+

If you wish to use a different basecaller, the rest of this walk-through assumes that the basecalls are in *fastq* format and stored in a directory ``basecalls``


BAM of Mapped Basecalls
-----------------------
From the set of basecalls, map to a reference so that a specific reference fragment for each read can be determined.


.. code-block:: bash

    minimap2 -I 16G -x map-ont -t 32 -a --secondary=no reference.fasta basecalls/*.fastq | samtools view -b -S -T reference.fasta - > basecalls.bam



minimap2
........
Requires a working installation for *minimap2*.  See https://lh3.github.io/minimap2/ for details.


+--------------------+---------------------------------------------------------------------------+
| -I 16G             |   Only split index every 16 gigabases                                     |
+--------------------+---------------------------------------------------------------------------+
| -x                 |   Preset for mapping ONT reads to a reference                             |
+--------------------+---------------------------------------------------------------------------+
| -t 32              |   Use 32 threads to run                                                   |
+--------------------+---------------------------------------------------------------------------+
| -a                 |   Output in SAM format                                                    |
+--------------------+---------------------------------------------------------------------------+
| --secondary=no     |   Don't output secondary alignments                                       |
+--------------------+---------------------------------------------------------------------------+
| reference.fasta    |   *fasta* format file containing reference sequence to map against        |
+--------------------+---------------------------------------------------------------------------+
| basecalls/\*.fastq |   Constructs a list of all *fastq* files with the ``basecalls`` directory |
+--------------------+---------------------------------------------------------------------------+

samtools view
.............
Requires a working installation for *samtools*.  See http://www.htslib.org for details.

+---------------------+----------------------------------------+
|  -b                 |   Output is BAM                        |
+---------------------+----------------------------------------+
|  -S                 |   Input is SAM                         |
+---------------------+----------------------------------------+
|  -T reference.fasta |   Location of reference mapped to      |
+---------------------+----------------------------------------+
|  \-                 |   Read input from *stdin*              |
+---------------------+----------------------------------------+
|  > basecalls.bam    |   Redirect output to ``basecalls.bam`` |
|                     |   (printed to screen by default)       |
+---------------------+----------------------------------------+



Extract Per-read References
---------------------------
Taiyaki requires a specific reference for each read, in the same orientation as the read.
The ``get_refs_from_sam.py`` script extracts a specific reference for each read, which is used as its *true sequence* for training.
A low coverage, proportion to the basecalled mapped, might indicate a mismapped read or issues with the reference, so we filter out these reads.

.. code-block:: bash

    get_refs_from_sam.py reference.fasta basecalls.bam --min_coverage 0.8 > read_references.fasta

+-------------------------+----------------------------------------------------+
| --min_coverage 0.8      |  Only output a reference for reads where more than |
|                         |  80% of the basecall maps to the reference         |
+-------------------------+----------------------------------------------------+
| > read_references.fasta |  Redirect output to ``read_references.fasta``      |
|                         |  (printed to screen by default)                    |
+-------------------------+----------------------------------------------------+


Create Per-read Scaling Parameters
----------------------------------
Taiyaki allows a great deal of flexibility is how reads are scaled and trimmed before mapping, parameters for each read being contained in a *tab-separated-variable* (tsv) file with the following columns:

- UUID
- trim_start
- trim_end
- shift
- scale


The ``generate_per_read_params.py`` script analyses a directory of reads and produces a compatible tsv file using a default scaling method.

.. code-block:: bash

    generate_per_read_params.py --jobs 32 reads > read_params.tsv


+----------------------------------------+-------------------------------------------------------------+
|  --jobs 32                             |  Run using 32 threads                                       |
+----------------------------------------+-------------------------------------------------------------+
|  reads                                 |  Directory containing *fast5* reads files                   |
+----------------------------------------+-------------------------------------------------------------+
| > read_params.tsv                      |  Redirect output to ``read_params.tsv`` file.               |
|                                        |  Default is write to ``stdout``                             |
+----------------------------------------+-------------------------------------------------------------+

Create Mapped Read File
-----------------------
Taiyaki's main input format is a file containing mapped reads and necessary data to select chunks of reads for training.
The ``prepare_mapped_reads.py`` script takes the previously prepared files and processes them into final input file.

.. code-block:: bash

    prepare_mapped_reads.py  --jobs 32 reads read_params.tsv mapped_reads.hdf5 pretrained/r941_dna_minion.checkpoint read_references.fasta


+----------------------------------------+-------------------------------------------------------------+
|  --jobs 32                             |  Run using 32 threads                                       |
+----------------------------------------+-------------------------------------------------------------+
|  reads                                 |  Directory containing *fast5* reads files                   |
+----------------------------------------+-------------------------------------------------------------+
|  read_params.tsv                       |  Per-read scaling and trimming parameters                   |
+----------------------------------------+-------------------------------------------------------------+
|  mapped_reads.hdf5                     |  Output file.  A HDF5 format file, structured               |
|                                        |  according to (docs/FILE_FORMATS.md)                        |
+----------------------------------------+-------------------------------------------------------------+
|  pretrained/r941_dna_minion.checkpoint |  Model file used for remapping reads to their references    |
+----------------------------------------+-------------------------------------------------------------+
|  read_references.fasta                 |  *fasta* file containing a reference specific for each read |
+----------------------------------------+-------------------------------------------------------------+



Train a Model
-------------
Having prepared the mapped read file, the ``train_flipflop.py`` script training a flip-flop model.
Progress is displayed on the screen and written to a log file in the ``training`` directory. 
Checkpoints are regularly saved and training can be restarted from a checkpoint by replacing the model description file with the checkpoint file on the command line.

- ``training/model.log``   Log file
- ``training/model.py``    Input model file
- ``training/model_checkpoint_xxxxx.checkpoint``   Checkpoint

Depending the speed of the GPU used, this process can take several days.

.. code-block:: bash

    train_flipflop.py --device 0 taiyaki/models/mGru_flipflop.py mapped_reads.hdf5

+--------------------------------------+-----------------------------------------------------------+
|  --device 0                          |  Use CUDA device 0                                        |
+--------------------------------------+-----------------------------------------------------------+
|  taiyaki/models/mGru_flipflop.py     |  Model definition file                                    |
+--------------------------------------+-----------------------------------------------------------+
|  mapped_reads.hdf5                   |  Mapped reads file created by ``prepare_mapped_reads.py`` |
+--------------------------------------+-----------------------------------------------------------+


Export to Guppy
---------------
Guppy requires a *json* format file, which can be easily created from the final model file (``training/model_final.checkpoint``)

.. code-block:: bash

    dump_json.py training/model_final.checkpoint > model.json

Basecall with New Model
-----------------------
By way of exmaple, use the new model to basecall the training reads.
It is not recommended to use these basecalls to assess model, please use an alternative set.

.. code-block:: bash

    guppy_basecaller -i reads -s basecalls_new -c /opt/ont/guppy/data/dna_r9.4.1_450bps_hac.cfg -m `pwd`/model.json --device cuda:0


+----------------------+---------------------------------------------------------------------------------+
|  \`pwd\`/model.json  |  Use new model file for training.                                               |
|                      |  Guppy requires the absolute path to the model, constructed by calling ``pwd``  |
+----------------------+---------------------------------------------------------------------------------+
