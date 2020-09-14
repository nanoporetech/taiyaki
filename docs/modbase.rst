Training a Modified Base Model
==============================

This walk-through trains a model capable of calling 5mC and 6mA occurring certain biological contexts:

- CpG methylation in human
- DAM (6mA) and DCM (5mC) methylation in E. coli.

Models can be created for other modifications and contexts given appropriate training data.

The model will be trained from a set of reads and read-specific references, marked up with the location and type of modification.
A pre-trained model is used to create a mapping from the read to the reference but this model need not be modification aware.  
The model provided is a canonical base model from R9.4.1 trained from native DNA.

For the impatient, the following commands will be run.  These will be explained, step-by-step, in following sections.

.. code-block:: bash

        # Download and unpack training data
        wget  https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_modbase.tar.gz
        tar zxvf taiyaki_modbase.tar.gz
        cd taiyaki_modbase

        # Obtain and install Taiyaki
        git clone https://github.com/nanoporetech/taiyaki
        cd taiyaki && make install && cd ..

        # Create Per-read Scaling Parameters
        generate_per_read_params.py --jobs 32 reads > modbase.tsv

        # Create Mapped Read File
        prepare_mapped_reads.py --jobs 32 --mod Z C 5mC --mod Y A 6mA reads modbase.tsv modbase.hdf5 pretrained/r941_dna_minion.checkpoint modbase_references.fasta

        # Train modified base model
        train_flipflop.py --device 0 taiyaki/models/mLstm_cat_mod_flipflop.py modbase.hdf5

        # Basecall
        basecall.py --device 0 --modified_base_output basecalls.hdf5 reads training/model_final.checkpoint > basecalls.fa


Download and Unpack Training Data
---------------------------------

Download the training data, consisting of 10k R9.4.1 reads from a DNA sample of the following organisms.

- \E. coli (K12)
- \H. sapiens (NA12878)

.. code-block:: bash

        wget  https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_modbase.tar.gz
        # On some platforms, curl may be installed instead of wget
        # curl -O https://s3-eu-west-1.amazonaws.com/ont-research/taiyaki_modbase.tar.gz
        tar zxvf taiyaki_modbase.tar.gz
        cd taiyaki_modbase

Unpacking the ``taiyaki_modbase.tar.gz`` archive creates a directory ``taiyaki_modbase`` containing the files needed for this walk-through. 
An additional directory ``taiyaki_modbase/intermediate_files`` contains examples of the outputs that will be created.

- ``intermediate_files`` (examples of file created during walk-through)
    + ``modbase.hdf5``
    + ``modbase.tsv``
- ``pretrained`` (pre-trained models used for mapping reads)
    + ``r941_dna_minion.checkpoint``
    + ``r941_rna_minion.checkpoint``
- ``reads``
    + 10k single-read *fast5* files for training
- ``modbase_references.fasta``


Obtain and install Taiyaki
--------------------------

Download the *Taiyaki* software and install into a Python virtual environment.
For further information, see https://github.com/nanoporetech/taiyaki

.. code-block:: bash

    git clone https://github.com/nanoporetech/taiyaki
    (cd taiyaki && make install)
    source taiyaki/venv/activate

The remainder of this walk-through assumes that the working directory is ``taiyaki_modbase``, containing the data to train from, and that the *taiyaki* virtual environment is activated.

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

    generate_per_read_params.py --jobs 32 reads > modbase.tsv

+----------------------------------------+-------------------------------------------------------------+
|  --jobs 32                             |  Run using 32 threads                                       |
+----------------------------------------+-------------------------------------------------------------+
|  reads                                 |  Directory containing *fast5* reads files                   |
+----------------------------------------+-------------------------------------------------------------+
| > modbase.tsv                          |  Redirect output to ``modbase.tsv`` file.                   |
|                                        |  Default is write to ``stdout``                             |
+----------------------------------------+-------------------------------------------------------------+

Create Mapped Read File
-----------------------
Taiyaki's main input format is a file containing mapped reads and necessary data to select chunks of reads for training.
The ``prepare_mapped_reads.py`` script takes the previously prepared files and processes them into final input file.

The mapped read file is prepared from a set of fast5 files and read-specific references stored in as *fasta* file.
Each read-specific reference should be marked up with the location of any modifications, represented by an alternative base 'letter', that is specified on the command line.

In reference file provided, 5mC is represented by 'Z' and 6mA by 'Y' but these should not be considered definitive or assumed to be compatible with other software.
There are few standards for what letters represent modifications in DNA / RNA sequences and the final choice is left to the user.

An example reference might look like:

.. code-block::

    >f7630a4a-de56-4081-b203-49832119a4a9
    ATCAGCATCCGCAAGCCZAGGGYTCACCCGGACATGTTGCAGCGAAAACTGACGACGTAATTGAGTTTCAT

The following creates the input data for training.
Notice that each modification is given as a separate argument, describing: the letter used to represent it in the *fasta* file, the canonical base it is "equivalent" to, and a convenient name.

.. code-block:: bash

    prepare_mapped_reads.py --jobs 32 --mod Z C 5mC --mod Y A 6mA reads modbase.tsv modbase.hdf5 pretrained/r941_dna_minion.checkpoint modbase_references.fasta

+---------------------------------------------+-------------------------------------------------------------+
| --jobs 32                                   |  Number of threads to run simultaneously                    |
+---------------------------------------------+-------------------------------------------------------------+
| --mod Z C 5mC                               |  Description of each modification. Letter used to           |
| --mod Y A 6mA                               |  represent modificaton in `modbase_references.fasta`, the   |
|                                             |  canonical base for the modification, and a name.           |
+---------------------------------------------+-------------------------------------------------------------+
| reads                                       |  Directory contain reads in *fast5* format                  |
+---------------------------------------------+-------------------------------------------------------------+
| modbases.tsv                                |  Per-read scaling and trimming parameters                   |
+---------------------------------------------+-------------------------------------------------------------+
| modbases.hdf5                               |  Output file. A HDF5 format file, structured                |
|                                             |  according to (docs/FILE_FORMATS.md)                        |
+---------------------------------------------+-------------------------------------------------------------+
| pretrained/r941_dna_minion.checkpoint       |  Model file used for remapping reads to their references    |
+---------------------------------------------+-------------------------------------------------------------+
| modbase_references.fasta                    |  *fasta* file containing a reference specific for each read |
|                                             |  marked up with modified base information                   |
+---------------------------------------------+-------------------------------------------------------------+

Train a Model
-------------

Having prepared the mapped read file, the ``train_flipflop.py`` script trains a flip-flop model.
Progress is displayed on the screen and written to a log file in output directory. 
Checkpoints are regularly saved and training can be restarted from a checkpoint by replacing the model description file with the checkpoint file on the command line.

- ``training/model.log``   Log file
- ``training/model.py``    Input model file
- ``training/model_checkpoint_xxxxx.checkpoint``   Model checkpoint files

Two rounds of training are performed:
the first round down-weights learning the modified bases in favour a good canonical call,
the second round then focuses on learning the conditional prediction of whether a base is modified.

Depending the speed of the GPU used, this process can take several days.

.. code-block:: bash

    train_flipflop.py --device 0 taiyaki/models/mLstm_cat_mod_flipflop.py modbase.hdf5

+----------------------------------------------+-------------------------------------------------------------+
|  --device 0                                  |  Use CUDA device 0                                          |
+----------------------------------------------+-------------------------------------------------------------+
|  taiyaki/models/mLstm_cat_mod_flipflop.py    |  Model definition file, ``training/model_final.checkpoint`` |
|                                              |  for second round of training.                              |
+----------------------------------------------+-------------------------------------------------------------+
|  modbase.hdf5                                |  Mapped reads file created by ``prepare_mapped_reads.py``   |
+----------------------------------------------+-------------------------------------------------------------+

Modified base calling
---------------------

The trained model can be exported for use with basecallers that support modified bases: either Guppy or Megalodon.
[Megalodon](https://nanoporetech.github.io/megalodon/index.html) is the recommended tool, and it uses either Guppy 
or Taiyaki as a backend for modified base calling. The Guppy backend is typically faster, but only a limited range 
of models are supported.

If you wish to use Taiyaki as your Megalodon backend, then you can use your trained model checkpoint directly.
You will need to have both Megalodon and Taiyaki installed, then
see [this section](https://nanoporetech.github.io/megalodon/advanced_arguments.html#taiyaki-backend-arguments)
of the Megalodon documentation for how to use Taiyaki as your backend.

The default backend for Megalodon is Guppy. See the Megalodon documentation for more details.
You will have to convert your model checkpoint to JSON format that can be read by Guppy:

.. code-block:: bash

     dump_json.py training/model_final.checkpoint --output training/model.json

+----------------------------------------------+-------------------------------------------------------------+
|  training/model_final.checkpoint             |  Trained model file                                         |
+----------------------------------------------+-------------------------------------------------------------+
|  --output training/model.json                |  Write model to this file in JSON format                    |
|                                              |  Default is to write to ``stdout``                          |
+----------------------------------------------+-------------------------------------------------------------+

After exporting your model to JSON, you should create a Guppy config file pointing to your JSON model. The
easiest way to do this is to copy and modify an existing Guppy config file.

