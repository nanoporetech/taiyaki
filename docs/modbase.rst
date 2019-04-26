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
        prepare_mapped_reads.py  --jobs 32 --mod Z C 5mC --mod Y A 6mA reads modbase.tsv modbase.hdf5  pretrained/r941_dna_minion.checkpoint modbase_references.fasta

        # Train modified base model
        train_mod_flipflop.py --device 0 --mod_factor 0.01 --outdir training taiyaki/models/mGru_cat_mod_flipflop.py modbase.hdf5
        train_mod_flipflop.py --device 0 --mod_factor 1.0 --outdir training2 training/model_final.checkpoint modbase.hdf5

        # Basecall
        basecall.py --device 0 --modified_base_output basecalls.hdf5 reads training2/model_final.checkpoint  > basecalls.fa



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


The following creates the input data for training.  Notice that each modification is given as a separate argument, describing: the letter used to represent it in the *fasta* file, the canonical base it is "equivalent" to, and a convenient name.

.. code-block:: bash

    prepare_mapped_reads.py  --jobs 32 --mod Z C 5mC --mod Y A 6mA reads modbase.tsv modbase.hdf5  pretrained/r941_dna_minion.checkpoint modbase_references.fasta


+---------------------------------------------+-------------------------------------------------------------+
| --jobs 32                                   |  Number of threads to run simultaneously                    |
+---------------------------------------------+-------------------------------------------------------------+
| --mod Z C 5mC                               |  Description of each modification.  Letter used to          |
| --mod Y A 6mA                               |  represent modificaton in `modbase_references.fasta`, the   |
|                                             |  canonical base for the modification, and a name.           |
+---------------------------------------------+-------------------------------------------------------------+
| reads                                       |  Directory contain reads in *fast5* format                  |
+---------------------------------------------+-------------------------------------------------------------+
| modbases.tsv                                |  Per-read scaling and trimming parameters                   |
+---------------------------------------------+-------------------------------------------------------------+
| modbases.hdf5                               |  Output file.  A HDF5 format file, structured               |
|                                             |  according to (docs/FILE_FORMATS.md)                        |
+---------------------------------------------+-------------------------------------------------------------+
| pretrained/r941_dna_minion.checkpoint       |  Model file used for remapping reads to their references    |
+---------------------------------------------+-------------------------------------------------------------+
| modbase_references.fasta                    |  *fasta* file containing a reference specific for each read |
|                                             |  marked up with modified base information                   |
+---------------------------------------------+-------------------------------------------------------------+


Train a Model
-------------
Having prepared the mapped read file, the ``train_mod_flipflop.py`` script training a flip-flop model.
Progress is displayed on the screen and written to a log file in output directory. 
Checkpoints are regularly saved and training can be restarted from a checkpoint by replacing the model description file with the checkpoint file on the command line.

- ``train1/model.log``   Log file
- ``train1/model.py``    Input model file
- ``train1/model_checkpoint_xxxxx.checkpoint``   Model checkpoint files

Two rounds of training are performed:
the first round down-weights learning the modified bases in favour a good canonical call,
the second round then focuses on learning the conditional prediction of whether a base is modified.

Depending the speed of the GPU used, this process can take several days.

.. code-block:: bash

    train_mod_flipflop.py --device 0 --mod_factor 0.01 --outdir training taiyaki/models/mGru_cat_mod_flipflop.py modbase.hdf5
    train_mod_flipflop.py --device 0 --mod_factor 1.0 --outdir training2 training/model_final.checkpoint modbase.hdf5

+----------------------------------------------+-------------------------------------------------------------+
|  --device 0                                  |  Use CUDA device 0                                          |
+----------------------------------------------+-------------------------------------------------------------+
|  --mod_factor 0.01                           |  Relative importance of modifications in training           |
|                                              |  criterion (0.0 == ignore, 1.0 == same weight as canonical) |
+----------------------------------------------+-------------------------------------------------------------+
|  --outdir                                    |  Name of directory to write output files                    |
+----------------------------------------------+-------------------------------------------------------------+
|  taiyaki/models/mGru_cat_mod_flipflop.py     |  Model definition file, ``training/model_final.checkpoint`` |
|                                              |  for second round of training.                              |
+----------------------------------------------+-------------------------------------------------------------+
|  training                                    |  Output directory for model checkpoints and training log    |
+----------------------------------------------+-------------------------------------------------------------+
|  modbase.hdf5                                |  Mapped reads file created by ``prepare_mapped_reads.py``   |
+----------------------------------------------+-------------------------------------------------------------+


Basecall
--------
.. _`file formats`: FILE_FORMATS.md

The basecalls produced use the canonical base alphabet, information about putative modifed base calls is written out to the specified file, ``basecalls.hdf5``.


.. code-block:: bash

     basecall.py --device 0 --modified_base_output basecalls.hdf5 reads training2/model_final.checkpoint  > basecalls.fa


+----------------------------------------------+-------------------------------------------------------------+
|  --device 0                                  |  Use CUDA device 0                                          |
+----------------------------------------------+-------------------------------------------------------------+
|  --modified_base_output basecalls.hdf5       |  Output modifed base information to ``basecalls.hdf5``      |
+----------------------------------------------+-------------------------------------------------------------+
|  reads                                       |  Directory contain reads in *fast5* format                  |
+----------------------------------------------+-------------------------------------------------------------+
|  training2/model_final.checkpoint            |  Trained model file                                         |
+----------------------------------------------+-------------------------------------------------------------+
|  > basecalls.fa                              |  Redirect output basecalls to ``modbase.tsv`` file.         |
|                                              |  Default is to write to ``stdout``                          |
+----------------------------------------------+-------------------------------------------------------------+


Modified Base File
..................

The modified base output file, ``basecalls.hdf5`` in this example, stores the information about the presence of modifications given the basecall.
The information is stored in a per-read dataset, containing the conditional (log) probability of modification for each position of the *basecall*.
The calls are ordered according to the names given in the ``mod_long_names`` dataset.
Impossible calls, where the canonical basecall position and modification are incompatible, are indicated by ``nan`` values.

.. code-block::

    HDF5_file/
    ├── dataset: mod_long_names
    └── group: Reads/
        ├── dataset: <read_id_1>
        ├── dataset: <read_id_2>
        ├── dataset: <read_id_3>
        .
        .


Quick analysis
..............


.. code-block:: python

    import h5py
    import numpy as np

    # Read in information for first 120 positions of a2cd3a8c-dc41-4404-9dda-8ebffc6fd9e0
    with h5py.File('intermediate_files/basecalls.hdf5', 'r') as h5:
        cond_logprobs = h5['Reads/a2cd3a8c-dc41-4404-9dda-8ebffc6fd9e0'][:120]
        print(h5['mod_long_names'][()])

    # > Reference
    #                   CTCTGTCTCTGAGTCTCTGTCTTCTZGGAAGGACAACAGTCAGTGGATZGGGCACTTTCTGZGCAAGCATTZGTTT-ACCCTAAZGTGCTCAZGGCTACATTA
    #                                            m                      m            m         m            m       m
    # > Basecall
    # ACCCACAGTTTGTGTGCTCTCTGTCTCTGAGTCTCTGTCTTCTCGGAAGGACAACAGTCAGTGGATCGGGCACTTTCTGCGCAAGCATTCGTTTTACC-TAACGTGCTCACGGCTACATTA 
    # Expecting 5mC modification at basecall positions: 43 66 79 89 101 109

    #  First column of cond_logprob corresponds to 6mA, second is 5mC
    #  Possible positions of methylation (non-nan entries)
    print(np.where(~np.isnan(cond_logprobs))[0])
    #  Probable methyation calls -- gives 43, 66, 79, 89, 101, 109
    print(np.where(cond_probs[:,1] > np.log(0.5)))
    #  Confident methyation calls -- gives 43, 66, 79, 89, 101, 109
    print(np.where(cond_probs[:,1] > np.log(0.9)))
    #  Most confident 6mA call -- gives 2.8e-06
    print(np.exp(np.nanmax(cond_probs[:,0])))
