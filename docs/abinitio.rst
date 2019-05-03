Ab Initio training
==================
.. _`walk through`: walkthrough.rst
This walk-through describes an alternative entry point for training models with lighter input requirements that for the full `walk through`_.
The models obtained will not achieve the same accuracy as the full training process, but is a useful starting point for basecalling and mapping reads as preparation for more rigorous training.

The input for ab initio training is a set of signal-sequence pairs:

- Fixed length chunks from reads
- A reference sequence trimmed for each chunk.

Three sets of input data are provided by way of example:

- R9.4.1 DNA, 1497098 chunks of 2000 samples

  + r941_dna/chunks.hdf5
  + r941_dna/chunks.fa

- R9.4.1 RNA, 44735 chunks of 10000 samples

  + r941_rna/chunks.hdf5
  + r941_rna/chunks.fa

- R10 DNA, 498224 chunks of 2000 samples

  + r10_dna/chunks.hdf5
  + r10_dna/chunks.fa

Training
--------

Training is as simple as:

.. code-block:: bash

    train_abinitio.py --device 0 mGru_flipflop.py signal_chunks.hdf5 references.fa

+----------------------+------------------------------------------------------------------+
|  --device            |  Run training on GPU 0                                           |
+----------------------+------------------------------------------------------------------+
|  mGru_flipflop.py    |  Model description file, see ``taiyaki/models``                  |
+----------------------+------------------------------------------------------------------+
|  signal_chunks.hdf5  |  Signal chunk file, formatted as described in `Chunk format`_.   |
+----------------------+------------------------------------------------------------------+
|  references.fa       |  Per-chunk reference sequence                                    |
+----------------------+------------------------------------------------------------------+

A ``Makefile`` is provided to demonstrate training for the example data sets provided.

.. code-block:: bash

    #  Run all examples
    make all
    #  Run single example.  Possible examples r941_dna, r941_rna, or r10_dna
    make r941_dna/training


Chunk format
------------
.. _HDF5: https://www.hdfgroup.org

Chunks are stored in a HDF5_ file as a 2D array, *chunks x samples*.  The  TODO

Creating this file and the corresponding reads TODO

For example, the training file for the R941 DNA consists of 1497098 chunks of 2000 samples.

.. code-block:: bash

     h5ls -r r941_dna/chunks.hdf5 
     /                        Group
     /chunks                  Dataset {1497098, 2000}


Scaling issues
..............
.. _`file formats`: FILE_FORMATS.md#per-read-parameter-files
.. _MAD: https://en.wikipedia.org/wiki/Median_absolute_deviation

For compatibilty with ONT's basecallers and the default tool-chain, it is recommended that each read (not chunk) is scaled as follows:

.. code-block:: bash

    signal_scaled = signal - median(signal)
                    -----------------------
                       1.4826 mad(signal)

where the 'MAD_' (median absolute deviation) has additional multiplicative factor of 1.4826 to scale it consistently with standard deviation.


Other scaling methods could be used if the user is will to create a pre-read parameter file for future training (see `file formats`_).


Reference format
----------------
The references are stored in a *fasta* format, one reference for each **chunk** trimmed to that chunk.
The name of each reference should be the index of its respective chunk.


For example, the training file for the R941 DNA consists of 1497098 chunks of 2000 samples.

.. code-block::

        >0
        AGACAGCGAGGTTTATCCAATATTTTACAAGACACAAGAACTTCATGTCCATGCTTCAGG
        AACAGGACGTCAGATAGCAAACAATGGGAAGTATATTTTTATAACCGAGCAACATCTCTA
        CGGAACAGCGTTATCGGTATACAAGTACTCTATATCTTTCAAACGGTGGCTGTTCGTGGG
        CTACTCAGACATTAGGGCCAAATACGGTATA
        >1
        GTATAAGGAGTGTCAAAGATCTCTTTGTTGGTAACTGTCCCTCTGTAAATAGCCCAGTGC
        TGACAATTCTTACTGATGACAATAACATTCAAACAATTCTTCTTAAATAAAGGTTAAGGA
        AATGTAAATAAAAAAATAACAGTGACATTAATTTGTATATATCTCAACTTCTTCACTTTA
        ACCTGTCTGAGCTGTTTGGTTTTGAACTG


Modified bases
--------------
.. _modbase: modbase.rst
Ab initio training does not yet support our modified base models.
While a model could be trained treating each modified base as an additional canonical base, the recommended proceedure is to train a canonical model using the ab initio process and then use this as the 'pre-trained' model in the modbase_ walk through.
