#! /bin/bash -eux
set -o pipefail

# Test workflow from fast5 files and modified base per-read reference FASTA to
#     trained model using flip-flop remapping
# This is done with just a few reads so the model won't be useful for anything.
# This script must be executed with the current directory being the
#     taiyaki base directory

echo ""
echo "Test of modified base flip-flop remap and basecall network training starting"
echo ""

# Execute the whole workflow: generating per-read-params, mapped-signal file generation and then training
READ_DIR=test/data/reads
USER_PER_READ_MOD_REFERENCE_FILE=test/data/per_read_references.mod_bases.fasta

echo "USER_PER_READ_MOD_REFERENCE_FILE=${USER_PER_READ_MOD_REFERENCE_FILE}"

TAIYAKI_DIR=`pwd`
RESULT_DIR=${TAIYAKI_DIR}/RESULTS/mod_train_remapuser_ref
envDir=${envDir:-$TAIYAKI_DIR}

rm -rf $RESULT_DIR
rm -rf ${TAIYAKI_DIR}/RESULTS/training_ingredients

#TAIYAKIACTIVATE=(nothing) makes the test run without activating the venv at each step. Necessary for running on the git server.
make -f workflow/Makefile NETWORK_SIZE=96 MAXREADS=10 READDIR=${READ_DIR} TAIYAKI_ROOT=${TAIYAKI_DIR} DEVICE=cpu MAX_TRAINING_ITERS=2 USER_PER_READ_MOD_REFERENCE_FILE=${USER_PER_READ_MOD_REFERENCE_FILE} SEED=1 TAIYAKIACTIVATE= mod_train_remapuser_ref envDir=${envDir}

# Check that training log exists and has enough rows for us to be sure something useful has happened


traininglog_lines=`wc -l ${RESULT_DIR}/model.log | cut -f1 -d' '`
echo "Number of lines in training log: ${traininglog_lines}"
if [ "$traininglog_lines" -lt "9" ]
then
    echo "Training log too short- training not started properly"
    exit 1
fi

echo ""
echo "Test of modified base flip-flop remap and basecall network training completed successfully"
echo ""
