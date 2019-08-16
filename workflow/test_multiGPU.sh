#! /bin/bash -eux
set -o pipefail

# Test multi-GPU operation of train_flipflop.py
# We can't run this test on the git server because it needs GPUs
# Script should be run in a Taiyaki environment.
# This script must be executed with the current directory being the taiyaki base directory
# Or you can change this variable to point to your taiyaki installation
TAIYAKI=.


# Note that this will not work if there is already a multiGPU training run going on the machine in use:
# In that case you need to set master_addr and master_port. For example,
# --master_addr 127.0.0.2 --master_port 29501
# See https://github.com/facebookresearch/maskrcnn-benchmark/issues/241


echo ""
echo "Test of multi-GPU training with train_flipflop.py"
echo ""

OPENBLAS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS
OMP_NUM_THREADS=4
export OMP_NUM_THREADS

#Choose which devices to use - not necessary if running on 0,1
#export CUDA_VISIBLE_DEVICES="0,5"
#On UGE this converts the space separated list into a comma separated list
#export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu// /,}
#Number of GPUs to use
NGPU=2
#Training data
MAPPEDREADFILE=${TAIYAKI}/test/data/mapped_signal_file/mapped_reads_1.hdf5
#Start model
MODEL=${TAIYAKI}/models/mGru_flipflop.py

#Where to put the results
RESULT_DIR=${TAIYAKI}/workflow/multiGPU_test_results

LR_MAX=0.003
LR_MIN=0.00015
LR_COSINE_ITERS=20000

ITERATIONS=100
WARMUP=10

python -m torch.distributed.launch --nproc_per_node ${NGPU}\
        --master_addr 127.0.0.2 --master_port 29501\
        ${TAIYAKI}/bin/train_flipflop.py\
        --overwrite --lr_cosine_iters ${LR_COSINE_ITERS}\
        --min_sub_batch_size 32\
        --warmup_batches ${WARMUP} --niteration ${ITERATIONS}\
        --lr_max ${LR_MAX} --lr_min ${LR_MIN}\
        --outdir ${RESULT_DIR}\
        ${MODEL} ${MAPPEDREADFILE}

# Check that batch log exists and has the right number of rows

batchlog_lines=`wc -l ${RESULT_DIR}/batch.log | cut -f1 -d' '`
echo "Number of lines in training batch log should be $((ITERATIONS+1)): ${batchlog_lines}"
if [ "$batchlog_lines" -ne "$((ITERATIONS+1))" ]
then
    echo "Training batch log has incorrect number of lines"
    exit 1
fi

echo ""
echo "Test of multi-GPU training completed successfully"
echo ""
