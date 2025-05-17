export NUMBER_OF_GPUS=1
export BATCH_SIZE=16

torchrun --nproc_per_node=$NUMBER_OF_GPUS --nnodes=1 train.py --device_bsz $BATCH_SIZE
