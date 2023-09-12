#!/bin/bash

rm -rf /tmp/*.pt
rm -rf /tmp/ray_ssd/*
rm -rf ./checkpoints/* 

# CUDA
export CUDA_HOME=/usr/local/cuda/
# MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps/nvidia-log

# torch==1.9.0 is ok, but 1.10.0 fails to run the native megatron-lm source code
# JIT cannot deal with input tensor without concrete number of dimensions
export PYTORCH_JIT=0

# Distributed Env
RANK=0
WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=6001
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=`seq -s ',' 0 1 $(( $GPUS_PER_NODE-1 ))`

# Data
# _BASE=/home/sys/STRONGHOLD/data
# DATA_PATH=${_BASE}/my-gpt2-en_text_document
# VOCAB_PATH=${_BASE}/gpt2-vocab.json
# MERGE_PATH=${_BASE}/gpt2-merges.txt
CHECKPOINT_PATH=./checkpoints/gpt2

VOCAB_PATH=/shared_ssd_storage/shilonglei/OOC/Megatron-DeepSpeed/dataset/data/gpt2-vocab.json
MERGE_PATH=/shared_ssd_storage/shilonglei/OOC/Megatron-DeepSpeed/dataset/data/gpt2-merges.txt
DATA_PATH=/shared_ssd_storage/shilonglei/OOC/Megatron-DeepSpeed/dataset/data/meg-gpt2-oscar-en-10k_text_document

# # Todo. Hard code. @gl
# PYTHON_LIB=/usr/local/lib/python3.8/dist-packages
# cp ./scripts/distributed_c10d._gl_.py ${PYTHON_LIB}/torch/distributed/distributed_c10d.py
# cp ./scripts/deepspeed_cpu_adam._gl_.py ${PYTHON_LIB}/deepspeed/ops/adam/cpu_adam.py

# Model defination
NUM_LAYERS=${1-24}
HIDDEN_SIZE=${2-2560}
HEADS=${3-16}
SEQ_LEN=${4-1024}
BATCH_SIZE=${5-4}

WINDOW_SIZE=${6-4}

# GLOBAL_BATCH_SIZE=$((8 * ${BATCH_SIZE} * ${WORLD_SIZE}))
GLOBAL_BATCH_SIZE=${BATCH_SIZE}

# --- SH run an MoE model ------------
PYTHONGIL=1 python /shared_ssd_storage/shilonglei/OOC/sc22-ae/SHv0/pretrain_gpt.py \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size 1 \
    --moe-expert-parallel-size 1 \
    --num-experts 1 \
    --moe-loss-coeff 0.01 \
    --moe-train-capacity-factor 1.0 \
    --moe-eval-capacity-factor 1.0 \
    --moe-min-capacity 4 \
    --init-method-std 0.014 \
    --lr-decay-tokens 300000000000 \
    --lr-warmup-tokens 375000000 \
    --micro-batch-size 4 \
    --exit-duration-in-mins 30000000 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${HEADS} \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --max-position-embeddings ${SEQ_LEN} \
    --train-tokens 300000000000 \
    --train-iters 30 \
    --lr 4.5e-4 \
    --min-lr 4.5e-06 \
    --lr-decay-style cosine \
    --split 98,2,0 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-iters 10 \
    --save-interval 10000 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers 0 \
    --fp16 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_PATH \
    --merge-file ${MERGE_PATH} \
    --checkpoint-activations \
    --create-moe-param-group \
    --data-impl mmap\
    --distributed-backend nccl\
    --deepspeed \
    --deepspeed_config ds_config_gpt_gpt-0.125B-lr-4.5e-4-minlr-4.5e-06-bs-256-gpus--1-mp-1-pp-1-ep-64-mlc-0.01-cap-1.0-drop-true.json \
    --pipeline-model-parallel-size 1 \
    --no-pipeline-parallel \
    --deepspeed-activation-checkpointing \
    --enable-gl \
    --use-cpu-initialization \
    --gl-world-size ${WORLD_SIZE} \
    --gl-window-size ${WINDOW_SIZE} \
    --gl-ray-max-concurrency 12 
echo $CMD
eval $CMD

#   &> /shared_ssd_storage/shilonglei/OOC/Megatron-DeepSpeed/examples_deepspeed/MoE/output/log/gpt-0.125B-lr-4.5e-4-minlr-4.5e-06-bs-256-gpus--1-mp-1-pp-1-ep-64-mlc-0.01-cap-1.0-drop-true_gnerv3_2023.09.06-21.27.22.log


# --- original SH ------
# CMD="PYTHONGIL=1 python pretrain_gpt.py \
#        --num-layers ${NUM_LAYERS} \
#        --hidden-size ${HIDDEN_SIZE} \
#        --num-attention-heads ${HEADS} \
#        --seq-length ${SEQ_LEN} \
#        --micro-batch-size ${BATCH_SIZE} \
#        --global-batch-size ${GLOBAL_BATCH_SIZE} \
#        --max-position-embeddings ${SEQ_LEN} \
#        --train-iters 50 \
#        --log-interval 10 \
#        --exit-interval 50 \
#        --lr-decay-iters 320000 \
#        --save $CHECKPOINT_PATH \
#        --load $CHECKPOINT_PATH \
#        --data-path $DATA_PATH \
#        --vocab-file $VOCAB_PATH \
#        --merge-file ${MERGE_PATH} \
#        --data-impl mmap \
#        --distributed-backend nccl \
#        --split 949,50,1 \
#        --lr 0.00015 \
#        --min-lr 0.00001 \
#        --lr-decay-style cosine \
#        --lr-warmup-fraction .01 \
#        --weight-decay 1e-2 \
#        --clip-grad 1.0 \
#        --log-interval 10 \
#        --save-interval 10000 \
#        --eval-interval 1000 \
#        --eval-iters 1000 \
#        --checkpoint-activations \
#        --activations-checkpoint-method 'uniform' \
#        --activations-checkpoint-num-layers 1 \
#        --enable-gl \
#        --use-cpu-initialization \
#        --gl-world-size ${WORLD_SIZE} \
#        --gl-window-size ${WINDOW_SIZE} \
#        --gl-ray-max-concurrency 12
#        "