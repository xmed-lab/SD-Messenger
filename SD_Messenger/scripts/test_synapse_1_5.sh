#!/bin/bash
dataset='synapse'
method='test_synapse'
config='synapse_1_5'
split='1_5'
now=$(date +"%Y%m%d_%H%M%S")
config=configs/${config}.yaml

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=$1 --master_addr=localhost --master_port=15522 $method.py \
--config=$config --checkpoint-path $2 --port 15522 2>&1 | tee $now.log