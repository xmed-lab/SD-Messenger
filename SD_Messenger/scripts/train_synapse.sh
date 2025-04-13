#!/bin/bash
dataset='synapse'
method='train_synapse'
config='synapse'
split='1_5'
now=$(date +"%Y%m%d_%H%M%S")
config=configs/${config}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$method/$dataset/$split

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$1 --master_addr=localhost --master_port=$2 $method.py \
--config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
--save-path $save_path --port $2 2>&1 | tee $save_path/$now.log