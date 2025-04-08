#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='synapse'
method='test_flops'
exp='synapse'
config='synapse'
split='2_5_aug'
config=configs/${config}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/train_synapse_aug_cls_imbalance/$exp/$split

mkdir -p $save_path

CUDA_VISIBLE_DEVICES="5" python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log