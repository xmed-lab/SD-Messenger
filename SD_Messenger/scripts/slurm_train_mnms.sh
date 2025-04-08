#!/bin/bash
#SBATCH --job-name=SD-messenger-mnms
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p partion_name
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00

dataset='mnms'
method='train'
exp='mnms_toA'
config='mnms'
split='test_toA'
now=$(date +"%Y%m%d_%H%M%S")
config=configs/${config}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

module purge
module load Anaconda3
source activate ratchet
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/toolkit/11.8.0

mkdir -p $save_path

torchrun --nproc_per_node=$1 --master_addr=localhost --master_port=$2 $method.py \
--config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
--save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
