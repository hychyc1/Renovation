#!/bin/bash
#SBATCH -c 2              # Number of cores (-c)
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu   # Partition to submit to
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logging/normal_cy_single_2.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logging/normal_cy_single_2.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
source activate ML

# run code
python train.py --config cfg/cfg_normal_gnn_l1balance_single.yaml --district 朝阳区 
# --checkpoint checkpoints/ckpt_normal_gnn_l1_bal_single_200_447.21.pt