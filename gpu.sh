#!/bin/bash
#SBATCH -c 8              # Number of cores (-c)
#SBATCH --gres gpu:nvidia_a100-sxm4-40gb:1
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue   # Partition to submit to
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logging/normal.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logging/normal.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
source activate ML

# run code
python train.py --checkpoint checkpoints/ckpt_normal_20_3405.52.pt