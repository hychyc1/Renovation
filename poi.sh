#!/bin/bash
#SBATCH -c 4              # Number of cores (-c)
#SBATCH --gres gpu:nvidia_a100-sxm4-40gb:1
#SBATCH -t 1-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue   # Partition to submit to
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logging/poi.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logging/poi.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
source activate ML

# run code
python train.py --config cfg/cfg_poi.yaml