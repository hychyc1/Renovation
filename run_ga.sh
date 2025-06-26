#!/bin/bash
#SBATCH -c 1              # Number of cores (-c)
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu   # Partition to submit to
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logging/ga.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logging/ga.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
source activate ML

# run code
# python ga.py
python ga_stupid.py --district 朝阳区 --config cfg/cfg_normal_gnn_l1balance.yaml 