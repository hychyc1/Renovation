#!/bin/bash
#SBATCH -c 1              # Number of cores (-c)
#SBATCH -t 1-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=300W00           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load python/3.10.9-fasrc01
mamba activate ML

# run code
python train.py