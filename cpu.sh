#!/bin/bash
#SBATCH -c 8              # Number of cores (-c)
#SBATCH -t 1-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue   # Partition to submit to
#SBATCH --mem=30G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o normal.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e normal.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type END 	    ## Slurm will email you when your job ends
#SBATCH --mail-type FAIL            ## Slurm will email you when your job fails
#SBATCH --mail-use=yichenhuang@g.harvard.edu

# load modules
module load python/3.10.9-fasrc01
source activate ML

# run code
python train.py