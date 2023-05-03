#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 32 processor core(s) per node 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH -A baskargroup-a100gpu
#SBATCH --no-requeue
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3/4.10.3-svrr7oe
source activate DLRG_Lightning

python test.py