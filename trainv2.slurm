#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=10:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=4   # number of nodes
#SBATCH --ntasks-per-node=4  # 32 processor core(s) per node 
#SBATCH --mem=369G  # maximum memory per node
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH -A baskargroup-a100gpu
#SBATCH --no-requeue
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
#export PL_TORCH_DISTRIBUTED_BACKEND=gloo
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3/4.10.3-svrr7oe
source activate DLRG_Lightning

# ntasks-per-node * nodes should be equal to the total number of gpus requested

# nodes=1 ntasks-per-node=1 gpus=1  jobid = 4453668
#srun python main.py -nodes 1 -g 1 -b 32 --num_workers 4 -ep 10 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001_playground_1x1 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

# nodes=1 ntasks-per-node=2 gpus=2 jobid = 4453672
#srun python main.py -nodes 1 -g 2 -b 32 --num_workers 4 -ep 10 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001_playground_1x2 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

# nodes=1 ntasks-per-node=4 gpus=4 jobid = 4453674
#srun python main.py -nodes 1 -g 4 -b 32 --num_workers 4 -ep 10 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001_playground_1x4 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

# nodes=2 ntasks-per-node=4 gpus=4 jonid = 4453680
#srun python main.py -nodes 2 -g 4 -b 32 --num_workers 4 -ep 10 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001_playground_2x4 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

# nodes=4 ntasks-per-node=4 gpus=4 jobid = 4453684
srun python main.py -nodes 4 -g 4 -b 32 --num_workers 4 -ep 10 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001_playground_4x4 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'
