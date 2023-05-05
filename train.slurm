#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4  # 32 processor core(s) per node 
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=priority-a100    # gpu node(s)
#SBATCH -A baskargroup-a100gpu
#SBATCH --no-requeue
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
#export PL_TORCH_DISTRIBUTED_BACKEND=gloo
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3/4.10.3-svrr7oe
source activate DLRG_Lightning

###### Only Seuence Embeddings as a Input to predict 3D structure ######
# python main.py -g 1 -ep 250 --name 'Only_SeqEmbeddings_lr_0.0003_less_params_v2_L1loss_l2_0.003'

# python main.py -g 1 -ep 250 --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/tensorboard_logs/version_3/last.ckpt'


python main.py -nodes 1 -g 4 --num_workers 4 -b 32 -ep 250 -OnlySeq --bigData --name 'BigData_Only_SeqEmbeddings_lr_0.001'

#python main.py -nodes 2 -g 4 --num_workers 4 -b 32 -ep 250 -OnlySeq --bigData --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/BigData_Only_SeqEmbeddings_bs_256_lr_0.001/version_0/last.ckpt'



####### Vision-Language Model #######
# python main.py -g 1 -ep 250

# python main.py -g 1 -ep 250 --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/tensorboard_logs/version_3/last.ckpt'


# python main.py -g 1 -ep 250 --name VL_img_enc_4x4x64

# python main.py -g 1 -ep 250 --name VL_img_enc_4x4x64 --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/VL_img_enc_4x4x64/version_2/last.ckpt'


# python main.py -g 1 -ep 250 --name VL_img_enc_4x4x64_with_bg -bg 

# python main.py -g 1 -ep 250 --name VL_img_enc_4x4x64_with_bg -bg --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/VL_img_enc_4x4x64_with_bg/version_2/last.ckpt'


###### AutoEncoders for Images ######

# python main.py -g 1 -ep 100 --proj_name AutoEncoder --name AE_WithoutBackground_bs128_ep100 -ae -b 128

# python main.py -g 1 -ep 100 --proj_name AutoEncoder --name AE_WithoutBackground_bs128_ep100 -ae -b 128 --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithoutBackground_bs128_ep100/version_6/last.ckpt'


# python main.py -g 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100 -ae -bg -b 128 

# python main.py -g 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100 -ae -bg -b 128 --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_6/last.ckpt'


# python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithoutBackground_bs128_ep100_GroupNorm -ae -b 128 --gn

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithoutBackground_bs128_ep100_GroupNorm -ae -b 128 --gn --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithoutBackground_bs128_ep100_GroupNorm/version_5/last.ckpt'

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100_GroupNorm -ae -bg -b 128 --gn

#python main.py -g 4 --num_workers 1 -ep 100 --proj_name AutoEncoder --name AE_WithBackground_bs128_ep100_GroupNorm -ae -bg -b 128 --gn --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100_GroupNorm/version_7/last.ckpt'

###### VL models with pretrained AutoEncoder ######
# python main.py -g 1 -ep 250 --name VL_img_pretrainedAE_4x4x256_v2 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

# python main.py -g 4 --num_workers 1 -ep 250 --name VL_img_pretrainedAE_4x4x256_v2 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt' --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/VL_img_pretrainedAE_4x4x256_v2/version_1/last.ckpt'


# python main.py -g 4 --num_workers 1 -ep 250 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

#python main.py -g 4 --num_workers 1 -ep 250 --name VL_img_pretrainedAE_4x4x256_v2_lr_0.0001 --lr 0.0001 -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt' --weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/VL_img_pretrainedAE_4x4x256_v2_lr_0.0001/version_5/last.ckpt'



######## Trying DDP ########
#python main.py -g 4 --num_workers 4 -ep 5 --name VL_img_pretrainedAE_4x4x256_v2_ddp_4gpus -bg --ae_weights '/work/mech-ai-scratch/jrrade/Protein/TmAlphaFold/Vision_Language_Model/logs/AE_WithBackground_bs128_ep100/version_7/epoch=72-step=901039.ckpt'

