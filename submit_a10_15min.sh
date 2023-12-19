#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua10
### -- set the job Name --
#BSUB -J testjob_SWA_clean
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:15
# request xxGB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address

#BSUB -u s194357@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o SWA_gpu_out_%J.out
#BSUB -e SWA_gpu_err_%J.err

# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load matplotlib/3.7.1-numpy-1.24.3-python-3.9.17
module load cuda/12.2
which pip3

# module load torch_geometric
#module load matplotlib
#module load numpy
# module load pytorch_lightning as pl
# source ~/Desktop/DeepLearningPaiNN/SWAvenv/bin/activate


# /appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

python3 SWAclean_hpc.py


