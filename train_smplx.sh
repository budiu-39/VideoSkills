#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J job                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=2-23:55             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=90G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --exclude=galvani-cn221
#SBATCH --output=/mnt/lustre/work/ponsmoll/pba936/result_sbatch/lr_%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/ponsmoll/pba936/result_sbatch/lr%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ENTER_YOUR_EMAIL   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
cd $HOME
source .bashrc
conda activate /home/ponsmoll/pba936/.conda/envs/isaac
cd /mnt/lustre/work/ponsmoll/pba936/VideoSkills
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

# Compute Phase
srun python videoskills/train.py --task=smplx --use_wandb --headless
# srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments
