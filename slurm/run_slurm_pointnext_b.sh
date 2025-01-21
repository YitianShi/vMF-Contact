#!/bin/bash
#SBATCH --job-name=vmf           # Job name
#SBATCH --output=logs_slurm/output_%j.log           # Output log file (%j expands to jobID)
#SBATCH --error=logs_slurm/error_%j.log             # Error log file
#SBATCH --mem=100G                       # Total memory per task
#SBATCH --time=2-00:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:4                     # Number of GPUs (if needed)
#SBATCH --partition=accelerated     # Partition to submit to
#SBATCH --account=hk-project-test-p0023465
#SBATCH --constraint=LSDF

cd ..
# Load the necessary modules and activate the conda environment
source ~/.bashrc

# Load any necessary modules
module load devel/cuda/12.2

# Activate the conda environment
conda activate c3d

# Print some information about the job
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_JOB_GPUS"

# Execute your script or command
<<<<<<< HEAD
python vmf_contact_main/train.py --point_backbone pointnext-b --experiment diffusion_pointnext_s # --ckpt epoch=78-step=30257 
=======
python vmf_contact_main/train.py --point_backbone pointnext-b --experiment diffusion_pointnext_b # --ckpt epoch=78-step=30257 
>>>>>>> ba0bdf2105f2c4629a750e75e6d249cbbc678718
