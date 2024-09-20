#!/bin/bash
#SBATCH --job-name=odg
#SBATCH -A ap_invilab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stuti.pathak@uantwerpen.be
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH -o logs/%x.stdout.%j
#SBATCH -e logs/%x.stderr.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stuti.pathak@uantwerpen.be


# module load PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0

source ~/.bashrc

conda activate 

python -c "import torch; print(torch.cuda.is_available())"

python $VSC_SCRATCH/Point-Cloud-Autoencoder/HPC_script.sh