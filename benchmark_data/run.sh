#!/bin/bash
#SBATCH --job-name=pcne
#SBATCH -A ap_invilab
#SBATCH -p pascal_gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH -o logs/output
#SBATCH -e logs/error

export PATH="${VSC_SCRATCH}/containers/pointnet_ae/bin:$PATH"
python -c "import torch; print(torch.cuda.is_available())"
python ours_to_pcn.py