#!/bin/bash
#SBATCH --job-name=pcne
#SBATCH -A ap_invilab
#SBATCH -p ampere_gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH -o logs/%x.stdout.%j
#SBATCH -e logs/%x.stderr.%j

# module load PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0

# source $VSC_HOME/.bashrc
# conda activate pointnet_ae
# pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git
# # module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
# # python -c "import torch; print(torch.cuda.is_available())"
# # python $VSC_SCRATCH/Point-Cloud-Autoencoder/PointCloudAEcat_comp.py
# conda deactivate


export PATH="${VSC_SCRATCH}/containers/pointnet_ae/bin:$PATH"
python -c "import torch; print(torch.cuda.is_available())"
python PointCloudAEcat_comp_new_loader.py --data data/final_splits/Dutch/easy/splits --batch_size 2048