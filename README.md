# Revisiting Point Cloud Completion: Are We Ready For The Real-World? [ICCV 2025]

Read our paper here:

[![Published Paper](https://img.shields.io/badge/Published-Paper-blue)](https://openaccess.thecvf.com/content/ICCV2025/papers/Pathak_Revisiting_Point_Cloud_Completion_Are_We_Ready_For_The_Real-World_ICCV_2025_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2303.15225-b31b1b.svg)](https://arxiv.org/pdf/2411.17580)

## 🔍 Overview

Point clouds acquired in constrained, challenging, uncontrolled, and multi-sensor real-world settings are noisy, incomplete, and non-uniformly sparse. This presents acute challenges for the vital task of point cloud completion. Using tools from Algebraic Topology and Persistent Homology ($\mathcal{PH}$), we demonstrate that current benchmark object point clouds lack rich topological features that are integral part of point clouds captured in realistic environments. To facilitate research in this direction, we contribute the first real-world industrial dataset for point cloud completion, RealPC - a diverse, rich and varied set of point clouds. It consists of $\sim$ 40,000 pairs across 21 categories of industrial structures  in railway establishments. Benchmark results on several strong baselines reveal that existing methods fail in real-world scenarios. We discover a striking observation - unlike current datasets, \real{} consists of multiple 0- and 1-dimensional $\mathcal{PH}$-based topological features. We prove that integrating these topological priors into existing works helps improve completion. We present how 0-dimensional $\mathcal{PH}$ priors extract the global topology of a complete shape in the form of a 3D skeleton and assist a model in generating topologically consistent complete shapes. Since computing Homology is expensive, we present a simple, yet effective Homology Sampler guided network, BOSHNet that bypasses the Homology computation by sampling proxy backbones akin to 0-dim $\mathcal{PH}$. These backbones provide similar benefits of 0-dim $\mathcal{PH}$ right from the start of the training, unlike similar methods where accurate backbones are obtained only during later phases of the training.

![Teaser](./teaser.png)

## 📊 Dataset

We convert the individual .pcd files to paired Numpy arrays, hence obtaining two .npy files each (complete.npy and partial.npy) for the train and the test set. The RealPC dataset ($\sim$ 40,000 point cloud pairs) can be downloaded from here: 

[![RealPC Dataset](https://img.shields.io/badge/RealPC-Dataset-yellow)](https://drive.google.com/file/d/1decE9LodqeZP1C4zIxgqXBgvDZWD87j2/view?usp=sharing)

We use a smaller subset of the dataset above for TopODGNet, which can be downloaded here:

[![TopODGNet Dataset](https://img.shields.io/badge/TopODGNet-Dataset-green)](https://drive.google.com/file/d/1L7NWkhkkaKP6q74o1jBO-2SxuAkeDBLn/view?usp=sharing)


## 🛠️ Setup

Environment setup instructions for TopODGNet and BOSHNet via Conda are provided below:

### TopODGNet

1) Clone this repository.
2) Extract the TopODGNet dataset given above and place it where the repository folder is located.
3) Follow this repository for the setting up the appropriate conda environment in a GPU enabled system: [ODGNet](https://github.com/corecai163/ODGNet)

Note: For installing necessary extensions you need to download the extension folder from the above ODGNet github link and paste it in your repository folder and then install the extensions using instructions from the same ODGNet link.

### BOSHNet

TODO

## 🚀 Quick Start

### TopODGNet


Training:
```bash
python main.py --config cfgs/our_models/UpTrans.yaml --exp_name test --val_freq 10 --val_interval 100
```
Testing:
```bash
python main.py --config cfgs/our_models/UpTrans.yaml --exp_name test --val_freq 10 --val_interval 100 --test --ckpts experiments/UpTrans/our_models/ckpt-best.pth
```

### BOSHNet

Running:
```bash
python BOSHnet.py --data [location_of_train] --log [name_of_log]
```

## 📝 Citation

Please consider citing the following if you find this work useful:

```bibtex
@InProceedings{Pathak_2025_ICCV,
    author    = {Pathak, Stuti and Kumar, Prashant and Baiju, Dheeraj and Mboga, Nicholus and Steenackers, Gunther and Penne, Rudi},
    title     = {Revisiting Point Cloud Completion: Are We Ready For The Real-World?},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {25388-25398}
}
```






