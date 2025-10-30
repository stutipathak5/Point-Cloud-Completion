python -m torch.distributed.launch --nproc_per_node=1 main.py --launcher pytorch --sync_bn --config ./cfgs/KITTI_models/UpTrans.yaml --exp_name Kitti_UP --val_freq 10 --val_interval 100 
