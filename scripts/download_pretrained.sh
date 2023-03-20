#!/usr/bin/env bash
mkdir checkpoints
cd checkpoints
pip install gdown
gdown https://drive.google.com/uc?id=1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1
unzip DRAEM_checkpoints.zip

python train_DRAEM.py --gpu_id 5 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path /home/yzy/MVTec --anomaly_source_path /home/yzy/DTD/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/