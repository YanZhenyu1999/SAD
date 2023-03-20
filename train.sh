python train_DRAEM.py --gpu_id 2 --obj_id 0 --lr 0.0001 --bs 8 --epochs 200 --dataset mvtec --data_path /home/yzy/Datasets/MVTec/ --anomaly_source_path /home/yzy/Datasets/DTD/dtd/images/ --checkpoint_path ./checkpoints/mvtec/ --log_path ./logs/mvtec/

python test_DRAEM.py --gpu_id 4 --base_model_name "DRAEM_test_0.0001_700_bs8_" --dataset mvtec --data_path /home/yzy/Datasets/MVTec/ --checkpoint_path ./checkpoints/mvtec/

python train_DRAEM.py --lr 0.0001 --bs 8 --epochs 200 --dataset mt --data_path /home/yzy/Datasets/Magnetic_Tile/ --anomaly_source_path /home/yzy/Datasets/DTD/dtd/images/ --checkpoint_path ./checkpoints/mt/ --log_path ./logs/mt --visualize --gpu_id 5 --model DRAEM --set_t linear

python test_DRAEM.py --gpu_id 5 --base_model_name "DRAEM_test_0.0001_200_bs8_" --dataset mt --data_path /home/yzy/Datasets/Magnetic_Tile/ --checkpoint_path ./checkpoints/mt

python train_DRAEM.py --gpu_id 4 --lr 0.0001 --bs 8 --epochs 300 --dataset aitex --data_path /home/yzy/Datasets/AITEX/ --anomaly_source_path /home/yzy/Datasets/DTD/dtd/images/ --checkpoint_path ./checkpoints/aitex/ --log_path ./logs/aitex --visualize

python test_DRAEM.py --gpu_id 4 --base_model_name "DRAEM_test_0.0001_200_bs8_" --dataset aitex --data_path /home/yzy/Datasets/AITEX/ --checkpoint_path ./checkpoints/aitex

tensorboard --logdir ./logs/mt

ssh -N -f -L localhost:6006:127.0.0.1:6006 yzy@172.18.166.43

python train_DRAEM.py --lr 0.0001 --bs 8 --epochs 200 --dataset mt --data_path /home/yzy/Datasets/Magnetic_Tile/ --checkpoint_path ./checkpoints/mt/ --log_path ./logs/mt/ --visualize --ad 0.1 --gpu_id 3

python train_DRAEM.py --lr 0.0001 --bs 8 --epochs 200 --dataset mvtec --data_path /home/yzy/Datasets/MVTec/ --checkpoint_path ./checkpoints/mvtec/ --log_path ./logs/mvtec/ --visualize --ad 0.1 --gpu_id 6 --obj_id 4 --adv l2_0.1 --model SAD