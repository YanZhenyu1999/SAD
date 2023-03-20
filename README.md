# SAD

## Datasets
To train on the MVtec Anomaly Detection dataset [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. 

To train on the [Magnetic-tile-defect](https://github.com/abin24/Magnetic-tile-defect-datasets.) dataset. 

If you use the basline method DRAEM, the [Describable Textures dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) was used as the anomaly source image set in most of the experiments in the paper. You can run the **download_dataset.sh** script from the project directory to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:

```
./scripts/download_dataset.sh
```

## Training

Pass the folder containing the training dataset to the **train.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```shell
python train.py --lr 0.0001 --bs 8 --epochs 200 --dataset mvtec --data_path /home/yzy/Datasets/MVTec/ --checkpoint_path ./checkpoints/mvtec/ --log_path ./logs/mvtec/ --visualize --ad 0.1 --gpu_id 6 --obj_id 4 --adv l2_0.1 --model SAD
```

The conda environement used in the project is decsribed in **requirements.txt**.


## Evaluating
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python test_DRAEM.py --gpu_id 0 --base_model_name "DRAEM_test_0.0001_200_bs8_" --data_path ./datasets/mvtec/ --checkpoint_path ./checkpoints/DRAEM_checkpoints/
```

