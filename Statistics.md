# Visualize statistics in BatchNorm

This tutorial explains how to visualize statistics of BN mentioned in our paper.

## 1. Visualization command

We provide a visualization tool for batch statistics. You can find it in [`cls/stat_util.py`](./cls/stat_util.py).

If you would like to visualize batch statistics, remember first replace `MABN` with naive `BN`. Then run `train.py` with option `--record_statistics`. The total command is like: 

```shell
python -m torch.distributed.launch --nproc_per_node=8 train.py  \       
    --save ./your_path_to_logs \
    --train_dir ./your_imagenet_training_dataset_dir \
    --val_dir ./your_imagenet_eval_dataset_dir \
    --gpu_num=8 --record_statistics --batch_size 256

```

## 2. Check the visualization result

We give a example in running experiments with batch size 2 and batch size 32. We sample one layer for visuialize, most layers are similar. Batch statistics **collapse much faster** in small batch setting.

Visualization of four statistics mentioned in our paper.
1. $\mu_{\mathcal{B}}$
<div align="center"><img src="https://user-images.githubusercontent.com/18145538/222640751-86b1affd-1f6f-4937-ae59-14b6864aa2c8.png" width="640"></div>

2. $\sigma^2_{\mathcal{B}}$
<div align="center"><img src="https://user-images.githubusercontent.com/18145538/222661511-218131c0-713c-4b60-b1c9-8e5aa34babc8.png" width="640"></div>

3. $g_{\mathcal{B}}$
<div align="center"><img src="https://user-images.githubusercontent.com/18145538/222662125-a2de4fd0-98f1-4402-87f5-6f4671093670.png" width="640"></div>

4. $\Psi_{\mathcal{B}}$
<div align="center"><img src="https://user-images.githubusercontent.com/18145538/222662327-2020404c-f864-4c31-9baf-9cfd077379a2.png" width="640"></div>
