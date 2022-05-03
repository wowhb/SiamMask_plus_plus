# SiamMask++

Visual object tracking

## Contents
1. [Environment Setup](#environment-setup)
2. [Training Models](#testing-models)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, Titan XP GPUs

- Clone the repository 
```
git clone https://github.com/wowhb/SiamMask_plus_plus.git && cd SiamMask_plus_plus
export SiamMask_plus_plus=$PWD
```
- Setup python environment
```
conda create -n siammask_pp python=3.6
source activate siammask_pp
pip install -r requirements.txt
bash make.sh
```
- Add the project to your PYTHONPATH
```
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Training

### Training Data 
- Download the [Youtube-VOS](https://youtube-vos.org/dataset/download/), 
[COCO](http://cocodataset.org/#download), 
[ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/), 
and [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/).

### Download the pre-trained model (174 MB)
(This model was trained on the ImageNet-1k Dataset)


### Training SiamMask_plus_plus base model
- [Setup](#environment-setup) your environment
- From the experiment directory, run
```
cd $SiamMask_plus_plus/experiments/siammask_plus_plus_base/
bash run.sh
```
- If you experience out-of-memory errors, you can reduce the batch size in `run.sh`.
- You can view progress on Tensorboard (<experiment\_dir>/logs/)
- evaluation(after training, you can test checkpoints on VOT dataset.)
```shell
bash test_all.sh -s 1 -e 20 -d VOT2019 -g 2  # test all snapshots with 2 GPUs
```
- Select best model for hyperparametric search.
```shell
bash test_all.sh -m [best_test_model] -d VOT2018 -n [thread_num] -g [gpu_num] # 4 threads with 2 GPUS
```

### Training SiamMask_plus_plus model with the binary module
- [Setup](#environment-setup) your environment
- In the experiment file, train with the best SiamMask_plus_plus_base model
```
cd $SiamMask_plus_plus/experiments/bi_siammask_plus_plus
bash run.sh <best_base_model>
bash run.sh checkpoint_e12.pth
```
- You can view progress on Tensorboard (<experiment\_dir>/logs/)
- evaluation(after training, you can test checkpoints on VOT dataset.)
```shell
bash test_all.sh -s 1 -e 20 -d VOT2019 -g 2
```

### Training SiamMask_plus_plus model with the refine module
- [Setup](#environment-setup) your environment
- In the experiment file, train with the best bi_SiamMask_plus_plus model
```
cd $SiamMask_plus_plus/experiments/re_siammask_plus_plus
bash run.sh <best_binary_model>
bash run.sh checkpoint_e12.pth
```
- You can view progress on Tensorboard (<experiment\_dir>/logs/)
- evaluation(after training, you can test checkpoints on VOT dataset.)
```shell
bash test_all.sh -s 1 -e 20 -d VOT2019 -g 2
