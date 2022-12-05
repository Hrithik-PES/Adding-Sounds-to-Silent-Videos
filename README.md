# Capstone Project - Team No 96

# Generating Visually Aligned Sound from Videos
# Adding Sounds To Silent Videos
# Contents
----

* [Usage Guide](#usage-guide)
   * [Getting Started](#getting-started)
      * [Installation](#installation)
      * [Download Datasets](#download-datasets)
      * [Data Preprocessing](#data-preprocessing)
   * [Training REGNET](#training-regnet)
   * [Generating Sound](#generating-sound)
* [Other Info](#other-info)
   * [Citation](#citation)
   * [Contact](#contact)


----
# Usage Guide

## Getting Started
[[back to top](#Generating-Visually-Aligned-Sound-from-Videos)]

### Installation

Create a new Conda environment.
```bash
conda create -n regnet python=3.7.1
conda activate regnet
```
Install [PyTorch][pytorch] and other dependencies.
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0
conda install ffmpeg -n regnet -c conda-forge
pip install -r requirements.txt
```

### Download Datasets

We collect 8 sound types (Dog, Fireworks, Drum, Baby form [VEGAS][vegas] and Gun, Sneeze, Cough, Hammer from [AudioSet][audioset]) to build our [Visually Aligned Sound (VAS)][VAS] dataset.
Please first download VAS dataset and unzip the data to *`$REGNET_ROOT/data/`*  folder.

For each sound type in AudioSet, we download all videos from Youtube and clean data on Amazon Mechanical Turk (AMT) using the same way as [VEGAS][visual_to_sound].


```bash
unzip ./data/VAS.zip -d ./data
```



### Data Preprocessing

Run `data_preprocess.sh` to preprocess data and extract RGB and optical flow features. 

```bash
source data_preprocess.sh
```


## Training REGNET

Training the REGNET from scratch. The results will be saved to `ckpt/dog`.

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
save_dir ckpt/dog \
auxiliary_dim 32 \ 
rgb_feature_dir data/features/dog/feature_rgb_bninception_dim1024_21.5fps \
flow_feature_dir data/features/dog/feature_flow_bninception_dim1024_21.5fps \
mel_dir data/features/dog/melspec_10s_22050hz \
checkpoint_path ''
```

In case that the program stops unexpectedly, you can continue training.
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
-c ckpt/dog/opts.yml \
checkpoint_path ckpt/dog/checkpoint_018081
```

## Generating Sound

The generated spectrogram and waveform will be saved at `ckpt/dog/inference_result`
```bash
CUDA_VISIBLE_DEVICES=7 python test.py \
-c ckpt/dog/opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```

Second, run the inference code.
```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
-c config/dog_opts.yml \ 
aux_zero True \ 
checkpoint_path ckpt/dog/checkpoint_041000 \ 
save_dir ckpt/dog/inference_result \
wavenet_path /path/to/wavenet_dog.pth
```
