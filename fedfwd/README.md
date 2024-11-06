
## Requirements

To install requirements:

```setup
python -m venv FedFwd
source FedFwd/bin/activate
pip install -r requirements.txt
```

```data  and checkpoint
mkdir data
mkdir checkpoints
```
then you need to download the [pretrained imagenet-21k](https://github.com/google-research/vision_transformer) model and rename it to imagenet21k_ViT-B_16.npz into checkpoints folder. You can also download it [here](https://drive.google.com/file/d/17CUMf4m8mNAvT8iyhytIVM0WXy1dINUE/view?usp=sharing).
## Datasets

We provide two federated benchmark datasets spanning image classification task CIFAR100 and officenet for label heterogentiy and feature heterogentiy respectively.

### CIFAR100
For CIFAR100 dataset, download and unzip data under 'data' file catalog or simply run corresponding algorithm and our program will download data automatically.

### office-caltech10
Please download our pre-processed datasets [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155149226_link_cuhk_edu_hk/EaBgx5UmvatMi0KdvfdLWsABC49vcjZ2n9oZkjwl8jPMyA?e=TDxqN5), put under data/ directory and perform following commands:
```
cd ./data
unzip office_caltech_10_dataset.zip
```

## Usage
```
sh train.sh
```
