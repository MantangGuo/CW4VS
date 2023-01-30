# CW4VS
PyTorch implementation of our paper "[Content-aware Warping for View Synthesis](https://arxiv.org/abs/2201.09023)"

## Requrements
- Python 3.6.10
- Pytorch 1.8.1+cu102

## Train and Test
### Data
We provide the preprocessed DTU dataset we used for training and testing on 2-inputs task, i.e., reconstruct the target view from 2 source views. Our training dataset can be downloaded from [here](https://drive.google.com/file/d/1W6OJfva8RAL9fMotyZtZ-E43oPTXFdNz/view?usp=sharing), and the testing dataset can be downloaded from [here](https://drive.google.com/file/d/1kYkqZC2q18rf5kc5ff787E3M7CB-luRL/view?usp=sharing). Put the downloaded datasets in the folder `./dataset/`

The dataset is organized in a .h5 file. Each scene contains:
- `Ks (3 x 3 x N)`: the `3x3` intrinsic matrices;
- `Rs (3 x 3 x N)`: the `3x3` rotation matrces from the world coordinate system to camera coordinate system;
- `Ts (3 x N)`: the translation vectors from the world coordinate system to camera coordinate system;
- `flows (H x W x ns x (ns-1) x 2 x N)`: the optical flow maps between source views. For each one of the `ns` source views, we calculate the optical flow maps from the current source view and another one of the `ns-1` remaining source views;
- `pose_maps (6 x N)`: the 6DoF vectors formed by 3-dimension translation vector and 3-dimension Euler angles to represent the camera pose;
- `source_clusters (ns x N)`: the indexes of source views. For each target view, we select `ns` views as its source views;
- `views (3 x H x W x N)`: the view images,

where `N` is the number of views of the scene, `ns` is the number of source views for each target view, and `H x W` is the size of the view image.


### Run CW4VS
We provide the pre-trained model on the 2-input task over the DTU dataset in the folder `./logs/DTU/`

- Inference:
```
python test.py --num_source 2 --model_name dtu_s2.pth --test_data_path ./dataset/test_DTU_RGB_18x49_flow_18x49x2x1_6dof_18x49x6_sc_18x49x2.h5
```

- Training:
```
python train.py --num_source 2 --training_data_path ./dataset/train_DTU_RGB_79x49_flow_79x49x2x1_6dof_79x49x6_sc_79x49x2.h5
```
