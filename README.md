<p align="center"> <img src="./imgs/main.svg" width="1000" /> </p>

Inverse-Based Approach to Explaining and Visualizing Convolutional Neural Networks
======================
Hyuk Jin Kwon, Hyung Il Koo, Jae Woong Soh, and Nam Ik Cho

[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9457245) 
 
---

This page provides the source code of the paper 
'Inverse-Based Approach to Explaining and Visualizing Convolutional Neural Networks'  

## Abstract

This paper presents a new method for understanding and visualizing convolutional neural networks (CNNs). Most existing approaches to this problem focus on a global score and evaluate the pixel-wise contribution of inputs to the score. 
The analysis of CNNs for multi-labeled outputs or regression has not yet been considered in the literature, despite their success on image classification tasks with well-defined global scores.
To address this problem, we propose a new inverse-based approach that computes the inverse of a feed-forward pass to identify activations-of-interest in lower-layers. We developed a layer-wise inverse procedure based on two observations: (1) inverse results should have consistent internal activations to the original forward pass, and (2) a small amount of activation in inverse results is desirable for human interpretability.
Experimental results show that the proposed method allows us to analyze CNNs for classification and regression in the same framework.
We demonstrated that our method successfully finds attributions in the inputs for image classification with comparable performance to state-of-the-art methods. 
To visualize the trade-off between various methods, we developed a novel plot that shows the trade-off between the amount of activations and the rate of class re-identification.
In the case of regression, 
our method showed that conventional CNNs for single image super-resolution overlook a portion of frequency bands that may result in performance degradation.

## Requirements
Our code requires the following environment.
1. Ubuntu 18.04 LTS
2. Cuda 10.0 or higher
3. Python 3.6

## Installation
You can install our code with the following steps.
1. Clone and extract our code. 
2. Install python packages with 'pip install -r requirements.txt'.
3. Download datasets with 'get_data.py'.

### Database
We used following datasets.
1. Validation split of ImageNet2012 classification dataset.
2. Set5
3. Set14
4. BSDS100
5. Urban100
For Set5, Set14, BSDS100, and Urban100, we used double-precision MAT files as in [[1](#ref-1)].

### VDSR Models
We included a pretrained VDSR in '.pth' format.
Our training of VDSR is based on [[1](#ref-1)].

## Experiments with the VGG
Experiments with VGG can be reproduced with **'inv_vgg.py'**.

[Options]
```
python inv_vgg.py --gpu_num [GPU_number] --start_idx [start index] --end_idx [end index] --idx_stride [stride for indexes] --output_root_dir [output path]

--gpu_num: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--start_idx: Start index of images in ImageNet2012 validation to be processed. [Default: 0]
--end_idx: End index of images in ImageNet2012 validation to be processed [Default: 50000]
--idx_stride: Stride of images in ImageNet2012 validation to be processed [Default: 1]
--output_root_dir: Output directory [Default: './output']
```

## Experiments with the VDSR
Experiments with VGG can be reproduced with **'inv_vdsr.py'**.

[Options]
```
python inv_vdsr.py --gpu_num [GPU_number] --dataset [SR dataset] --scale_factor [SR scale] --output_root_dir [output path]

--gpu_num: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--dataset: Name of dataset to be tested [Choices: Artificial, Set5, Set14, BSDS100, Urban100]
--scale_factor: Scale factor of dataset [Choices: 2, 3, 4]
--output_root_dir: Output directory [Default: './output']
```

## Citation
```
@article{kwon2021invcnn,
  title={{Inverse-Based Approach to Explaining and Visualizing Convolutional Neural Networks}},
  author={Hyuk Jin Kwon, Hyung Il Koo, Jae Woong Soh, and Nam Ik, Cho},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021}
}
```

## Acknowledgment  
Our implementation of deletion and insertion metrics originated in [[2](#ref-2)].
****

## References
<a name="ref-1"></a>[1] https://github.com/twtygqyy/pytorch-vdsr \
<a name="ref-2"></a>[2] https://github.com/eclique/RISE