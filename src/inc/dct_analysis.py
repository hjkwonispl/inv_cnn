# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
import sys
sys.path.append("common")
sys.path.append("inverse")
sys.path.append("denoising")

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from patchify import patchify


def unravel_zigzag(m):
  """ Zigzag scanning of DCT results """
  return np.concatenate([np.diagonal(m[::-1,:], k)[::(2*(k % 2)-1)] \
     for k in range(1-m.shape[0], m.shape[0])])


def preprocess_input_image(input_img):
  """ Preprocess input image (from mat to torch.tensor) 
  
  Args:
    input_img (np.array): Image in numpy format.
    device (str): device name for pytorch.
  """
  input_img = input_img.astype(float)
  input_img = input_img/255.

  return input_img


def img_into_patches(img, pat_h_sz, pat_w_sz, stride):
  """ Divide image into patchs of size 'pat_h_sz' x 'pat_w_sz' with stride

  Args:
    img (np.array): Input image.
    pat_h_sz (int): Height of patch.
    pat_w_sz (int): Width of patch.
    stride (int): Stride for patches.
  """
  img_h_sz, img_w_sz = img.shape
  num_rows = np.floor(img_h_sz / pat_h_sz)
  num_cols = np.floor(img_w_sz / pat_w_sz)

  img_h_sz = int(num_rows * pat_h_sz)
  img_w_sz = int(num_cols * pat_w_sz)

  img = img[0:img_h_sz, 0:img_w_sz]
  patches = patchify(img, (pat_h_sz, pat_w_sz), step=stride)
  
  num_pat_row, num_pat_col, _, _ = patches.shape
  mean_patches = patches.reshape(num_pat_row, num_pat_col, -1).mean(axis=2)
  
  return patches, mean_patches, (img_h_sz, img_w_sz)


def dct_analysis(data_dir, output_dir, dataset, scale_factor):
  """ DCT analysis of focused and unfocused patches (size: 8x8) in the input
  image for the VDSR. 
  We first compute attribution and decompose it into 8x8 non-overlapping 
  patches. Patches are divided into focused and nonfocused whether the 
  mean l1 norm of each patch is over the average or not. Then, we DCT each 
  patches and compute average DCT coefficient of focused and unfocused patches.
  Finally, the average DCT coefficients are zigzag scanned to be displayed.

  Args:
    data_dir (str): Path for input images.
    output_dir (str): Path for the output results.
    dataset (str): Name of dataset to be processed.
    scale_factor (str): Scale of SISR for the VDSR.
  """
  pat_h_sz = 8
  pat_w_sz = 8
  pat_area = pat_h_sz * pat_w_sz
  pat_stride = 8

  focused_psd_list = []
  unfocused_psd_list = []
  input_psd_list = []

  input_mat_path = os.path.join(data_dir, dataset, 'x{}'.format(scale_factor))
  inv_mat_path = output_dir
  fig_list = glob(os.path.join(input_mat_path, '*.mat'))

  for fig_name in fig_list:

    unfocused_psd = np.zeros((pat_h_sz, pat_w_sz))
    focused_psd = np.zeros((pat_h_sz, pat_w_sz))
    input_psd = np.zeros((pat_h_sz, pat_w_sz))

    # Get input and label pair
    input_img = preprocess_input_image(sio.loadmat(fig_name)['im_b_y'])

    # Get inversed img
    inv_img = sio.loadmat(
      os.path.join(inv_mat_path, 'inv_' + os.path.basename(fig_name)) + '.mat')
    inv_img = inv_img[list(inv_img.keys())[3]].astype(float)

    # Get attribution
    attr = np.abs(inv_img / (input_img + 1e-6))
    
    # Attr into patches
    attr_pats, attr_mean_pats, img_sz = img_into_patches(
      img=attr, 
      pat_h_sz=pat_h_sz, 
      pat_w_sz=pat_w_sz, 
      stride=pat_stride
    )
    num_row_patches, num_col_patches = attr_mean_pats.shape

    # Input image into patches
    input_patches, _, _ = img_into_patches(
      img=input_img, 
      pat_h_sz=pat_h_sz, 
      pat_w_sz=pat_w_sz, 
      stride=pat_stride
    )

    # DCT over patches
    pat_img_dct_list = []
    pat_img_avg_alpha_list = []
    for r in range(num_row_patches):
      for c in range(num_col_patches):

        # DWT with db2
        pat_img = input_patches[r,c,:,:]
        pat_attr = attr_pats[r,c,:,:]
        pat_img_dct = dct(
          dct(pat_img, axis=0, norm='ortho'), axis=1, norm='ortho'
        )
        pat_img_dct = np.abs(pat_img_dct)
        pat_img_avg_alpha = (np.abs(pat_attr).sum() / pat_area)

        pat_img_dct_list.append(pat_img_dct)
        pat_img_avg_alpha_list.append(pat_img_avg_alpha)

    num_focused = 0
    num_unfocused = 0
    focus_thres = np.mean(pat_img_avg_alpha_list) # Define threshold for focused

    for pat_img_dct, pat_img_avg_alpha in \
      zip(pat_img_dct_list, pat_img_avg_alpha_list):
      
      if pat_img_avg_alpha < focus_thres:
        unfocused_psd += pat_img_dct
        num_unfocused += 1
      else:
        focused_psd += pat_img_dct
        num_focused += 1
      input_psd += pat_img_dct
    
    if num_unfocused != 0:
      unfocused_psd = unfocused_psd / num_unfocused
    if num_focused != 0:
      focused_psd = focused_psd / num_focused
    input_psd = input_psd / (num_focused + num_unfocused)

    unfocused_psd_list.append(unfocused_psd)
    focused_psd_list.append(focused_psd)
    input_psd_list.append(input_psd)

  # Convert to np.array
  unfocused_psd_list = np.array(unfocused_psd_list)
  focused_psd_list = np.array(focused_psd_list)
  input_psd_list = np.array(input_psd_list)
  
  # Compute average
  avg_unfocused_psd_list = unfocused_psd_list.mean(0)
  avg_focused_psd_list = focused_psd_list.mean(0)
  avg_input_psd_list = input_psd_list.mean(0)

  # Zigzag scanning
  unfocused_zigzag = unravel_zigzag(avg_unfocused_psd_list)
  focused_zigzag = unravel_zigzag(avg_focused_psd_list)
  input_zigzag = unravel_zigzag(avg_input_psd_list)

  # Plot the image PSD
  ax = plt.figure().add_subplot(111)
  ax.plot(unfocused_zigzag, color='g', label='Unfocused', linewidth=1)
  ax.plot(focused_zigzag, color='r', label='Focused', linewidth=1)
  ax.plot(input_zigzag, '--', color='k', label='All', linewidth=1)
  ax.set_yscale('log')
  ax.set_xlim(0, 63)
  ax.set_ylim(1e-6, 5)
  ax.grid()
  plt.ylabel('Power spectral density')
  plt.xlabel('Frequency bands')
  plt.legend()
  plt.tight_layout()
  
  # Save frequency domain analysis result.
  plt.savefig(
    os.path.join(
      output_dir, 
      'freq_domain_analysis_{}_x{}.png'.format(dataset, scale_factor)
    )
  )


# Test code
if __name__ == '__main__':

  dct_analysis(
    data_dir='./data', 
    output_dir='./output/inv_vdsr/vdsr_Urban100_x4_0.15_700_210731_1037', 
    dataset='Urban100', 
    scale_factor=4,
  )