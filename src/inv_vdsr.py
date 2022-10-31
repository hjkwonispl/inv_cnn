# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('src/inc')

import numpy as np
import argparse
import logging
import torch
import torchvision

from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import scipy.io as sio
import glob
import math
import time

import load_nn
from inv_nn import inverse_vdsr
from dct_analysis import dct_analysis

# Constants.
DATA_DIR = './data'
SCRIPT_NAME = os.path.basename(os.path.splitext(__file__)[0])

# Argument parsing.
parser = argparse.ArgumentParser(description='Set Ablation Constants')

# Exp settings.
parser.add_argument('--dataset', '-ds', default='Artificial', type=str,
  help='Name of dataset to be tested (Choices: Artificial, Set5, Set14, BSDS100, Urban100)')
parser.add_argument('--scale_factor', '-sf', default=2, type=int,
  help='Scale factor of dataset (Choices: 2,3,4)')
parser.add_argument('--model_name', '-mn', default='vdsr', type=str,
  help='Name of model to be inversed (default=vdsr)')
parser.add_argument('--num_exp', '-ne', default=700, type=int,
  help='Number of experiment (identifier')
parser.add_argument('--gpu_num', '-gn', default='0', type=str,
  help='Number of GPU to be used.')
parser.add_argument('--output_root_dir',
                    '-od',
                    default=os.path.join('./output', SCRIPT_NAME),
                    type=str)

# Remark for experiment.
parser.add_argument('--remark', '-rk', default='release', type=str)

# Options for inverse_cascaded_models_for_reconstruction.
parser.add_argument('--mask_weight', '-mw', default=0.15, type=float,
  help='Weight of activation norm for the convolution layer')
parser.add_argument('--max_iter', '-mi', default=3000, type=int, 
  help='Maximun number of iteration of GPA')
parser.add_argument('--scale', '-sc', default='True', type=str,
  help='Enable feature magnitude scaling for GPA')
parser.add_argument('--clip_op', '-co', default='+', type=str, 
  help='Type of clipping operation for projection (Choices: ''+'', ''none'')')
parser.add_argument('--cost_seq_bufsz', '-cb', default=10, type=int,
  help='Termination condition for GPA (GPA loss buffer size)')
parser.add_argument('--cost_seq_std_th', '-ch', default=0.1, type=float,
  help='Termination condition for GPA (GPA loss standard dev threshold)')
parser.add_argument('--log_interval', '-li', default=10, type=int)
exp_args = parser.parse_args()

# Env settings.
os.environ['CUDA_VISIBLE_DEVICES'] = exp_args.gpu_num
logging.basicConfig(
  format='[%(asctime)s] %(message)s',
  datefmt='%y/%m/%d %I:%M:%S',
  filename=os.path.join('./log', '{}_{}_{}.log'.format(
     SCRIPT_NAME,
     exp_args.model_name,
     exp_args.remark
  )),
  level=logging.DEBUG,
)


# List of inverse option for 1st conv.
def exp_inverse_vdsr(
  exp_args,
  device,
):
  """ Inverse VDSR experiment.

  Args:
    exp_args (argparse.args): arguments for exp
    device (str): device for exp
  """
  # Output dir
  exp_time_stamp = datetime.today().strftime('%y%m%d_%H%M')
  exp_output_dir = '{}_{}_{}_{}_{}_{}'.format(
    exp_args.model_name,
    exp_args.dataset,
    'x{}'.format(exp_args.scale_factor),
    exp_args.mask_weight,
    exp_args.num_exp,
    exp_time_stamp,
  )
  exp_output_dir = os.path.join(exp_args.output_root_dir, exp_output_dir)
  os.makedirs(exp_output_dir, exist_ok=True)

  # Define device, model and db.
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_nn.CascadedModels(model_name=exp_args.model_name, device='cpu')
  pretrained_model = model.pretrained_model().to(device)
  layerwise_model = model.layerwise_model()

  # Get image list (in 'mat' form)
  image_list = reversed(sorted(
    glob.glob(
      os.path.join(
        DATA_DIR, exp_args.dataset, 'x{}'.format(exp_args.scale_factor), '*.mat'
      )
    ),
    key=os.path.getsize,
  ))
  image_list = list(image_list)

  # For statistics
  avg_psnr_pred = 0.0
  avg_ssim_pred = 0.0
  avg_psnr_ipred = 0.0
  avg_ssim_ipred = 0.0
  avg_psnr_res = 0.0
  avg_ssim_res = 0.0
  avg_psnr_inputs = 0.0
  avg_ssim_inputs = 0.0
  avg_psnr_hrs = 0.0
  avg_ssim_hrs = 0.0
  avg_mse_res = 0.0
  avg_mse_inv = 0.0
  avg_mse_hr = 0.0
  avg_mse_pred = 0.0
  avg_mse_ipred = 0.0
  total_time = 0.0
  img_count = 0.0

  for img_idx, img_name in enumerate(reversed(image_list)):

    img_count += 1
    print("Processing ", img_name)

    # Get input and label pair
    label_img = sio.loadmat(img_name)['im_gt_y'].astype(float)
    input_img = preprocess_input_image(sio.loadmat(img_name)['im_b_y'], 'cpu')
    label_img /= 255.0

    # Get prediction
    pred_img = model.forward_layerwise_model(
      in_f=input_img,
      layerwise_model=layerwise_model,
      device='cpu',
    )

    # Get inverse
    start_time = time.time()
    inv_input_img = inverse_vdsr(
      layerwise_model=layerwise_model,
      output=pred_img,
      max_iter=exp_args.max_iter,
      mask_weight=exp_args.mask_weight,
      scale=exp_args.scale,
      cost_seq_bufsz=exp_args.cost_seq_bufsz,
      cost_seq_std_th=exp_args.cost_seq_std_th,
      clip_op=exp_args.clip_op,
      log_interval=exp_args.log_interval,
      device=device,
    )
    elapsed_time = time.time() - start_time
    total_time += elapsed_time

    # Free GPU memory
    model.empty_io(layerwise_model=layerwise_model)
    torch.cuda.empty_cache()

    # To device
    pretrained_model = pretrained_model.to(device)
    inv_input_img = inv_input_img.to(device)
    input_img = input_img.to(device)

    # Get predictions
    with torch.no_grad():
      pred_img = pretrained_model(input_img)
      inv_pred_img = pretrained_model(inv_input_img)

    # Get statistics of prediction (x + y) dD
    hr_img_pred = pred_img + input_img
    psnr_pred, ssim_pred = get_psnr_and_ssim(
      gt=label_img,
      pred=hr_img_pred.clone().detach(),
      shave_border=exp_args.scale_factor)
    avg_psnr_pred += psnr_pred
    avg_ssim_pred += ssim_pred

    # Get statistics of prediction of inverse (x + hat y) DE
    hr_img_ipred = inv_pred_img + input_img
    psnr_ipred, ssim_ipred = get_psnr_and_ssim(
      gt=label_img,
      pred=hr_img_ipred.clone().detach(),
      shave_border=exp_args.scale_factor)
    avg_psnr_ipred += psnr_ipred
    avg_ssim_ipred += ssim_ipred

    # Get statistics between input images ( x to hat x ) dA
    psnr_inputs, ssim_inputs = get_psnr_and_ssim(
      gt=post_process_hr_image(input_img),
      pred=inv_input_img,
      shave_border=exp_args.scale_factor)
    avg_psnr_inputs += psnr_inputs
    avg_ssim_inputs += ssim_inputs

    # Get statistics between predictions ( y to hat y ) dB
    psnr_res, ssim_res = get_psnr_and_ssim(
      gt=post_process_hr_image(pred_img),
      pred=inv_pred_img,
      shave_border=exp_args.scale_factor)
    avg_psnr_res += psnr_res
    avg_ssim_res += ssim_res

    # Get statistics of HRs  ( x + y to to hat x + hat y ) dC
    psnr_hrs, ssim_hrs = get_psnr_and_ssim(
      gt=post_process_hr_image(hr_img_pred),
      pred=hr_img_ipred,
      shave_border=exp_args.scale_factor)
    avg_psnr_hrs += psnr_hrs
    avg_ssim_hrs += ssim_hrs

    # MSE (x to hat x)
    diff_inv = (input_img - inv_input_img).squeeze()
    height, width = diff_inv.shape[:2]
    diff_inv = diff_inv[
      exp_args.scale_factor:(height - exp_args.scale_factor),
      exp_args.scale_factor:(width - exp_args.scale_factor)
    ]
    mse_inv = torch.mean(diff_inv ** 2)
    avg_mse_inv += mse_inv

    # MSE (y to hat y)
    diff_pred = (pred_img - inv_pred_img).squeeze()
    height, width = diff_pred.shape[:2]
    diff_pred = diff_pred[
      exp_args.scale_factor:height - exp_args.scale_factor,
      exp_args.scale_factor:width - exp_args.scale_factor
    ]
    mse_res = torch.mean(diff_pred ** 2)
    avg_mse_res += mse_res

    # MSE (x + y to x + hat y)
    diff_hr = (hr_img_pred - hr_img_ipred).squeeze()
    height, width = diff_hr.shape[:2]
    diff_hr = diff_hr[
      exp_args.scale_factor:height - exp_args.scale_factor,
      exp_args.scale_factor:width - exp_args.scale_factor
    ]
    mse_hr = torch.mean(diff_hr ** 2)
    avg_mse_hr += mse_hr

    # MSE (x + y to z)
    diff_pred = (hr_img_pred - torch.tensor(label_img).to(device)).squeeze()
    height, width = diff_pred.shape[:2]
    diff_pred = diff_pred[
      exp_args.scale_factor:height - exp_args.scale_factor,
      exp_args.scale_factor:width - exp_args.scale_factor
    ]
    mse_pred = torch.mean(diff_pred ** 2)
    avg_mse_pred += mse_pred

    # Get statistics of prediction of inverse (x + hat y)
    diff_ipred = (hr_img_ipred - torch.tensor(label_img).to(device)).squeeze()
    height, width = diff_ipred.shape[:2]
    diff_ipred = diff_ipred[
      exp_args.scale_factor:height - exp_args.scale_factor,
      exp_args.scale_factor:width - exp_args.scale_factor
    ]
    mse_ipred = torch.mean(diff_ipred ** 2)
    avg_mse_ipred += mse_ipred

    # Save exp results
    output_img_name = os.path.join(exp_output_dir, os.path.basename(img_name))
    write_exp_result(
      exp_output_dir=exp_output_dir,
      img_name=output_img_name,
      pred_statistics=(psnr_pred, ssim_pred),
      ipred_statistics=(psnr_ipred, ssim_ipred),
      res_statistics=(psnr_res, ssim_res),
      inputs_statistics=(psnr_inputs, ssim_inputs),
      hr_statistics=(psnr_hrs, ssim_hrs),
      mse_res_statistics=mse_res,
      mse_hr_statistics=mse_hr,
      mse_inv_statistics=mse_inv,
      mse_pred_statistics=mse_pred,
      mse_ipred_statistics=mse_ipred,
      elapsed_time=elapsed_time,
      exp_args=exp_args,
    )

    # Save result images
    input_img_ycbcr = sio.loadmat(img_name)['im_b_ycbcr']
    save_figs(input_img, input_img_ycbcr, output_img_name, 'input')
    save_figs(pred_img, input_img_ycbcr, output_img_name, 'pred')
    save_figs(inv_input_img, input_img_ycbcr, output_img_name, 'inv')
    save_figs(inv_pred_img, input_img_ycbcr, output_img_name, 'pred_from_inv')
    save_figs(hr_img_pred, input_img_ycbcr, output_img_name, 'hr_from_pred')
    save_figs(hr_img_ipred, input_img_ycbcr, output_img_name, 'hr_from_inv')

    # Save result matrix
    save_as_mat(inv_input_img, 
      os.path.join(exp_output_dir, 'inv_' + os.path.basename(img_name)))
    save_as_mat(hr_img_pred, 
      os.path.join(exp_output_dir, 'pred_' + os.path.basename(img_name)))

    # Log results
    log_str = '{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} '.format(
      psnr_pred, psnr_ipred, psnr_res, psnr_inputs, psnr_hrs)
    log_str += '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
      ssim_pred, ssim_ipred, ssim_res, ssim_inputs, ssim_hrs)
    print(log_str)

  # DCT analysis
  dct_analysis(
    data_dir=DATA_DIR, 
    output_dir=exp_output_dir,
    dataset=exp_args.dataset, 
    scale_factor=exp_args.scale_factor,
  )

  # Save summary
  write_exp_summary(
    exp_output_dir=exp_output_dir,
    img_count=img_count,
    pred_statistics=(avg_psnr_pred, avg_ssim_pred),
    ipred_statistics=(avg_psnr_ipred, avg_ssim_ipred),
    res_statistics=(avg_psnr_res, avg_ssim_res),
    inputs_statistics=(avg_psnr_inputs, avg_ssim_inputs),
    hr_statistics=(avg_psnr_hrs, avg_ssim_hrs),
    mse_res_statistics=avg_mse_res,
    mse_hr_statistics=avg_mse_hr,
    mse_inv_statistics=avg_mse_inv,
    mse_pred_statistics=avg_mse_pred,
    mse_ipred_statistics=avg_mse_ipred,
    total_time=total_time,
    exp_args=exp_args,
  )

def save_as_mat(mat_data, mat_name):
  """ Save mat_data as mat

  Args:
    output_dir (str): Output dir
    mat_data (np.array): Data to save as mat file
    mat_name (str): Name of mat file
  """
  sio.savemat('{}.mat'.format(mat_name),
    {mat_name.split('/')[-1].split('.')[0]:
     mat_data.detach().clone().cpu().squeeze().numpy()
    }
  )

def save_figs(img_y, img_ycbcr, output_img_name, postfix):
  """ Save 2 figures. One is gray scale image for PSNR and SSIM.
  The other is color image generated by SRed Y channel.ArithmeticError

  Args:
    img_y (torch.tensor): SRed Y channel.
    img_ycbcr (np.array): Bicubic interpolated image in YCBCR
    output_img_name (str): output image (include path)
    postfix (str): postfix added to output image name.
  """
  # Save grayscale images (used in calculate PSNR, SSIM)
  torchvision.utils.save_image(img_y, output_img_name + '_' + postfix + '.png')

  # Save color images
  img_color = colorize(
    ycbcr=clip_to_uint8(img_ycbcr),
    y=clip_to_uint8(post_process_hr_image(img_y)),
  )
  img_color.save(output_img_name + '_' + postfix + '_color.png')


def clip_to_uint8(img_np_array):
  """ Clipping numpy image into uint8 limit (0 to 255)
  Reference: https://github.com/twtygqyy/pytorch-vdsr

  Args:
    img_np_array (np.array): Image in numpy format.
  """
  img_np_array = img_np_array * 255.
  img_np_array[img_np_array < 0] = 0
  img_np_array[img_np_array > 255.] = 255.

  return img_np_array


def get_psnr_and_ssim(gt, pred, shave_border):
  """ Get PSNR and SSIM 

  Args:
    gt (torch.tensor): Ground truth image.
    pred (torch.tensor): Test image truth image.
    shave_border (int): Size of border to be excluded.
  """
  post_proc_pred = post_process_hr_image(pred)
  psnr = psnr_vdsr(gt=gt, pred=post_proc_pred, shave_border=shave_border)
  ssim = ssim_vdsr(gt=gt, pred=post_proc_pred, shave_border=shave_border)

  return psnr, ssim


def preprocess_input_image(input_img, device):
  """ Preprocess input image (from mat to torch.tensor) 
  
  Args:
    input_img (np.array): Image in numpy format.
    device (str): device name for pytorch.
  """
  input_img = input_img.astype(float)
  input_img = input_img/255.
  input_img = torch.from_numpy(input_img).float()
  input_img = input_img.reshape(1, -1, input_img.shape[0], input_img.shape[1])
  input_img = input_img.to(device)

  return input_img


def post_process_hr_image(hr_img_tensor):
  """ Convert to numpy.array image
  
  Args:
    hr_img_tensor (torch.tensor): HR image in torch.tensor format. 
  """
  hr_img = hr_img_tensor.cpu()
  hr_img = hr_img.data[0].numpy().astype(np.float32)
  hr_img = hr_img[0, :, :]

  return hr_img


def colorize(y, ycbcr):
  """ Draw color image from Y channel SR 
  Reference: https://github.com/twtygqyy/pytorch-vdsr

  Args:
    y (np.array): Y channel image. 
    ycbcr (np.array): YCbCr image. 
  """
  img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
  img[:, :, 0] = y
  img[:, :, 1] = ycbcr[:, :, 1]
  img[:, :, 2] = ycbcr[:, :, 2]
  img = Image.fromarray(img, "YCbCr").convert("RGB")

  return img


def psnr_vdsr(pred, gt, shave_border=0):
  """ Compute PSNR for VDSR results 
  Reference: https://github.com/twtygqyy/pytorch-vdsr

  Args:
    gt (torch.tensor): Ground truth image.
    pred (torch.tensor): Test image truth image.
    shave_border (int): Size of border to be excluded.
  """
  height, width = pred.shape[:2]
  pred = pred[shave_border:height - shave_border,
              shave_border:width - shave_border]
  gt = gt[shave_border:height - shave_border,
          shave_border:width - shave_border]
  imdff = pred - gt
  rmse = math.sqrt(np.mean(imdff ** 2))
  if rmse == 0:
    return 100
  return 20 * math.log10(1.0 / rmse)


def ssim_vdsr(pred, gt, shave_border=0):
  """ Compute SSIM for VDSR results 
  Reference: https://scikit-image.org/docs/dev/api/skimage.metrics.html

  Args:
    gt (torch.tensor): Ground truth image.
    pred (torch.tensor): Test image truth image.
    shave_border (int): Size of border to be excluded. 
  """
  height, width = pred.shape[:2]
  pred = pred[shave_border:height - shave_border,
              shave_border:width - shave_border]
  gt = gt[shave_border:height - shave_border,
          shave_border:width - shave_border]

  return ssim(
    pred.astype(np.float),
    gt.astype(np.float),
    multichannel=False,
    gaussian_weights=True,
    sigma=1.5,
    use_sample_covariance=False,
    data_range=1.0
  )

def write_exp_result(
  exp_output_dir,
  img_name,
  pred_statistics,
  ipred_statistics,
  res_statistics,
  inputs_statistics,
  hr_statistics,
  mse_res_statistics,
  mse_hr_statistics,
  mse_inv_statistics,
  mse_pred_statistics,
  mse_ipred_statistics,
  elapsed_time,
  exp_args,
):
  """ Write result of each experiment in txt format

  Args:
    exp_output_dir (str): Directory for text file.
    img_name (str): Name of image under exp.
    pred_statistics (turple): PSNR, SSIM of (x + y)
    ipred_statistics (turple): PSNR, SSIM of (x + hat y)
    res_statistics (turple): PSNR, SSIM between y and hat y
    inputs_statistics (turple): PSNR, SSIM between x and hat x
    hr_statistics (turple): PSNR, SSIM between (x + y) and (x + hat y)
    mse_res_statistics (float): MSE of (x + y)
    mse_hr_statistics (float): MSE of (x + hat y)
    mse_inv_statistics (float): MSE between y and hat y
    mse_pred_statistics (float): MSE between x and hat x
    mse_ipred_statistics (float): MSE between (x + y) and (x + hat y)
    elapsed_time (float): Processing time
    exp_args (argparse.args): Arguments for exp.
  """
  exp_result_file = '{}.txt'.format(exp_output_dir)
  is_exist = os.path.exists(exp_result_file)

  with open(exp_result_file, 'a+') as exp_result_file:

    # Argument of exps.
    if not is_exist:
      for arg in vars(exp_args):
        exp_result_file.write('{} {} '.format(arg, getattr(exp_args, arg)))
      exp_result_file.write('\n')

      exp_result_file.write('input ')
      exp_result_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('PSNR'))
      exp_result_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('SSIM'))
      exp_result_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('MSE'))
      exp_result_file.write('proc_time\n')

    exp_result_file.write('{} '.format(os.path.basename(img_name)))
    exp_result_file.write('{} '.format(inputs_statistics[0]))
    exp_result_file.write('{} '.format(res_statistics[0]))
    exp_result_file.write('{} '.format(hr_statistics[0]))
    exp_result_file.write('{} '.format(pred_statistics[0]))
    exp_result_file.write('{} '.format(ipred_statistics[0]))
    exp_result_file.write('{} '.format(inputs_statistics[1]))
    exp_result_file.write('{} '.format(res_statistics[1]))
    exp_result_file.write('{} '.format(hr_statistics[1]))
    exp_result_file.write('{} '.format(pred_statistics[1]))
    exp_result_file.write('{} '.format(ipred_statistics[1]))
    exp_result_file.write('{} '.format(mse_inv_statistics))    
    exp_result_file.write('{} '.format(mse_res_statistics))
    exp_result_file.write('{} '.format(mse_hr_statistics))
    exp_result_file.write('{} '.format(mse_pred_statistics))
    exp_result_file.write('{} '.format(mse_ipred_statistics))
    exp_result_file.write('{} '.format(elapsed_time))
    exp_result_file.write('\n')


def write_exp_summary(
  exp_output_dir,
  img_count,
  pred_statistics,
  ipred_statistics,
  res_statistics,
  inputs_statistics,
  hr_statistics,
  mse_res_statistics,
  mse_hr_statistics,
  mse_inv_statistics,
  mse_pred_statistics,
  mse_ipred_statistics,
  total_time,
  exp_args,
):
  """ Write result of each experiment in txt format

  Args:
    exp_output_dir (str): Directory for text file.
    img_count (int): Number of images in experiment.
    pred_statistics (turple): PSNR, SSIM of (x + y)
    ipred_statistics (turple): PSNR, SSIM of (x + hat y)
    res_statistics (turple): PSNR, SSIM between y and hat y
    inputs_statistics (turple): PSNR, SSIM between x and hat x
    hr_statistics (turple): PSNR, SSIM between (x + y) and (x + hat y)
    mse_res_statistics (float): MSE of (x + y)
    mse_hr_statistics (float): MSE of (x + hat y)
    mse_inv_statistics (float): MSE between y and hat y
    mse_pred_statistics (float): MSE between x and hat x
    mse_ipred_statistics (float): MSE between (x + y) and (x + hat y)
    total_time (float): Total processing time
    exp_args (argparse.args): Arguments for exp.
  """
  exp_summary_file = '{}.txt'.format(exp_output_dir)
  exp_summary_file = os.path.join(
    os.path.dirname(exp_summary_file),
    '{}_summary.txt'.format(exp_args.model_name),
  )

  # Write summary for exp
  with open(exp_summary_file, 'a+') as exp_summary_file:

    # Argument of exp
    for arg in vars(exp_args):
      exp_summary_file.write('{} {} '.format(arg, getattr(exp_args, arg)))
    exp_summary_file.write('\n')

    exp_summary_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('PSNR'))
    exp_summary_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('SSIM'))
    exp_summary_file.write('dA_{0} dB_{0} dC_{0} dD_{0} dE_{0} '.format('MSE'))
    exp_summary_file.write('total_time\n')

    exp_summary_file.write('{} '.format(inputs_statistics[0] / img_count))
    exp_summary_file.write('{} '.format(res_statistics[0] / img_count))
    exp_summary_file.write('{} '.format(hr_statistics[0] / img_count))
    exp_summary_file.write('{} '.format(pred_statistics[0] / img_count))
    exp_summary_file.write('{} '.format(ipred_statistics[0] / img_count))
    exp_summary_file.write('{} '.format(inputs_statistics[1] / img_count))
    exp_summary_file.write('{} '.format(res_statistics[1] / img_count))
    exp_summary_file.write('{} '.format(hr_statistics[1] / img_count))
    exp_summary_file.write('{} '.format(pred_statistics[1] / img_count))
    exp_summary_file.write('{} '.format(ipred_statistics[1] / img_count))
    exp_summary_file.write('{} '.format(mse_inv_statistics/ img_count))
    exp_summary_file.write('{} '.format(mse_res_statistics/ img_count))
    exp_summary_file.write('{} '.format(mse_hr_statistics/ img_count))
    exp_summary_file.write('{} '.format(mse_pred_statistics/ img_count))
    exp_summary_file.write('{} '.format(mse_ipred_statistics/ img_count))        
    exp_summary_file.write('{} '.format(total_time))
    exp_summary_file.write('\n')


if __name__ == '__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  exp_inverse_vdsr(exp_args=exp_args, device=device)