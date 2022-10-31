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
import torch.nn.functional as F
import torchvision
from datetime import datetime
from matplotlib import pyplot as plt
import scipy.io as sio
import re

import load_nn
import load_db
from delins_score import delins_test_with_heatmap
from inv_nn import inverse_vgg
from utils import torch_to_np
from utils import torch_to_image

# Constants.
DATA_DIR = './data'
IMAGENET_VAL_PATH = os.path.join(DATA_DIR, 'imagenet2012')
SCRIPT_NAME = os.path.basename(os.path.splitext(__file__)[0])
VAL_GT_PATH = './res/ILSVRC2012_validation_ground_truth.txt'
VAL_GT_TXT_PATH = './res/ILSVRC2012_validation_ground_truth_text.txt'
EPS = 1e-6

# Argument parsing.
parser = argparse.ArgumentParser(description='Set Exp Constants')

# Arguments for experimental settings
parser.add_argument('--gpu_num', '-gn', default=0, type=int,
  help='Number of GPU to be used.')
parser.add_argument('--num_exp', '-ne', default=700, type=int,
  help='Number of experiment (identifier')
parser.add_argument('--model_name', '-mn', default='vgg16', type=str,
  help='Name of model to be inversed (Default: vgg16, Choices: vgg11, vgg16, vgg19)')
parser.add_argument('--start_idx', '-si', default=1, type=int,
  help='Start index of images in ImageNet2012 validation to be processed')
parser.add_argument('--end_idx', '-ei', default=50000, type=int,
  help='End index of images in ImageNet2012 validation to be processed')
parser.add_argument('--idx_stride', '-ie', default=10, type=int,
  help='Stride of images in ImageNet2012 validation to be processed')
parser.add_argument('--output_root_dir', '-od',
  default=os.path.join('./output', SCRIPT_NAME), type=str,
  help='Output directory')
parser.add_argument('--log_interval', '-li', default=100, type=int,
  help='Logging interval')
parser.add_argument('--remark', '-rk', default='release', type=str,
  help='Remark for the experiment')

# Arguments for inverse
parser.add_argument('--max_iter', '-mi', default=3000, type=int, 
  help='Maximun number of iteration of GPA')
parser.add_argument('--mask_weight', '-mw', default=0.7, type=float,
  help='Weight of activation norm for the convolution layer')
parser.add_argument('--scale', '-sc', default='True', type=str,
  help='Enable feature magnitude scaling for GPA')
parser.add_argument('--clip_op', '-co', default='+', type=str, 
  help='Type of clipping operation for projection (Choices: ''+'', ''none'')')
parser.add_argument('--cost_seq_bufsz', '-cb', default=10, type=int,
  help='Termination condition for GPA (GPA loss buffer size)')
parser.add_argument('--cost_seq_std_th', '-ch', default=0.1, type=float,
  help='Termination condition for GPA (GPA loss standard dev threshold)')
exp_args = parser.parse_args()

# Env settings.
os.environ['CUDA_VISIBLE_DEVICES'] = str(exp_args.gpu_num)
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


def exp_inverse_vgg(exp_args, device):
  """ Inverse VGG experiment

  Args:
    exp_args (argparse): Arguments for the experiment.
    device(str): Device for torch.
  """

  # Output dir
  exp_time_stamp = datetime.today().strftime('%y%m%d_%H%M')
  exp_output_dir = '{}_{}_{}'.format(
    exp_args.model_name,
    exp_args.num_exp,
    exp_time_stamp,
  )
  exp_output_dir = os.path.join(exp_args.output_root_dir, exp_output_dir)
  heatmap_dir = os.path.join(exp_output_dir, 'attribution')
  invmat_dir = os.path.join(exp_output_dir, 'inverse_result')
  os.makedirs(exp_output_dir, exist_ok=True)
  os.makedirs(heatmap_dir, exist_ok=True)
  os.makedirs(invmat_dir, exist_ok=True)

  # Define device, model and db.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = load_nn.CascadedModels(model_name=exp_args.model_name, device=device)
  imagenet = load_db.ImageNet(
    val_dir=os.path.join(DATA_DIR, 'imagenet2012'),
    val_gt_path=VAL_GT_PATH,
    val_gt_txt_path=VAL_GT_TXT_PATH,
    device=device
  )

  # List of failed samples.
  failed_sample_list = []

  # Layerwise_model
  layerwise_model = model.layerwise_model()

  # Number of inversed samples
  n_inversed_samples = 0

  # Summary accuracy of each layer (= each depth) prediction from inverse
  top1_correct_list = np.zeros(len(layerwise_model) - 8)
  top5_correct_list = np.zeros(len(layerwise_model) - 8)

  # Write result for each inverses
  for i, data_idx in enumerate(
    range(exp_args.start_idx, exp_args.end_idx, exp_args.idx_stride)):

    print('Processing {:08d}th file {}'.format(i, data_idx))
    logging.info('Processing {:08d}th file {}'.format(i, data_idx))

    # Forward layerwise_model.
    input_img, label, label_txt = imagenet.validation_data(data_idx)
    inference = model.forward_layerwise_model(
      in_f=imagenet.pre_processing(input_img),
      layerwise_model=layerwise_model,
      device=device,
    )
    torchvision.utils.save_image( # Save input image
      imagenet.pre_processing(input_img, False),
      os.path.join(
        exp_output_dir,
        imagenet.val_img_name_template.format(data_idx) + '_input.jpg',
      )
    )

    # Get Logits and idx and probs for exp.
    logits = F.softmax(inference, dim=1)
    pred_idx = logits.argmax()
    pred_prob = logits.max()

    # Inverse
    inv_f_list = inverse_vgg(
      target_class_idx=pred_idx,
      layerwise_model=layerwise_model,
      max_iter=exp_args.max_iter,
      mask_weight=exp_args.mask_weight,
      cost_seq_bufsz=exp_args.cost_seq_bufsz,
      cost_seq_std_th=exp_args.cost_seq_std_th,
      clip_op=exp_args.clip_op,
      log_interval=exp_args.log_interval,
      device=device,
    )
    if inv_f_list == False: # Exception handling for inverse process.
      logging.error('Problem in inverse process')
      print('Problem in inverse process')
      failed_sample_list.append(data_idx)
      continue
    inv_f_list.reverse()

    # Get inversed input image
    inv_input_img = inv_f_list[0].clone().detach()
    inv_input_img = (inv_input_img - inv_input_img.min())
    inv_input_img = inv_input_img / (inv_input_img.max() - inv_input_img.min())
    torchvision.utils.save_image(
      inv_input_img.clone().detach(),
      os.path.join(
        exp_output_dir,
        imagenet.val_img_name_template.format(data_idx) + '_inv.jpg',
      )
    )

    # Test for reconstruction.
    inv_f_pred_list = []
    inv_f_pred_top5_list = []
    inv_f_prob_list = []

    # Forward layerwise model to freeze ReLU and Max positions.
    _ = model.forward_layerwise_model(
      in_f=imagenet.pre_processing(input_img),
      layerwise_model=layerwise_model,
      device=device,
    )

    # Get prediction from each inversed features in laywewise_model
    for f_idx in range(len(layerwise_model) - 8):

      # Get partial model and its input feature
      input_feature = inv_f_list[f_idx]
      partial_layerwise_model = layerwise_model[f_idx:]

      # Prediction from inversed feature.
      inv_f_forward_pass = model.forward_layerwise_model(
        in_f=input_feature,
        layerwise_model=partial_layerwise_model,
        device=device,
      )
      inv_f_inference = inv_f_forward_pass[-1]

      # Store top 1 result and its probability
      inv_logits = F.softmax(inv_f_inference, dim=0)
      inv_f_pred = torch.argmax(inv_logits)
      inv_f_prob = torch.max(inv_logits)
      inv_f_pred_list.append(int(torch_to_np(inv_f_pred)))
      inv_f_prob_list.append(float(torch_to_np(inv_f_prob)))

      # Store top 5 results
      _, inv_f_pred_top5 = torch.topk(inv_f_inference, k=5)
      inv_f_pred_top5_list.append(inv_f_pred_top5.unsqueeze(0))

    # Count correct number of samples and store prediction (top 1)
    is_f_pred_correct = (torch_to_np(pred_idx) == inv_f_pred_list)
    top1_correct_list += is_f_pred_correct.astype(float).squeeze()

    # Count correct number of samples and store prediction (top 5)
    inv_f_pred_top5_list = torch_to_np(torch.cat(inv_f_pred_top5_list, dim=0))
    is_f_pred_top5_correct = (torch_to_np(pred_idx) == inv_f_pred_top5_list)
    is_f_pred_top5_correct = is_f_pred_top5_correct.sum(axis=1)
    top5_correct_list += is_f_pred_top5_correct.astype(float).squeeze()

    # Count total number of inversed samples
    n_inversed_samples += 1

    # Save exp results
    write_exp_result(
      exp_output_dir=exp_output_dir,
      imagenet=imagenet,
      input_img=input_img,
      label=int(torch_to_np(label)),
      pred_idx=int(torch_to_np(pred_idx)),
      pred_prob=float(torch_to_np(pred_prob)),
      inv_input_img=inv_input_img,
      inv_f_list=inv_f_list,
      inv_f_pred_list=inv_f_pred_list,
      inv_f_prob_list=inv_f_prob_list,
      data_idx=data_idx,
      exp_args=exp_args,
    )

  # Perform deletion/insertion test
  delins_test_result = delins_test_with_heatmap(
    pretrained_model=model.pretrained_model(),
    inv_dir=os.path.join(exp_output_dir, 'attribution'),
    batch_sz=20,
    val_gt_txt_path=VAL_GT_TXT_PATH,
    val_gt_path=VAL_GT_PATH,
    imagenet_val_path=IMAGENET_VAL_PATH,
    device=device,
  )

  # Compute mean_l1_norm
  mean_l1_norm = compute_mean_l1_norm(
    exp_output_dir=exp_output_dir,
    imagenet_val_path=IMAGENET_VAL_PATH,
    val_gt_path=VAL_GT_PATH,
    val_gt_txt_path=VAL_GT_TXT_PATH
  )

  # Save summary of exp
  write_summary(
    exp_output_dir=exp_output_dir,
    exp_time_stamp=exp_time_stamp,
    top1_correct_list=top1_correct_list,
    top5_correct_list=top5_correct_list,
    failed_sample_list=failed_sample_list,
    n_inversed_samples=n_inversed_samples,
    exp_args=exp_args,
    delins_test_result=delins_test_result,
    mean_l1_norm=mean_l1_norm,
  )


def write_summary(
  exp_output_dir,
  exp_time_stamp,
  top1_correct_list,
  top5_correct_list,
  failed_sample_list,
  n_inversed_samples,
  exp_args,
  delins_test_result,
  mean_l1_norm,
  ):
  """ Write summary of experiment

  Args:
    exp_output_dir (str): Directory for text file.
    exp_time_stamp (str): Time stamp string for exp.
    top1_correct_list (list): Accuracy of prediction for each layer (top1).
    top5_correct_list (list): Accuracy of prediction for each layer (top5).
    failed_sample_list (list): List of failed sample index.
    n_inversed_samples (int): Number of inversed test examples.
    exp_args (argparse.args): Arguments for exp.
    delins_test_result (list): List of deletion/insertion metric score.
    mean_l1_norm (float): Mean L1 norm of inversed predictions.
  """
  exp_result_file = '{}.txt'.format(exp_output_dir)
  summary_file = os.path.join(
    os.path.dirname(exp_result_file),
    '{}_summary.txt'.format(exp_args.model_name),
  )

  # Write summary for exp
  with open(summary_file, 'a+') as summary_file:

    # General info
    summary_file.write('{} '.format(exp_args.model_name))
    summary_file.write('{} '.format(exp_args.remark))
    summary_file.write('{} '.format(exp_time_stamp))
    summary_file.write('{} '.format(n_inversed_samples))
    summary_file.write('{} '.format(failed_sample_list))

    # Argument of exp
    for arg in vars(exp_args):
      summary_file.write('{} {} '.format(arg, getattr(exp_args, arg)))
    summary_file.write('\n')

    # Column names
    summary_file.write('Exp_num Top1_accuracy Top5_accuracy ')
    summary_file.write('Mean_Del.score Mean_Ins.score Mean_L1_norm')
    summary_file.write('\n')

    # Summary results
    summary_file.write('{:d} '.format(exp_args.num_exp))
    top1_accuracy = top1_correct_list[0] / n_inversed_samples * 100
    summary_file.write('{:.2f} '.format(top1_accuracy))
    top5_accuracy = top5_correct_list[0] / n_inversed_samples * 100
    summary_file.write('{:.2f} '.format(top5_accuracy))
    summary_file.write('{:f} {:f} '.format(
      delins_test_result['deletion_scores'].mean(),
      delins_test_result['insertion_scores'].mean(),
    ))
    summary_file.write('{:f}'.format(mean_l1_norm))
    summary_file.write('\n')


def write_exp_result(
  exp_output_dir,
  imagenet,
  input_img,
  label,
  pred_idx,
  pred_prob,
  inv_input_img,
  inv_f_list,
  inv_f_pred_list,
  inv_f_prob_list,
  data_idx,
  exp_args,
):
  """ Write result of each inverse

  Args:
    exp_output_dir (str): Directory for text file.
    imagenet (ImageNet): ImageNet obj in load_db.
    input_img (PIL Image): Input image.
    label (int): Label of given input.
    pred_idx (int): Prediction of given input w.r.t. model.
    pred_prob (float): Prob of pred of given input w.r.t. model.
    inv_input_img (tensor): Inverse of prediction.
    inv_f_list (list of tensor): Inverse of each feature.
    inv_f_pred_list (list of int): Prediction from each inversed feature.
    inv_f_prob_list (list of tensor): Prob of pred from each inversed feature.
    data_idx (int): Data idx of imput image in ImageNet.
    exp_args (argparse.args): Arguments for exp.
  """
  exp_result_file = '{}.txt'.format(exp_output_dir)
  is_exist = os.path.exists(exp_result_file)

  exp_result_file = open(exp_result_file, 'a+')

  # Argument of exps.
  if not is_exist:
    for arg in vars(exp_args):
      exp_result_file.write('{} {} '.format(arg, getattr(exp_args, arg)))
    exp_result_file.write('\n')

    exp_result_file.write('Input_Img_Idx Label Pred Prob Inv_Pred Inv_Prob')
    exp_result_file.write('\n')

  # Data index and label.
  exp_result_file.write('{:05d} {:04d} '.format(data_idx, label))

  # Prediction.
  exp_result_file.write('{:04d} {:.4f} '.format(pred_idx, pred_prob))

  # List of prediction from each layer's reconstruction
  exp_result_file.write('{:04d} {:.4f} '.format(
    inv_f_pred_list[0], inv_f_prob_list[0])
  )
  exp_result_file.write('\n')
  exp_result_file.close()
  
  # Save output image
  cropped_input_img = imagenet.pre_processing(
    input_img,
    with_normalization=False,
  )
  # Get inversed input image.
  fig = plt.figure(figsize=(16, 8))
  ax = fig.subplots(1, 2)
  ax[0].set_title(r'$\bf Input$ (Pred: {}, Prob: {:.3f})'.format(
    imagenet.get_label_txt(pred_idx), pred_prob),
    fontsize=20
  )
  ax[0].set_axis_off()
  ax[0].imshow(torch_to_image(cropped_input_img))
  ax[1].set_title(r'$\bf Inverse$ (Pred: {}, Prob: {:.3f})'.format(
    imagenet.get_label_txt(inv_f_pred_list[0]),
    inv_f_prob_list[0]),
    fontsize=20
  )
  ax[1].set_axis_off()
  ax[1].imshow(torch_to_image(inv_input_img))
  fig.suptitle(r'$\bf Label$: {}, {}'.format(
    label, imagenet.get_label_txt(label)),
    fontsize=20
  )
  fig.savefig(
    os.path.join(
      exp_output_dir,
      imagenet.val_img_name_template.format(data_idx) + '_compare.jpg'
    ),
  )
  plt.close(fig)

  # Save output heatmap as mat
  heatmap_dir = os.path.join(exp_output_dir, 'attribution')
  inv_img = inv_f_list[0].clone().detach()
  input_img_tensor = imagenet.pre_processing(input_img)
  inv_img_heatmap = (abs(inv_img) / (abs(input_img_tensor) + 1e-6))
  inv_img_heatmap = inv_img_heatmap.squeeze().sum(dim=0)
  inv_img_heatmap = inv_img_heatmap / inv_img_heatmap.max()

  save_as_mat(
    output_dir=heatmap_dir,
    mat_data=torch_to_np(inv_img_heatmap.squeeze()),
    mat_name=(imagenet.val_img_name_template + '_hm').format(data_idx),
  )

  # Save inverse output as mat
  invmat_dir = os.path.join(exp_output_dir, 'inverse_result')
  save_as_mat(
    output_dir=invmat_dir,
    mat_data=torch_to_image(inv_f_list[0].squeeze()),
    mat_name=(imagenet.val_img_name_template).format(data_idx),
  )


def save_as_mat(output_dir, mat_data, mat_name):
  """ Save mat_data as mat

  Args:
    output_dir (str): Output dir
    mat_data (np.array): Data to save as mat file
    mat_name (str): Name of mat file
  """
  sio.savemat(os.path.join(output_dir, '{}.mat'.format(mat_name)),
                   {mat_name.split('.')[0]: mat_data.squeeze()})


def compute_mean_l1_norm(
  exp_output_dir,
  imagenet_val_path,
  val_gt_path,
  val_gt_txt_path,
):
  """ Compute mean l1 norm

  Args:
    exp_output_dir (str): Output dir has inverse results.
    imagenet_val_path (str): Directory where validation split of ImageNet images
    val_gt_path (str): Ground truth text file for val split
    val_gt_txt_path(str): Text description of GT of val split
  """
  # Load DB
  imagenet = load_db.ImageNet(
    val_dir=imagenet_val_path,
    val_gt_path=val_gt_path,
    val_gt_txt_path=val_gt_txt_path,
    device='cpu'
  )

  # Regular expression and directiry for inverse resuts (mat files)
  mat_file = re.compile('.*.mat')
  abs_path = os.path.join(exp_output_dir, 'inverse_result')

  avg_l1_norm = 0
  num_files = 0

  if os.path.isdir(abs_path):

    file_list = os.listdir(abs_path)
    mat_file_name_list = list(filter(mat_file.match, file_list))
    num_files = len(mat_file_name_list)

    for mat_file_name in mat_file_name_list:

      # Get inverse result
      mat_file_path = os.path.join(abs_path, mat_file_name)
      mat_contents = sio.loadmat(mat_file_path)
      mat_keys = list(mat_contents.keys())
      inv_img = mat_contents[mat_keys[3]]
      inv_img = torch.tensor(np.transpose(inv_img, (2, 0, 1)))

      # Get input image
      raw_img, _, _ = imagenet.validation_data(
        imagenet.get_validation_data_idx(mat_file_name)
      )
      input_img = imagenet.pre_processing(raw_img).squeeze()
      n_ch = input_img.shape[0]
      img_sz = input_img.shape[1] * input_img.shape[2]

      # Compute attribution and normalize it into [0, 1]
      norm_img = (abs(inv_img) / (abs(input_img) + EPS))
      norm_img_min, _ = norm_img.reshape([n_ch, img_sz]).min(dim=1)
      norm_img = norm_img - norm_img_min.reshape([n_ch, 1, 1])
      norm_img = norm_img.sum(dim=0)
      norm_img = norm_img / norm_img.max()

      # Compute average l1 norm
      l1_norm = torch.sum(norm_img)
      avg_l1_norm += l1_norm / img_sz

    if num_files != 0:
      avg_l1_norm /= num_files

  return avg_l1_norm


if __name__ == '__main__':

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  exp_inverse_vgg(exp_args=exp_args, device=device)