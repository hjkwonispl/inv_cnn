# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'), "common"))

import numpy as np
import logging
import torch
import torch.nn.functional as F
from inv_layer import(
  inv_adaptive_maxpool2d,
  inv_maxpool2d,
  inv_relu,
  inv_fc_activations,
  inv_fc_softmax,
  inv_conv_activations,
)


def forward_laywerwise_model(input_image, layerwise_model, device):
  """ Get forward pass information of layerwise_model.

  Args:
    input_image (torch.tensor): Input of network.
    layerwise_model (list): List of dictionary of CNN model.
    device (str): Device for torch.

  Returns:
    Forward information of layerwise_model
    For example,
    [{'layer_idx': 0,
      'layer_type': 'Linear',
      'in_f': torch.tensor,
      'out_f': torch.tensor},
     {'layer_idx': 0,
      'layer_type': 'Conv2d',
      'in_f': torch.tensor,
      'out_f': torch.tensor},
     ...,
     ]
  """

  # Forward given batch before classifier
  in_f = input_image.to(device)
  in_f.requires_grad = False
  out_f = []
  forward_pass = [] # Store forward_pass data

  # Store forward pass information
  with torch.no_grad():
    for layer_idx, layer_dict in enumerate(layerwise_model):

      layer = layer_dict['op'].to(device)
      layer_type = layer_dict['type']

      if layer_type == 'Linear':
        # (batch_num, channels, _, _) = in_f.shape
        batch_num = in_f.shape[0]
        in_f = in_f.view(batch_num, -1)
        out_f = layer(in_f)
      else:
        out_f = layer(in_f)

      forward_pass.append({
        'layer_type': layer_type,
        'layer_idx': layer_idx,
        'in_f': in_f.clone().detach().to(device),
        'out_f': out_f.clone().detach().to(device)})

      in_f = out_f

  return forward_pass


def inverse_vgg(
  target_class_idx,
  layerwise_model,
  max_iter,
  mask_weight,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  log_interval,
  device,
):
  """ Get inverse feature of given netowork upto kth layer w.r.t
  onehot(target_class_idx).

  Args:
    forward_pass (list): Return value of forward_laywerwise_model.
    target_class_idx (torch.tensor): Scalar value indicate idx to be inversed.
    layerwise_model (list): Layerwise model.
    scale (str): Scale previously inversed feature when start succeediing
      'True': Scale invesed result to have same norm as output in forward pass.
      'False': Do not apply scaling.
    reg_weight (float): Weight for regularization.
    mask_weight (float): Weight for regularization mask.
    max_iter (int): Maximum iteration for GPA.
    cost_seq_bufsz (int): The lenth of cost series which updated according
      to the iteration.
    cost_seq_std_th (float): Theshold for standard deviation of cost.
      If the standard deviation of cost lesser than cost_seq_std_th,
      terminate iteration with success flag.
    clip_op (str): Option for clipping x.
    device (str): device for torch.
    log_interval: Polling interval for logging GPA process.

  Returns:
    List of tensors contain invered feature map of
    each layer in layerwise model
    For example,
      [torch.tensor([3,224,224]),
       torch.tensor([64,112,112]),
       torch.tensor([128,64,64]),
       torch.tensor([256,32,32]),
       torch.tensor([512,16,16]),
       torch.tensor([512,16,16]),
       ...
       ]
  """
  # Set Initialization method for GPA
  init_method = 'avgpool'

  # Set Initial inversemap.
  x_inv = torch.zeros([1000]).to(device)
  x_inv[target_class_idx] = 1

  # Inverse feature part laywerwisely.
  inv_feature_list = []
  mask = None

  for inv_idx, layer_dict in enumerate(reversed(layerwise_model)):

    # Retrieve layerwise information
    layer_idx = layer_dict['idx']
    layer_type = layer_dict['type']
    layer = layer_dict['op'].to(device)
    in_f = layer_dict['in_f'].to(device)
    out_f = layer_dict['out_f'].to(device)

    logging.info('Inverse {}th layer ({})'.format(inv_idx, layer_type))

    # Inverse Dropout
    if layer_type in ['Dropout', 'Flatten', 'ReLU']:
      logging.info('Skip \'{}\' layer.'.format(layer_type))
      inv_feature_list.append(x_inv)
      continue

    # Rescale x_inv
    x_inv = x_inv * torch.norm(out_f) / (torch.norm(x_inv) + 1e-10)
    in_f = in_f * torch.norm(out_f) / (torch.norm(x_inv) + 1e-10)

    # Inverse FC
    if layer_type == 'Linear':

      in_f = in_f.squeeze()

      # Predicator for overdetermined ness.
      in_f_size = torch.tensor(in_f.shape).float().prod()
      out_f_size = torch.tensor(out_f.shape).float().prod()

      if inv_idx == 0:
        inv_fc_ftn = inv_fc_softmax
      else:
        inv_fc_ftn = inv_fc_activations

      # Inverse FC layer.
      x_inv = inv_fc_ftn(
        target_idx=target_class_idx.squeeze(),
        y=x_inv,
        x_bound=in_f,
        op=layer,
        reg_weight=mask_weight,
        cost_seq_bufsz=cost_seq_bufsz,
        cost_seq_std_th=cost_seq_std_th,
        clip_op=clip_op,
        max_iter=max_iter,
        device=device,
        log_interval=log_interval,
        init_method=init_method,
      ).view(in_f.shape)

    if layer_type == 'Conv2d':

      mask = x_inv
      mask = mask.abs().sum(dim=1).unsqueeze(0)
      mask /= mask.max()
      mask = 1 - mask

      # Apply negative variation for input space
      if inv_idx == len(layerwise_model) - 1:
        clip_op = '+-'

      x_inv = inv_conv_activations(
        y=x_inv,
        x_bound=in_f,
        op=layer,
        mask=mask,
        mask_weight=mask_weight,
        cost_seq_bufsz=cost_seq_bufsz,
        cost_seq_std_th=cost_seq_std_th,
        clip_op=clip_op,
        max_iter=max_iter,
        device=device,
        log_interval=log_interval,
        init_method=init_method,
      ).view(in_f.shape)

    # Inverse AdaptiveMaxPool2d
    if layer_type in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']:
      x_inv = x_inv.view(out_f.shape)
      x_inv = inv_adaptive_maxpool2d(
        y=x_inv,
        x_bound=in_f,
        clip_op=clip_op,
        device=device,
      ).view(in_f.shape)

    # Inverse MaxPool2d
    if layer_type == 'MaxPool2d':
      x_inv = inv_maxpool2d(
        y=x_inv,
        x_bound=in_f,
        op=layer,
        output_size=in_f.shape,
        clip_op=clip_op,
        device=device,
      ).view(in_f.shape)

    # If x_inv has nan -> terminate
    if torch.isnan(x_inv).any() > 0:
      err_str = 'NAN detected : {}th layer, type = {}'.format(
        inv_idx,
        layer_type
      )
      logging.error(err_str)
      print(err_str)
      return False

    # If x_inv is zero image -> terminate
    if torch.sum(x_inv.abs().view(-1)) == 0:
      err_str = 'Zero sum: {}th layer, type = {}'.format(
        inv_idx,
        layer_type
      )
      logging.error(err_str)
      print(err_str)
      return False

    # Store inverse result
    inv_feature_list.append(x_inv)

  return inv_feature_list


def inverse_vdsr(
  layerwise_model,
  output,
  max_iter,
  scale,
  mask_weight,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  device,
  log_interval,
  **kwargs,
):
  """ Get inverse DnCNN.

  Args:
    layerwise_model (list): Layerwise model.
    max_iter (int): Maximum iteration for GPA.
    scale (str): Scale previously inversed feature when start succeediing
      'True': Scale invesed result to have same norm as output in forward pass.
      'False': Do not apply scaling.
    reg_type (str): Type of regularization for over-determined case.
      'norm' = No addictional cost,
      'l1' = l1_weight * torch.abs().sum(),
      'l2' = l2_weight * torch.norm(),
    reg_weight (float): Weight for regularization.
    mask_weight (float): Weight for regularization mask.
    cost_seq_bufsz (int): The lenth of cost series which updated according
      to the iteration.
    cost_seq_std_th (float): Theshold for standard deviation of cost.
      If the standard deviation of cost lesser than cost_seq_std_th,
      terminate iteration with success flag.
    clip_op (str): Option for clipping x.
    device (str): device for torch.
    log_interval: Polling interval for logging GPA process.

  Returns:
    List of tensors contain invered feature map of
    each layer in layerwise model
    For example,
      [torch.tensor([3,224,224]),
       torch.tensor([64,112,112]),
       torch.tensor([128,64,64]),
       torch.tensor([256,32,32]),
       torch.tensor([512,16,16]),
       torch.tensor([512,16,16]),
       ...
       ]
  """
  # Set Initialization method for GPA
  init_method = 'avgpool'

  # Set Initial inversemap.
  x_inv = output.to(device)

  # Inverse feature part laywerwisely.
  inv_feature_list = []
  mask = None

  for inv_idx, layer_dict in enumerate(reversed(layerwise_model)):

    # Retrieve layerwise information
    layer_idx = layer_dict['idx']
    layer_type = layer_dict['type']
    layer = layer_dict['op'].to(device)
    in_f = layer_dict['in_f'].to(device)
    out_f = layer_dict['out_f'].to(device)

    logging.info('Inverse {}th layer ({})'.format(inv_idx, layer_type))

    # Inverse Dropout
    if layer_type in ['Dropout', 'Flatten']:
      logging.info('Skip \'{}\' layer.'.format(layer_type))
      inv_feature_list.append(x_inv)
      continue

    # Rescale x_inv
    scale_value = 1.0
    if scale == 'True':
      scale_value = torch.norm(out_f) / (torch.norm(x_inv) + 1e-10)
      x_inv = x_inv * scale_value
      in_f = in_f * scale_value

    if layer_type == 'Conv2d':

      mask = x_inv
      mask = mask.abs().sum(dim=1).unsqueeze(0)
      mask /= mask.max()
      mask = 1 - mask

      _, _, conv_mask_w, conv_mask_h = in_f.shape
      conv_mask = F.interpolate(
        mask,
        size=(conv_mask_w, conv_mask_h),
        mode='nearest',
        align_corners=None,
      )

      # Apply negative variation for input space
      if inv_idx == len(layerwise_model) - 1:
        clip_op = '+-'

      x_inv = inv_conv_activations(
        y=x_inv.to(device),
        x_bound=in_f,
        op=layer,
        mask=conv_mask.to(device),
        mask_weight=mask_weight,
        cost_seq_bufsz=cost_seq_bufsz,
        cost_seq_std_th=cost_seq_std_th,
        clip_op=clip_op,
        max_iter=max_iter,
        device=device,
        log_interval=log_interval,
        init_method=init_method,
      ).view(in_f.shape)

    # Inverse ReLU
    if layer_type == 'ReLU':
      x_inv = inv_relu(
        y=x_inv,
        x_bound=in_f,
        clip_op=clip_op,
        device=device,
      ).view(in_f.shape)

    # If x_inv has nan -> terminate
    if torch.isnan(x_inv).any() > 0:
      err_str = 'NAN detected : {}th layer, type = {}'.format(
        inv_idx,
        layer_type
      )
      logging.error(err_str)
      print(err_str)
      return False

    # If x_inv is zero image -> terminate
    if torch.sum(x_inv.abs().view(-1)) == 0:
      err_str = 'Zero sum: {}th layer, type = {}'.format(
        inv_idx,
        layer_type
      )
      logging.error(err_str)
      print(err_str)
      return False

  return x_inv