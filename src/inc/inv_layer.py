# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.abspath('..'), "common"))
sys.path.append(os.path.join(os.path.abspath('..'), "inverse"))

import time
import numpy as np
import logging
import itertools
import torch
import torch.nn.functional as F


def clip_tensor(x, x_bound, clip_op, device):
  """ Clip tensors with following rules.
  no option: 0 < x < max(0, x_bound)
  allow_lbd: |x| < |x_bound|
  unbounded: no_clipping

  Args:
    x (tf.tensor): Input tensor.
    x_bound (tf.tensor): Upper bound of x.
    clip_op (str): Cliping option.
      '+': Clip x w.r.t 0 < x < x_o
      '+-': Clip x w.r.t |x| < |x_o|
      'none': No clipping

  Returns:
    Clippled torch.tensor
  """
  def positive_clip(x, x_bound):
    """ Implementation of clip_op='+'

    Args:
      x (tf.tensor): Input tensor.
      x_bound (tf.tensor): Upper bound of x.

    Returns:
      Clippled torch.tensor
    """
    x = torch.clamp(input=x, min=0)
    x_bound = torch.clamp(input=x_bound, min=0)
    x = x * (x <= x_bound) + x_bound * (x > x_bound)

    return x

  def unpreprocess(x):
    """ Unpreprocess for input feature """
    x = x * torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(device)
    x = x + torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(device)
    return x

  def preprocess(x):
    """ Unpreprocess for input feature """
    x = x - torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(device)
    x = x / torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(device)
    return x

  clipped_x = None
  if clip_op == 'none':
    clipped_x = x
  elif clip_op == '+':
    clipped_x = positive_clip(x, x_bound)
  elif clip_op == '+-':
    clipped_x = positive_clip(x, x_bound) - positive_clip(-x, -x_bound)
  elif clip_op == 'up':
    clipped_x = preprocess(positive_clip(unpreprocess(x), unpreprocess(x_bound)))
  else:
    logging.error('Error in Clipping option.')
    clipped_x = None

  return clipped_x


def inv_adaptive_maxpool2d(y, x_bound, clip_op, device):
  """ Inverse torch.nn.AdaptiveMaxpool2d

  Args:
    y (torch.tensor): Target value.
    x_bound (torch.tensor): Upper bound for x.
    clip_op (str): Cliping option.
    device (str): Device for torch.

  Returns:
    Unmaxpooled torch.tensor.
  """
  # Calculate pool size to approcimate adaptive maxpool2d behavior.
  (_, _, out_h, out_w) = y.shape
  (_, _, in_h, in_w) = x_bound.shape
  pool_sz_h = np.ceil(in_h / out_h).astype(np.int32)
  pool_sz_w = np.ceil(in_w / out_w).astype(np.int32)
  pad_sz_h = np.ceil(0.5 * (pool_sz_h * out_h - in_h)).astype(np.int32)
  pad_sz_w = np.ceil(0.5 * (pool_sz_w * out_w - in_w)).astype(np.int32)

  # Generate max pool index with torch.nn.MaxPool2d.
  l_adap_max2d = torch.nn.AdaptiveMaxPool2d(
    output_size=(out_h, out_w),
    return_indices=True
    )

  _, x_pool_idx = l_adap_max2d(x_bound)

  # Define torch.nn.MaxUnpool2d.
  l_unpool = torch.nn.MaxUnpool2d(
    kernel_size=(pool_sz_h, pool_sz_w),
    stride=(pool_sz_h, pool_sz_w),
    padding=(pad_sz_h, pad_sz_w)
    )

  # Unmaxpool with torch.nn.MaxUnpool2d.
  x_unpool = l_unpool(y, x_pool_idx, output_size=(in_h, in_w))

  return x_unpool.float()


def inv_maxpool2d(y,
                  x_bound,
                  op,
                  output_size,
                  clip_op,
                  device,
                 ):
  """ Inverse torch.nn.MaxPool2d

  Args:
    y (torch.tensor): Target value.
    x_bound (torch.tensor): Upper bound for x.
    op (torch.nn.MaxPool2d): MaxPool2d layer applied to x_bound in forward pass.
    output_size (torch.Size): Argument for torch.nn.MaxUnpool2d.
    clip_op (str): Cliping option.
    device (str): Device for torch.

  Returns:
    Unmaxpooled torch.tensor.
  """
  # Generate max pool index with torch.nn.MaxPool2d.
  l_max2d = torch.nn.MaxPool2d(
    kernel_size=op.kernel_size,
    stride=op.stride,
    padding=op.padding,
    return_indices=True,
  )

  x_pool, x_pool_idx = l_max2d(x_bound)

  # Define torch.nn.MaxUnpool2d.
  l_unpool = torch.nn.MaxUnpool2d(
    kernel_size=op.kernel_size,
    stride=op.stride,
    padding=op.padding
    )

  # Unmaxpool with torch.nn.MaxUnpool2d.
  if x_pool.shape == y.shape:
    x_unpool = l_unpool(y, x_pool_idx, output_size=output_size)
  else:
    x_unpool = l_unpool(x_pool, x_pool_idx, output_size=output_size)

  return x_unpool.float()


def inv_relu(y, x_bound, clip_op, device):
  """ Inverse torch.nn.ReLU

  Args:
    y (torch.tensor): Target value.
    x_bound (torch.tensor): Upper bound for x.
    clip_op (str): Cliping option.
    device (str): Device for torch.

  Returns:
    UnReLUed torch.tensor.
  """
  # Calculate ReLU mask
  relu_mask = (x_bound > 0).float()
  x = torch.mul(relu_mask, y)

  return x.float()


def inv_fc_activations(
  y,
  x_bound,
  op,
  reg_weight,
  max_iter,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  device,
  log_interval,
  init_method,
  **kwargs,
):
  """ Inverse over-determined problem by solving optimization problem of form,

    argmin_x: |y - op(x)|_2 + regularization(x)
    subject to: clip_tensor(x),

  with Gradient Projection Algorithm (GPA).

  Args:
    y (torch.tensor): Target value.
    x_bound (torch.tensor): Upper bound for x.
    op (torch.nn.Conv2d or torch.nn.Linear): Opretaion that
      applied to x_bound in forward pass.
    reg_weight (float): Weight for regularization.
    max_iter (int): Maximum number of iteration.
    cost_seq_bufsz (int): Length of cost series.
    cost_seq_std_th (float): Threshold for termination.
      We terminate GPA if cost_seq_std_th < std of cost series.
    clip_op (str): Option for clipping x.
    device (str): Device for torch.
    log_interval: Polling interval for logging GPA process.

  Returns:
    Inversed value from solution of optimization problem.
  """
  # Define objective function for GPA.
  def objective_ftn(x_c):
   
    fc_f = F.relu(op(x_c))
    target_cost = torch.norm(y - fc_f)
    reg_cost = torch.norm(x_c)

    return target_cost + reg_weight * reg_cost

  # GPA
  x = gradient_projection_algorithm(
    x_bound=x_bound,
    objective_ftn=objective_ftn,
    max_iter=max_iter,
    cost_seq_bufsz=cost_seq_bufsz,
    cost_seq_std_th=cost_seq_std_th,
    clip_op=clip_op,
    device=device,
    log_interval=log_interval,
    init_method=init_method,
  )

  return x


def inv_fc_softmax(
  x_bound,
  op,
  target_idx,
  max_iter,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  device,
  log_interval,
  init_method,
  **kwargs,
):
  def objective_ftn(x_c):
    y = op(x_c)
    y_max = y[target_idx]
    y_remainder = torch.cat([y[0:target_idx], y[target_idx+1:]])

    return torch.max(y_remainder) - y_max

  x_inv = gradient_projection_algorithm(
    x_bound=x_bound,
    objective_ftn=objective_ftn,
    max_iter=max_iter,
    cost_seq_bufsz=cost_seq_bufsz,
    cost_seq_std_th=cost_seq_std_th,
    clip_op=clip_op,
    device=device,
    log_interval=log_interval,
    init_method=init_method,
  )

  return x_inv


def inv_conv_activations(
  y,
  x_bound,
  op,
  mask,
  mask_weight,
  max_iter,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  device,
  log_interval,
  init_method,
  **kwargs,
):
  """ Inverse over-determined problem by solving optimization problem of form,

    argmin_x: |y - op(x)|_2 + regularization(x)
    subject to: clip_tensor(x),

  with Gradient Projection Algorithm (GPA).

  Args:
    y (torch.tensor): Target value.
    x_bound (torch.tensor): Upper bound for x.
    op (torch.nn.Conv2d or torch.nn.Linear): Opretaion that
      applied to x_bound in forward pass.
    mask (torch.tensor): Mask for regularization.
    mask_weight (float): Weight for mask regularization.
    max_iter (int): Maximum number of iteration.
    cost_seq_bufsz (int): Length of cost series.
    cost_seq_std_th (float): Threshold for termination.
      We terminate GPA if cost_seq_std_th < std of cost series.
    clip_op (str): Option for clipping x.
    device (str): Device for torch.
    log_interval: Polling interval for logging GPA process.
    init_method: Method of initialize x.

  Returns:
    Inversed value from solution of optimization problem.
  """
  # Define objective function for GPA.
  def objective_ftn(x_c):
   
    conv_f = F.relu(op(x_c))
    target_cost = torch.norm(y - conv_f)
    reg_cost = torch.norm(mask * x_c)

    return target_cost + mask_weight * reg_cost

  x = gradient_projection_algorithm(
    x_bound=x_bound,
    objective_ftn=objective_ftn,
    max_iter=max_iter,
    cost_seq_bufsz=cost_seq_bufsz,
    cost_seq_std_th=cost_seq_std_th,
    clip_op=clip_op,
    device=device,
    log_interval=log_interval,
    init_method=init_method,
  )

  return x


def log_gpa(log_dict, delimeter):
  """ Log processes in GPA.

  Args:
    log_dict (dict): Dictionary with string and one argument.
      Keyword should be string and value can be 'float', 'int' and 'str'.
      Example: {'Num Iter:': 10, 'Step Size =': 0.11, '-----':'', ...}
    delimeter (char): Delimeter of each log string.
  """
  log_str = ''
  for key, value in log_dict.items():
    if isinstance(value, torch.Tensor):
      value = value.float().detach().cpu().numpy()
      log_str += '{} {:.6f}, {}'.format(key, value, delimeter)
    if isinstance(value, float):
      log_str += '{} {:.6f}, {}'.format(key, value, delimeter)
    if isinstance(value, int):
      log_str += '{} {:06d}, {}'.format(key, value, delimeter)
    if isinstance(value, str):
      log_str += '{} {} {}'.format(key, value, delimeter)
  log_str = log_str[:-1]

  logging.info(log_str)


def gradient_projection_algorithm(
  x_bound,
  objective_ftn,
  max_iter,
  cost_seq_bufsz,
  cost_seq_std_th,
  clip_op,
  device,
  log_interval,
  init_method='ones',
):
  """ Inverse torch.nn.Linear or torch.nn.Conv2d by solving
  optimization problem of form,

    argmin_x: |y - op(x)|_2 + regularization(x)
    subject to: clip_tensor(x).

  with Gradient Projection Algorithm (GPA).
  (Dimitri P. Bertsekas, Convex Optimization Algorithms p.322)

  We terminate if the standard deviation of series of cost (with length
  cost_seq_bufsz) is lesser than cost_seq_std_th.

  Args:
    x_bound (torch.tensor): Upper bound for x.
    objective_ftn (callable): Objective function for x.
    max_iter (int): Maximum number of iteration.
    cost_seq_bufsz (int): Length of cost series.
    cost_seq_std_th (float): Threshold for termination.
      We terminate GPA if cost_seq_std_th < std of cost series.
    clip_op (str): Option for clipping x.
    device (str): Device for torch.
    log_interval: Polling interval for logging GPA process.
    init_method: Method of initialize x.

  Returns:
    Solution of optimization problem.
  """
  # Time stamp for GPA.
  gpa_start_time = time.time()

  # Set initial value of x.
  if init_method == 'avgpool':
  # Initial value is given by averaging x with factor 5.
    if len(x_bound.shape) == 1:
      avgpool = torch.nn.AvgPool1d(5, stride=1, padding=2)
      x = avgpool(x_bound.unsqueeze(dim=0).unsqueeze(dim=0))
      x = x.squeeze()
    else:
      avgpool = torch.nn.AvgPool2d(5, stride=1, padding=2)
      x = avgpool(x_bound)
  elif init_method == 'ones':
    x = torch.ones_like(x_bound)
  elif init_method == 'zeros':
    x = torch.zeros_like(x_bound)
  elif init_method == 'rand':
    x = torch.randn(x_bound.shape)
  elif init_method == 'he':
    x = torch.nn.init.kaiming_normal_(torch.zeros_like(x_bound))
  else:
    logging.error('Error in initialization')
    return None
  x = x.to(device)

  # GPA step size (abbreviation of lambda (l)).
  l = 100

  # Previous value of x (xp).
  xp = x

  # Extrapolation beta for GPA (e_beta).
  e_beta = 0

  # Series of cost to determine termination
  cost_series = torch.tensor([]).to(device)

  # Threshold to terminate iteration
  cost_series_std = None

  for n in itertools.count():

    # Calculate gradient
    x.requires_grad = True
    f_x = objective_ftn(x)
    f_x.backward(retain_graph=True)
    df_dx = x.grad.clone().detach()

    # Approximate Lipshitz constant (L) with ddf_dx and use 1/(2L) as l.
    f_x.backward()
    ddf_dx = x.grad.clone().detach()

    l_criterion = 0.5 / (torch.max(torch.abs(ddf_dx)) + 1e-6)
    if n == 0:
      l = 0.5 * l_criterion
    else:
      if l > l_criterion:
        l = 0.5 * l_criterion

    ddf_dx = None

    with torch.no_grad():

      # Project x_(n+1)
      xl = clip_tensor(
        x=x + e_beta * (x - xp) - l * df_dx,
        x_bound=x_bound,
        clip_op=clip_op,
        device=device,
      )
      f_xl = objective_ftn(xl)

      # Compute extrapolation beta
      e_beta = (n - 1) / (n + 2)

      # Termination condition
      if n < cost_seq_bufsz: # Store cost values until cost_seq_bufsz
        cost_series = torch.cat([cost_series, f_xl.reshape([1])])
      else:
        # Terminate when reached max_iter
        if n > max_iter:
          logging.info('Iteration Failed')
          x = xl
          break
        # Terminate if cost_std lesser than thresold
        cost_series_std = cost_series.std()
        if cost_series_std < cost_seq_std_th: # method 3
          logging.info('Iteration Converged')
          x = xl
          break
        # If not terminate update cost_series
        cost_series = torch.cat([cost_series[1:], f_xl.reshape([1])])

      # Log iteration step
      if n == 0:
        log_gpa(log_dict={'Initial: Cost': f_xl}, delimeter='\n')
        log_interval_start_time = time.time()
      elif n % log_interval == 0:
        log_interval_time = time.time() - log_interval_start_time
        rate = 1 / log_interval_time
        log_gpa(
          log_dict={
            'Num Iter:': n,
            'Step Size =': l,
            'Cost =': f_xl,
            'Std of costs =': cost_series_std,
            'TH of Cost std =': cost_seq_std_th,
            'Rate =': rate
          },
          delimeter='',
          )
        log_interval_start_time = time.time()

      # Update x
      x.grad.zero_()
      xp = x
      x = xl

  # Log summary
  total_time = time.time() - gpa_start_time
  log_gpa(
    log_dict={
      '+--------------------------------': '',
      '\t\t    | Final Num Iter': n,
      '\t\t    | Total Time': total_time,
      '\t\t    | Final Step Size': l,
      '\t\t    | Final Cost': f_xl,
      '\t\t    | Final Std of costs': cost_series_std,
      '\t\t    | TH of Cost std': cost_seq_std_th,
      '\t\t    +--------------------------------.': '',
    },
    delimeter='\n',
  )

  return x