# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def torch_to_np(tensor):
  """ Return torch tensor as numpy.

  Args:
    tensor (torch.tensor): Input tensor.

  Returns:
    Numpy array of given tensor.
  """

  return tensor.clone().detach().cpu().numpy()


def torch_to_image(tensor):
  """ Return torch tensor as numpy for image.

  Args:
    tensor (torch.tensor): Input tensor.

  Returns:
    Numpy array of given tensor in shape [W, H, 3]
  """
  return torch_to_np(tensor.squeeze().transpose(0,2).transpose(0,1))