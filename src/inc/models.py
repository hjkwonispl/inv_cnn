# -*- coding: utf-8 -*-
import torch.nn as nn

class Conv_ReLU_Block(nn.Module):
  """ Conv and ReLU block for the VDSR implementation """  
  def __init__(self):
    super(Conv_ReLU_Block, self).__init__()
    self.conv = nn.Conv2d(
      in_channels=64,
      out_channels=64,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.conv(x))


class VDSR(nn.Module):
  """ VDSR implementation
  [CAUTION]: Input image is not added the output. Outputs are only residuals.
  """  
  def __init__(self):
    super(VDSR, self).__init__()
    self.input = nn.Conv2d(
      in_channels=1,
      out_channels=64,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )
    self.relu = nn.ReLU(inplace=True)
    self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
    self.output = nn.Conv2d(
      in_channels=64,
      out_channels=1,
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False
    )

  def make_layer(self, block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
      layers.append(block())
    return nn.Sequential(*layers)

  def forward(self, x):
    residual = x
    out = self.relu(self.input(x))
    out = self.residual_layer(out)
    out = self.output(out)
    return out