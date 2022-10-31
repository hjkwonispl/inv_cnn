# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torchvision

import models

class CascadedModels:
  """ Get cascade type torchvision.models and modify it into layerwised forms.
  Supported models are
    vgg11, vgg11bn, vgg16, vgg16bn, vgg19, vgg19bn, VDSR

  Attributes:
    __model_name (str): Name of torchvision model
    __device (str): Device for torch
      torchvision.models with modification on AdaptiveAvgPool to 
      AdaptiveMaxPool2d

  Properties:
    model_name (str): self.__model_name
    device (str): self.__device
  """
  def __init__(self, model_name, device='cuda'):
    """ Initialization

    Args:
      model_name (str): Name of torchvision model
      device (str): Device for torch
    """
    self.__model_name = model_name
    self.__device = device

    logging.info('Model Name = {:s}'.format(model_name))

  def pretrained_model(self):
    """ Get pretrained model from torchvision.models and
    replace AdaptiveAvgPool -> AdaptiveMaxPool2d to inverse pooling w.r.t
    unmaxpool operation

    Returns:
      __pretrained_model (torchvision.models): Pretrained model from
        torchvision.models with modification on
        AdaptiveAvgPool -> AdaptiveMaxPool2d
    """
    # Load pre-trained models in torchvision.models
    if self.__model_name == 'VGG11' or self.__model_name == 'vgg11':
      pretrained_model = torchvision.models.vgg11(pretrained=True)
    if self.__model_name == 'VGG11BN' or self.__model_name == 'vgg11bn':
      pretrained_model = torchvision.models.vgg11_bn(pretrained=True)
    if self.__model_name == 'VGG16' or self.__model_name == 'vgg16':
      pretrained_model = torchvision.models.vgg16(pretrained=True)
    if self.__model_name == 'VGG16BN' or self.__model_name == 'vgg16bn':
      pretrained_model = torchvision.models.vgg16_bn(pretrained=True)
    if self.__model_name == 'VGG19' or self.__model_name == 'vgg19':
      pretrained_model = torchvision.models.vgg19(pretrained=True)
    if self.__model_name == 'VGG19BN' or self.__model_name == 'vgg19bn':
      pretrained_model = torchvision.models.vgg19_bn(pretrained=True)
    if self.__model_name == 'VDSR' or self.__model_name == 'vdsr':
      pretrained_model = models.VDSR()
      pretrained_model.load_state_dict(
        torch.load('./models/vdsr.pth')
      )

    # Let model in evaluation mode and proper device
    pretrained_model = pretrained_model.to(self.__device).eval()

    # Replace AdaptiveAvgPool2d with AdaptiveMaxPool2d
    if hasattr(pretrained_model, 'avgpool'):
      avgpool_output_size = pretrained_model.avgpool.output_size
      pretrained_model.avgpool = torch.nn.AdaptiveMaxPool2d(
        output_size=avgpool_output_size).to(self.__device).eval()

    return pretrained_model

  def layerwise_model(self):
    """ Return layerwised version of torchvision.models.

    Returns:
      A list of dict.
      For example,
      [
        {'layer_type':'Linear', 'layer': torch.nn.Linear},
        {'layer_type':'Conv2d', 'layer': torch.nn.Conv2d},
        {'layer_type':'ReLU', 'layer': torch.nn.ReLU},
        {'layer_type':'MaxPool2d', 'layer': torch.nn.MaxPool2d},
        ...
      ]
    """
    layerwise_model = []
    pretrained_model = self.pretrained_model()
    pretrained_modules = [module for module in pretrained_model.modules()]
    idx = 0

    for op in pretrained_modules:

      op_type = type(op).__name__

      if op_type in ['VGG', 'Sequential', 'Identity', 'VDSR', 'Conv_ReLU_Block']:
        continue

      layer_dict = {
        'type': op_type,
        'idx': idx,
        'op': op.eval(),
        'in_f': None,
        'out_f': None,
      }
      layerwise_model.append(layer_dict)
      idx += 1

      # Add flatten layer before FC.
      if op_type == 'AdaptiveMaxPool2d':
        layer_dict = {
          'type': 'Flatten',
          'idx': idx,
          'op': torch.nn.Flatten(),
          'in_f': None,
          'out_f': None,
        }
        layerwise_model.append(layer_dict)
        idx += 1

    return layerwise_model

  def layerwise_model_with_classifier(self):
    """ Return layerwised version of torchvision.models. which has classifier
      part. (e.g. VGG)
      Currently, only works for VGG16.

    Returns:
      A list of dict.
      For example,
      [
        {'layer_type':'Linear', 'layer': torch.nn.Linear},
        {'layer_type':'Conv2d', 'layer': torch.nn.Conv2d},
        {'layer_type':'ReLU', 'layer': torch.nn.ReLU},
        {'layer_type':'MaxPool2d', 'layer': torch.nn.MaxPool2d},
        ...
        {'layer_type':'Classifier', 'layer': vgg16.classifier},
      ]
    """
    layerwise_model = []
    pretrained_model = self.pretrained_model()
    pretrained_model_features = pretrained_model.features
    pretrained_model_features.add_module('31', pretrained_model.avgpool)
    pretrained_model_classifier = pretrained_model.classifier

    pretrained_modules = [module for module in pretrained_model_features]
    idx = 0

    for op in pretrained_modules:

      op_type = type(op).__name__
      layer_dict = {
        'type': op_type,
        'idx': idx,
        'op': op.eval(),
        'in_f': None,
        'out_f': None,
      }
      layerwise_model.append(layer_dict)
      idx += 1

      # Add flatten layer before FC.
      if op_type == 'AdaptiveMaxPool2d':
        layer_dict = {
          'type': 'Flatten',
          'idx': idx,
          'op': torch.nn.Flatten(),
          'in_f': None,
          'out_f': None,
        }
        layerwise_model.append(layer_dict)
        idx += 1
        
    classifier_dict = {
      'type': 'Classifier',
      'idx': idx,
      'op': pretrained_model_classifier,
      'in_f': None,
      'out_f': None,
    }

    layerwise_model.append(classifier_dict)

    return layerwise_model

  def forward_layerwise_model(self, in_f, layerwise_model, device):
    """ Forward layerwise_model w.r.t. given inputs.

    Args:
      in_f (torch.tensor): Input feature of network.
      layerwise_model (list): List of dictionary of CNN model.
      device (str): Device for torch.

    Returns:
      Inference result from layerwise model.
    """
    # Forward given batch before classifier
    x = in_f.to(device)

    # Store forward pass information
    with torch.no_grad():
      for layer_dict in layerwise_model:
        op = layer_dict['op'].to(device)
        layer_dict['in_f'] = x.clone().detach().to(device)
        x = op(x)
        layer_dict['out_f'] = x.clone().detach().to(device)

    return x

  def empty_io(self, layerwise_model):
    """ Empty layerwise_model w.r.t. given inputs.

    Args:
      layerwise_model (list): List of dictionary of CNN model.

    Returns:
      Inference result from layerwise model.
    """
    # Store forward pass information
    with torch.no_grad():
      for layer_dict in layerwise_model:
        layer_dict['in_f'] = None
        layer_dict['out_f'] = None

  @property
  def model_name(self):
    return self.__model_name

  @property
  def device(self):
    return self.__device

  @device.setter
  def device(self, device):
    self.__device = device