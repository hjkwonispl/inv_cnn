""" -*- coding: utf-8 -*-"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gdown

urls = [
  'https://drive.google.com/uc?id=1U_6NGTnuIoUTy76haVtMFXaIpkhEBOHj',#ImageNet
  'https://drive.google.com/uc?id=1E5H8t3Zlae9jlZMCYGdhXW32iC10hqBE',#Set5
  'https://drive.google.com/uc?id=1qcapaDljpKrlSRoMdlKeJRsgcPjQNj2-',#Set14
  'https://drive.google.com/uc?id=18MFak5L30uOsPMvJjyjW28XIcUGpSZOv',#BSDS100
  'https://drive.google.com/uc?id=1fE0xeTf-WVB1EggBa2KEsZuDdAfsAPD6',#Urban100
  'https://drive.google.com/uc?id=1L-zPx2cN89bCjyTSq74Y8Sn710TYTOX4',#Artificial
]
dataset_tgzs = [
  './data/imagenet2012.tgz',
  './data/Set5.tgz',
  './data/Set14.tgz',
  './data/BSDS100.tgz',
  './data/Urban100.tgz',
  './data/Artificial.tgz',
]

for url, dataset_tgz in zip(urls, dataset_tgzs):
  gdown.download(url, dataset_tgz, quiet=False)

# Extract tgz files
for dataset_tgz in dataset_tgzs:
  os.system('tar -xvf {} -C ./data'.format(dataset_tgz))