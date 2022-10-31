#!/bin/bash

echo "[install.sh] Make directories for data, log, and output"
mkdir -p data log output

echo "[install.sh] Create inv_cnn environment"
conda create -n inv_cnn python=3.6 -y

echo "[install.sh] Activate inv_cnn environment"
source activate inv_cnn

echo "[install.sh] Install inv_cnn packages"
pip install -r requirements.txt

source deactivate