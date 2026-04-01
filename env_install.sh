#!/bin/bash

rm ~/.condarc

echo "
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/r
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud
  pytorch: https://mirrors.ustc.edu.cn/anaconda/cloud
" > ~/.condarc

conda clean -i -y



conda create -y -p /opt/conda/envs/nnscaler_h3c python=3.10
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /opt/conda/envs/nnscaler_h3c
conda install -y cudatoolkit=11.8 -c nvidia
conda install -y pytorch==2.3.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pushd /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group2
pushd /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/nnscaler-h3c
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install -e .
export NNSCALER_HOME=$(pwd)
export PYTHONPATH=${NNSCALER_HOME}:$PYTHONPATH
popd
pip install importlib-resources
pip install transformers==4.47.0 tensorboard datasets==2.20.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install /data/haiqwa/zevin_nfs/andy/flash-attn/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/apex-24.4.1+cu118torch2.3.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# pip install /data/haiqwa/zevin_nfs/andy/flash-attn/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -i https://pypi.mirrors.ustc.edu.cn/simple/
# torch._C._GLIBCXX_USE_CXX11_ABI
conda install -y -c conda-forge gcc=12.1.0

mkdir ~/.config/clangd
cp /data/haiqwa/zevin_nfs/andy/Auto-Parallelization/config.yaml ~/.config/clangd/
export LD_LIBRARY_PATH=/opt/conda/envs/nnscaler_h3c/lib:$LD_LIBRARY_PATH