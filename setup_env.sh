#!/usr/bin/env bash
# Sets up the conda environment for LongCat-Video.
# Requires a machine with conda and an NVIDIA GPU (CUDA 12.4 by default;
# adjust the torch --index-url below to match your CUDA version).
set -euo pipefail

# create conda environment
conda create -n longcat-video python=3.10
conda activate longcat-video

# install torch (configure according to your CUDA version)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# install flash-attn-2
pip install ninja
pip install psutil
pip install packaging
pip install flash_attn==2.7.4.post1

# install other requirements
pip install -r requirements.txt

# install longcat-video-avatar requirements
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
pip install -r requirements_avatar.txt
