#!/bin/bash
cd ~/multi-client-l2h
source ~/miniconda3/etc/profile.d/conda.sh
conda activate l2h-b
CUDA_VISIBLE_DEVICES=0 python experiments/test_lora_value.py
