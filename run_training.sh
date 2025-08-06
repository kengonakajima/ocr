#!/bin/bash
# 警告を抑制してトレーニングを実行

source venv/bin/activate

# Protobuf警告を抑制
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# TensorFlow警告を抑制  
export TF_CPP_MIN_LOG_LEVEL=2

# 実行
python step5_train.py 2>/dev/null