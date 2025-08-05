#!/usr/bin/env python3
import torch
import torchvision
import numpy as np

print("=== PyTorch環境確認 ===")
print(f"PyTorchバージョン: {torch.__version__}")
print(f"TorchVisionバージョン: {torchvision.__version__}")
print(f"NumPyバージョン: {np.__version__}")

print("\n=== デバイス確認 ===")
if torch.cuda.is_available():
    print("CUDA利用可能")
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
elif torch.backends.mps.is_available():
    print("Apple Metal Performance Shaders (MPS) 利用可能")
    device = torch.device("mps")
    print("MacのGPUアクセラレーションが使用できます")
else:
    print("CPU使用")

print("\n=== 簡単な動作確認 ===")
x = torch.randn(2, 3)
print(f"テンソルサイズ: {x.shape}")
print(f"テンソルの内容:\n{x}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x_gpu = x.to(device)
    print(f"\nMPSデバイス上のテンソル: {x_gpu.device}")
    
print("\n=== メモリ使用量 ===")
import psutil
process = psutil.Process()
print(f"現在のメモリ使用量: {process.memory_info().rss / 1024**3:.2f} GB")