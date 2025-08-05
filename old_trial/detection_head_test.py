#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models

def create_backbone():
    """MobileNetV3-Smallバックボーンを作成"""
    mobilenet = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    backbone = mobilenet.features
    return backbone

class DetectionHead(nn.Module):
    """文字検出用のヘッド"""
    def __init__(self, in_channels=576):
        super().__init__()
        
        # 特徴マップから文字の存在確率と位置を予測
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # 出力チャンネル
        # - 1ch: 文字存在確率（objectness）
        # - 4ch: バウンディングボックス回帰（x, y, w, h）
        self.conv_obj = nn.Conv2d(128, 1, kernel_size=1)  # 文字か背景か
        self.conv_box = nn.Conv2d(128, 4, kernel_size=1)  # bbox座標
        
    def forward(self, features):
        x = self.conv1(features)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        # 文字存在確率
        objectness = self.conv_obj(x)
        objectness = torch.sigmoid(objectness)  # 0-1の確率に
        
        # バウンディングボックス
        bbox = self.conv_box(x)
        
        return objectness, bbox

def test_detection_head():
    """検出ヘッドの動作確認"""
    print("=== 検出ヘッドテスト ===")
    
    # バックボーンと検出ヘッドを作成
    backbone = create_backbone()
    detection_head = DetectionHead(in_channels=576)
    
    # テスト入力
    test_input = torch.randn(1, 3, 480, 640)
    print(f"入力画像サイズ: {test_input.shape}")
    
    # バックボーンを通す
    with torch.no_grad():
        features = backbone(test_input)
        print(f"バックボーン出力: {features.shape}")
        
        # 検出ヘッドを通す
        objectness, bbox = detection_head(features)
        print(f"\n検出ヘッド出力:")
        print(f"  文字存在確率マップ: {objectness.shape}")
        print(f"  バウンディングボックス: {bbox.shape}")
    
    # 出力の解釈
    print("\n--- 出力の解釈 ---")
    print(f"各ピクセルでの予測:")
    print(f"  - 15×20 = 300個の位置")
    print(f"  - 各位置で「文字があるか」の確率（0-1）")
    print(f"  - 各位置でのbbox予測（x, y, w, h）")
    
    # 統計情報
    print(f"\n--- 統計情報 ---")
    print(f"文字存在確率:")
    print(f"  最小: {objectness.min():.3f}")
    print(f"  最大: {objectness.max():.3f}")
    print(f"  平均: {objectness.mean():.3f}")
    
    # パラメータ数
    total_params = sum(p.numel() for p in detection_head.parameters())
    print(f"\n検出ヘッドのパラメータ数: {total_params:,}")
    
    # メモリ使用量
    import psutil
    process = psutil.Process()
    print(f"メモリ使用量: {process.memory_info().rss / 1024**3:.2f} GB")
    
    # バッチサイズ8での確認
    print("\n--- バッチサイズ8での確認 ---")
    batch_input = torch.randn(8, 3, 480, 640)
    with torch.no_grad():
        features = backbone(batch_input)
        objectness, bbox = detection_head(features)
    print(f"バッチ入力: {batch_input.shape}")
    print(f"文字存在確率: {objectness.shape}")
    print(f"バウンディングボックス: {bbox.shape}")

if __name__ == "__main__":
    test_detection_head()