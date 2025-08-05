#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models

def create_backbone():
    """MobileNetV3-Smallバックボーンを作成"""
    # 事前学習済みモデルを読み込み
    mobilenet = models.mobilenet_v3_small(pretrained=True)
    
    # 分類層を除いたバックボーン部分のみ取得
    # features部分には畳み込み層が含まれる
    backbone = mobilenet.features
    
    return backbone

def test_backbone():
    """バックボーンの動作確認"""
    print("=== MobileNetV3-Small バックボーンテスト ===")
    
    # バックボーンを作成
    backbone = create_backbone()
    
    # モデル構造を簡単に確認
    print("\n--- モデル構造（最初と最後の層）---")
    layers = list(backbone.children())
    print(f"最初の層: {layers[0]}")
    print(f"...")
    print(f"最後の層: {layers[-1]}")
    print(f"総層数: {len(layers)}")
    
    # テスト入力を作成（バッチサイズ1, チャンネル3, 高さ480, 幅640）
    test_input = torch.randn(1, 3, 480, 640)
    print(f"\n入力サイズ: {test_input.shape}")
    
    # forward passを実行
    with torch.no_grad():
        output = backbone(test_input)
    
    print(f"出力サイズ: {output.shape}")
    
    # 特徴マップの統計情報
    print(f"\n--- 出力特徴マップの統計 ---")
    print(f"最小値: {output.min():.3f}")
    print(f"最大値: {output.max():.3f}")
    print(f"平均値: {output.mean():.3f}")
    
    # パラメータ数を計算
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\n--- パラメータ数 ---")
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    
    # メモリ使用量
    import psutil
    process = psutil.Process()
    print(f"\nメモリ使用量: {process.memory_info().rss / 1024**3:.2f} GB")
    
    # 異なる解像度での出力サイズを確認
    print("\n--- 異なる入力サイズでのテスト ---")
    test_sizes = [(224, 224), (320, 320), (480, 640)]
    for h, w in test_sizes:
        test_input = torch.randn(1, 3, h, w)
        with torch.no_grad():
            output = backbone(test_input)
        print(f"入力: {test_input.shape} -> 出力: {output.shape}")
    
    return backbone

if __name__ == "__main__":
    backbone = test_backbone()