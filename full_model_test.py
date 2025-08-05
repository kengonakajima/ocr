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
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_obj = nn.Conv2d(128, 1, kernel_size=1)
        self.conv_box = nn.Conv2d(128, 4, kernel_size=1)
        
    def forward(self, features):
        x = self.conv1(features)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        objectness = torch.sigmoid(self.conv_obj(x))
        bbox = self.conv_box(x)
        return objectness, bbox

class RecognitionHead(nn.Module):
    """文字認識用のヘッド"""
    def __init__(self, in_channels=576, num_classes=62):
        super().__init__()
        # 62クラス = A-Z(26) + a-z(26) + 0-9(10)
        
        # 特徴をさらに圧縮
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # 各位置での文字クラス分類
        self.conv_cls = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, features):
        x = self.conv1(features)
        x = self.relu1(x)
        
        # 各位置での文字クラスの確率
        class_scores = self.conv_cls(x)
        return class_scores

class OCRModel(nn.Module):
    """完全なOCRモデル"""
    def __init__(self, num_classes=62):
        super().__init__()
        self.backbone = create_backbone()
        self.detection_head = DetectionHead(in_channels=576)
        self.recognition_head = RecognitionHead(in_channels=576, num_classes=num_classes)
        
    def forward(self, images):
        # バックボーンで特徴抽出
        features = self.backbone(images)
        
        # 検出ヘッド
        objectness, bbox = self.detection_head(features)
        
        # 認識ヘッド
        class_scores = self.recognition_head(features)
        
        return {
            'objectness': objectness,
            'bbox': bbox,
            'class_scores': class_scores,
            'features': features
        }

def test_full_model():
    """フルモデルの動作確認"""
    print("=== フルモデルテスト ===")
    
    # モデルを作成
    model = OCRModel(num_classes=62)
    model.eval()  # 評価モード
    
    # テスト入力
    test_input = torch.randn(1, 3, 480, 640)
    print(f"入力画像サイズ: {test_input.shape}")
    
    # Forward pass
    print("\n--- Forward Pass ---")
    with torch.no_grad():
        outputs = model(test_input)
    
    print("出力サイズ:")
    print(f"  特徴マップ: {outputs['features'].shape}")
    print(f"  文字存在確率: {outputs['objectness'].shape}")
    print(f"  バウンディングボックス: {outputs['bbox'].shape}")
    print(f"  文字クラススコア: {outputs['class_scores'].shape}")
    
    # 出力の解釈
    print("\n--- 出力の解釈 ---")
    print(f"各位置（15×20=300箇所）で:")
    print(f"  - 文字があるか: {outputs['objectness'].shape} → 各位置で0-1の確率")
    print(f"  - 文字の位置: {outputs['bbox'].shape} → 各位置で(x,y,w,h)")
    print(f"  - 文字の種類: {outputs['class_scores'].shape} → 各位置で62クラスの確率")
    
    # 文字クラスの確認
    print("\n--- 文字クラスの詳細 ---")
    print("0-25: A-Z")
    print("26-51: a-z")
    print("52-61: 0-9")
    
    # パラメータ数
    print("\n--- パラメータ数 ---")
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    detection_params = sum(p.numel() for p in model.detection_head.parameters())
    recognition_params = sum(p.numel() for p in model.recognition_head.parameters())
    
    print(f"バックボーン: {backbone_params:,}")
    print(f"検出ヘッド: {detection_params:,}")
    print(f"認識ヘッド: {recognition_params:,}")
    print(f"合計: {total_params:,}")
    
    # メモリ使用量
    import psutil
    process = psutil.Process()
    print(f"\nメモリ使用量: {process.memory_info().rss / 1024**3:.2f} GB")
    
    # バッチサイズ8での確認
    print("\n--- バッチサイズ8での確認 ---")
    batch_input = torch.randn(8, 3, 480, 640)
    with torch.no_grad():
        outputs = model(batch_input)
    
    print(f"バッチ入力: {batch_input.shape}")
    for key, value in outputs.items():
        if key != 'features':  # 特徴マップは省略
            print(f"{key}: {value.shape}")

if __name__ == "__main__":
    test_full_model()