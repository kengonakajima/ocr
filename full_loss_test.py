#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import OCRDataset, DataLoader, custom_collate_fn
from full_model_test import OCRModel
import string

class OCRLoss(nn.Module):
    """OCR用の統合損失関数（検出＋認識）"""
    def __init__(self, num_classes=62):
        super().__init__()
        # 検出用
        self.bce_loss = nn.BCELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        # 認識用
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # 文字クラスのマッピング
        self.char_to_class = {}
        # A-Z: 0-25
        for i, char in enumerate(string.ascii_uppercase):
            self.char_to_class[char] = i
        # a-z: 26-51
        for i, char in enumerate(string.ascii_lowercase):
            self.char_to_class[char] = i + 26
        # 0-9: 52-61
        for i, char in enumerate(string.digits):
            self.char_to_class[char] = i + 52
            
    def forward(self, predictions, targets, image_paths):
        """
        predictions: モデルの出力
        targets: バウンディングボックスのリスト
        image_paths: 画像ファイルパス（文字情報を取得するため）
        """
        objectness_pred = predictions['objectness']
        bbox_pred = predictions['bbox']
        class_scores_pred = predictions['class_scores']
        
        batch_size, _, height, width = objectness_pred.shape
        device = objectness_pred.device
        
        # ターゲットマップを作成
        objectness_target = torch.zeros_like(objectness_pred)
        bbox_target = torch.zeros_like(bbox_pred)
        class_target = torch.zeros(batch_size, height, width, dtype=torch.long, device=device)
        
        total_obj_loss = 0
        total_bbox_loss = 0
        total_cls_loss = 0
        num_positive = 0
        
        # 各画像について処理
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
            
            # 画像パスから文字情報を推測（ランダム生成なので実際の文字は不明）
            # 今回はダミーでランダムな文字を割り当て
            num_chars = len(targets[b])
            
            # 各GTボックスについて
            for idx, gt_box in enumerate(targets[b]):
                # GT box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = gt_box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                # 特徴マップ上の位置に変換
                fx = int(cx * width / 640)
                fy = int(cy * height / 480)
                
                # 範囲チェック
                if 0 <= fx < width and 0 <= fy < height:
                    # 検出ターゲット
                    objectness_target[b, 0, fy, fx] = 1.0
                    bbox_target[b, 0, fy, fx] = cx
                    bbox_target[b, 1, fy, fx] = cy
                    bbox_target[b, 2, fy, fx] = w
                    bbox_target[b, 3, fy, fx] = h
                    
                    # 認識ターゲット（ダミー：ランダムなクラス）
                    # 実際の実装では、画像生成時の文字情報を使う必要がある
                    dummy_class = torch.randint(0, 62, (1,)).item()
                    class_target[b, fy, fx] = dummy_class
                    
                    num_positive += 1
        
        # 1. Objectness loss
        obj_loss = self.bce_loss(objectness_pred, objectness_target)
        obj_loss = obj_loss.mean()
        
        # 2. Bounding box loss（正解位置のみ）
        if num_positive > 0:
            pos_mask = objectness_target > 0.5
            pos_mask_bbox = pos_mask.expand_as(bbox_pred)
            
            bbox_loss = self.smooth_l1_loss(
                bbox_pred[pos_mask_bbox], 
                bbox_target[pos_mask_bbox]
            )
            bbox_loss = bbox_loss.mean()
        else:
            bbox_loss = torch.tensor(0.0, device=device)
        
        # 3. Classification loss（正解位置のみ）
        if num_positive > 0:
            # (B, C, H, W) -> (B*H*W, C)
            class_scores_flat = class_scores_pred.permute(0, 2, 3, 1).reshape(-1, 62)
            class_target_flat = class_target.reshape(-1)
            
            # 正解位置のマスク
            pos_mask_flat = (objectness_target > 0.5).squeeze(1).reshape(-1)
            
            if pos_mask_flat.sum() > 0:
                cls_loss = self.ce_loss(
                    class_scores_flat[pos_mask_flat],
                    class_target_flat[pos_mask_flat]
                )
                cls_loss = cls_loss.mean()
            else:
                cls_loss = torch.tensor(0.0, device=device)
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # 合計損失
        total_loss = obj_loss + bbox_loss + cls_loss
        
        return {
            'total_loss': total_loss,
            'obj_loss': obj_loss,
            'bbox_loss': bbox_loss,
            'cls_loss': cls_loss,
            'num_positive': num_positive
        }

def test_full_loss():
    """統合損失の動作確認"""
    print("=== 統合損失テスト（検出＋認識）===")
    
    # データセットとローダー
    dataset = OCRDataset(data_dir='output', start_idx=1, end_idx=10)
    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # モデルと損失関数
    model = OCRModel()
    loss_fn = OCRLoss()
    
    # 1バッチでテスト
    batch = next(iter(dataloader))
    images = batch['images']
    bboxes = batch['bboxes']
    image_paths = batch['image_paths']
    
    print(f"入力画像: {images.shape}")
    print(f"バウンディングボックス数: {[len(bb) for bb in bboxes]}")
    
    # Forward pass
    outputs = model(images)
    
    # 損失計算
    losses = loss_fn(outputs, bboxes, image_paths)
    
    print("\n--- 損失値 ---")
    print(f"合計損失: {losses['total_loss']:.4f}")
    print(f"検出損失:")
    print(f"  - Objectness: {losses['obj_loss']:.4f}")
    print(f"  - BBox回帰: {losses['bbox_loss']:.4f}")
    print(f"認識損失:")
    print(f"  - 分類: {losses['cls_loss']:.4f}")
    print(f"正解位置数: {losses['num_positive']}")
    
    # 期待される値の確認
    print("\n--- 期待値との比較 ---")
    print("学習前なので:")
    print("- Objectness損失 ≈ 0.69")
    print("- 分類損失 ≈ 4.13 (log(62)、62クラスのランダム予測)")
    
    # 実際に計算
    import math
    print(f"\n実際の期待値:")
    print(f"- log(2) = {math.log(2):.3f}")
    print(f"- log(62) = {math.log(62):.3f}")
    
    # 勾配確認
    print("\n--- 勾配確認 ---")
    model.zero_grad()
    total_loss = losses['total_loss']
    total_loss.backward()
    
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                if 'backbone' in name:
                    grad_info['backbone'] = True
                elif 'detection' in name:
                    grad_info['detection'] = True
                elif 'recognition' in name:
                    grad_info['recognition'] = True
    
    print(f"勾配が流れた部分: {list(grad_info.keys())}")

if __name__ == "__main__":
    test_full_loss()