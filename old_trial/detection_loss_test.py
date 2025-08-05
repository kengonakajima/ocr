#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import OCRDataset, DataLoader, custom_collate_fn
from full_model_test import OCRModel

class DetectionLoss(nn.Module):
    """文字検出用の損失関数"""
    def __init__(self):
        super().__init__()
        # Binary Cross Entropy for objectness
        self.bce_loss = nn.BCELoss(reduction='none')
        # Smooth L1 Loss for bounding box regression
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, predictions, targets):
        """
        predictions: モデルの出力
            - objectness: (B, 1, H, W) 文字存在確率
            - bbox: (B, 4, H, W) バウンディングボックス
        targets: 正解データ
            - bboxes: 各画像のバウンディングボックスのリスト
        """
        objectness_pred = predictions['objectness']
        bbox_pred = predictions['bbox']
        
        batch_size, _, height, width = objectness_pred.shape
        device = objectness_pred.device
        
        # ターゲットマップを作成
        objectness_target = torch.zeros_like(objectness_pred)
        bbox_target = torch.zeros_like(bbox_pred)
        
        total_obj_loss = 0
        total_bbox_loss = 0
        num_positive = 0
        
        # 各画像について処理
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
                
            # 各GTボックスについて
            for gt_box in targets[b]:
                # GT box: [x1, y1, x2, y2]
                x1, y1, x2, y2 = gt_box
                cx = (x1 + x2) / 2  # 中心x
                cy = (y1 + y2) / 2  # 中心y
                w = x2 - x1  # 幅
                h = y2 - y1  # 高さ
                
                # 特徴マップ上の位置に変換（640x480 -> 20x15）
                fx = int(cx * width / 640)
                fy = int(cy * height / 480)
                
                # 範囲チェック
                if 0 <= fx < width and 0 <= fy < height:
                    # その位置を正解（文字あり）とする
                    objectness_target[b, 0, fy, fx] = 1.0
                    
                    # バウンディングボックスの正解
                    bbox_target[b, 0, fy, fx] = cx  # x中心
                    bbox_target[b, 1, fy, fx] = cy  # y中心
                    bbox_target[b, 2, fy, fx] = w   # 幅
                    bbox_target[b, 3, fy, fx] = h   # 高さ
                    
                    num_positive += 1
        
        # Objectness loss (全ピクセル)
        obj_loss = self.bce_loss(objectness_pred, objectness_target)
        obj_loss = obj_loss.mean()
        
        # Bounding box loss (正解位置のみ)
        if num_positive > 0:
            pos_mask = objectness_target > 0.5
            pos_mask = pos_mask.expand_as(bbox_pred)
            
            bbox_loss = self.smooth_l1_loss(
                bbox_pred[pos_mask], 
                bbox_target[pos_mask]
            )
            bbox_loss = bbox_loss.mean()
        else:
            bbox_loss = torch.tensor(0.0, device=device)
        
        # 合計損失
        total_loss = obj_loss + bbox_loss
        
        return {
            'total_loss': total_loss,
            'obj_loss': obj_loss,
            'bbox_loss': bbox_loss,
            'num_positive': num_positive
        }

def test_detection_loss():
    """検出損失の動作確認"""
    print("=== 検出損失テスト ===")
    
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
    loss_fn = DetectionLoss()
    
    # 1バッチでテスト
    batch = next(iter(dataloader))
    images = batch['images']
    bboxes = batch['bboxes']
    
    print(f"入力画像: {images.shape}")
    print(f"バウンディングボックス数: {[len(bb) for bb in bboxes]}")
    
    # Forward pass（勾配計算のためno_gradを外す）
    outputs = model(images)
    
    # 損失計算
    losses = loss_fn(outputs, bboxes)
    
    print("\n--- 損失値 ---")
    print(f"合計損失: {losses['total_loss']:.4f}")
    print(f"Objectness損失: {losses['obj_loss']:.4f}")
    print(f"BBox損失: {losses['bbox_loss']:.4f}")
    print(f"正解位置数: {losses['num_positive']}")
    
    # 期待される値の確認
    print("\n--- 期待値との比較 ---")
    print("学習前なので:")
    print("- Objectness損失 ≈ 0.69 (log(2)、ランダムな予測)")
    print("- BBox損失: 正解位置での回帰誤差")
    
    # 勾配が流れるか確認
    print("\n--- 勾配確認 ---")
    model.zero_grad()
    total_loss = losses['total_loss']
    total_loss.backward()
    
    # 勾配が計算されたか確認
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().max() > 0:
            has_grad = True
            break
    
    print(f"勾配が計算された: {has_grad}")

if __name__ == "__main__":
    test_detection_loss()