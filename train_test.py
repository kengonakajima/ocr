#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import OCRDataset, custom_collate_fn
from full_model_test import OCRModel
from full_loss_test import OCRLoss

def test_training_with_labels():
    """文字ラベル付きデータで学習テスト"""
    print("=== 文字ラベル付きデータで学習テスト ===")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 少量のデータでテスト
    dataset = OCRDataset(data_dir='output', start_idx=1, end_idx=100)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # モデルと最適化
    model = OCRModel().to(device)
    criterion = OCRLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5バッチだけ学習
    model.train()
    print("\nバッチごとの損失:")
    print("バッチ | 合計損失 | Obj損失 | Box損失 | Cls損失")
    print("-" * 50)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break
            
        # データ準備
        images = batch['images'].to(device)
        bboxes = batch['bboxes']
        chars = batch['chars']
        
        # 学習ステップ
        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, bboxes, chars)
        losses['total_loss'].backward()
        optimizer.step()
        
        # 結果表示
        print(f"{batch_idx+1:^6} | {losses['total_loss'].item():^9.4f} | "
              f"{losses['obj_loss'].item():^8.4f} | {losses['bbox_loss'].item():^8.4f} | "
              f"{losses['cls_loss'].item():^8.4f}")
    
    print("\n期待される動き:")
    print("- Cls損失が4.13付近から始まり、徐々に減少する")
    print("- これは実際の文字ラベルを学習している証拠")

if __name__ == "__main__":
    test_training_with_labels()