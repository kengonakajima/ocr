#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import psutil
from data_loader import OCRDataset, custom_collate_fn
from full_model_test import OCRModel
from full_loss_test import OCRLoss

def train_one_epoch():
    """1エポックだけの学習"""
    print("=== 1エポック学習テスト ===")
    
    # デバイス設定
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データセット（少量でテスト）
    print("\nデータセット準備中...")
    train_dataset = OCRDataset(data_dir='output', start_idx=1, end_idx=100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # モデルと最適化
    print("モデル準備中...")
    model = OCRModel().to(device)
    criterion = OCRLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # メモリ使用量（初期）
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"初期メモリ使用量: {initial_memory:.2f} GB")
    
    # 学習ループ
    model.train()
    epoch_losses = []
    batch_times = []
    
    print("\n学習開始...")
    print("-" * 60)
    
    for batch_idx, batch in enumerate(train_loader):
        start_time = time.time()
        
        # データをデバイスに移動
        images = batch['images'].to(device)
        bboxes = batch['bboxes']
        image_paths = batch['image_paths']
        
        # 勾配リセット
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # 損失計算
        losses = criterion(outputs, bboxes, image_paths)
        total_loss = losses['total_loss']
        
        # Backward
        total_loss.backward()
        
        # パラメータ更新
        optimizer.step()
        
        # 統計
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        epoch_losses.append(total_loss.item())
        
        # 進捗表示（5バッチごと）
        if batch_idx % 5 == 0:
            current_memory = process.memory_info().rss / 1024**3
            print(f"バッチ [{batch_idx+1}/{len(train_loader)}] "
                  f"損失: {total_loss.item():.4f} "
                  f"(Obj: {losses['obj_loss'].item():.3f}, "
                  f"Box: {losses['bbox_loss'].item():.3f}, "
                  f"Cls: {losses['cls_loss'].item():.3f}) "
                  f"時間: {batch_time:.2f}秒 "
                  f"メモリ: {current_memory:.2f}GB")
    
    print("-" * 60)
    
    # エポック統計
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_time = sum(batch_times) / len(batch_times)
    max_memory = process.memory_info().rss / 1024**3
    
    print(f"\n=== エポック完了 ===")
    print(f"平均損失: {avg_loss:.4f}")
    print(f"平均バッチ時間: {avg_time:.2f}秒")
    print(f"最大メモリ使用量: {max_memory:.2f} GB")
    
    # 損失の推移を確認
    print(f"\n損失の推移:")
    print(f"最初の5バッチ平均: {sum(epoch_losses[:5])/5:.4f}")
    print(f"最後の5バッチ平均: {sum(epoch_losses[-5:])/5:.4f}")
    
    # 損失が減少したか確認
    if epoch_losses[-1] < epoch_losses[0]:
        print("✓ 損失が減少しました！")
        print(f"  初期: {epoch_losses[0]:.4f} → 最終: {epoch_losses[-1]:.4f}")
    else:
        print("✗ 損失が増加しています")
    
    # パラメータの更新を確認
    print("\n=== パラメータ更新の確認 ===")
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
    
    print(f"総勾配ノルム: {total_grad_norm:.4f}")
    
    return model, epoch_losses

if __name__ == "__main__":
    model, losses = train_one_epoch()