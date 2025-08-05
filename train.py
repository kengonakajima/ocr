#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import OCRDataset, custom_collate_fn
from full_model_test import OCRModel
from full_loss_test import OCRLoss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"使用デバイス: {self.device}")
        
        # データセット準備
        self.prepare_data()
        
        # モデル、損失関数、最適化
        self.model = OCRModel().to(self.device)
        self.criterion = OCRLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # 学習率スケジューラ（コサインアニーリング）
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs']
        )
        
        # 記録用
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def prepare_data(self):
        """データセットの準備"""
        print("データセット準備中...")
        
        # 全データセット
        full_dataset = OCRDataset(
            data_dir='output', 
            start_idx=1, 
            end_idx=5000
        )
        
        # Train/Val/Test分割
        train_size = 4000
        val_size = 500
        test_size = 500
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # データローダー
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
    def train_epoch(self, epoch):
        """1エポックの学習"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # データ準備
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes']
            image_paths = batch['image_paths']
            
            # 学習ステップ
            self.optimizer.zero_grad()
            outputs = self.model(images)
            losses = self.criterion(outputs, bboxes, image_paths)
            total_loss = losses['total_loss']
            
            total_loss.backward()
            
            # 勾配クリッピング（安定化のため）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # 記録
            epoch_losses.append(total_loss.item())
            
            # 進捗表示
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{self.config['epochs']}] "
                      f"Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"(Obj: {losses['obj_loss'].item():.3f}, "
                      f"Box: {losses['bbox_loss'].item():.3f}, "
                      f"Cls: {losses['cls_loss'].item():.3f})")
        
        return sum(epoch_losses) / len(epoch_losses)
    
    def validate(self):
        """検証セットでの評価"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                bboxes = batch['bboxes']
                image_paths = batch['image_paths']
                
                outputs = self.model(images)
                losses = self.criterion(outputs, bboxes, image_paths)
                val_losses.append(losses['total_loss'].item())
        
        return sum(val_losses) / len(val_losses)
    
    def save_checkpoint(self, epoch, val_loss):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 保存ディレクトリ
        os.makedirs('checkpoints', exist_ok=True)
        
        # 最新のチェックポイント
        torch.save(checkpoint, 'checkpoints/latest.pth')
        
        # ベストモデル
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, 'checkpoints/best.pth')
            print(f"✓ ベストモデル更新！ Val Loss: {val_loss:.4f}")
    
    def plot_losses(self):
        """損失のグラフ化"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.close()
    
    def train(self):
        """学習のメインループ"""
        print(f"\n=== 学習開始 ===")
        print(f"エポック数: {self.config['epochs']}")
        print(f"バッチサイズ: {self.config['batch_size']}")
        print(f"学習率: {self.config['lr']}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # 学習
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 検証
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 学習率更新
            self.scheduler.step()
            
            # 時間計測
            epoch_time = time.time() - start_time
            
            # ログ出力
            print(f"\nEpoch [{epoch}/{self.config['epochs']}] "
                  f"Time: {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # チェックポイント保存
            self.save_checkpoint(epoch, val_loss)
            
            # グラフ更新（5エポックごと）
            if epoch % 5 == 0:
                self.plot_losses()
        
        print("\n=== 学習完了 ===")
        print(f"最終Train Loss: {self.train_losses[-1]:.4f}")
        print(f"最終Val Loss: {self.val_losses[-1]:.4f}")
        print(f"ベストVal Loss: {self.best_val_loss:.4f}")

def main():
    # 設定
    config = {
        'epochs': 10,  # テスト用に少なめ
        'batch_size': 8,
        'lr': 0.001
    }
    
    # 学習実行
    trainer = Trainer(config)
    trainer.train()
    
    # 結果保存
    results = {
        'config': config,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n結果を training_results.json に保存しました")

if __name__ == "__main__":
    main()