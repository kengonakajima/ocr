#!/usr/bin/env python3
from data_loader import OCRDataset, DataLoader, custom_collate_fn
import torch

def test_multiple_batches():
    """複数バッチでの動作確認"""
    print("=== 複数バッチテスト ===")
    
    # 全データでデータセットを作成
    dataset = OCRDataset(data_dir='output', start_idx=1, end_idx=5000)
    
    # Train/Val/Testに分割
    train_size = 4000
    val_size = 500
    test_size = 500
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 訓練用データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # 3バッチ分確認
    print("\n=== 訓練データローダーの確認 ===")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"\nバッチ {i+1}:")
        print(f"  画像shape: {batch['images'].shape}")
        print(f"  文字数: {batch['num_chars']}")
        print(f"  最小値: {batch['images'].min():.3f}, 最大値: {batch['images'].max():.3f}")
    
    # データの一貫性チェック
    print("\n=== データ一貫性チェック ===")
    batch = next(iter(train_loader))
    assert batch['images'].shape[0] == 8, "バッチサイズが正しくない"
    assert batch['images'].shape[1:] == (3, 480, 640), "画像サイズが正しくない"
    assert 0 <= batch['images'].min() <= 1, "正規化が正しくない"
    assert 0 <= batch['images'].max() <= 1, "正規化が正しくない"
    print("✓ すべてのチェックをパス")

if __name__ == "__main__":
    test_multiple_batches()