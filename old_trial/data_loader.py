#!/usr/bin/env python3
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class OCRDataset(Dataset):
    """OCR用のデータセット"""
    
    def __init__(self, data_dir='output', start_idx=1, end_idx=5000):
        self.data_dir = data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.image_size = (640, 480)
        
        # ファイルリストを作成
        self.samples = []
        for i in range(start_idx, end_idx + 1):
            img_name = f"{i:05d}.png"
            json_name = f"bb_{i:05d}.json"
            img_path = os.path.join(data_dir, img_name)
            json_path = os.path.join(data_dir, json_name)
            
            if os.path.exists(img_path) and os.path.exists(json_path):
                self.samples.append((img_path, json_path))
        
        print(f"データセット作成: {len(self.samples)}個のサンプル")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        
        # 画像を読み込み
        image = Image.open(img_path).convert('RGB')
        # numpy配列に変換してからテンソルに
        image_array = np.array(image)
        # (H, W, C) -> (C, H, W)に変換
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        # 0-255 -> 0-1に正規化
        image_tensor = image_tensor / 255.0
        
        # JSONを読み込み
        with open(json_path, 'r') as f:
            bboxes = json.load(f)
        
        # バウンディングボックスと文字情報を分離
        bbox_list = []
        char_list = []
        for bbox in bboxes:
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox['w']
            y2 = y1 + bbox['h']
            bbox_list.append([x1, y1, x2, y2])
            
            # 文字情報がある場合は追加（後方互換性のため）
            if 'char' in bbox:
                char_list.append(bbox['char'])
            else:
                char_list.append(None)  # 古いデータの場合
        
        if bbox_list:
            bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        else:
            # 空の場合は形状を保つため
            bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'bboxes': bbox_tensor,
            'chars': char_list,  # 文字情報を追加
            'num_chars': len(bboxes),
            'image_path': img_path
        }


def test_dataloader():
    """データローダーのテスト"""
    print("=== データローダーテスト ===")
    
    # データセットを作成（最初の100個だけ）
    dataset = OCRDataset(data_dir='output', start_idx=1, end_idx=100)
    
    # データローダーを作成
    dataloader = DataLoader(
        dataset, 
        batch_size=8,
        shuffle=True,
        num_workers=0,  # まずは0で動作確認
        collate_fn=custom_collate_fn
    )
    
    # 1バッチ取得して確認
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- バッチ {batch_idx + 1} ---")
        print(f"画像テンソルサイズ: {batch['images'].shape}")
        print(f"バッチ内の画像数: {len(batch['bboxes'])}")
        print(f"各画像の文字数: {batch['num_chars']}")
        
        # 最初の画像の情報
        print(f"\n最初の画像:")
        print(f"  パス: {batch['image_paths'][0]}")
        print(f"  バウンディングボックス数: {len(batch['bboxes'][0])}")
        if len(batch['bboxes'][0]) > 0:
            print(f"  最初のbbox: {batch['bboxes'][0][0].tolist()}")
        
        # メモリ使用量を確認
        import psutil
        process = psutil.Process()
        print(f"\nメモリ使用量: {process.memory_info().rss / 1024**3:.2f} GB")
        
        # 1バッチだけで終了
        break


def custom_collate_fn(batch):
    """バッチ内のデータをまとめる関数"""
    # 画像は通常通りスタック
    images = torch.stack([item['image'] for item in batch])
    
    # bboxesはリストのまま（各画像で数が異なるため）
    bboxes = [item['bboxes'] for item in batch]
    
    # 文字情報もリストのまま
    chars = [item['chars'] for item in batch]
    
    # その他の情報
    num_chars = [item['num_chars'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes,
        'chars': chars,  # 文字情報を追加
        'num_chars': num_chars,
        'image_paths': image_paths
    }


if __name__ == "__main__":
    test_dataloader()