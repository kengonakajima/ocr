import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

class OCRDataLoader:
    """OCRデータのローダークラス"""
    
    def __init__(self, data_dir: str = 'output'):
        self.data_dir = data_dir
        self.char_to_label = self._create_char_mapping()
        self.label_to_char = {v: k for k, v in self.char_to_label.items()}
        
        # ファイルリストを取得
        self.image_files = sorted([f for f in os.listdir(data_dir) 
                                 if f.endswith('.png')])
        self.json_files = sorted([f for f in os.listdir(data_dir) 
                                if f.startswith('bb_') and f.endswith('.json')])
        
        print(f"データローダー初期化完了:")
        print(f"  画像ファイル数: {len(self.image_files)}")
        print(f"  JSONファイル数: {len(self.json_files)}")
        print(f"  文字クラス数: {len(self.char_to_label)} + 背景クラス1 = {len(self.char_to_label) + 1}")
    
    def _create_char_mapping(self) -> Dict[str, int]:
        """文字から数値ラベルへのマッピングを作成"""
        chars = []
        
        # A-Z (65-90)
        for i in range(26):
            chars.append(chr(ord('A') + i))
        
        # a-z (97-122)
        for i in range(26):
            chars.append(chr(ord('a') + i))
        
        # 0-9 (48-57)
        for i in range(10):
            chars.append(str(i))
        
        # 文字をラベル（0-61）にマッピング
        char_to_label = {char: idx for idx, char in enumerate(chars)}
        
        # 背景クラスは62
        char_to_label['background'] = 62
        
        return char_to_label
    
    def load_image_and_labels(self, index: int) -> Tuple[np.ndarray, List[Dict]]:
        """指定インデックスの画像とラベルを読み込む"""
        
        # ファイル名を取得
        img_file = self.image_files[index]
        img_num = img_file.replace('.png', '')
        json_file = f'bb_{img_num}.json'
        
        # 画像を読み込み
        img_path = os.path.join(self.data_dir, img_file)
        image = Image.open(img_path)
        image_array = np.array(image)
        
        # ラベルを読み込み
        json_path = os.path.join(self.data_dir, json_file)
        with open(json_path, 'r') as f:
            bboxes = json.load(f)
        
        # 文字ラベルを数値に変換
        for bbox in bboxes:
            char = bbox['char']
            bbox['label'] = self.char_to_label.get(char, 62)  # 不明な文字は背景扱い
        
        return image_array, bboxes
    
    def get_batch(self, indices: List[int]) -> Tuple[List[np.ndarray], List[List[Dict]]]:
        """複数の画像とラベルをバッチで取得"""
        images = []
        labels_list = []
        
        for idx in indices:
            image, labels = self.load_image_and_labels(idx)
            images.append(image)
            labels_list.append(labels)
        
        return images, labels_list
    
    def get_data_stats(self) -> Dict:
        """データセットの統計情報を取得"""
        total_chars = 0
        char_counts = {}
        
        # 最初の100枚から統計を取得
        for i in range(min(100, len(self.image_files))):
            _, bboxes = self.load_image_and_labels(i)
            total_chars += len(bboxes)
            
            for bbox in bboxes:
                char = bbox['char']
                char_counts[char] = char_counts.get(char, 0) + 1
        
        return {
            'total_images': len(self.image_files),
            'sample_char_count': total_chars,
            'average_chars_per_image': total_chars / min(100, len(self.image_files)),
            'unique_chars': len(char_counts),
            'char_distribution_sample': dict(sorted(char_counts.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:10])
        }


def test_data_loader():
    """データローダーのテスト"""
    print("="*50)
    print("ステップ2: データローダーのテスト")
    print("="*50)
    
    # データローダーを初期化
    loader = OCRDataLoader('output')
    
    # 文字マッピングを確認
    print("\n=== 文字マッピングの確認 ===")
    print(f"最初の10個のマッピング:")
    for char, label in list(loader.char_to_label.items())[:10]:
        print(f"  '{char}' -> {label}")
    print(f"...")
    print(f"  'background' -> {loader.char_to_label['background']}")
    
    # 10個のデータを読み込んでテスト
    print("\n=== 10個のデータ読み込みテスト ===")
    test_indices = list(range(10))
    
    for idx in test_indices:
        image, bboxes = loader.load_image_and_labels(idx)
        print(f"画像 {idx:04d}: サイズ={image.shape}, 文字数={len(bboxes)}")
        
        # 最初の3文字の情報を表示
        for i, bbox in enumerate(bboxes[:3]):
            print(f"  文字{i+1}: '{bbox['char']}' (label={bbox['label']}), " +
                  f"位置=({bbox['x']}, {bbox['y']}), サイズ=({bbox['w']}×{bbox['h']})")
    
    # データ統計を表示
    print("\n=== データセット統計 ===")
    stats = loader.get_data_stats()
    for key, value in stats.items():
        if key == 'char_distribution_sample':
            print(f"{key}:")
            for char, count in value.items():
                print(f"  '{char}': {count}回")
        else:
            print(f"{key}: {value}")
    
    # バッチ読み込みテスト
    print("\n=== バッチ読み込みテスト ===")
    batch_indices = [0, 1, 2, 3, 4]
    images, labels_list = loader.get_batch(batch_indices)
    print(f"バッチサイズ: {len(images)}")
    print(f"各画像のサイズ: {[img.shape for img in images]}")
    print(f"各画像の文字数: {[len(labels) for labels in labels_list]}")
    
    print("\n=== ステップ2完了 ===")
    print("✓ 文字ラベルマッピング作成完了")
    print("✓ 画像とラベルの読み込み機能実装完了")
    print("✓ バッチ処理機能実装完了")


if __name__ == "__main__":
    test_data_loader()