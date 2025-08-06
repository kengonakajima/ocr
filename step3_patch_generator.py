import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import json

# step2で作成したデータローダーをインポート
from step2_data_loader import OCRDataLoader


class PatchGenerator:
    """16×16パッチを生成するクラス"""
    
    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
        
    def extract_char_patches(self, image: np.ndarray, bboxes: List[Dict]) -> Tuple[List[np.ndarray], List[int]]:
        """文字位置から16×16パッチを切り出す"""
        patches = []
        labels = []
        
        for bbox in bboxes:
            x, y = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            label = bbox['label']
            
            # 文字の中心を計算
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 16×16パッチの左上座標を計算（中心が文字の中心になるように）
            patch_x = center_x - self.patch_size // 2
            patch_y = center_y - self.patch_size // 2
            
            # 画像境界のチェック
            if (patch_x >= 0 and patch_y >= 0 and 
                patch_x + self.patch_size <= image.shape[1] and 
                patch_y + self.patch_size <= image.shape[0]):
                
                # パッチを切り出し
                patch = image[patch_y:patch_y+self.patch_size, 
                            patch_x:patch_x+self.patch_size]
                
                # サイズが正しいか確認
                if patch.shape == (self.patch_size, self.patch_size, 3):
                    patches.append(patch)
                    labels.append(label)
        
        return patches, labels
    
    def extract_background_patches(self, image: np.ndarray, bboxes: List[Dict], 
                                 num_patches: int) -> Tuple[List[np.ndarray], List[int]]:
        """背景からランダムにパッチを切り出す"""
        patches = []
        labels = []
        
        # 文字領域のマスクを作成
        char_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        for bbox in bboxes:
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            # 文字領域を少し広めにマーク（余白を考慮）
            margin = 4
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            char_mask[y1:y2, x1:x2] = True
        
        # 背景パッチを生成
        attempts = 0
        max_attempts = num_patches * 10
        
        while len(patches) < num_patches and attempts < max_attempts:
            attempts += 1
            
            # ランダムな位置を選択
            x = random.randint(0, image.shape[1] - self.patch_size)
            y = random.randint(0, image.shape[0] - self.patch_size)
            
            # パッチ領域が文字と重ならないかチェック
            patch_region = char_mask[y:y+self.patch_size, x:x+self.patch_size]
            
            # 重なりが少ない（10%未満）場合のみ採用
            if np.sum(patch_region) < (self.patch_size * self.patch_size * 0.1):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                labels.append(62)  # 背景ラベル
        
        return patches, labels
    
    def normalize_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """パッチを正規化（0-255 → 0-1）"""
        normalized = np.array(patches, dtype=np.float32) / 255.0
        return normalized
    
    def generate_training_patches(self, loader: OCRDataLoader, 
                                 num_images: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """複数の画像から学習用パッチを生成"""
        all_patches = []
        all_labels = []
        
        for i in range(min(num_images, len(loader.image_files))):
            # 画像とラベルを読み込み
            image, bboxes = loader.load_image_and_labels(i)
            
            # 文字パッチを抽出
            char_patches, char_labels = self.extract_char_patches(image, bboxes)
            
            # 背景パッチを抽出（文字と同数）
            bg_patches, bg_labels = self.extract_background_patches(
                image, bboxes, len(char_patches))
            
            # 結合
            all_patches.extend(char_patches)
            all_labels.extend(char_labels)
            all_patches.extend(bg_patches)
            all_labels.extend(bg_labels)
            
            if (i + 1) % 10 == 0:
                print(f"  処理済み: {i+1}/{num_images} 画像")
        
        # NumPy配列に変換して正規化
        X = self.normalize_patches(all_patches)
        y = np.array(all_labels, dtype=np.int32)
        
        return X, y


def visualize_patches(patches: np.ndarray, labels: np.ndarray, 
                     loader: OCRDataLoader, num_samples: int = 20):
    """生成したパッチを可視化"""
    
    # ランダムにサンプルを選択
    indices = random.sample(range(len(patches)), min(num_samples, len(patches)))
    
    # 図を作成
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # パッチを表示（0-1から0-255に戻す）
        patch_display = (patches[idx] * 255).astype(np.uint8)
        axes[i].imshow(patch_display)
        
        # ラベルを文字に変換
        label = labels[idx]
        if label == 62:
            char = 'BG'
        else:
            char = loader.label_to_char[label]
        
        axes[i].set_title(f'Label: {label} ({char})', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('step3_patch_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("パッチの可視化結果を 'step3_patch_visualization.png' に保存しました")


def test_patch_generator():
    """パッチ生成機能のテスト"""
    print("="*50)
    print("ステップ3: パッチ生成機能のテスト")
    print("="*50)
    
    # データローダーとパッチジェネレーターを初期化
    loader = OCRDataLoader('output')
    generator = PatchGenerator(patch_size=16)
    
    # 単一画像でのテスト
    print("\n=== 単一画像でのパッチ生成テスト ===")
    image, bboxes = loader.load_image_and_labels(0)
    
    # 文字パッチを抽出
    char_patches, char_labels = generator.extract_char_patches(image, bboxes)
    print(f"文字パッチ数: {len(char_patches)}")
    print(f"文字ラベル: {char_labels[:5]}...")
    
    # 背景パッチを抽出
    bg_patches, bg_labels = generator.extract_background_patches(image, bboxes, len(char_patches))
    print(f"背景パッチ数: {len(bg_patches)}")
    print(f"背景ラベル: {bg_labels[:5]}...")
    
    # パッチサイズの確認
    if char_patches:
        print(f"パッチサイズ: {char_patches[0].shape}")
    
    # 複数画像からパッチを生成（10枚に削減してテスト）
    print("\n=== 複数画像からのパッチ生成 ===")
    print("10枚の画像からパッチを生成中...")
    X_train, y_train = generator.generate_training_patches(loader, num_images=10)
    
    print(f"\n生成結果:")
    print(f"  総パッチ数: {len(X_train)}")
    print(f"  パッチ形状: {X_train.shape}")
    print(f"  ラベル形状: {y_train.shape}")
    print(f"  値の範囲: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # クラス分布を確認
    print("\n=== クラス分布 ===")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    
    # 文字クラスと背景クラスを分けて集計
    char_count = sum(counts[unique_labels < 62])
    bg_count = counts[unique_labels == 62][0] if 62 in unique_labels else 0
    
    print(f"文字パッチ数: {char_count}")
    print(f"背景パッチ数: {bg_count}")
    print(f"文字:背景の比率: 1:{bg_count/char_count:.2f}")
    
    # 各文字クラスのサンプル数（上位10個）
    print("\n上位10クラスの分布:")
    sorted_indices = np.argsort(counts)[::-1][:10]
    for idx in sorted_indices:
        label = unique_labels[idx]
        count = counts[idx]
        if label == 62:
            char = 'Background'
        else:
            char = loader.label_to_char[label]
        print(f"  クラス {label:2d} ({char:>10s}): {count:4d} サンプル")
    
    # パッチを可視化
    print("\n=== パッチの可視化 ===")
    visualize_patches(X_train, y_train, loader, num_samples=20)
    
    print("\n=== ステップ3完了 ===")
    print("✓ 文字パッチ抽出機能実装完了")
    print("✓ 背景パッチ抽出機能実装完了")
    print("✓ 正規化処理実装完了")
    print("✓ バランスの取れたデータセット生成完了")


if __name__ == "__main__":
    test_patch_generator()