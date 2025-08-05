import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def check_data_structure():
    # データディレクトリのパス
    data_dir = 'output'
    
    # 画像ファイルとJSONファイルをリスト化
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    json_files = sorted([f for f in os.listdir(data_dir) if f.startswith('bb_') and f.endswith('.json')])
    
    print(f"=== データ構造の確認 ===")
    print(f"画像ファイル数: {len(image_files)}")
    print(f"JSONファイル数: {len(json_files)}")
    print(f"最初の5つの画像: {image_files[:5]}")
    print(f"最初の5つのJSON: {json_files[:5]}")
    
    # ファイル名の対応を確認
    print(f"\n=== ファイル名の対応確認 ===")
    for i in range(min(5, len(image_files))):
        img_num = image_files[i].replace('.png', '')
        json_name = f'bb_{img_num}.json'
        if json_name in json_files:
            print(f"✓ {image_files[i]} ←→ {json_name}")
        else:
            print(f"✗ {image_files[i]} ←→ JSONファイルが見つかりません")
    
    # 文字の種類を集計
    print(f"\n=== 文字種の集計 ===")
    char_counts = {}
    for json_file in json_files[:100]:  # 最初の100ファイルで集計
        with open(os.path.join(data_dir, json_file), 'r') as f:
            bboxes = json.load(f)
            for bbox in bboxes:
                char = bbox['char']
                char_counts[char] = char_counts.get(char, 0) + 1
    
    print(f"検出された文字種数: {len(char_counts)}")
    print(f"文字種のサンプル: {list(char_counts.keys())[:10]}")
    
    return image_files, json_files

def visualize_random_samples(image_files, json_files, num_samples=5):
    """ランダムに選んだ画像でバウンディングボックスを可視化"""
    
    # ランダムにサンプルを選択
    sample_indices = random.sample(range(min(100, len(image_files))), num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for idx, sample_idx in enumerate(sample_indices):
        # 画像とJSONを読み込み
        img_path = os.path.join('output', image_files[sample_idx])
        img_num = image_files[sample_idx].replace('.png', '')
        json_path = os.path.join('output', f'bb_{img_num}.json')
        
        # 画像を表示
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(f'{image_files[sample_idx]}')
        
        # バウンディングボックスを描画
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                bboxes = json.load(f)
                
            for bbox in bboxes:
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                char = bbox['char']
                
                # 矩形を描画
                rect = patches.Rectangle((x, y), w, h, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none')
                axes[idx].add_patch(rect)
                
                # 文字ラベルを表示
                axes[idx].text(x, y-2, char, color='red', 
                             fontsize=10, weight='bold')
        
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('step1_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n可視化結果を 'step1_sample_visualization.png' に保存しました")

def check_image_properties(image_files):
    """画像のプロパティを確認"""
    print(f"\n=== 画像プロパティの確認 ===")
    
    # 最初の5枚の画像をチェック
    for i in range(min(5, len(image_files))):
        img_path = os.path.join('output', image_files[i])
        img = Image.open(img_path)
        
        print(f"{image_files[i]}: サイズ={img.size}, モード={img.mode}")

def main():
    print("ステップ1: データ構造の理解と確認")
    print("="*50)
    
    # データ構造をチェック
    image_files, json_files = check_data_structure()
    
    # 画像プロパティを確認
    check_image_properties(image_files)
    
    # ランダムサンプルを可視化
    print(f"\n=== ランダムサンプルの可視化 ===")
    visualize_random_samples(image_files, json_files, num_samples=5)
    
    print(f"\n=== ステップ1完了 ===")
    print(f"✓ データ構造の確認完了")
    print(f"✓ 画像とJSONの対応確認完了")
    print(f"✓ バウンディングボックスの可視化完了")

if __name__ == "__main__":
    main()