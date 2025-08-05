#!/usr/bin/env python3
import os
import json
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = 'output'

def check_data_consistency():
    """データの一貫性を確認"""
    print("=== データ一貫性チェック ===")
    
    # ファイル数を確認
    png_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    json_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])
    
    print(f"PNG画像数: {len(png_files)}")
    print(f"JSONファイル数: {len(json_files)}")
    
    # 対応関係を確認
    missing_json = []
    missing_png = []
    
    for png_file in png_files[:10]:  # 最初の10個だけチェック
        base_name = png_file.replace('.png', '')
        json_file = f"bb_{base_name}.json"
        if json_file not in json_files:
            missing_json.append(json_file)
    
    if missing_json:
        print(f"警告: JSONファイルが見つかりません: {missing_json}")
    else:
        print("✓ PNG-JSON対応関係: OK")
    
    return png_files, json_files

def load_and_check_sample(image_path, json_path):
    """サンプルデータを読み込んで確認"""
    print(f"\n=== サンプルデータ確認: {os.path.basename(image_path)} ===")
    
    # 画像を読み込み
    img = Image.open(image_path)
    print(f"画像サイズ: {img.size}")
    print(f"画像モード: {img.mode}")
    
    # JSONを読み込み
    with open(json_path, 'r') as f:
        bboxes = json.load(f)
    
    print(f"バウンディングボックス数: {len(bboxes)}")
    
    # 最初の3つのbboxを表示
    for i, bbox in enumerate(bboxes[:3]):
        print(f"  bbox {i}: x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
    
    return img, bboxes

def visualize_bboxes(img, bboxes, save_path=None):
    """バウンディングボックスを可視化"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 画像を表示
    ax.imshow(img, cmap='gray')
    
    # バウンディングボックスを描画
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), 
            bbox['w'], bbox['h'],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_title(f"Bounding Boxes ({len(bboxes)} characters)")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"可視化画像を保存: {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    # データ一貫性チェック
    png_files, json_files = check_data_consistency()
    
    # ランダムに3つのサンプルを確認
    print("\n=== ランダムサンプルの詳細確認 ===")
    sample_indices = random.sample(range(min(100, len(png_files))), 3)
    
    for idx in sample_indices:
        png_file = png_files[idx]
        base_name = png_file.replace('.png', '')
        json_file = f"bb_{base_name}.json"
        
        image_path = os.path.join(OUTPUT_DIR, png_file)
        json_path = os.path.join(OUTPUT_DIR, json_file)
        
        img, bboxes = load_and_check_sample(image_path, json_path)
        
        # 可視化
        vis_path = f"check_{base_name}.png"
        visualize_bboxes(img, bboxes, vis_path)
    
    # 統計情報
    print("\n=== 統計情報 ===")
    bbox_counts = []
    for i in range(min(100, len(json_files))):
        json_path = os.path.join(OUTPUT_DIR, json_files[i])
        with open(json_path, 'r') as f:
            bboxes = json.load(f)
            bbox_counts.append(len(bboxes))
    
    print(f"平均文字数/画像: {sum(bbox_counts)/len(bbox_counts):.1f}")
    print(f"最小文字数: {min(bbox_counts)}")
    print(f"最大文字数: {max(bbox_counts)}")

if __name__ == "__main__":
    main()