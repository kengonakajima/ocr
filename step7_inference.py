import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time
import json
import warnings
warnings.filterwarnings('ignore')

# 前のステップのモジュールをインポート
from step2_data_loader import OCRDataLoader


class SimplestOCRInference:
    """最も素朴なピクセル完全スキャンOCR推論"""
    
    def __init__(self, model_path='step5_trained_model.h5'):
        """
        Args:
            model_path: 学習済みモデルのパス
        """
        print("=== OCR推論システム初期化 ===")
        self.model = keras.models.load_model(model_path)
        self.loader = OCRDataLoader('output')
        self.patch_size = 16
        print(f"✓ モデル読み込み完了")
        print(f"✓ パッチサイズ: {self.patch_size}×{self.patch_size}")
        
    def full_scan(self, image_array, batch_size=1000, confidence_threshold=0.5):
        """
        画像を完全スキャンして文字を検出
        
        Args:
            image_array: 入力画像 (480, 640, 3)
            batch_size: バッチ処理のサイズ
            confidence_threshold: 検出の信頼度しきい値
        
        Returns:
            検出結果のリスト
        """
        height, width = image_array.shape[:2]
        
        # スキャン可能な位置数を計算
        scan_height = height - self.patch_size + 1  # 465
        scan_width = width - self.patch_size + 1    # 625
        total_positions = scan_height * scan_width  # 290,625
        
        print(f"\n=== 完全スキャン開始 ===")
        print(f"画像サイズ: {width}×{height}")
        print(f"スキャン位置数: {total_positions:,}")
        
        all_detections = []
        patches_batch = []
        positions_batch = []
        
        start_time = time.time()
        processed = 0
        
        # すべての位置をスキャン
        for y in range(scan_height):
            for x in range(scan_width):
                # パッチを切り出し
                patch = image_array[y:y+self.patch_size, x:x+self.patch_size]
                patch_normalized = patch.astype(np.float32) / 255.0
                
                patches_batch.append(patch_normalized)
                positions_batch.append((x, y))
                
                # バッチが満杯になったら推論実行
                if len(patches_batch) >= batch_size or (y == scan_height-1 and x == scan_width-1):
                    # 推論実行
                    batch_array = np.array(patches_batch)
                    predictions = self.model.predict(batch_array, verbose=0)
                    
                    # 結果を処理
                    for i, (px, py) in enumerate(positions_batch):
                        pred_class = np.argmax(predictions[i])
                        confidence = np.max(predictions[i])
                        
                        # 背景でなく、信頼度がしきい値以上の場合のみ記録
                        if pred_class != 62 and confidence >= confidence_threshold:
                            all_detections.append({
                                'x': px,
                                'y': py,
                                'class': pred_class,
                                'char': self.loader.label_to_char.get(pred_class, '?'),
                                'confidence': float(confidence)
                            })
                    
                    processed += len(patches_batch)
                    if processed % 10000 == 0:
                        elapsed = time.time() - start_time
                        progress = processed / total_positions * 100
                        eta = elapsed / processed * (total_positions - processed)
                        print(f"  処理済み: {processed:,}/{total_positions:,} ({progress:.1f}%) - ETA: {eta:.1f}秒")
                    
                    # バッチをクリア
                    patches_batch = []
                    positions_batch = []
        
        scan_time = time.time() - start_time
        print(f"スキャン完了: {scan_time:.2f}秒")
        print(f"検出数（NMS前）: {len(all_detections)}")
        
        return all_detections
    
    def non_maximum_suppression(self, detections, distance_threshold=8):
        """
        Non-Maximum Suppression (NMS) で重複検出を除去
        
        Args:
            detections: 検出結果のリスト
            distance_threshold: 同一文字とみなす距離のしきい値
        
        Returns:
            NMS後の検出結果
        """
        print(f"\n=== NMS処理 ===")
        print(f"距離しきい値: {distance_threshold}ピクセル")
        
        if not detections:
            return []
        
        # 信頼度でソート（高い順）
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        suppressed = set()
        
        for i, det in enumerate(sorted_detections):
            if i in suppressed:
                continue
            
            # この検出を採用
            final_detections.append(det)
            
            # 近傍の検出を抑制
            for j in range(i + 1, len(sorted_detections)):
                if j in suppressed:
                    continue
                
                other = sorted_detections[j]
                
                # 同じ文字クラスで距離が近い場合は抑制
                if det['class'] == other['class']:
                    distance = np.sqrt((det['x'] - other['x'])**2 + (det['y'] - other['y'])**2)
                    if distance < distance_threshold:
                        suppressed.add(j)
        
        print(f"NMS前: {len(detections)} → NMS後: {len(final_detections)}")
        
        return final_detections
    
    def visualize_detections(self, image_array, detections, save_path='step7_detection_result.png'):
        """
        検出結果を可視化
        
        Args:
            image_array: 元画像
            detections: 検出結果
            save_path: 保存パス
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image_array)
        
        # 各検出に対してバウンディングボックスを描画
        for det in detections:
            x, y = det['x'], det['y']
            char = det['char']
            conf = det['confidence']
            
            # 矩形を描画
            rect = patches.Rectangle((x, y), self.patch_size, self.patch_size,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # 文字ラベルを表示
            ax.text(x, y-2, f'{char}:{conf:.2f}', color='red', 
                   fontsize=8, weight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_title(f'OCR Detection Results ({len(detections)} characters)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"検出結果を '{save_path}' に保存しました")
    
    def compare_with_ground_truth(self, detections, ground_truth, iou_threshold=0.5):
        """
        検出結果とGround Truthを比較
        
        Args:
            detections: 検出結果
            ground_truth: 正解データ
            iou_threshold: IoUしきい値
        
        Returns:
            評価メトリクス
        """
        print(f"\n=== 精度評価 ===")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_gt = set()
        
        # 各検出に対して最も近いGTを探す
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                # IoUを計算（簡略化：中心距離で近似）
                distance = np.sqrt((det['x'] - gt['x'])**2 + (det['y'] - gt['y'])**2)
                
                # 距離が近い and 同じ文字クラス
                if distance < 8 and det['char'] == gt['char']:
                    best_gt_idx = gt_idx
                    break
            
            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives = len(ground_truth) - len(matched_gt)
        
        # メトリクス計算
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def test_on_sample_image(image_idx=200):
    """サンプル画像でテスト"""
    print("="*50)
    print("ステップ7: 推論パイプラインのテスト")
    print("="*50)
    
    # 推論システムを初期化
    ocr = SimplestOCRInference()
    
    # テスト画像を読み込み
    loader = OCRDataLoader('output')
    image_array, ground_truth = loader.load_image_and_labels(image_idx)
    
    print(f"\nテスト画像: {loader.image_files[image_idx]}")
    print(f"Ground Truth文字数: {len(ground_truth)}")
    
    # 完全スキャン実行（デモ用に小さい領域のみ）
    # 実際の完全スキャンは時間がかかるため、一部領域のみ
    demo_mode = True
    if demo_mode:
        print("\n[デモモード: 300×300領域のみスキャン]")
        image_crop = image_array[:300, :300]
        detections = ocr.full_scan(image_crop, batch_size=500, confidence_threshold=0.5)  # しきい値を下げる
    else:
        detections = ocr.full_scan(image_array, batch_size=1000, confidence_threshold=0.7)
    
    # NMS適用
    final_detections = ocr.non_maximum_suppression(detections, distance_threshold=10)
    
    # 結果を可視化
    if demo_mode:
        ocr.visualize_detections(image_crop, final_detections)
    else:
        ocr.visualize_detections(image_array, final_detections)
    
    # Ground Truthとの比較（デモモードでは一部のみ）
    if demo_mode:
        # 300×300領域内のGTのみ抽出
        gt_in_region = [gt for gt in ground_truth if gt['x'] < 284 and gt['y'] < 284]
        if gt_in_region:
            metrics = ocr.compare_with_ground_truth(final_detections, gt_in_region)
    else:
        metrics = ocr.compare_with_ground_truth(final_detections, ground_truth)
    
    print("\n=== ステップ7完了 ===")
    print(f"✓ 完全スキャン実装完了")
    print(f"✓ NMS実装完了")
    print(f"✓ 最終検出数: {len(final_detections)}")
    
    return final_detections


if __name__ == "__main__":
    # サンプル画像でテスト
    detections = test_on_sample_image(image_idx=200)
    
    # 検出結果を保存
    with open('step7_detections.json', 'w') as f:
        json.dump(detections, f, indent=2)
    print("\n検出結果を 'step7_detections.json' に保存しました")