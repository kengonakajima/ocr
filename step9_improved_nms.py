import numpy as np
import time
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 前のステップのモジュールをインポート
from step2_data_loader import OCRDataLoader
from step7_inference import SimplestOCRInference


class ImprovedNMS:
    """改良版Non-Maximum Suppression"""
    
    def __init__(self):
        self.stats = {
            'total_detections': 0,
            'after_confidence': 0,
            'after_grouping': 0,
            'suppressed_count': 0
        }
    
    def apply_nms(self, detections: List[Dict], 
                  confidence_threshold: float = 0.5,
                  distance_threshold: int = 16) -> List[Dict]:
        """
        改良版NMSを適用
        
        Args:
            detections: 検出結果のリスト
            confidence_threshold: 確信度のしきい値
            distance_threshold: グループ化の距離しきい値（16ピクセル）
        
        Returns:
            NMS後の検出結果
        """
        print("\n=== 改良版NMS処理開始 ===")
        self.stats['total_detections'] = len(detections)
        print(f"入力検出数: {len(detections)}")
        
        # ステップ1: 確信度でフィルタリング
        confident_detections = self._filter_by_confidence(detections, confidence_threshold)
        
        # ステップ2: 空間的グループ化
        groups = self._group_nearby_detections(confident_detections, distance_threshold)
        
        # ステップ3: 各グループから最良を選択
        final_detections = self._select_best_from_groups(groups)
        
        # 統計情報を表示
        self._print_statistics()
        
        return final_detections
    
    def _filter_by_confidence(self, detections: List[Dict], 
                             threshold: float) -> List[Dict]:
        """確信度が0.5以上の検出のみを残す"""
        filtered = [d for d in detections if d['confidence'] >= threshold]
        self.stats['after_confidence'] = len(filtered)
        
        print(f"\nステップ1: 確信度フィルタリング")
        print(f"  しきい値: {threshold}")
        print(f"  残った検出数: {len(filtered)} ({len(filtered)/len(detections)*100:.1f}%)")
        
        # 確信度分布を分析
        if filtered:
            confidences = [d['confidence'] for d in filtered]
            print(f"  確信度範囲: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  平均確信度: {np.mean(confidences):.3f}")
        
        return filtered
    
    def _group_nearby_detections(self, detections: List[Dict], 
                                distance_threshold: int) -> List[List[Dict]]:
        """16ピクセル以内の重複検出をグループ化"""
        print(f"\nステップ2: 空間的グループ化")
        print(f"  距離しきい値: {distance_threshold}ピクセル")
        
        if not detections:
            return []
        
        # 検出を位置でソート（左上から右下へ）
        sorted_detections = sorted(detections, key=lambda d: (d['y'], d['x']))
        
        groups = []
        used = set()
        
        for i, det in enumerate(sorted_detections):
            if i in used:
                continue
            
            # 新しいグループを開始
            group = [det]
            used.add(i)
            
            # 近傍の検出を同じグループに追加
            for j in range(i + 1, len(sorted_detections)):
                if j in used:
                    continue
                
                other = sorted_detections[j]
                
                # グループ内のいずれかの検出と近い場合は追加
                for member in group:
                    distance = np.sqrt((member['x'] - other['x'])**2 + 
                                     (member['y'] - other['y'])**2)
                    
                    # 同じ文字クラスで距離が近い場合
                    if member['class'] == other['class'] and distance <= distance_threshold:
                        group.append(other)
                        used.add(j)
                        break
            
            groups.append(group)
        
        self.stats['after_grouping'] = len(groups)
        
        # グループ統計
        group_sizes = [len(g) for g in groups]
        print(f"  形成されたグループ数: {len(groups)}")
        print(f"  グループサイズ分布:")
        for size in range(1, min(6, max(group_sizes) + 1)):
            count = sum(1 for s in group_sizes if s == size)
            if count > 0:
                print(f"    サイズ{size}: {count}グループ")
        if max(group_sizes) > 5:
            large_groups = sum(1 for s in group_sizes if s > 5)
            print(f"    サイズ6以上: {large_groups}グループ")
        
        return groups
    
    def _select_best_from_groups(self, groups: List[List[Dict]]) -> List[Dict]:
        """各グループで最も確信度の高いものを選択"""
        print(f"\nステップ3: 各グループから最良を選択")
        
        final_detections = []
        total_suppressed = 0
        
        for group in groups:
            # グループ内で最も確信度の高い検出を選択
            best = max(group, key=lambda d: d['confidence'])
            final_detections.append(best)
            
            suppressed = len(group) - 1
            total_suppressed += suppressed
            
            # 詳細ログ（グループサイズが大きい場合のみ）
            if len(group) > 3:
                chars = [d['char'] for d in group]
                confidences = [d['confidence'] for d in group]
                print(f"  大規模グループ検出: {best['char']} at ({best['x']}, {best['y']})")
                print(f"    グループサイズ: {len(group)}, 確信度範囲: {min(confidences):.3f}-{max(confidences):.3f}")
        
        self.stats['suppressed_count'] = total_suppressed
        print(f"  最終検出数: {len(final_detections)}")
        print(f"  抑制された検出数: {total_suppressed}")
        
        return final_detections
    
    def _print_statistics(self):
        """NMS処理の統計情報を表示"""
        print("\n=== NMS統計サマリー ===")
        print(f"入力検出数: {self.stats['total_detections']}")
        print(f"確信度フィルタ後: {self.stats['after_confidence']} "
              f"({self.stats['after_confidence']/max(1, self.stats['total_detections'])*100:.1f}%)")
        print(f"グループ化後: {self.stats['after_grouping']} "
              f"({self.stats['after_grouping']/max(1, self.stats['total_detections'])*100:.1f}%)")
        print(f"抑制された検出: {self.stats['suppressed_count']}")
        
        reduction_rate = (1 - self.stats['after_grouping']/max(1, self.stats['total_detections'])) * 100
        print(f"削減率: {reduction_rate:.1f}%")
    
    def visualize_nms_process(self, image_array: np.ndarray, 
                             original_detections: List[Dict],
                             final_detections: List[Dict],
                             save_path: str = 'step9_nms_comparison.png'):
        """NMS前後の比較を可視化"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # NMS前
        ax = axes[0]
        ax.imshow(image_array)
        for det in original_detections:
            if det['confidence'] >= 0.5:  # 確信度フィルタ後のみ表示
                rect = patches.Rectangle((det['x'], det['y']), 16, 16,
                                        linewidth=1, edgecolor='yellow', 
                                        facecolor='none', alpha=0.5)
                ax.add_patch(rect)
        ax.set_title(f'Before NMS ({len([d for d in original_detections if d["confidence"] >= 0.5])} detections)')
        ax.axis('off')
        
        # NMS後
        ax = axes[1]
        ax.imshow(image_array)
        for det in final_detections:
            rect = patches.Rectangle((det['x'], det['y']), 16, 16,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(det['x'], det['y']-2, f"{det['char']}", 
                   color='red', fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax.set_title(f'After NMS ({len(final_detections)} detections)')
        ax.axis('off')
        
        plt.suptitle('NMS Comparison: Before vs After')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nNMS比較画像を '{save_path}' に保存しました")


def test_improved_nms():
    """改良版NMSのテスト"""
    print("="*50)
    print("ステップ9: 改良版NMS（Non-Maximum Suppression）の実装")
    print("="*50)
    
    # 推論システムを初期化
    ocr = SimplestOCRInference('step5_trained_model.h5')
    loader = OCRDataLoader('output')
    
    # テスト画像を選択
    test_image_idx = 250
    image_array, ground_truth = loader.load_image_and_labels(test_image_idx)
    
    print(f"\nテスト画像: {loader.image_files[test_image_idx]}")
    print(f"Ground Truth文字数: {len(ground_truth)}")
    
    # 部分領域でスキャン（デモ用）
    print("\n[300×300領域をスキャン]")
    image_crop = image_array[:300, :300]
    
    # 完全スキャン実行
    detections = ocr.full_scan(image_crop, batch_size=500, confidence_threshold=0.3)
    
    # 改良版NMSを適用
    nms = ImprovedNMS()
    final_detections = nms.apply_nms(
        detections,
        confidence_threshold=0.5,
        distance_threshold=16
    )
    
    # 可視化
    nms.visualize_nms_process(image_crop, detections, final_detections)
    
    # 元のNMSとの比較
    print("\n=== 元のNMSとの比較 ===")
    original_nms_result = ocr.non_maximum_suppression(detections, distance_threshold=8)
    print(f"元のNMS結果: {len(original_nms_result)}検出")
    print(f"改良版NMS結果: {len(final_detections)}検出")
    
    # 精度評価
    gt_in_region = [gt for gt in ground_truth if gt['x'] < 284 and gt['y'] < 284]
    if gt_in_region:
        print("\n改良版NMSの精度評価:")
        metrics = ocr.compare_with_ground_truth(final_detections, gt_in_region)
    
    # 結果を保存
    result_data = {
        'original_detections': int(len(detections)),
        'after_confidence_filter': int(nms.stats['after_confidence']),
        'after_grouping': int(nms.stats['after_grouping']),
        'final_detections': int(len(final_detections)),
        'suppressed_count': int(nms.stats['suppressed_count']),
        'reduction_rate': float((1 - len(final_detections)/len(detections)) * 100 if detections else 0),
        'detections': [{
            'x': int(d['x']),
            'y': int(d['y']),
            'char': d['char'],
            'confidence': float(d['confidence'])
        } for d in final_detections[:10]]  # 最初の10個のみ保存
    }
    
    with open('step9_nms_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print("\n=== ステップ9完了 ===")
    print("✓ 確信度0.5以上でフィルタリング完了")
    print("✓ 16ピクセル以内の重複検出をグループ化完了")
    print("✓ 各グループで最も確信度の高いものを選択完了")
    print(f"✓ NMS前後の検出数比較: {len(detections)} → {len(final_detections)}")
    
    return final_detections


if __name__ == "__main__":
    test_improved_nms()