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
from step9_improved_nms import ImprovedNMS


class FinalEvaluator:
    """最終評価システム"""
    
    def __init__(self, model_path='step5_trained_model.h5'):
        print("=== 最終評価システム初期化 ===")
        self.ocr = SimplestOCRInference(model_path)
        self.nms = ImprovedNMS()
        self.loader = OCRDataLoader('output')
        print("✓ システム準備完了\n")
    
    def evaluate_image(self, image_idx: int, use_improved_nms: bool = True) -> Dict:
        """
        単一画像で評価を実行
        
        Args:
            image_idx: 画像インデックス
            use_improved_nms: 改良版NMSを使用するか
        
        Returns:
            評価結果の辞書
        """
        # 画像とGround Truthを読み込み
        image_array, ground_truth = self.loader.load_image_and_labels(image_idx)
        
        print(f"画像 {self.loader.image_files[image_idx]} を評価中...")
        print(f"  Ground Truth文字数: {len(ground_truth)}")
        
        # 部分領域でスキャン（評価用に400×400領域）
        eval_region_size = 400
        image_crop = image_array[:eval_region_size, :eval_region_size]
        
        # Ground Truthも対応する領域に制限
        gt_in_region = [
            gt for gt in ground_truth 
            if gt['x'] < eval_region_size - 16 and gt['y'] < eval_region_size - 16
        ]
        
        # 完全スキャン実行
        start_time = time.time()
        detections = self.ocr.full_scan(
            image_crop, 
            batch_size=1000, 
            confidence_threshold=0.3  # 低めの閾値で広く検出
        )
        scan_time = time.time() - start_time
        
        # NMS適用
        if use_improved_nms:
            final_detections = self.nms.apply_nms(
                detections,
                confidence_threshold=0.5,
                distance_threshold=16
            )
        else:
            final_detections = self.ocr.non_maximum_suppression(
                detections, 
                distance_threshold=8
            )
        
        # 評価メトリクス計算
        metrics = self.calculate_metrics(final_detections, gt_in_region)
        metrics['scan_time'] = scan_time
        metrics['detections_before_nms'] = len(detections)
        metrics['detections_after_nms'] = len(final_detections)
        metrics['gt_in_region'] = len(gt_in_region)
        metrics['image_file'] = self.loader.image_files[image_idx]
        
        return metrics, image_crop, final_detections, gt_in_region
    
    def calculate_metrics(self, detections: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Precision, Recall, F1スコアを計算
        
        Args:
            detections: 検出結果
            ground_truth: 正解データ
        
        Returns:
            評価メトリクス
        """
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        matched_gt = set()
        matched_detections = []
        
        # 各検出に対して最も近いGTを探す
        for det in detections:
            best_match = None
            best_distance = float('inf')
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                # 文字が一致し、距離が近い場合
                if det['char'] == gt['char']:
                    distance = np.sqrt((det['x'] - gt['x'])**2 + (det['y'] - gt['y'])**2)
                    if distance < 12:  # 12ピクセル以内なら一致とみなす
                        if distance < best_distance:
                            best_distance = distance
                            best_match = gt_idx
            
            if best_match is not None:
                tp += 1
                matched_gt.add(best_match)
                matched_detections.append((det, ground_truth[best_match]))
            else:
                fp += 1
        
        fn = len(ground_truth) - len(matched_gt)
        
        # メトリクス計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matched_pairs': matched_detections
        }
    
    def evaluate_multiple_images(self, image_indices: List[int], 
                               use_improved_nms: bool = True) -> Dict:
        """
        複数画像で評価を実行
        
        Args:
            image_indices: 評価する画像のインデックスリスト
            use_improved_nms: 改良版NMSを使用するか
        
        Returns:
            総合評価結果
        """
        all_results = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_time = 0
        
        print(f"\n{len(image_indices)}枚の画像で評価を実行...")
        print("=" * 50)
        
        for i, idx in enumerate(image_indices):
            print(f"\n[{i+1}/{len(image_indices)}] ", end="")
            metrics, _, _, _ = self.evaluate_image(idx, use_improved_nms)
            
            all_results.append(metrics)
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            total_time += metrics['scan_time']
            
            print(f"  結果: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        # 総合メトリクス計算
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                    if (overall_precision + overall_recall) > 0 else 0
        
        avg_metrics = {
            'num_images': len(image_indices),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'avg_time_per_image': total_time / len(image_indices),
            'individual_results': all_results
        }
        
        return avg_metrics
    
    def visualize_results(self, image_idx: int, save_path: str = 'step10_final_result.png'):
        """
        評価結果を可視化
        
        Args:
            image_idx: 表示する画像のインデックス
            save_path: 保存パス
        """
        # 画像で評価を実行
        metrics, image_crop, detections, ground_truth = self.evaluate_image(image_idx)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Ground Truth
        ax = axes[0]
        ax.imshow(image_crop)
        for gt in ground_truth:
            rect = patches.Rectangle((gt['x'], gt['y']), 16, 16,
                                    linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(gt['x'], gt['y']-2, gt['char'], color='green', 
                   fontsize=8, weight='bold')
        ax.set_title(f'Ground Truth ({len(ground_truth)} chars)')
        ax.axis('off')
        
        # 2. Detections
        ax = axes[1]
        ax.imshow(image_crop)
        for det in detections:
            rect = patches.Rectangle((det['x'], det['y']), 16, 16,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(det['x'], det['y']-2, det['char'], color='red', 
                   fontsize=8, weight='bold')
        ax.set_title(f'Detections ({len(detections)} chars)')
        ax.axis('off')
        
        # 3. Metrics
        ax = axes[2]
        ax.axis('off')
        metrics_text = f"""
Performance Metrics:
-------------------
True Positives:  {metrics['tp']}
False Positives: {metrics['fp']}
False Negatives: {metrics['fn']}

Precision: {metrics['precision']:.3f}
Recall:    {metrics['recall']:.3f}
F1 Score:  {metrics['f1']:.3f}

Processing Time: {metrics['scan_time']:.2f}s
Before NMS: {metrics['detections_before_nms']}
After NMS:  {metrics['detections_after_nms']}
        """
        ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
               verticalalignment='center')
        ax.set_title('Evaluation Results')
        
        plt.suptitle(f'Final OCR Evaluation - {metrics["image_file"]}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n評価結果を '{save_path}' に保存しました")


def main():
    print("="*60)
    print("ステップ10: 最終評価の実施")
    print("="*60)
    
    # 評価システムを初期化
    evaluator = FinalEvaluator()
    
    # テスト画像セットを選択（学習に使用していない画像）
    test_indices = list(range(400, 410))  # 10枚でテスト
    
    # 複数画像で評価
    print("\n=== 複数画像での評価 ===")
    avg_metrics = evaluator.evaluate_multiple_images(test_indices, use_improved_nms=True)
    
    # 結果サマリーを表示
    print("\n" + "="*60)
    print("評価結果サマリー")
    print("="*60)
    print(f"評価画像数: {avg_metrics['num_images']}")
    print(f"総True Positives: {avg_metrics['total_tp']}")
    print(f"総False Positives: {avg_metrics['total_fp']}")
    print(f"総False Negatives: {avg_metrics['total_fn']}")
    print("-"*40)
    print(f"全体Precision: {avg_metrics['overall_precision']:.4f}")
    print(f"全体Recall: {avg_metrics['overall_recall']:.4f}")
    print(f"全体F1スコア: {avg_metrics['overall_f1']:.4f}")
    print(f"平均処理時間: {avg_metrics['avg_time_per_image']:.2f}秒/画像")
    
    # F1スコアの判定
    print("\n" + "="*60)
    if avg_metrics['overall_f1'] >= 0.5:
        print("✓ 成功: F1スコアが0.5以上を達成しました！")
        print(f"  達成値: {avg_metrics['overall_f1']:.4f}")
    else:
        print("△ 改善必要: F1スコアが0.5未満です")
        print(f"  現在値: {avg_metrics['overall_f1']:.4f}")
        print("\n改善案:")
        print("  1. より多くの学習データを使用")
        print("  2. データ拡張の適用")
        print("  3. モデルアーキテクチャの改良")
        print("  4. ハイパーパラメータの調整")
    
    # 代表的な画像で可視化
    print("\n代表画像での詳細評価...")
    evaluator.visualize_results(test_indices[0])
    
    # 結果をJSON保存
    save_data = {
        'test_indices': test_indices,
        'overall_metrics': {
            'precision': float(avg_metrics['overall_precision']),
            'recall': float(avg_metrics['overall_recall']),
            'f1': float(avg_metrics['overall_f1']),
            'avg_time': float(avg_metrics['avg_time_per_image'])
        },
        'counts': {
            'tp': int(avg_metrics['total_tp']),
            'fp': int(avg_metrics['total_fp']),
            'fn': int(avg_metrics['total_fn'])
        },
        'individual_results': [
            {
                'image': r['image_file'],
                'precision': float(r['precision']),
                'recall': float(r['recall']),
                'f1': float(r['f1']),
                'tp': int(r['tp']),
                'fp': int(r['fp']),
                'fn': int(r['fn'])
            }
            for r in avg_metrics['individual_results']
        ]
    }
    
    with open('step10_evaluation_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print("\n詳細結果を 'step10_evaluation_results.json' に保存しました")
    
    print("\n" + "="*60)
    print("ステップ10完了")
    print("="*60)
    print("✓ テスト画像での文字検出を実行")
    print("✓ 検出結果とGround Truthを比較")
    print("✓ 精度・再現率・F1スコアを計算")
    if avg_metrics['overall_f1'] >= 0.5:
        print("✓ F1スコアが0.5以上を達成")
    else:
        print(f"△ F1スコア: {avg_metrics['overall_f1']:.4f} (目標: 0.5以上)")
    
    print("\n🎉 すべてのステップが完了しました！")
    
    return avg_metrics


if __name__ == "__main__":
    main()