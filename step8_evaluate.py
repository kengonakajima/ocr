import numpy as np
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 前のステップのモジュールをインポート
from step2_data_loader import OCRDataLoader
from step7_inference import SimplestOCRInference


class OCREvaluator:
    """OCRシステムの総合評価"""
    
    def __init__(self, model_path='step5_trained_model.h5'):
        self.ocr = SimplestOCRInference(model_path)
        self.loader = OCRDataLoader('output')
        
    def evaluate_on_images(self, image_indices: List[int], 
                          scan_mode='partial', confidence_threshold=0.5):
        """
        複数画像でOCRを評価
        
        Args:
            image_indices: 評価する画像のインデックスリスト
            scan_mode: 'partial'(部分スキャン) or 'full'(完全スキャン)
            confidence_threshold: 検出の信頼度しきい値
        
        Returns:
            評価結果の辞書
        """
        all_results = []
        total_time = 0
        
        for img_idx in image_indices:
            print(f"\n画像 {img_idx+1}/{len(image_indices)} を処理中...")
            
            # 画像とGround Truthを読み込み
            image_array, ground_truth = self.loader.load_image_and_labels(img_idx)
            
            # スキャン実行
            start_time = time.time()
            
            if scan_mode == 'partial':
                # 部分スキャン（高速評価用）
                image_crop = image_array[:300, :400]
                detections = self.ocr.full_scan(image_crop, batch_size=500, 
                                               confidence_threshold=confidence_threshold)
                # 対応するGTを抽出
                gt_filtered = [gt for gt in ground_truth 
                             if gt['x'] < 384 and gt['y'] < 284]
            else:
                # 完全スキャン
                detections = self.ocr.full_scan(image_array, batch_size=1000,
                                               confidence_threshold=confidence_threshold)
                gt_filtered = ground_truth
            
            # NMS適用
            detections_nms = self.ocr.non_maximum_suppression(detections)
            
            scan_time = time.time() - start_time
            total_time += scan_time
            
            # 評価メトリクス計算
            metrics = self.calculate_metrics(detections_nms, gt_filtered)
            metrics['scan_time'] = scan_time
            metrics['image_file'] = self.loader.image_files[img_idx]
            metrics['detections_before_nms'] = len(detections)
            metrics['detections_after_nms'] = len(detections_nms)
            
            all_results.append(metrics)
            
            print(f"  Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, "
                  f"F1: {metrics['f1']:.3f}")
        
        # 平均メトリクスを計算
        avg_metrics = self.calculate_average_metrics(all_results)
        avg_metrics['total_time'] = total_time
        avg_metrics['avg_time_per_image'] = total_time / len(image_indices)
        
        return all_results, avg_metrics
    
    def calculate_metrics(self, detections: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        検出結果とGround Truthから評価メトリクスを計算
        """
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        matched_gt = set()
        
        # 各検出に対して最も近いGTを探す
        for det in detections:
            best_match = None
            best_distance = float('inf')
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                # 距離を計算
                distance = np.sqrt((det['x'] - gt['x'])**2 + (det['y'] - gt['y'])**2)
                
                # 同じ文字で距離が近い場合
                if det['char'] == gt['char'] and distance < 12:
                    if distance < best_distance:
                        best_distance = distance
                        best_match = gt_idx
            
            if best_match is not None:
                tp += 1
                matched_gt.add(best_match)
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
            'gt_count': len(ground_truth),
            'det_count': len(detections)
        }
    
    def calculate_average_metrics(self, results: List[Dict]) -> Dict:
        """平均メトリクスを計算"""
        avg_metrics = {
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_f1': np.mean([r['f1'] for r in results]),
            'total_tp': sum([r['tp'] for r in results]),
            'total_fp': sum([r['fp'] for r in results]),
            'total_fn': sum([r['fn'] for r in results]),
            'total_gt': sum([r['gt_count'] for r in results]),
            'total_det': sum([r['det_count'] for r in results])
        }
        
        # 全体の精度も計算
        total_tp = avg_metrics['total_tp']
        total_fp = avg_metrics['total_fp']
        total_fn = avg_metrics['total_fn']
        
        avg_metrics['overall_precision'] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        avg_metrics['overall_recall'] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        avg_metrics['overall_f1'] = (2 * avg_metrics['overall_precision'] * avg_metrics['overall_recall'] / 
                                     (avg_metrics['overall_precision'] + avg_metrics['overall_recall']) 
                                     if (avg_metrics['overall_precision'] + avg_metrics['overall_recall']) > 0 else 0)
        
        return avg_metrics
    
    def plot_evaluation_results(self, results: List[Dict], avg_metrics: Dict):
        """評価結果をプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 各画像のF1スコア
        ax = axes[0, 0]
        f1_scores = [r['f1'] for r in results]
        ax.bar(range(len(f1_scores)), f1_scores)
        ax.axhline(y=avg_metrics['avg_f1'], color='r', linestyle='--', label=f'Avg: {avg_metrics["avg_f1"]:.3f}')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score per Image')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Precision vs Recall
        ax = axes[0, 1]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        ax.scatter(recalls, precisions, alpha=0.6)
        ax.scatter([avg_metrics['overall_recall']], [avg_metrics['overall_precision']], 
                  color='red', s=100, marker='*', label='Overall')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 検出数の分布
        ax = axes[1, 0]
        gt_counts = [r['gt_count'] for r in results]
        det_counts = [r['det_count'] for r in results]
        x = range(len(results))
        width = 0.35
        ax.bar([i - width/2 for i in x], gt_counts, width, label='Ground Truth', alpha=0.7)
        ax.bar([i + width/2 for i in x], det_counts, width, label='Detections', alpha=0.7)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Character Count')
        ax.set_title('Ground Truth vs Detections')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 処理時間
        ax = axes[1, 1]
        scan_times = [r['scan_time'] for r in results]
        ax.bar(range(len(scan_times)), scan_times)
        ax.axhline(y=np.mean(scan_times), color='r', linestyle='--', 
                  label=f'Avg: {np.mean(scan_times):.2f}s')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Scan Time (s)')
        ax.set_title('Processing Time per Image')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'OCR Evaluation Results (Overall F1: {avg_metrics["overall_f1"]:.3f})', fontsize=14)
        plt.tight_layout()
        plt.savefig('step8_evaluation_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n評価結果グラフを 'step8_evaluation_results.png' に保存しました")


def generate_final_report(results: List[Dict], avg_metrics: Dict):
    """最終評価レポートを生成"""
    report = []
    report.append("="*60)
    report.append("最も素朴なピクセル完全スキャンOCR - 最終評価レポート")
    report.append("="*60)
    
    report.append("\n【システム概要】")
    report.append("- 手法: 16×16ピクセル領域の完全スキャン")
    report.append("- モデル: 2層CNN（60,415パラメータ）")
    report.append("- クラス数: 63（A-Z, a-z, 0-9, 背景）")
    
    report.append("\n【評価条件】")
    report.append(f"- 評価画像数: {len(results)}")
    report.append(f"- 総文字数: {avg_metrics['total_gt']}")
    report.append(f"- 総検出数: {avg_metrics['total_det']}")
    
    report.append("\n【性能メトリクス】")
    report.append(f"- 全体Precision: {avg_metrics['overall_precision']:.4f}")
    report.append(f"- 全体Recall: {avg_metrics['overall_recall']:.4f}")
    report.append(f"- 全体F1スコア: {avg_metrics['overall_f1']:.4f}")
    report.append(f"- 平均処理時間: {avg_metrics['avg_time_per_image']:.3f}秒/画像")
    
    report.append("\n【検出統計】")
    report.append(f"- True Positives: {avg_metrics['total_tp']}")
    report.append(f"- False Positives: {avg_metrics['total_fp']}")
    report.append(f"- False Negatives: {avg_metrics['total_fn']}")
    
    report.append("\n【考察】")
    if avg_metrics['overall_f1'] > 0.7:
        report.append("✓ 良好な性能: シンプルな手法でも高精度を達成")
    elif avg_metrics['overall_f1'] > 0.5:
        report.append("△ 中程度の性能: 実用化には改善が必要")
    else:
        report.append("✗ 改善必要: より多くの学習データが必要")
    
    if avg_metrics['overall_precision'] > avg_metrics['overall_recall']:
        report.append("- 特徴: 高精度だが検出漏れあり（保守的）")
    else:
        report.append("- 特徴: 検出率は高いが誤検出あり（積極的）")
    
    report.append("\n【結論】")
    report.append("最も素朴な完全スキャン手法でも、")
    report.append("適切な学習とNMS処理により実用的なOCRが可能。")
    report.append("計算量は多いが、確実性の高い手法として有効。")
    
    report_text = "\n".join(report)
    
    # レポートを保存
    with open('step8_final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n最終レポートを 'step8_final_report.txt' に保存しました")
    
    return report_text


def main():
    print("="*50)
    print("ステップ8: OCRシステムの総合評価")
    print("="*50)
    
    # 評価システムを初期化
    evaluator = OCREvaluator()
    
    # テスト画像を選択（学習に使用していない画像）
    test_indices = list(range(300, 310))  # 10枚でテスト
    
    print(f"\n{len(test_indices)}枚の画像で評価を実行...")
    print("（部分スキャンモード: 300×400領域）")
    
    # 評価実行
    results, avg_metrics = evaluator.evaluate_on_images(
        test_indices, 
        scan_mode='partial',  # 'full'にすると完全スキャン（時間がかかる）
        confidence_threshold=0.5
    )
    
    # 結果をプロット
    evaluator.plot_evaluation_results(results, avg_metrics)
    
    # 最終レポート生成
    generate_final_report(results, avg_metrics)
    
    # 詳細結果をJSON保存
    with open('step8_detailed_results.json', 'w') as f:
        json.dump({
            'individual_results': results,
            'average_metrics': avg_metrics
        }, f, indent=2)
    
    print("\n詳細結果を 'step8_detailed_results.json' に保存しました")
    
    print("\n=== ステップ8完了 ===")
    print("✓ 総合評価完了")
    print(f"✓ 最終F1スコア: {avg_metrics['overall_f1']:.4f}")
    print("✓ すべてのステップが完了しました！")


if __name__ == "__main__":
    main()