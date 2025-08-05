#!/usr/bin/env python3
import torch
import numpy as np

def calculate_iou(box1, box2):
    """
    2つのボックスのIoU（Intersection over Union）を計算
    box: [x1, y1, x2, y2]
    """
    # 交差領域の座標
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 交差領域の面積
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 各ボックスの面積
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 和集合の面積
    union = area1 + area2 - intersection
    
    # IoU
    if union == 0:
        return 0
    return intersection / union

def evaluate_detection(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    検出結果の評価
    
    pred_boxes: 予測ボックス [[x1,y1,x2,y2], ...]
    pred_scores: 各ボックスの信頼度スコア
    gt_boxes: 正解ボックス [[x1,y1,x2,y2], ...]
    """
    if len(pred_boxes) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': len(gt_boxes)
        }
    
    # スコアでソート（高い順）
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    
    # マッチング用のフラグ
    gt_matched = [False] * len(gt_boxes)
    
    tp = 0  # True Positive
    fp = 0  # False Positive
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # 最もIoUが高いGTボックスを探す
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # マッチング判定
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = len(gt_boxes) - tp  # False Negative
    
    # メトリクス計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def test_metrics():
    """メトリクスの動作確認"""
    print("=== 評価メトリクステスト ===")
    
    # テストケース1: 完全一致
    print("\n--- テストケース1: 完全一致 ---")
    pred_boxes = [[10, 10, 30, 30], [50, 50, 70, 70]]
    pred_scores = [0.9, 0.8]
    gt_boxes = [[10, 10, 30, 30], [50, 50, 70, 70]]
    
    results = evaluate_detection(pred_boxes, pred_scores, gt_boxes)
    print(f"予測: {pred_boxes}")
    print(f"正解: {gt_boxes}")
    print(f"結果: Precision={results['precision']:.2f}, Recall={results['recall']:.2f}, F1={results['f1']:.2f}")
    print(f"     TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
    
    # テストケース2: 部分的な重なり
    print("\n--- テストケース2: 部分的な重なり ---")
    pred_boxes = [[10, 10, 30, 30], [45, 45, 65, 65]]  # 2つ目が少しずれている
    pred_scores = [0.9, 0.8]
    gt_boxes = [[10, 10, 30, 30], [50, 50, 70, 70]]
    
    # IoU計算例
    iou = calculate_iou([45, 45, 65, 65], [50, 50, 70, 70])
    print(f"2つ目のボックスのIoU: {iou:.3f}")
    
    results = evaluate_detection(pred_boxes, pred_scores, gt_boxes)
    print(f"結果: Precision={results['precision']:.2f}, Recall={results['recall']:.2f}, F1={results['f1']:.2f}")
    
    # テストケース3: 誤検出と見逃し
    print("\n--- テストケース3: 誤検出と見逃し ---")
    pred_boxes = [[10, 10, 30, 30], [100, 100, 120, 120]]  # 2つ目は誤検出
    pred_scores = [0.9, 0.7]
    gt_boxes = [[10, 10, 30, 30], [50, 50, 70, 70]]  # 2つ目を見逃し
    
    results = evaluate_detection(pred_boxes, pred_scores, gt_boxes)
    print(f"予測: {pred_boxes}")
    print(f"正解: {gt_boxes}")
    print(f"結果: Precision={results['precision']:.2f}, Recall={results['recall']:.2f}, F1={results['f1']:.2f}")
    print(f"     TP={results['tp']}, FP={results['fp']}, FN={results['fn']}")
    
    # IoU閾値の影響
    print("\n--- IoU閾値の影響 ---")
    pred_box = [12, 12, 32, 32]  # 少しずれた予測
    gt_box = [10, 10, 30, 30]
    iou = calculate_iou(pred_box, gt_box)
    print(f"IoU = {iou:.3f}")
    
    for threshold in [0.3, 0.5, 0.7]:
        results = evaluate_detection([pred_box], [0.9], [gt_box], iou_threshold=threshold)
        print(f"閾値{threshold}: TP={results['tp']}, FP={results['fp']}")

if __name__ == "__main__":
    test_metrics()