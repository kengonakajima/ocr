import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 前のステップのモジュールをインポート
from step2_data_loader import OCRDataLoader
from step3_patch_generator import PatchGenerator


def load_trained_model(model_path='step5_trained_model.h5'):
    """学習済みモデルを読み込み"""
    print("=== モデル読み込み ===")
    model = keras.models.load_model(model_path)
    print(f"✓ モデル '{model_path}' を読み込みました")
    return model


def create_test_data(num_images=20):
    """テスト用データを作成"""
    print("\n=== テストデータ作成 ===")
    
    loader = OCRDataLoader('output')
    generator = PatchGenerator(patch_size=16)
    
    # 学習に使用していない画像からパッチを生成
    # （実際は学習データと重複しているが、デモ用として）
    print(f"{num_images}枚の画像からテストデータを生成中...")
    
    all_patches = []
    all_labels = []
    
    # 後ろの方の画像を使用（学習では最初の50枚を使用）
    start_idx = 100
    for i in range(start_idx, min(start_idx + num_images, len(loader.image_files))):
        image, bboxes = loader.load_image_and_labels(i)
        
        # 文字パッチを抽出
        char_patches, char_labels = generator.extract_char_patches(image, bboxes)
        # 背景パッチを抽出
        bg_patches, bg_labels = generator.extract_background_patches(
            image, bboxes, len(char_patches))
        
        all_patches.extend(char_patches)
        all_labels.extend(char_labels)
        all_patches.extend(bg_patches)
        all_labels.extend(bg_labels)
    
    X_test = generator.normalize_patches(all_patches)
    y_test = np.array(all_labels, dtype=np.int32)
    
    print(f"テストデータ: {X_test.shape}")
    print(f"テストラベル: {y_test.shape}")
    
    return X_test, y_test, loader


def analyze_predictions(model, X_test, y_test, loader):
    """予測結果を詳細に分析"""
    print("\n=== 予測分析 ===")
    
    # 予測実行
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # 基本精度
    accuracy = np.mean(predicted_classes == y_test)
    print(f"全体精度: {accuracy:.4f}")
    
    # 文字と背景の精度を分けて計算
    char_mask = y_test < 62
    bg_mask = y_test == 62
    
    char_accuracy = np.mean(predicted_classes[char_mask] == y_test[char_mask])
    bg_accuracy = np.mean(predicted_classes[bg_mask] == y_test[bg_mask])
    
    print(f"文字認識精度: {char_accuracy:.4f}")
    print(f"背景認識精度: {bg_accuracy:.4f}")
    
    # 信頼度の分析
    print(f"\n信頼度スコア統計:")
    print(f"  平均: {np.mean(confidence_scores):.4f}")
    print(f"  最小: {np.min(confidence_scores):.4f}")
    print(f"  最大: {np.max(confidence_scores):.4f}")
    
    # 低信頼度の予測を分析
    low_conf_threshold = 0.5
    low_conf_mask = confidence_scores < low_conf_threshold
    print(f"\n信頼度 < {low_conf_threshold} のサンプル: {np.sum(low_conf_mask)}/{len(confidence_scores)}")
    
    return predicted_classes, confidence_scores


def plot_confusion_matrix(y_true, y_pred, loader, num_classes=10):
    """混同行列を可視化（上位クラスのみ）"""
    print("\n=== 混同行列作成 ===")
    
    # 最も頻度の高いクラスを選択
    unique_classes, counts = np.unique(y_true, return_counts=True)
    top_classes = unique_classes[np.argsort(counts)[-num_classes:]]
    
    # フィルタリング
    mask = np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # 混同行列を計算
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
    
    # ラベル名を作成
    labels = []
    for cls in top_classes:
        if cls == 62:
            labels.append('BG')
        else:
            labels.append(loader.label_to_char.get(cls, f'{cls}'))
    
    # プロット
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Top {num_classes} Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('step6_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()  # showの代わりにclose
    print("混同行列を 'step6_confusion_matrix.png' に保存しました")


def visualize_predictions(model, X_test, y_test, loader, num_samples=16):
    """予測結果を視覚化"""
    print("\n=== 予測結果の可視化 ===")
    
    # ランダムにサンプルを選択
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # 予測実行
    predictions = model.predict(X_test[indices], verbose=0)
    
    # プロット
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # パッチを表示
        patch = (X_test[idx] * 255).astype(np.uint8)
        axes[i].imshow(patch)
        
        # 真のラベルと予測ラベル
        true_label = y_test[idx]
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        
        # ラベルを文字に変換
        true_char = 'BG' if true_label == 62 else loader.label_to_char.get(true_label, f'{true_label}')
        pred_char = 'BG' if pred_label == 62 else loader.label_to_char.get(pred_label, f'{pred_label}')
        
        # 正解/不正解で色分け
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].set_title(f'True: {true_char}\nPred: {pred_char} ({confidence:.2f})', 
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Prediction Results', fontsize=14)
    plt.tight_layout()
    plt.savefig('step6_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()  # showの代わりにclose
    print("予測結果を 'step6_predictions.png' に保存しました")


def analyze_errors(model, X_test, y_test, loader):
    """エラー分析"""
    print("\n=== エラー分析 ===")
    
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    # 誤分類のマスク
    error_mask = predicted_classes != y_test
    error_indices = np.where(error_mask)[0]
    
    print(f"誤分類数: {len(error_indices)}/{len(y_test)} ({100*len(error_indices)/len(y_test):.2f}%)")
    
    if len(error_indices) > 0:
        # エラーの種類を分析
        error_types = {}
        for idx in error_indices[:50]:  # 最初の50個のエラーを分析
            true_label = y_test[idx]
            pred_label = predicted_classes[idx]
            
            true_char = 'BG' if true_label == 62 else loader.label_to_char.get(true_label, str(true_label))
            pred_char = 'BG' if pred_label == 62 else loader.label_to_char.get(pred_label, str(pred_label))
            
            error_key = f"{true_char} -> {pred_char}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        # 頻度の高いエラーを表示
        print("\n頻繁なエラーパターン（上位10個）:")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error_pattern, count in sorted_errors[:10]:
            print(f"  {error_pattern}: {count}回")
        
        # エラーの信頼度分析
        error_confidences = confidence_scores[error_mask]
        print(f"\n誤分類の信頼度:")
        print(f"  平均: {np.mean(error_confidences):.4f}")
        print(f"  最小: {np.min(error_confidences):.4f}")
        print(f"  最大: {np.max(error_confidences):.4f}")


def save_validation_report(results, filename='step6_validation_report.txt'):
    """検証レポートを保存"""
    with open(filename, 'w') as f:
        f.write("="*50 + "\n")
        f.write("ステップ6: モデル検証レポート\n")
        f.write("="*50 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n検証レポートを '{filename}' に保存しました")


def main():
    print("="*50)
    print("ステップ6: 検証機能の実装")
    print("="*50)
    
    # モデルを読み込み
    model = load_trained_model()
    
    # テストデータを作成
    X_test, y_test, loader = create_test_data(num_images=20)
    
    # 予測分析
    predicted_classes, confidence_scores = analyze_predictions(model, X_test, y_test, loader)
    
    # 混同行列
    plot_confusion_matrix(y_test, predicted_classes, loader)
    
    # 予測結果の可視化
    visualize_predictions(model, X_test, y_test, loader)
    
    # エラー分析
    analyze_errors(model, X_test, y_test, loader)
    
    # 結果をまとめる
    accuracy = np.mean(predicted_classes == y_test)
    results = {
        "テストサンプル数": len(y_test),
        "全体精度": f"{accuracy:.4f}",
        "文字クラス精度": f"{np.mean(predicted_classes[y_test<62] == y_test[y_test<62]):.4f}",
        "背景クラス精度": f"{np.mean(predicted_classes[y_test==62] == y_test[y_test==62]):.4f}",
        "平均信頼度": f"{np.mean(confidence_scores):.4f}",
        "誤分類率": f"{np.mean(predicted_classes != y_test):.4f}"
    }
    
    # レポート保存
    save_validation_report(results)
    
    print("\n=== ステップ6完了 ===")
    print("✓ モデル検証完了")
    print("✓ 混同行列作成完了")
    print("✓ エラー分析完了")
    print(f"✓ 最終テスト精度: {accuracy:.4f}")


if __name__ == "__main__":
    main()