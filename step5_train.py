import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import os

# 前のステップのモジュールをインポート
from step2_data_loader import OCRDataLoader
from step3_patch_generator import PatchGenerator
from step4_cnn_model import create_ocr_cnn_model, compile_model


def prepare_training_data(num_images=100):
    """学習データを準備"""
    print("=== データ準備 ===")
    
    # データローダーとパッチジェネレーターを初期化
    loader = OCRDataLoader('output')
    generator = PatchGenerator(patch_size=16)
    
    # パッチを生成
    print(f"{num_images}枚の画像からパッチを生成中...")
    X, y = generator.generate_training_patches(loader, num_images=num_images)
    
    # 訓練用と検証用に分割（8:2）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"訓練データ: {X_train.shape}, ラベル: {y_train.shape}")
    print(f"検証データ: {X_val.shape}, ラベル: {y_val.shape}")
    
    return X_train, X_val, y_train, y_val


def create_data_generator(X, y, batch_size=128):
    """データジェネレーターを作成（メモリ効率化）"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    """モデルを学習"""
    print("\n=== 学習開始 ===")
    print(f"エポック数: {epochs}")
    print(f"バッチサイズ: {batch_size}")
    
    # データジェネレーターを作成
    train_dataset = create_data_generator(X_train, y_train, batch_size)
    val_dataset = create_data_generator(X_val, y_val, batch_size)
    
    # コールバックの設定
    callbacks = [
        # 早期終了（検証損失が改善しない場合）
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # 学習率の自動調整
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # 学習実行
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n学習時間: {training_time:.2f}秒")
    
    return history


def plot_training_history(history):
    """学習曲線を描画"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 損失のプロット
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 精度のプロット
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training History - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('step5_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("学習曲線を 'step5_training_history.png' に保存しました")


def evaluate_model(model, X_test, y_test, loader):
    """モデルを評価"""
    print("\n=== モデル評価 ===")
    
    # 予測実行
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 精度計算
    accuracy = np.mean(predicted_classes == y_test)
    print(f"テスト精度: {accuracy:.4f}")
    
    # クラスごとの精度
    print("\n=== クラスごとの精度（上位10クラス） ===")
    class_accuracies = {}
    
    for class_id in np.unique(y_test):
        mask = y_test == class_id
        if np.sum(mask) > 0:
            class_acc = np.mean(predicted_classes[mask] == y_test[mask])
            if class_id == 62:
                char = 'Background'
            else:
                char = loader.label_to_char.get(class_id, f'Class{class_id}')
            class_accuracies[char] = class_acc
    
    # 精度でソート
    sorted_acc = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    for char, acc in sorted_acc[:10]:
        print(f"  {char:>10s}: {acc:.4f}")
    
    return accuracy


def save_trained_model(model, filename='step5_trained_model.h5'):
    """学習済みモデルを保存"""
    model.save(filename)
    print(f"\n学習済みモデルを '{filename}' に保存しました")
    
    # モデルサイズを確認
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"ファイルサイズ: {file_size:.2f} MB")


def main():
    print("="*50)
    print("ステップ5: 学習ループの実装")
    print("="*50)
    
    # データを準備（少量でテスト）
    X_train, X_val, y_train, y_val = prepare_training_data(num_images=50)
    
    # モデルを作成
    print("\n=== モデル作成 ===")
    model = create_ocr_cnn_model()
    model = compile_model(model, learning_rate=0.001)
    print("✓ モデル作成完了")
    
    # 学習実行
    history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val,
        epochs=10,
        batch_size=128
    )
    
    # 学習曲線を描画
    plot_training_history(history)
    
    # モデルを評価
    loader = OCRDataLoader('output')
    final_accuracy = evaluate_model(model, X_val, y_val, loader)
    
    # モデルを保存
    save_trained_model(model)
    
    # 最終結果
    print("\n=== ステップ5完了 ===")
    print(f"✓ 学習完了")
    print(f"✓ 最終検証精度: {final_accuracy:.4f}")
    print(f"✓ 総エポック数: {len(history.history['loss'])}")
    
    # 学習の成否を判定
    if final_accuracy > 0.7:
        print("✓ 学習成功（精度70%以上）")
    elif final_accuracy > 0.5:
        print("△ 学習は進んでいるが改善の余地あり")
    else:
        print("✗ 学習が不十分（データ量を増やすことを推奨）")


if __name__ == "__main__":
    main()