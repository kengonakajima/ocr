import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

def create_ocr_cnn_model(input_shape=(16, 16, 3), num_classes=63):
    """
    最も素朴なOCR用CNNモデルを作成
    
    Args:
        input_shape: 入力画像のサイズ (16, 16, 3)
        num_classes: 分類クラス数 (62文字 + 1背景 = 63)
    
    Returns:
        Kerasモデル
    """
    
    model = models.Sequential([
        # 第1畳み込み層
        layers.Conv2D(32, kernel_size=3, activation='relu', 
                     input_shape=input_shape, name='conv1'),
        
        # 第1プーリング層
        layers.MaxPooling2D(pool_size=2, name='pool1'),
        
        # 第2畳み込み層
        layers.Conv2D(64, kernel_size=3, activation='relu', name='conv2'),
        
        # 第2プーリング層
        layers.MaxPooling2D(pool_size=2, name='pool2'),
        
        # 平坦化
        layers.Flatten(name='flatten'),
        
        # 第1全結合層
        layers.Dense(128, activation='relu', name='dense1'),
        
        # ドロップアウト（過学習防止）
        layers.Dropout(0.5, name='dropout'),
        
        # 出力層
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    モデルをコンパイル
    
    Args:
        model: Kerasモデル
        learning_rate: 学習率
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def analyze_model(model):
    """
    モデルの詳細を分析
    """
    print("\n=== モデル構造の詳細分析 ===")
    
    # 各層の出力形状を取得
    print("\n層ごとの出力形状:")
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            print(f"  {layer.name:12s}: {layer.output_shape}")
    
    # パラメータ数の詳細
    print("\n層ごとのパラメータ数:")
    total_params = 0
    for layer in model.layers:
        if layer.count_params() > 0:
            params = layer.count_params()
            print(f"  {layer.name:12s}: {params:,} パラメータ")
            total_params += params
    
    print(f"\n総パラメータ数: {total_params:,}")
    
    # メモリ使用量の推定（float32を仮定）
    memory_mb = total_params * 4 / (1024 * 1024)
    print(f"推定メモリ使用量: {memory_mb:.2f} MB")
    
    return total_params


def test_model_inference(model):
    """
    モデルの推論をテスト
    """
    print("\n=== 推論テスト ===")
    
    # ダミーデータを作成（バッチサイズ5）
    dummy_input = np.random.random((5, 16, 16, 3)).astype(np.float32)
    
    # 推論実行
    predictions = model.predict(dummy_input, verbose=0)
    
    print(f"入力形状: {dummy_input.shape}")
    print(f"出力形状: {predictions.shape}")
    print(f"出力例（最初のサンプル、上位5クラス）:")
    
    # 最初のサンプルの上位5クラスを表示
    top5_indices = np.argsort(predictions[0])[-5:][::-1]
    for i, idx in enumerate(top5_indices):
        print(f"  クラス {idx:2d}: {predictions[0][idx]:.4f}")
    
    # 確率の合計が1になることを確認
    print(f"\n確率の合計: {np.sum(predictions[0]):.6f}")


def visualize_model_architecture():
    """
    モデルアーキテクチャの視覚的な表現
    """
    print("\n=== モデルアーキテクチャの視覚化 ===")
    print("""
    入力 (16×16×3)
         ↓
    Conv2D(32) → 14×14×32
         ↓
    MaxPool(2) → 7×7×32
         ↓
    Conv2D(64) → 5×5×64
         ↓
    MaxPool(2) → 2×2×64
         ↓
    Flatten → 256
         ↓
    Dense(128) → 128
         ↓
    Dropout(0.5)
         ↓
    Dense(63) → 63
         ↓
    出力 (63クラスの確率)
    """)


def main():
    print("="*50)
    print("ステップ4: CNNモデルの定義")
    print("="*50)
    
    # モデルを作成
    print("\n=== モデルの作成 ===")
    model = create_ocr_cnn_model()
    print("✓ モデル作成完了")
    
    # モデルをコンパイル
    model = compile_model(model)
    print("✓ モデルコンパイル完了")
    
    # モデルのサマリーを表示
    print("\n=== モデルサマリー ===")
    model.summary()
    
    # モデルの詳細分析
    total_params = analyze_model(model)
    
    # 目標との比較
    print("\n=== 目標との比較 ===")
    target_params = 100000
    if total_params < target_params:
        print(f"✓ パラメータ数 {total_params:,} < 目標 {target_params:,}")
    else:
        print(f"✗ パラメータ数 {total_params:,} > 目標 {target_params:,}")
    
    # モデルアーキテクチャの視覚化
    visualize_model_architecture()
    
    # 推論テスト
    test_model_inference(model)
    
    # モデルを保存
    print("\n=== モデルの保存 ===")
    model.save('step4_ocr_model.h5')
    print("✓ モデルを 'step4_ocr_model.h5' に保存しました")
    
    print("\n=== ステップ4完了 ===")
    print("✓ CNNモデルの定義完了")
    print("✓ 16×16入力、63クラス出力")
    print(f"✓ 総パラメータ数: {total_params:,}")
    print("✓ モデルの保存完了")


if __name__ == "__main__":
    main()