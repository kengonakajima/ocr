import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

def inspect_h5_structure(filename='step4_ocr_model.h5'):
    """H5ファイルの構造を表示"""
    print("="*50)
    print("H5ファイルの中身を調査")
    print("="*50)
    
    # H5ファイルを開く
    with h5py.File(filename, 'r') as f:
        print("\n=== ルートキー ===")
        for key in f.keys():
            print(f"- {key}")
        
        print("\n=== 詳細構造 ===")
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/")
        
        f.visititems(print_structure)
        
        # 重みの例を表示
        if 'model_weights' in f:
            print("\n=== 重みの例（conv1層） ===")
            try:
                conv1_path = 'model_weights/conv1/sequential/conv1'
                if conv1_path in f:
                    kernel = f[f'{conv1_path}/kernel'][:]
                    bias = f[f'{conv1_path}/bias'][:]
                    print(f"カーネル形状: {kernel.shape}")
                    print(f"バイアス形状: {bias.shape}")
                    print(f"カーネルの一部: {kernel[0,0,0,:5]}")
                    print(f"バイアスの一部: {bias[:5]}")
            except Exception as e:
                print(f"重みの読み込みエラー: {e}")


def inspect_with_keras(filename='step4_ocr_model.h5'):
    """Kerasを使ってモデル情報を表示"""
    print("\n=== Kerasでの読み込み ===")
    
    # モデルを読み込み
    model = keras.models.load_model(filename)
    
    # 各層の重みを確認
    print("\n層ごとの重み情報:")
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            print(f"\n{layer.name}:")
            for i, w in enumerate(weights):
                if i == 0:
                    print(f"  重み: shape={w.shape}, 最小={w.min():.4f}, 最大={w.max():.4f}")
                else:
                    print(f"  バイアス: shape={w.shape}, 最小={w.min():.4f}, 最大={w.max():.4f}")


def extract_weights_to_numpy(filename='step4_ocr_model.h5'):
    """重みをNumPy配列として抽出"""
    print("\n=== 重みの抽出 ===")
    
    model = keras.models.load_model(filename)
    
    # 重みを辞書に保存
    weights_dict = {}
    for layer in model.layers:
        if layer.get_weights():
            weights_dict[layer.name] = layer.get_weights()
    
    # conv1の重みを保存
    if 'conv1' in weights_dict:
        np.save('conv1_weights.npy', weights_dict['conv1'][0])
        print("✓ conv1の重みを 'conv1_weights.npy' に保存")
    
    return weights_dict


def main():
    # H5ファイルの構造を調査
    inspect_h5_structure()
    
    # Kerasで読み込み
    inspect_with_keras()
    
    # 重みを抽出
    weights = extract_weights_to_numpy()
    
    print("\n=== サマリー ===")
    print(f"保存された層数: {len(weights)}")
    total_params = sum(sum(np.prod(w.shape) for w in layer_weights) 
                      for layer_weights in weights.values())
    print(f"総パラメータ数: {total_params:,}")


if __name__ == "__main__":
    main()