#!/usr/bin/env python3
import numpy as np
import sys

def view_npy(filename):
    """NPYファイルの内容を表示"""
    try:
        data = np.load(filename)
        
        print(f"ファイル: {filename}")
        print(f"形状: {data.shape}")
        print(f"データ型: {data.dtype}")
        print(f"サイズ: {data.size}要素")
        print(f"メモリ: {data.nbytes / 1024:.2f} KB")
        print(f"最小値: {data.min():.6f}")
        print(f"最大値: {data.max():.6f}")
        print(f"平均値: {data.mean():.6f}")
        print(f"標準偏差: {data.std():.6f}")
        
        # 小さいデータは全表示、大きいデータは一部表示
        if data.size < 100:
            print("\n全データ:")
            print(data)
        else:
            print("\n最初の要素（一部）:")
            if data.ndim == 1:
                print(data[:10])
            elif data.ndim == 2:
                print(data[:3, :3])
            elif data.ndim == 3:
                print(data[0, 0, :5])
            elif data.ndim == 4:
                print(f"data[0,0,0,:5] = {data[0,0,0,:5]}")
                print(f"data[0,0,1,:5] = {data[0,0,1,:5]}")
                
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_npy(sys.argv[1])
    else:
        # デフォルトでconv1_weights.npyを表示
        view_npy('conv1_weights.npy')