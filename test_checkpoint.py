#!/usr/bin/env python3
import torch
from full_model_test import OCRModel

def test_save_load():
    """モデルの保存と読み込みテスト"""
    print("=== モデル保存/読み込みテスト ===")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # モデル作成
    model1 = OCRModel().to(device)
    
    # ダミーの入力でforward
    dummy_input = torch.randn(1, 3, 480, 640).to(device)
    with torch.no_grad():
        output1 = model1(dummy_input)
    
    # チェックポイント作成
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model1.state_dict(),
        'test_value': 12345
    }
    
    # 保存
    import os
    os.makedirs('test_checkpoints', exist_ok=True)
    save_path = 'test_checkpoints/test.pth'
    torch.save(checkpoint, save_path)
    print(f"✓ モデルを保存: {save_path}")
    
    # 新しいモデルを作成して読み込み
    model2 = OCRModel().to(device)
    
    # 読み込み前の出力（異なるはず）
    with torch.no_grad():
        output2_before = model2(dummy_input)
    
    # チェックポイント読み込み
    loaded_checkpoint = torch.load(save_path, weights_only=False)
    model2.load_state_dict(loaded_checkpoint['model_state_dict'])
    print(f"✓ モデルを読み込み: {save_path}")
    print(f"✓ エポック: {loaded_checkpoint['epoch']}")
    print(f"✓ テスト値: {loaded_checkpoint['test_value']}")
    
    # 読み込み後の出力（同じはず）
    with torch.no_grad():
        output2_after = model2(dummy_input)
    
    # 出力の比較
    diff_before = torch.abs(output1['objectness'] - output2_before['objectness']).max()
    diff_after = torch.abs(output1['objectness'] - output2_after['objectness']).max()
    
    print(f"\n出力の差:")
    print(f"読み込み前: {diff_before:.6f} (異なる)")
    print(f"読み込み後: {diff_after:.6f} (同じはず)")
    
    if diff_after < 1e-6:
        print("\n✓ 保存と読み込みが正常に動作しています！")
    else:
        print("\n✗ 保存と読み込みに問題があります")
    
    # ファイルサイズ確認
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nチェックポイントサイズ: {file_size:.2f} MB")

if __name__ == "__main__":
    test_save_load()