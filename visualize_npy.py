import numpy as np
import matplotlib.pyplot as plt

# NPYファイルを読み込み
weights = np.load('conv1_weights.npy')

# 最初の8個のフィルタを可視化
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(8):
    # RGB 3チャンネルの平均を取って2D表示
    filter_avg = weights[:, :, :, i].mean(axis=2)
    
    im = axes[i].imshow(filter_avg, cmap='RdBu', vmin=-0.15, vmax=0.15)
    axes[i].set_title(f'Filter {i}')
    axes[i].axis('off')

plt.colorbar(im, ax=axes)
plt.suptitle('Conv1層のフィルタ（最初の8個）')
plt.tight_layout()
plt.savefig('conv1_filters.png')
plt.show()
print("フィルタの可視化を 'conv1_filters.png' に保存しました")