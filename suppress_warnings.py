"""
TensorFlowの警告を抑制するユーティリティ
"""

import warnings
import os

def suppress_tf_warnings():
    """TensorFlowの各種警告を抑制"""
    
    # Protobuf警告を抑制
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
    
    # TensorFlow のログレベルを設定（0=全表示, 1=INFO非表示, 2=WARNING非表示, 3=ERROR非表示）
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # TensorFlowの警告を抑制
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

# 使用例：
# スクリプトの最初に以下を追加
# from suppress_warnings import suppress_tf_warnings
# suppress_tf_warnings()