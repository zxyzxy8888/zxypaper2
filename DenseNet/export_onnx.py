"""
ONNX模型导出脚本
使用前请确保已安装: pip install onnxscript onnx
"""

import torch
from model import DenseNet3D

def export_to_onnx(model_path='densenet3d_model.pth', output_path='densenet3d_model.onnx'):
    """导出模型为ONNX格式"""
    
    # 创建模型实例
    model = DenseNet3D(num_classes=2, in_channels=2)
    
    # 加载预训练权重
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 2, 128, 128, 128)
    
    # 导出为ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['logits', 'features'],
            opset_version=18,  # 使用最新的opset版本
            do_constant_folding=True,
            verbose=False
        )
        print(f"✓ 模型已成功导出为ONNX格式: {output_path}")
        return True
    except Exception as e:
        print(f"✗ 导出失败: {e}")
        return False

if __name__ == '__main__':
    export_to_onnx()
