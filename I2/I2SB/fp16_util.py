import torch.nn as nn


def convert_module_to_f16(module: nn.Module) -> None:
    """Recursively convert module parameters/buffers to fp16."""
    module.half()
    for child in module.children():
        convert_module_to_f16(child)


def convert_module_to_f32(module: nn.Module) -> None:
    """Recursively convert module parameters/buffers to fp32."""
    module.float()
    for child in module.children():
        convert_module_to_f32(child)
