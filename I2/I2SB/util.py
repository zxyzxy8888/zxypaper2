import numpy as np
import torch
from pathlib import Path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def unsqueeze_xdim(x, xdim):
    """将形状 [B] 的张量扩展到 [B, 1, 1, ...] 以匹配数据维度。"""
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if x.dim() == 0:
        x = x.unsqueeze(0)
    return x.view(*x.shape, *([1] * len(xdim)))


def space_indices(total_steps, num_indices):
    """在 [0, total_steps-1] 上等间隔采样索引，包含两端。"""
    if total_steps <= 0:
        return []
    if num_indices <= 1:
        return [0]

    idx = np.linspace(0, total_steps - 1, num_indices, dtype=int)
    idx = np.unique(idx)
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != total_steps - 1:
        idx = np.append(idx, total_steps - 1)
    return idx.tolist()


class _WriterAdapter:
    def __init__(self, writer):
        self.writer = writer

    def add_scalar(self, step, tag, value):
        self.writer.add_scalar(tag, float(value), int(step))

    def add_image(self, step, tag, image):
        self.writer.add_image(tag, image, int(step))

    def close(self):
        self.writer.close()


class _NoOpWriter:
    def add_scalar(self, step, tag, value):
        return None

    def add_image(self, step, tag, image):
        return None

    def close(self):
        return None


def build_log_writer(opt):
    """构建与 Runner 兼容的 TensorBoard writer。"""
    try:
        from torch.utils.tensorboard import SummaryWriter

        default_dir = Path(opt.ckpt_path) / "tb"
        log_dir = Path(getattr(opt, "tb_logdir", default_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        return _WriterAdapter(SummaryWriter(log_dir=str(log_dir)))
    except Exception:
        return _NoOpWriter()
