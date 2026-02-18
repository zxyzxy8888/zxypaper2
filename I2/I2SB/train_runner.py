import argparse
import logging
import os
import random
import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MRIPETDataset
from split_dataset import get_paired_files_with_subjects, split_by_subject
from runner import Runner


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("I2SBRunner")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def build_opt(args, device, num_itr, ckpt_path: Path, run_stamp: str):
    beta_max = args.beta_max
    if beta_max is None:
        beta_max = args.beta_end * args.timesteps

    # Runner 依赖的配置字段
    opt = argparse.Namespace(
        device=device,
        global_rank=0,
        distributed=False,
        ckpt_path=ckpt_path,
        load=args.load,
        use_fp16=args.use_fp16,
        ema=args.ema,
        lr=args.learning_rate,
        l2_norm=args.weight_decay,
        lr_gamma=args.lr_gamma,
        lr_step=args.lr_step,
        num_itr=num_itr,
        interval=args.timesteps,
        beta_max=beta_max,
        t0=args.t0,
        T=args.T,
        ot_ode=args.ot_ode,
        eval_nfe=args.eval_nfe,
        run_stamp=run_stamp,
        tb_logdir=ckpt_path / f"tb_{run_stamp}",
    )
    return opt


def main():
    parser = argparse.ArgumentParser(description="Runner 入口：标准 I2SB MRI->PET 训练/验证/测试")
    parser.add_argument("--mri_dir", type=str, required=True, help="MRI目录")
    parser.add_argument("--pet_dir", type=str, required=True, help="PET目录")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV路径")

    # 兼容你旧命令，当前 runner 流程不使用 mask_dir
    parser.add_argument("--mask_dir", type=str, default=None, help="兼容参数，当前脚本不使用")

    parser.add_argument("--output_dir", type=str, default=r"D:\zxyself\output", help="输出目录")
    parser.add_argument("--exp_name", type=str, default="I2SB_runner", help="实验名")
    parser.add_argument("--run_stamp", type=str, default=None, help="运行时间戳，默认自动生成")

    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="训练 batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数")

    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--lr_gamma", type=float, default=1.0, help="StepLR gamma；1.0=关闭")
    parser.add_argument("--lr_step", type=int, default=1000, help="StepLR step_size")

    parser.add_argument("--timesteps", type=int, default=700, help="扩散步数")
    parser.add_argument("--beta_end", type=float, default=2e-2, help="beta schedule 终值")
    parser.add_argument("--beta_max", type=float, default=None, help="若给定则覆盖 beta_end*timesteps")
    parser.add_argument("--t0", type=float, default=1e-4, help="noise_levels 起点")
    parser.add_argument("--T", type=float, default=1.0, help="noise_levels 终点")
    parser.add_argument("--ema", type=float, default=0.999, help="EMA 衰减")

    parser.add_argument("--image_size", type=int, default=96, help="立方体目标尺寸")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--eval_nfe", type=int, default=49, help="验证/测试采样步数")
    parser.add_argument("--ot_ode", action="store_true", help="使用 ODE 模式")
    parser.add_argument("--use_fp16", action="store_true", help="启用 fp16")

    parser.add_argument("--load", type=str, default="", help="加载 checkpoint 路径")
    parser.add_argument("--test_only", action="store_true", help="仅测试，不训练")

    args = parser.parse_args()

    set_seed(args.seed)
    run_stamp = args.run_stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.output_dir) / args.exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    log = setup_logger(ckpt_path / f"train_{run_stamp}.log")
    log.info("Preparing dataset...")

    paired_files, subject_to_files = get_paired_files_with_subjects(args.mri_dir, args.pet_dir, args.csv_path)
    train_files, val_files, test_files = split_by_subject(
        paired_files,
        subject_to_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
    )

    target_shape = (args.image_size, args.image_size, args.image_size)
    train_dataset = MRIPETDataset(train_files, target_shape=target_shape)
    val_dataset = MRIPETDataset(val_files, target_shape=target_shape)
    test_dataset = MRIPETDataset(test_files, target_shape=target_shape)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    if len(train_loader) == 0:
        raise RuntimeError("Train loader is empty. Please check dataset paths and split ratios.")

    num_itr = args.num_epochs * len(train_loader)
    opt = build_opt(args, device, num_itr, ckpt_path, run_stamp)

    log.info(
        f"Run config | stamp={run_stamp} | device={device} | epochs={args.num_epochs} | num_itr={num_itr} | "
        f"batch={args.batch_size} | timesteps={args.timesteps}"
    )

    runner = Runner(opt, log)

    if not args.test_only:
        runner.train(opt, train_loader, val_loader)

    runner.test(test_loader)


if __name__ == "__main__":
    main()
