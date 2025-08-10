from typing import Dict, Literal
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


##################################################################################################
def warmup_lr_scheduler(
        config : Dict[str, Dict[str, str]],  # Configurations for the scheduler
        optimizer: optim.Optimizer  # Optimizer to be wrapped by the scheduler
    ) -> LRScheduler:
    """
    Linearly ramps up the learning rate within warmup_epochs
    number of epochs.
    """
    warmup_epochs = int(config["warmup_scheduler"]["warmup_epochs"])
    # 除以 warmup_epochs 並強制為 float 避免整數除法
    lambda1 = lambda epoch: (epoch + 1) / max(1, warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # 移除 verbose 參數，避免不支援錯誤
    return scheduler


##################################################################################################
def training_lr_scheduler(
        config: Dict[str, Dict[str, str]],  # Configurations for the scheduler
        optimizer : optim.Optimizer  # Optimizer to be wrapped by the scheduler
        ) -> LRScheduler:
    """
    Wraps a normal scheduler based on config
    """
    scheduler_type = config["train_scheduler"]["scheduler_type"].lower()

    if scheduler_type == "reduceonplateau" or scheduler_type == "reducelronplateau":
        # 支援 "reduceonplateau" 與錯字 "reducelronplateau" 都容錯
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            mode=config["train_scheduler"].get("mode", "min"),
            patience=int(config["train_scheduler"].get("patience", 10)),
            # verbose 參數依 PyTorch 版本調整，必要時可移除
            verbose=False,
            min_lr=float(config["train_scheduler"]["scheduler_args"].get("min_lr", 0)),
        )
        return scheduler

    elif scheduler_type == "cosine_annealing_wr":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config["train_scheduler"]["scheduler_args"].get("t_0_epochs", 10)),
            T_mult=int(config["train_scheduler"]["scheduler_args"].get("t_mult", 1)),
            eta_min=float(config["train_scheduler"]["scheduler_args"].get("min_lr", 0)),
            last_epoch=-1,
            # 同樣移除 verbose 避免錯誤，最新版有支援可加回
        )
        return scheduler

    else:
        raise NotImplementedError(f"Specified Scheduler '{scheduler_type}' Is Not Implemented")


##################################################################################################
def build_scheduler(
    scheduler_type: Literal["warmup_scheduler", "training_scheduler"],
    optimizer: optim.Optimizer,
    config: Dict[str, Dict[str, str]],
) -> LRScheduler:
    """generates the learning rate scheduler

    Args:
        optimizer (optim.Optimizer): pytorch optimizer
        scheduler_type (str): type of scheduler
        config (dict): configuration dictionary for scheduler

    Returns:
        LRScheduler
    """
    if config is None:
        raise ValueError("Config dictionary cannot be None")

    if scheduler_type == "warmup_scheduler":
        return warmup_lr_scheduler(config=config, optimizer=optimizer)

    elif scheduler_type == "training_scheduler":
        return training_lr_scheduler(config=config, optimizer=optimizer)

    else:
        raise ValueError("Invalid Input -- Check scheduler_type")

