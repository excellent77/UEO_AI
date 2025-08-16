from typing import Dict, Literal
import torch.optim as optim
from torch.optim import lr_scheduler 



SCH_LIST = (
    "StepLR",
    "MultiStepLR", 
    "ExponentialLR", 
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "OneCycleLR",
    "PolynomialLR",
    "CosineAnnealingWarmRestarts",
    "warmup_scheduler"
)



##################################################################################################
def warmup_lr_scheduler(
        optimizer: optim.Optimizer,  # Optimizer to be wrapped by the scheduler
        warmup_epochs: int = 10,  # Number of warmup epochs
    ) -> lr_scheduler.LRScheduler:
    """
    Linearly ramps up the learning rate within warmup_epochs
    number of epochs.
    """
    # 除以 warmup_epochs 並強制為 float 避免整數除法
    lambda1 = lambda epoch: (epoch + 1) / max(1, warmup_epochs)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # 移除 verbose 參數，避免不支援錯誤
    return scheduler

##################################################################################################
def build_scheduler(
    scheduler_type: Literal[
        "StepLR",
        "MultiStepLR", 
        "ExponentialLR", 
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "CyclicLR",
        "OneCycleLR",
        "PolynomialLR",
        "CosineAnnealingWarmRestarts",
        "warmup_scheduler"
    ],
    optimizer: optim.Optimizer,
    config: Dict = {},
) -> lr_scheduler.LRScheduler:
    """generates the learning rate scheduler

    Args:
        optimizer (optim.Optimizer): pytorch optimizer
        scheduler_type (str): type of scheduler
        config (dict): configuration dictionary for scheduler

    Returns:
        LRScheduler
    """
    if scheduler_type == "warmup_scheduler":
        
        return warmup_lr_scheduler(
            optimizer=optimizer,
            warmup_epochs=config.get("warmup_epochs", 10)
        )

    elif scheduler_type == "StepLR":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "MultiStepLR":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [30, 80]),
            gamma=config.get("gamma", 0.1),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "ExponentialLR":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get("gamma", 0.99),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 50),
            eta_min=config.get("eta_min", 0),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 1),
            eta_min=config.get("eta_min", 0),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CyclicLR":
        return lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.get("base_lr", 0.001),
            max_lr=config.get("max_lr", 0.01),
            step_size_up=config.get("step_size_up", 2000),
            step_size_down=config.get("step_size_down", None),
            mode=config.get("mode", "triangular"),
            gamma=config.get("gamma", 1.0),
            scale_fn=config.get("scale_fn", None),
            scale_mode=config.get("scale_mode", "cycle"),
            cycle_momentum=config.get("cycle_momentum", True),
            base_momentum=config.get("base_momentum", 0.8),
            max_momentum=config.get("max_momentum", 0.9),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "OneCycleLR":
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 0.01),
            total_steps=config.get("total_steps", None),
            epochs=config.get("epochs", 10),
            steps_per_epoch=config.get("steps_per_epoch", None),
            pct_start=config.get("pct_start", 0.3),
            anneal_strategy=config.get("anneal_strategy", "cos"),
            cycle_momentum=config.get("cycle_momentum", True),
            base_momentum=config.get("base_momentum", 0.85),
            max_momentum=config.get("max_momentum", 0.95),
            div_factor=config.get("div_factor", 25.0),
            final_div_factor=config.get("final_div_factor", 10000.0),
            three_phase=config.get("three_phase", False),
            last_epoch=config.get("last_epoch", -1),
        )

    elif scheduler_type == "PolynomialLR":
        return lr_scheduler.PolynomialLR(
            optimizer,
            lr_lambda=config.get("lr_lambda", lambda epoch: (1 - epoch / config.get("max_epochs", 100)) ** 0.9),
            last_epoch=config.get("last_epoch", -1),
        )

    else:
        raise ValueError("Invalid Input -- Check scheduler_type")

