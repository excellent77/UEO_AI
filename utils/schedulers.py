from typing import Dict, Literal
import torch.optim as optim
from torch.optim import lr_scheduler 



SCH_LIST = (
    "WarmupReduceLROnPlateau",
    "StepLR",
    "MultiStepLR", 
    "ExponentialLR", 
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "OneCycleLR",
    "PolynomialLR",
    "CosineAnnealingWarmRestarts"
)


class SchedulerWrapper:
    """
    包裝一個 PyTorch scheduler，使其 step 方法可以接受額外的參數而不報錯。
    """
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def step(self, *args, **kwargs):
        # 只呼叫原 scheduler 的 step，忽略所有傳入的參數
        self._scheduler.step()

    def __getattr__(self, name):
        # 將所有其他屬性請求代理到原始的 scheduler 物件
        return getattr(self._scheduler, name)

##################################################################################################
class WarmupReduceLROnPlateau:
    """
    一個結合了線性 Warmup 和 ReduceLROnPlateau 的排程器。
    在前 warmup_epochs 個週期內，學習率會線性增加。
    之後，它會切換到 ReduceLROnPlateau 的行為。
    """
    def __init__(self, optimizer, warmup_epochs, reduce_lr_config):
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        self.main_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **reduce_lr_config)
        self._optimizer = optimizer
        self.last_epoch = -1

    def step(self, metrics=None):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        elif metrics is not None:
            self.main_scheduler.step(metrics)

    def get_last_lr(self):
        return self._optimizer.param_groups[0]['lr']

    # 為了相容性，提供 state_dict 和 load_state_dict
    def state_dict(self):
        return {
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'main_scheduler': self.main_scheduler.state_dict(),
            'last_epoch': self.last_epoch
        }

    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict['warmup_scheduler'])
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        self.last_epoch = state_dict['last_epoch']

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
        "warmup_scheduler",
        "WarmupReduceLROnPlateau"
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
    scheduler = None
    if scheduler_type == "WarmupReduceLROnPlateau":
        reduce_lr_config = {
            'mode': config.get('mode', 'min'),
            'factor': config.get('factor', 0.5),
            'patience': config.get('patience', 3)
        }
        return WarmupReduceLROnPlateau(
            optimizer=optimizer,
            warmup_epochs=config.get("warmup_epochs", 30),
            reduce_lr_config=reduce_lr_config
        ) # 這個類別已經處理了 metrics，不需要包裝

    elif scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get("milestones", [30, 80]),
            gamma=config.get("gamma", 0.1),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get("gamma", 0.99),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 50),
            eta_min=config.get("eta_min", 0),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("T_0", 10),
            T_mult=config.get("T_mult", 1),
            eta_min=config.get("eta_min", 0),
            last_epoch=config.get("last_epoch", -1),
        )
    
    elif scheduler_type == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(
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
        scheduler = lr_scheduler.OneCycleLR(
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
        scheduler = lr_scheduler.PolynomialLR(
            optimizer,
            lr_lambda=config.get("lr_lambda", lambda epoch: (1 - epoch / config.get("max_epochs", 100)) ** 0.9),
            last_epoch=config.get("last_epoch", -1),
        )

    else:
        raise ValueError("Invalid Input -- Check scheduler_type")
    
    return SchedulerWrapper(scheduler)
