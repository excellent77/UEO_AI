from typing import Dict, Literal
import torch.optim as optim
import torch.nn as nn



# 獲取 optimizers.py 中定義的所有優化器名稱
OPTIM_LIST = ("adam", "adamw", "sgd")



def build_optimizer(
        optimizer_type: Literal["adam", "adamw", "sgd"], # Type of optimizer,
        model: nn.Module, # Model to optimize
        *args, # Additional arguments
        **kwargs, # Additional keyword arguments
    ) -> optim.Optimizer:
    '''
    builds the optimizer based on the type and arguments provided
    return:
         the optimizer
    '''

    if optimizer_type == "adam":
        return optim.Adam(
            model.parameters(),
            *args,
            **kwargs
            )
    
    elif optimizer_type == "adamw":
        return optim.AdamW (
            model.parameters(),
            *args,
            **kwargs
            )
    
    elif optimizer_type == "sgd":
        return optim.SGD(
            model.parameters(),
            *args,
            **kwargs
            )
    
    else:
        raise ValueError("must be adam or adamw for now")



if __name__ == '__main__':
    pass
