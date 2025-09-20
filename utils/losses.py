from typing import Literal
import torch
import torch.nn as nn



LOSS_LIST = ("MSE", "BCEW", "L1Loss")



#####################################################################
class CrossEntropyLoss(nn.Module):
    '''
    Cross Entropy Loss for classification tasks.
    Uses mean reduction by default.
    Can be used for multi-class classification tasks.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(*args, **kwargs)

    def __call__(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


#####################################################################
class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(*args, **kwargs)

    def __call__(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


#####################################################################
class MSELoss:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._loss = nn.MSELoss(*args, **kwargs)

    def __call__(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


#####################################################################
class L1Loss:
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._loss = nn.L1Loss(*args, **kwargs)

    def __call__(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss



#####################################################################
class Compose_Loss(nn.Module):
    '''
    Composes multiple loss functions into a single loss function.
    '''
    def __init__(self,
                 losses:list,
                 weights:list=None
                 ):
        super().__init__()
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)
        self.weights = torch.tensor(self.weights/sum(self.weights), dtype=torch.float32)

    def __call__(self, predictions, targets):
        total_loss = 0.0
        for i, loss_fn in enumerate(self.losses):
            loss = loss_fn(predictions, targets)
            total_loss += self.weights[i] * loss
        return total_loss
    


def build_loss(
        loss_type: Literal[
            "BCEW",
            "MSE",
            "L1Loss"
        ], # Type of optimizer
        *args, # Additional arguments
        **kwargs, # Additional keyword arguments
    ):
    '''
    builds the loss based on the type and arguments provided
    return:
         the loss function object
    '''

    if loss_type == "BCEW":
        return BinaryCrossEntropyWithLogits(*args, **kwargs)
    elif loss_type == "MSE":
        return MSELoss(*args, **kwargs)
    elif loss_type == "L1Loss":
        return L1Loss(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")