import torch
import numpy as np


class AlphaZeroLoss(torch.nn.Module):
    
    def __init__(self, c: float = 1):
        super(AlphaZeroLoss, self).__init__()
        self.c = c

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        d: torch.Tensor = y_pred - y_true
        z: torch.Tensor = (d / self.c).pow(2) + 1
        return torch.mean(torch.log(z))
    
class MSELoss(torch.nn.Module):
    
    def __init__(self, c: float = 1):
        super(MSELoss, self).__init__()
        self.c = c
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        d: torch.Tensor = y_pred - y_true
        return torch.mean(0.5 * (d / self.c).pow(2))
    
class WelschLoss(torch.nn.Module):
    
    def __init__(self, c: float = 1):
        super(WelschLoss, self).__init__()
        self.c = c
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        d: torch.Tensor = y_pred - y_true
        return torch.mean(1 - torch.exp(-0.5 * (d / self.c).pow(2)))
    
class GeneralLoss(torch.nn.Module):
    
    def __init__(self, alpha: float = 2, c = 1):
        super(GeneralLoss, self).__init__()
        self.alpha = alpha
        self.c = c
        self.a = abs(self.alpha - 2) / self.alpha
        self.b = abs(self.alpha - 2)
            
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        d: torch.Tensor = ( (y_pred - y_true) / self.c ).pow(2)
        e: torch.Tensor = self.a * ( ( d / self.b + 1 ).pow(self.alpha / 2) - 1 )
        return torch.mean(e)
        

class RobustGeneralLoss(torch.nn.Module):
    
    def __init__(self, loss_type: str = "none", alpha: float = 2, c = 1):
        super(RobustGeneralLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
        if loss_type == 'mse' or np.isclose(alpha, 2):
            self.loss = torch.nn.MSELoss(c)
        elif loss_type == 'cauchy' or np.isclose(alpha, 0):
            self.loss = AlphaZeroLoss(c)
        elif loss_type == 'welsch' or np.isclose(alpha, abs(np.inf)):
            self.loss = WelschLoss(c)
        else:
            self.loss = GeneralLoss(alpha, c)
            
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.loss(y_pred, y_true)
    
    
if __name__ == "__main__":
    
    c = 10
    
    y_pred = torch.randn(5, 2, 3) * 10
    y_true = torch.randn(5, 2, 3) * 10
    
    loss = RobustGeneralLoss(loss_type = "none", alpha = 0, c = c)
    print(loss(y_pred, y_true))
    
    loss = RobustGeneralLoss(loss_type = "none", alpha = 2, c = c)
    print(loss(y_pred, y_true))
    
    loss = RobustGeneralLoss(loss_type = "none", alpha = 3, c = c)
    print(loss(y_pred, y_true))
    
    loss = RobustGeneralLoss(loss_type = "none", alpha = np.inf, c = c)
    print(loss(y_pred, y_true))