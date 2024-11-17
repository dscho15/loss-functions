import torch
import einops
import math

class MixtureOf2LaplaceLoss(torch.nn.Module):
    
    def __init__(self, reduce = 'mean'):
        super(MixtureOf2LaplaceLoss, self).__init__()
        self.min = torch.tensor([-1])
        self.max = torch.tensor([3])
        self.reduce = reduce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):     
        mu = y_pred[:, :3, :, :]
        alpha = y_pred[:, 3:5, :, :]
        beta = y_pred[:, 5:, :, :]
        
        weight = torch.softmax(alpha, dim=1)
                
        if self.min.device != alpha.device:
            self.min = self.min.to(alpha.device)
            self.max = self.max.to(alpha.device)
        
        beta2 = torch.clamp(beta, self.min, self.max)
        beta1 = torch.ones_like(beta) * self.min
        beta = torch.cat([beta1, beta2], dim=1)
        
        term2 = (y_true - mu).abs().unsqueeze(2) * torch.exp(-beta).unsqueeze(1)
        term1 = weight - math.log(2) - beta
        
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        
        return torch.mean(nf_loss)