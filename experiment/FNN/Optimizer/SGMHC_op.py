import torch 
from torch.optim import Optimizer
import math
import numpy as np

class SGHMC_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma)
        self.device=device
        super(SGHMC_op,self).__init__(params,defaults)

    def step(self,closure=None,curr_iter_count=0.0):
        loss=None
        if closure is not  None:
            loss=closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                param_grad=param.grad.data
                eta=group['lr_a']*(curr_iter_count)**(-group['lr_gamma'])
                noise=torch.randn_like(param)*math.sqrt(2*eta)
                param.data.add_(-eta,param_grad)
                param.data.add_(noise)
                
        
        return loss
