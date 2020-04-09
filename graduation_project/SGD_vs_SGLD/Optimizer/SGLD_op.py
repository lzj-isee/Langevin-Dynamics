import torch 
from torch.optim import Optimizer
import math
import numpy as np

class SGLD_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma)
        self.device=device
        super(SGLD_op,self).__init__(params,defaults)

    def step(self,closure=None,eta=1.0):
        loss=None
        if closure is not  None:
            loss=closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                param_grad=param.grad.data
                noise=torch.randn_like(param)*math.sqrt(2*eta)
                param.data.add_(-eta,param_grad)
                param.data.add_(noise)     
        return loss