import torch 
from torch.optim import Optimizer
import math
import numpy as np
import copy

class SVRG_LD_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma)
        self.device=device
        super(SVRG_LD_op,self).__init__(params,defaults)
        '''
        #add another group v
        param_group_v=copy.deepcopy(self.param_groups[0])
        param_group_v['name']='v'
        for param in param_group_v['params']:
            param.requires_grad=False
        self.param_groups.append(param_group_v)
        '''


    def step(self,closure=None,curr_iter_count=0.0):
        loss=None
        if closure is not  None:
            loss=closure()
        '''
        group_x=self.param_groups[0]
        group_v=self.param_groups[1]
        for param_x, param_v in zip(group_x['params'],group_v['params']):
            if param_x.grad is None:
                continue
            g=param_x.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            # gamma=1, beta=1
            noise=torch.randn_like(param_x)*math.sqrt(2*eta)
            param_v.data=param_v.data-eta*(1*param_v.data+g)+noise.data
            param_x.data.add_(eta,param_v.data)
        '''
        
        return loss
