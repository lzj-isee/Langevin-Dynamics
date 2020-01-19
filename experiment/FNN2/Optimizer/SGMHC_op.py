import torch 
from torch.optim import Optimizer
import math
import numpy as np
import copy

class SGHMC_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma,name='x')
        self.device=device
        super(SGHMC_op,self).__init__(params,defaults)
        #add another group v
        param_group_v=copy.deepcopy(self.param_groups[0])
        param_group_v['name']='v'
        for param in param_group_v['params']:
            param.requires_grad=False
            param.data=torch.zeros_like(param.data)
        self.param_groups.append(param_group_v)


    def step(self,closure=None,curr_iter_count=0.0):
        loss=None
        if closure is not  None:
            loss=closure()
        
        group_x=self.param_groups[0]
        group_v=self.param_groups[1]
        for param_x, param_v in zip(group_x['params'],group_v['params']):
            if param_x.grad is None:
                continue
            g=param_x.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            # beta=1
            gamma=1/group_x['lr_a'] #tune gamma to get a better performance
            noise=torch.randn_like(param_x)*math.sqrt(2*eta*gamma)
            param_x.data.add_(eta,param_v.data)
            param_v.data=param_v.data-eta*(gamma*param_v.data+g)+noise.data
            

        '''
        # code in SGLD_op
        for param in group_x['params']:
            if param.grad is None:
                continue
            param_grad=param.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            noise=torch.randn_like(param)*math.sqrt(2*eta)
            param.data.add_(-eta,param_grad)
            param.data.add_(noise)
        '''     
        
        return loss
