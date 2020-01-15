import torch 
from torch.optim import Optimizer
import math
import numpy as np
import copy

class SVRG_LD_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma)
        self.device=device
        #self.param_groups[0] is main_model, self.param_groups[1] is snapshot_model
        # and self.param_groups[2] save the all_grad for snapshot_model and it doesn't need to calc the grad, just save
        super(SVRG_LD_op,self).__init__(params,defaults)
        param_group_all=copy.deepcopy(self.param_groups[1])
        param_group_all['name']='all'
        for param in param_group_all['params']:
            param.requires_grad=False
            param.grad=torch.zeros_like(param.data)
        self.param_groups.append(param_group_all)



    def step(self,closure=None,curr_iter_count=0.0):
        loss=None
        if closure is not  None:
            loss=closure()
        group_x=self.param_groups[0]
        group_snapshot=self.param_groups[1]
        group_all=self.param_groups[2]
        for param_x, param_snapshot, param_all in zip(group_x['params'],group_snapshot['params'],group_all['params']):
            if param_x.grad is None:
                continue
            g=param_x.grad.data
            alpha_snap=param_snapshot.grad.data
            alpha_all=param_all.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            noise=torch.randn_like(param_x)*math.sqrt(2*eta)
            grad=g-alpha_snap+alpha_all
            param_x.data.add_(-eta,grad)
            param_x.data.add_(noise)
        
        return loss

    def updata_snapshot_grad(self):
        group_snapshot=self.param_groups[1]
        group_all=self.param_groups[2]
        for param_snapshot,param_all in zip(group_snapshot['params'],group_all['params']):
            if param_snapshot.grad is None:
                continue
            # save the grad in self.param_groups[2] from snapshot_model
            param_all.grad.data=param_snapshot.grad.data
        
    def update_snapshot_weight(self):
        group_x=self.param_groups[0]
        group_snapshot=self.param_groups[1]
        for param_x, param_snapshot in zip(group_x['params'],group_snapshot['params']):
            param_snapshot.data=param_x.data