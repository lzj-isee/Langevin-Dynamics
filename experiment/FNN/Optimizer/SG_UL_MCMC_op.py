import torch 
from torch.optim import Optimizer
import math
import numpy as np
import copy

class SG_UL_MCMC_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma)
        self.device=device
        #self.param_groups[0] is main_x, self.param_groups[1] save main_v
        super(SG_UL_MCMC_op,self).__init__(params,defaults)
        param_group_v=copy.deepcopy(self.param_groups[0])
        #append a new param_group to save v
        param_group_v['name']='v'
        for param in param_group_v['params']:
            param.requires_grad=False
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
            grad=param_x.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            shape=param_x.data.shape
            # default:  gamma=1 and  u=1
            gamma=1
            u=1
            noise_x,noise_v=self._gen_noise(shape,eta,gamma,u)
            #update
            param_x.data=param_x.data+\
                gamma*(1-np.exp(-gamma*eta))*param_v.data+\
                u*gamma**(-2)*(gamma*eta+np.exp(-gamma*eta)-1)*grad+noise_x
            param_v.data=param_v.data*np.exp(-gamma*eta)-\
                u*gamma**(-1)*(1-np.exp(-gamma*eta))*grad+noise_v
        return loss

    def _gen_noise(self,shape,eta,gamma,u):
        cov=np.ones((2,2))*\
            (u*gamma**(-1)*(1-2*np.exp(-gamma*eta)+np.exp(-2*gamma*eta)))
        stdx=np.ones(1)*\
            (u*gamma**(-2)*(2*gamma*eta+4*np.exp(-gamma*eta)-np.exp(-2*gamma*eta)-3))
        stdv=np.ones(1)*\
            (u*(1-np.exp(-2*gamma*eta)))
        std=np.concatenate((stdx,stdv))
        m=cov-np.diag(np.diag(cov))+np.diag(std)
        noise=np.random.multivariate_normal(np.zeros(2),m,shape)
        if len(shape)==1:
            return torch.Tensor(noise[:,0]), torch.Tensor(noise[:,1])
        else:
            return torch.Tensor(noise[:,:,0]), torch.Tensor(noise[:,:,1])

        