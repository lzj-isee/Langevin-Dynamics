import torch 
from torch.optim import Optimizer
import math
import numpy as np
import copy

class SRM_HMC_op(Optimizer):
    def __init__(self,params,lr_a,lr_gamma,p,friction,device="cpu"):
        defaults=dict(lr_a=lr_a,lr_gamma=lr_gamma,p=p,friction=friction)
        self.device=device
        #self.param_groups[0] is main_model, self.param_groups[1] is last_model
        # and self.param_groups[2] save the v of main_model and it has no need to Calc the grad
        super(SRM_HMC_op,self).__init__(params,defaults)
        param_group_v=copy.deepcopy(self.param_groups[0])
        #append a new param_group to save v and g_k
        param_group_v['name']='main_v'
        for param in param_group_v['params']:
            param.requires_grad=False
            param.grad=torch.zeros_like(param.data)
            param.data=torch.zeros_like(param.data)
        self.param_groups.append(param_group_v)
        #init the grad of last_model to be zero, or it will be None
        for param in self.param_groups[1]['params']:
            param.grad=torch.zeros_like(param.data)



    def step(self,closure=None,curr_iter_count=0.0):# Don't use this function
        loss=None
        if closure is not  None:
            loss=closure()
        group_x=self.param_groups[0]
        group_last=self.param_groups[1]
        group_v=self.param_groups[2]
        for param_x, param_last, param_v in zip(group_x['params'],group_last['params'],group_v['params']):
            if param_x.grad is None:
                continue
            g=param_v.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            gamma=group_x['friction']/group_x['lr_a']# gamma=1
            # save x_k to update param_last
            param_last.data=param_x.data
            #update
            '''
            u=1
            shape=param_x.data.shape
            noise_x,noise_v=self._gen_noise(shape,eta,gamma,u)
            param_x.data=param_x.data+gamma*(1-np.exp(-gamma*eta))*param_v.data+u*gamma**(-2)*(gamma*eta+np.exp(-gamma*eta)-1)*grad+noise_x
            param_v.data=param_v.data*np.exp(-gamma*eta)-u*gamma**(-1)*(1-np.exp(-gamma*eta))*grad+noise_v
            '''
            noise=torch.randn_like(param_x)*math.sqrt(2*eta*gamma)
            param_x.data.add_(eta,param_v.data)
            param_v.data=param_v.data-eta*(gamma*param_v.data+g)+noise.data
            #Note: param_groups_v save g_k
            rho=1/(curr_iter_count)**group_x['p']
            param_v.grad.data=param_x.grad.data+(1-rho)*(param_v.grad.data-param_last.grad.data)
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

    def init_grad(self):
        group_x=self.param_groups[0]
        group_v=self.param_groups[2]
        for param_x, param_v in zip(group_x['params'],group_v['params']):
            if param_x.grad is None:
                continue
            param_v.grad.data=param_x.grad.data
        
    def updata_x_v(self,curr_iter_count):
        group_x=self.param_groups[0]
        group_last=self.param_groups[1]
        group_v=self.param_groups[2]
        for param_x, param_last, param_v in zip(group_x['params'],group_last['params'],group_v['params']):
            if param_x.grad is None:
                continue
            g=param_v.grad.data
            eta=group_x['lr_a']*(curr_iter_count)**(-group_x['lr_gamma'])
            gamma=group_x['friction']/group_x['lr_a']# gamma=1
            # save x_k to update param_last
            param_last.data=param_x.data
            #update
            noise=torch.randn_like(param_x)*math.sqrt(2*eta*gamma)
            param_x.data.add_(eta,param_v.data)
            param_v.data=param_v.data-eta*(gamma*param_v.data+g)+noise.data
    
    def updata_g(self,curr_iter_count):
        group_x=self.param_groups[0]
        group_last=self.param_groups[1]
        group_v=self.param_groups[2]
        for param_x, param_last, param_v in zip(group_x['params'],group_last['params'],group_v['params']):
            if param_x.grad is None:
                continue
            #Note: param_groups_v save g_k
            rho=1/(curr_iter_count)**group_x['p']
            param_v.grad.data=param_x.grad.data+(1-rho)*(param_v.grad.data-param_last.grad.data)

