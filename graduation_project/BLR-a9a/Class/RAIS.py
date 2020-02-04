import torch
import numpy as np
 


class model_RAIS(object):
    def __init__(self,curr_x,train_num,alpha,d,device):
        self.curr_x=curr_x.to(device)
        self.dim=len(curr_x)
        self.train_num=train_num
        self.alpha=alpha
        self.d=d
        self.r=0
        self.t=0
        self.gn=torch.zeros(self.train_num)  # gradient norm
        self.v=torch.zeros(self.train_num)  
        self.p=torch.ones(self.train_num)/train_num # Init uniform sample
        self.EgR=0
        self.EgR1=0
        self.EgU1=0

    def initialize(self,grads):
        self.gn=torch.norm(grads,dim=1)

    def avg_grad(self,grads,indices):
        w=(1/self.p/self.train_num).numpy()
        avg_grad = torch.Tensor(np.average(grads.to('cpu').numpy(),axis=0,weights=w[indices]))
        self.EgR=self.EgR*(1-self.alpha)+self.alpha*torch.norm(avg_grad)**2
        return avg_grad


    def update(self,grads,indices):
        batchSize=len(indices)
        grads_norm=torch.norm(grads,dim=1)
        # update gradient norm
        self.gn[indices]=grads_norm
        # update v
        k=torch.sqrt(self.train_num/torch.sum(self.gn))
        self.v=(1+k*self.d)*self.gn
        # update p
        self.p=self.v/torch.sum(self.v)
        # update r
        wR=((1/self.p/self.train_num)**2).numpy()
        wU=(1/self.p/self.train_num).numpy()
        self.EgR1=self.EgR1*(1-self.alpha)+self.alpha*\
            torch.Tensor(np.average((grads_norm**2).numpy().reshape(-1,1),axis=0,weights=wR[indices]))
        self.EgU1=self.EgU1*(1-self.alpha)+self.alpha*\
            torch.Tensor(np.average((grads_norm**2).numpy().reshape(-1,1),axis=0,weights=wU[indices]))
        self.r=1+(self.EgU1-self.EgR1)/self.EgR/batchSize
        self.t+=self.r

        
        







    
    

    