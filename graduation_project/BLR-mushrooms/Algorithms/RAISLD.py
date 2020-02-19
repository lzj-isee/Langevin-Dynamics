from Algorithms.SGLD import Alg_SGLD
import torch
import numpy as np
import scipy.sparse

class Alg_RAISLD(Alg_SGLD):
    def __init__(self,curr_x,train_num,alpha,d,device):
        super(Alg_RAISLD,self).__init__(curr_x,device)
        self.train_num=train_num
        self.alpha=alpha    # param of exponential moving average
        self.d=d    # param to calc v
        self.r=0    # gain ratio
        self.t=0    # effective iteration number
        self.gn=torch.zeros(self.train_num) # gradient norm
        self.v=torch.zeros(self.train_num)  # param to update p
        self.p=torch.ones(self.train_num)/train_num # Init uniform sample
        self.EgR=0  # param to update r
        self.EgR1=0 # param to update r
        self.EgU1=0 # param to update r

    def initialize(self,trainSet):
        Labels,Features=self.Datas_Transform(trainSet)
        grads=self.Grads_Calc_r(Features,Labels)
        self.gn=torch.norm(grads,dim=1)

    def average_grads(self):
        w=(1/self.p/self.train_num).numpy()
        self.grad_avg = torch.Tensor(\
            np.average(self.grads.to('cpu').numpy(),axis=0,weights=w[self.indices]))
        self.EgR=self.EgR*(1-self.alpha)+self.alpha*torch.norm(self.grad_avg)**2


    def update(self):
        batchSize=len(self.indices)
        grads_norm=torch.norm(self.grads,dim=1)
        # update gradient norm
        self.gn[self.indices]=grads_norm
        # update v
        k=torch.sqrt(self.train_num/torch.sum(self.gn))
        self.v=(1+k*self.d)*self.gn
        # update p
        self.p=self.v/torch.sum(self.v)
        # update r
        wR=((1/self.p/self.train_num)**2).numpy()
        wU=(1/self.p/self.train_num).numpy()
        self.EgR1=self.EgR1*(1-self.alpha)+self.alpha*\
            torch.Tensor(\
                np.average((grads_norm**2).numpy().reshape(-1,1),axis=0,weights=wR[self.indices]))
        self.EgU1=self.EgU1*(1-self.alpha)+self.alpha*\
            torch.Tensor(\
                np.average((grads_norm**2).numpy().reshape(-1,1),axis=0,weights=wU[self.indices]))
        self.r=1+(self.EgU1-self.EgR1)/self.EgR/batchSize
        self.t+=self.r