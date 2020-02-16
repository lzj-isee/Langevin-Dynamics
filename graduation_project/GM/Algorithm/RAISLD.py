from Algorithm.SGLD import Alg_SGLD
import numpy as np

class Alg_RAISLD(Alg_SGLD):
    def __init__(self,curr_x,train_num,alpha,d):
        super(Alg_RAISLD,self).__init__(curr_x)
        self.train_num=train_num
        self.alpha=alpha    # param of exponential moving average
        self.d=d    # param to calc v
        self.r=0    # gain ratio
        self.t=0    # effective iteration number
        self.gn=np.zeros(self.train_num) # gradient norm
        self.v=np.zeros(self.train_num)  # param to update p
        self.p=np.ones(self.train_num)/train_num # Init uniform sample
        self.EgR=0  # param to update r
        self.EgR1=0 # param to update r
        self.EgU1=0 # param to update r
    
    def initialize(self,trainSet):
        grads=self.Grads_Calc_r(self.curr_x,trainSet)
        self.gn=np.linalg.norm(grads,axis=1)

    def average_grads(self):
        w=1/self.p/self.train_num
        self.grad_avg=np.average(self.grads,axis=0,weights=w[self.indices])
        self.EgR=self.EgR*(1-self.alpha)+self.alpha*np.linalg.norm(self.grad_avg)**2

    def update(self):
        batchSize=len(self.indices)
        grads_norm=np.linalg.norm(self.grads,axis=1)
        # update gradient norm
        self.gn[self.indices]=grads_norm
        # update v
        k=np.sqrt(self.train_num/np.sum(self.gn))
        self.v=(1+k*self.d)*self.gn
        # update p
        self.p=self.v/np.sum(self.v)
        # update r
        wR=(1/self.p/self.train_num)**2
        wU=1/self.p/self.train_num
        self.EgR1=self.EgR1*(1-self.alpha)+self.alpha*\
            np.average((grads_norm**2).reshape(-1,1),axis=0,weights=wR[self.indices])
        self.EgU1=self.EgU1*(1-self.alpha)+self.alpha*\
            np.average((grads_norm**2).reshape(-1,1),axis=0,weights=wU[self.indices])
        self.r=1+(self.EgU1-self.EgR1)/self.EgR/batchSize
        self.t+=self.r
    