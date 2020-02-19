from Algorithms.SGLD import Alg_SGLD
import torch
import numpy as np
import scipy.sparse

class Alg_SVRGLD(Alg_SGLD):
    def __init__(self,curr_x,snap_x,device):
        self.snap_x=snap_x.to(device)
        super(Alg_SVRGLD,self).__init__(curr_x,device)
    
    def update(self,trainSet):
        # Update grad_alpha and snap_x
        # If memory is not enough, rewrite this function
        Labels,Features=self.Datas_Transform(trainSet)
        Z= -torch.matmul(Features,self.curr_x)*Labels
        A= torch.sigmoid(Z)
        grads= -(A*Labels).view(-1,1)*Features
        self.grad_alpha=grads.mean(0)
        self.snap_x=self.curr_x

    def snap_Grads_Calc(self):
        # Calculate the gradients of snap_x
        Z= -torch.matmul(self.Features,self.snap_x)*self.Labels
        A= torch.sigmoid(Z)
        self.grad_snap= (-(A*self.Labels).view(-1,1)*self.Features).mean(0)



