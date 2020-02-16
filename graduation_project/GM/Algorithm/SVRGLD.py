from Algorithm.SGLD import Alg_SGLD
import numpy as np

class Alg_SVRGLD(Alg_SGLD):
    def __init__(self,curr_x,snap_x):
        self.snap_x=snap_x
        super(Alg_SVRGLD,self).__init__(curr_x)
    
    def update(self,trainSet):
        # Update grad_alpha and snap_x
        grads=self.Grads_Calc_r(self.curr_x,trainSet)
        self.grad_alpha=grads.mean(0)
        self.snap_x=self.curr_x
    
    def snap_Grads_Calc(self):
        # Calculate the gradients of snap_x
        grads=self.Grads_Calc_r(self.snap_x,self.datas)
        self.grad_snap=grads.mean(0)