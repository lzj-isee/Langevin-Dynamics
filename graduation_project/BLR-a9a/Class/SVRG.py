import torch
 


class model_SVRG(object):
    def __init__(self,curr_x,last_x,device):
        self.curr_x=curr_x.to(device)
        self.snap_x=last_x.to(device)
        self.dim=len(curr_x)
        self.alpha=torch.zeros(self.dim).to(device)
    