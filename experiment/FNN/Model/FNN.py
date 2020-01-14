import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self):
        super(FNN,self).__init__()
        self.fc1=nn.Linear(28*28,100)
        self.fc2=nn.Linear(100,10)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    