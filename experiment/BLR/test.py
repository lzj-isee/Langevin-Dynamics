import numpy as np
import scipy.sparse as sp
import torch
from  torch.utils.data.dataloader import DataLoader
import torch.sparse as tsp

trainDatas=sp.load_npz('./dataset/trainDatas.npz')
coor=torch.LongTensor([trainDatas.row,trainDatas.col])
data=torch.FloatTensor(trainDatas.data)
size0=trainDatas.shape[0]
size1=trainDatas.shape[1]
trainDatas=torch.sparse.FloatTensor(coor,data,torch.Size([size0,size1]))
trainloader=DataLoader(trainDatas,batch_size=10,shuffle=True)
for i,data in enumerate(trainloader):
    a=data
    b=data.to_dense().clone().detach().numpy()
    pause=1
    


















pause=1
