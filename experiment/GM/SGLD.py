import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
from  functions import*
torch.manual_seed(2020)
path='./result_samples/SGLD/samples.npy'

a=torch.Tensor(np.load('./dataset/a.npy'))

num_epoch=20000
batchSize=10
dim=2
factor_a=1
factor_b=0
factor_gamma=0.1

maxIteration=num_epoch*len(a)/batchSize
datas=DataLoader(a,batch_size=batchSize,shuffle=True)
x_list=[]
x=torch.ones(dim,requires_grad=True)*0

for epoch in tqdm(range(num_epoch)):
    for i ,data in enumerate(datas):
        grads=torch.zeros(dim)
        for  j in range(len(data)):
            grads+=ng_grad_f(x,a[j])
        grad_avg=grads/len(data)
        eta=factor_a*(factor_b+epoch*len(a)+i+1)**(-factor_gamma)
        noise=np.sqrt(2*eta)*torch.randn(2)
        x=x-eta*grad_avg+noise
        x_list.append(x.clone().detach().numpy())

np.save(path,np.array(x_list))
