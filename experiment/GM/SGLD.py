import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
from  functions import*
import os
torch.manual_seed(2020)
if not os.path.exists('./result_samples/SGLD'):
    os.makedirs('./result_samples/SGLD')
path='./result_samples/SGLD/samples.npy'

a=torch.Tensor(np.load('./dataset/a.npy'))

num_epoch=20000
batchSize=10
dim=2
factor_a=1
factor_b=0
factor_gamma=0.1
#----------------------------------------------------------------------------
datas=DataLoader(a,batch_size=batchSize,shuffle=True)
x_list=[]
x=np.ones(dim)*0

for epoch in tqdm(range(num_epoch)):
    for i ,data in enumerate(datas):
        grads=grad_f(x,data.clone().detach().numpy().astype('float64'))
        grad_avg=grads.mean(0)
        eta=factor_a*(factor_b+epoch*len(a)+i+1)**(-factor_gamma)
        noise=(np.sqrt(2*eta)*torch.randn(2)).clone().detach().numpy()
        x=x-eta*grad_avg+noise
        x_list.append(x)

np.save(path,np.array(x_list))
