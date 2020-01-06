import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
from  torch.utils.data.sampler import RandomSampler
from  functions import*
torch.manual_seed(2020)
path='./result_samples/SG_UL_MCMC/samples.npy'

a=torch.Tensor(np.load('./dataset/a.npy'))

num_epoch=20000
batchSize=10
dim=2
factor_a=0.35
factor_b=0
factor_gamma=0
u=1
gamma=1

maxIteration=num_epoch*len(a)/batchSize
datas=DataLoader(a,batch_size=batchSize,shuffle=True)
x_list=[]
x=torch.ones(dim,requires_grad=True)*0
v=torch.ones(dim)*0

for epoch in tqdm(range(num_epoch)):
    for i ,data in enumerate(datas):
        grads=torch.zeros(dim)
        for  j in range(len(data)):
            grads+=ng_grad_f(x,a[j])
        grad_avg=grads/len(data)
        #----
        eta=factor_a*(factor_b+epoch*len(a)+i+1)**(-factor_gamma)
        noise_v=u*(1-np.exp(-2*gamma*eta))*torch.randn(2)
        noise_x=u*gamma**(-2)*(2*gamma*eta+4*np.exp(-gamma*eta)-np.exp(-2*gamma*eta)-3)\
            *torch.randn(2)
        x=x+gamma*(1-np.exp(-gamma*eta))*v+\
            u*gamma**(-2)*(gamma*eta+np.exp(-gamma*eta)-1)*grad_avg+noise_x
        v=v*np.exp(-gamma*eta)-u*gamma**(-1)*(1-np.exp(-gamma*eta))*grad_avg+noise_v
        x_list.append(x.clone().detach().numpy())

np.save(path,np.array(x_list))
