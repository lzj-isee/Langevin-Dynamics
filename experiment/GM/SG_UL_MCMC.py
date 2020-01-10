import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
from  functions import*
import os
if not os.path.exists('./result_samples/SG_UL_MCMC'):
    os.makedirs('./result_samples/SG_UL_MCMC')
torch.manual_seed(2020)
path='./result_samples/SG_UL_MCMC/samples.npy'

a=torch.Tensor(np.load('./dataset/a.npy'))

num_epoch=20000
batchSize=10
dim=2
factor_a=1.5
factor_b=0
factor_gamma=0.08
u=1
gamma=1
#-----------------------------------------------------------------------------------
datas=DataLoader(a,batch_size=batchSize,shuffle=True)
x_list=[]
x=np.ones(dim)*0
v=np.ones(dim)*0

for epoch in tqdm(range(num_epoch)):
    for i ,data in enumerate(datas):
        grads=grad_f(x,data.clone().detach().numpy().astype('float64'))
        grad_avg=grads.mean(0)
        #----
        eta=factor_a*(factor_b+epoch*len(a)+i+1)**(-factor_gamma)
        noise_v=(u*(1-np.exp(-2*gamma*eta))*torch.randn(2)).clone().detach().numpy().astype('float64')
        noise_x=(u*gamma**(-2)*(2*gamma*eta+4*np.exp(-gamma*eta)-np.exp(-2*gamma*eta)-3)\
            *torch.randn(2)).clone().detach().numpy().astype('float64')
        x=x+gamma*(1-np.exp(-gamma*eta))*v+\
            u*gamma**(-2)*(gamma*eta+np.exp(-gamma*eta)-1)*grad_avg+noise_x
        v=v*np.exp(-gamma*eta)-u*gamma**(-1)*(1-np.exp(-gamma*eta))*grad_avg+noise_v
        x_list.append(x)

np.save(path,np.array(x_list))
