import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from  functions import*
import os

def SGLD_it(trainDatas,num_epoch,dim,factor_a,factor_b,factor_gamma,Num):
    param=np.ones(dim)*0
    t=0
    for epoch in range(num_epoch):
        for i,data in enumerate(trainDatas):
            t+=1
            batchSize=data.shape[0]
            grad_l=grad_Calc(data,param).mean(0)*Num/batchSize
            grad_p=np.mean(param,0)/2
            grad=grad_l+grad_p
            eta=factor_a*(factor_b+t)**(-factor_gamma)
            noise=(np.sqrt(2*eta)*torch.randn(dim)).numpy()
            param=param-eta*grad+noise
    
    

def SGLD_train(random_seed,train_setting,save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.manual_seed(random_seed['pytorch'])
    num_epoch=train_setting['num_epoch']
    batchSize=train_setting['batchSize']
    dim=train_setting['dim']
    factor_a=train_setting['factor_a']
    factor_b=train_setting['factor_b']
    factor_gamma=train_setting['factor_gamma']
    trains=dataLoad('./dataset/trainDatas.npz')
    Num=trains.shape[0]
    trainDatas=DataLoader(trains,batch_size=batchSize,shuffle=True)
    SGLD_it(trainDatas,num_epoch,dim,factor_a,factor_b,factor_gamma,Num)








if __name__ == "__main__":
    random_seed={'pytorch':2020}
    train_setting={
        'num_epoch':20,
        'batchSize':64,
        'dim':123+1,
        'factor_a':0.001,
        'factor_b':0,
        'factor_gamma':0.1
        }
    save_folder='./SGLD_result'
    SGLD_train(random_seed,train_setting,save_folder)