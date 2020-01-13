import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
from  torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from  functions import*
import os
from tqdm import tqdm

def SGLD_it(trainDatas,num_epoch,dim,factor_a,factor_b,factor_gamma,Num,testDatas,test_interval):
    nll_list=[]
    param=np.ones(dim)*0
    t=0
    for epoch in range(num_epoch):
        for i,data in enumerate(trainDatas):
            if t%test_interval==0:
                nll=nll_Cala(testDatas,param)
                information_print(epoch,num_epoch,i,len(trainDatas),nll)
                nll_list.append(nll)
            t+=1
            batchSize=data[0].shape[0]
            '''
            #SGLD
            '''
            grad_l=grad_Calc(data,param).mean(0)*Num/batchSize
            grad_p=param
            grad=grad_l+grad_p
            eta=factor_a*(factor_b+t)**(-factor_gamma)
            noise=(np.sqrt(2*eta)*torch.randn(dim)).numpy()
            param=param-eta*grad+noise
    
    return np.array(nll_list)
    
    

def SGLD_train(random_seed,train_setting,test_interval,save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.manual_seed(random_seed['pytorch'])
    num_epoch=train_setting['num_epoch']
    batchSize=train_setting['batchSize']
    dim=train_setting['dim']
    factor_a=train_setting['factor_a']
    factor_b=train_setting['factor_b']
    factor_gamma=train_setting['factor_gamma']
    save_name=save_folder+'/'+\
        'seed[{:}]'.format(random_seed['pytorch'])+\
        'setting[{},{},{}]'.format(factor_a,factor_b,factor_gamma)
    trains=dataLoad('./dataset/train_features.npz','./dataset/train_labels.npy')
    tests=dataLoad('./dataset/test_features.npz','./dataset/test_labels.npy')
    Num=(trains.tensors)[0].shape[0]
    trainDatas=DataLoader(trains,batch_size=batchSize,shuffle=True)
    #SGLD
    nll_list=SGLD_it(trainDatas,num_epoch,dim,factor_a,factor_b,factor_gamma,Num,tests,test_interval)
    #SAVE
    np.save(save_name+'.npy',nll_list)


if __name__ == "__main__":
    random_seed={'pytorch':2020}
    train_setting={
        'num_epoch':10,
        'batchSize':64,
        'dim':123+1,
        'factor_a':0.005,
        'factor_b':0,
        'factor_gamma':0.7
        }
    test_interval=20
    save_folder='./SGLD_result'
    SGLD_train(random_seed,train_setting,test_interval,save_folder)