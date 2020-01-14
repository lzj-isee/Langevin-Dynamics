import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from  functions import*
import os

def NSRM_HMC_it(datas,num_epoch,dim,batchSize,factor_a,factor_b,factor_gamma,factor_i,u,gamma):
    Num=len(datas)
    t=0
    tk=0
    x_list=[]
    x_last=np.ones(dim)*0
    x=np.ones(dim)*0
    v=np.ones(dim)*0
    for epoch in tqdm(range(num_epoch)):
        for k in range(int(Num/batchSize)):
            if k==0 and epoch%factor_i==0:
                tk=0
                g=grad_f(x,datas[np.random.choice(Num,size=batchSize,replace=False)]).mean(0)
            t+=1
            tk+=1
            eta=factor_a*(factor_b+t)**(-factor_gamma)
            noise_x,noise_v=noise_Gen1(u,gamma,eta,dim)
            x_last=x
            x=x+gamma*(1-np.exp(-gamma*eta))*v+\
                u*gamma**(-2)*(gamma*eta+np.exp(-gamma*eta)-1)*g+noise_x
            v=v*np.exp(-gamma*eta)-u*gamma**(-1)*(1-np.exp(-gamma*eta))*g+noise_v
            datas_choice=datas[np.random.choice(Num,size=batchSize,replace=False)]
            g=grad_f(x,datas_choice).mean(0)+(1-1/tk)*(g-grad_f(x_last,datas_choice).mean(0))
            x_list.append(x)
    x_list=np.array(x_list)[500000:1000000]
    return x_list
    

def save_figure(samples,save_name):
    Ys=np.load('./ref.npy')
    Xs=np.linspace(-6,6,len(Ys))
    plt.figure(figsize=(6.4,6.4))
    ax1=plt.subplot2grid((15,15),(0,4),rowspan=11,colspan=11)
    ax1.hist2d(samples[:,0],samples[:,1],bins=60,range=[[-6,6],[-6,6]],cmap='Reds')
    ax2=plt.subplot2grid((15,15),(0,0),rowspan=11,colspan=3)
    ax2.hist(samples[:,1],bins=60,range=[-6,6], orientation='horizontal',density=True)
    ax2.plot(Ys[:,1],Xs,color='r')
    plt.ylim((-6,6))
    ax3=plt.subplot2grid((15,15),(12,4),rowspan=3,colspan=11)
    ax3.hist(samples[:,0],bins=60,range=[-6,6], orientation='vertical',density=True,label='NSRM_HMC')
    ax3.plot(Xs,Ys[:,0],color='r',label='ref')
    plt.xlim((-6,6))
    plt.legend(bbox_to_anchor=(-0.12,0.7))
    plt.savefig(save_name+'.png')
    plt.close()

def NSRM_HMC_sample(random_seed,train_setting,save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.manual_seed(random_seed['pytorch'])
    np.random.seed(random_seed['numpy'])
    num_epoch=train_setting['num_epoch']
    batchSize=train_setting['batchSize']
    dim=train_setting['dim']
    factor_a=train_setting['factor_a']
    factor_b=train_setting['factor_b']
    factor_gamma=train_setting['factor_gamma']
    factor_i=train_setting['factor_i']
    u=train_setting['u']
    gamma=train_setting['gamma']
    save_name=save_folder+'/'+\
        'seed[{:},{:}]'.format(random_seed['pytorch'],random_seed['numpy'])+\
        'setting[{:},{:},{:.2f}]'.format(factor_a,factor_b,factor_gamma)+\
        'interval[{:}]'.format(factor_i)
    datas=np.load('./dataset/a.npy')
    #SGLD
    x_list=NSRM_HMC_it(datas,num_epoch,dim,batchSize,factor_a,factor_b,factor_gamma,factor_i,u,gamma)
    #SAVE
    np.save(save_name+'.npy',x_list)
    save_figure(x_list,save_name)
    



if __name__ == "__main__":
    random_seed={
        'pytorch':2020,
        'numpy':2020}
    train_setting={
        'num_epoch':20000,
        'batchSize':10,
        'dim':2,
        'factor_a':0.06,
        'factor_b':0,
        'factor_gamma':0,
        'u':1,
        'gamma':1,
        'factor_i':10
        }
    save_folder='./NSRM_HMC_result'
    NSRM_HMC_sample(random_seed,train_setting,save_folder)