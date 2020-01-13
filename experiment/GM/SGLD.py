import numpy as np
import torch
from tqdm import tqdm
from  torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from  functions import*
import os

def SGLD_it(datas,num_epoch,dim,factor_a,factor_b,factor_gamma):
    x_list=[]
    x=np.ones(dim)*0
    t=0
    for epoch in tqdm(range(num_epoch)):
        for i ,data in enumerate(datas):
            t+=1
            grad_avg=grad_f(x,data.numpy()).mean(0)
            eta=factor_a*(factor_b+t)**(-factor_gamma)
            noise=(np.sqrt(2*eta)*torch.randn(2)).numpy()
            x=x-eta*grad_avg+noise
            x_list.append(x)
    x_list=np.array(x_list)
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
    ax3.hist(samples[:,0],bins=60,range=[-6,6], orientation='vertical',density=True,label='SGLD')
    ax3.plot(Xs,Ys[:,0],color='r',label='ref')
    plt.xlim((-6,6))
    plt.legend(bbox_to_anchor=(-0.12,0.7))
    plt.savefig(save_name+'.png')

def SGLD_sample(random_seed,train_setting,save_folder):
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
        'setting[{:},{:},{:}]'.format(factor_a,factor_b,factor_gamma)
    datas_np=torch.Tensor(np.load('./dataset/a.npy'))
    datas=DataLoader(datas_np,batch_size=batchSize,shuffle=True)
    #SGLD
    x_list=SGLD_it(datas,num_epoch,dim,factor_a,factor_b,factor_gamma)
    #SAVE
    np.save(save_name+'.npy',x_list)
    save_figure(x_list,save_name)
    



if __name__ == "__main__":
    random_seed={'pytorch':2020}
    train_setting={
        'num_epoch':20000,
        'batchSize':10,
        'dim':2,
        'factor_a':1,
        'factor_b':0,
        'factor_gamma':0.1
        }
    save_folder='./SGLD_result'
    SGLD_sample(random_seed,train_setting,save_folder)