import numpy as np
import matplotlib.pyplot as plt

class Alg_SGLD(object):
    def __init__(self,curr_x):
        self.curr_x=curr_x
        self.dim=len(curr_x)
    
    def Sample_Datas(self,trainSet,train_num,batchSize,weights):
        # Sample Datas from T=trainSet and preprocess
        self.indices=np.random.choice(
            a=np.arange(train_num),
            size=batchSize,
            replace=True,
            p=weights)
        self.datas=trainSet[self.indices,:]

    def Grads_Calc(self):
        self.batchSize=len(self.indices)
        X=np.repeat(np.reshape(self.curr_x,(-1,self.dim)),
            repeats=self.batchSize,axis=0)
        z=np.reshape((1+2*np.exp(2*np.dot(self.curr_x,self.datas.T))),(-1,1))
        self.grads=X-self.datas\
            +2*self.datas/np.repeat(z,repeats=self.dim,axis=1)

    def Grads_Calc_r(self,input_x,datas):
        self.batchSize=len(datas)
        X=np.repeat(np.reshape(input_x,(-1,self.dim)),
            repeats=self.batchSize,axis=0)
        z=np.reshape((1+2*np.exp(2*np.dot(input_x,datas.T))),(-1,1))
        return X-datas+2*datas/np.repeat(z,repeats=self.dim,axis=1)

    def save_figure(self,samples,Ys,save_path,name):
        Xs=np.linspace(-6,6,len(Ys))
        plt.figure(figsize=(6.4,6.4))
        ax1=plt.subplot2grid((15,15),(0,4),rowspan=11,colspan=11)
        ax1.hist2d(samples[:,0],samples[:,1],bins=120,range=[[-6,6],[-6,6]],cmap='Reds')
        ax2=plt.subplot2grid((15,15),(0,0),rowspan=11,colspan=3)
        ax2.hist(samples[:,1],bins=120,range=[-6,6], orientation='horizontal',density=True)
        ax2.plot(Ys[:,1],Xs,color='r')
        plt.ylim((-6,6))
        ax3=plt.subplot2grid((15,15),(12,4),rowspan=3,colspan=11)
        ax3.hist(samples[:,0],bins=120,range=[-6,6], orientation='vertical',density=True,label=name)
        ax3.plot(Xs,Ys[:,0],color='r',label='ref')
        plt.xlim((-6,6))
        plt.legend(bbox_to_anchor=(-0.12,0.7))
        plt.savefig(save_path+'.png')
        plt.close

