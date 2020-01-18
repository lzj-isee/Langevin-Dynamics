import numpy as np
import matplotlib.pyplot as plt



def plot_fig(samples,name):
    Ys=np.load('./ref.npy')
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
    plt.show()

data=np.load('./SRM_HMC_result4/seed[2020,2020]setting[0.08,0,0.05]pow[0.50].npy')
data=data[len(data)-500000:len(data)]
plot_fig(data,'SRM_HMC')