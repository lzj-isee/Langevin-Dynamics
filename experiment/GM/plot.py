import numpy as np
import matplotlib.pyplot as plt
lens=500000
Ys=np.load('./ref.npy')
Xs=np.linspace(-6,6,len(Ys))
sgld=np.load('./result_samples/SGLD/samples.npy')
sgulmcmc=np.load('./result_samples/SG_UL_MCMC/samples.npy')
sgld=sgld[len(sgld)-lens:len(sgld)]
sgulmcmc=sgulmcmc[len(sgulmcmc)-lens:len(sgulmcmc)]

#SGLD
plt.figure(figsize=(6.4,6.4))
ax1=plt.subplot2grid((15,15),(0,4),rowspan=11,colspan=11)
ax1.hist2d(sgld[:,0],sgld[:,1],bins=60,range=[[-6,6],[-6,6]],cmap='Reds')
ax2=plt.subplot2grid((15,15),(0,0),rowspan=11,colspan=3)
ax2.hist(sgld[:,1],bins=60,range=[-6,6], orientation='horizontal',density=True)
ax2.plot(Ys[:,1],Xs,color='r')
plt.ylim((-6,6))
ax3=plt.subplot2grid((15,15),(12,4),rowspan=3,colspan=11)
ax3.hist(sgld[:,0],bins=60,range=[-6,6], orientation='vertical',density=True,label='SGLD')
ax3.plot(Xs,Ys[:,0],color='r',label='ref')
plt.xlim((-6,6))
plt.legend(bbox_to_anchor=(-0.12,0.7))

#SG_UL_MCMC
plt.figure(figsize=(6.4,6.4))
ax1=plt.subplot2grid((15,15),(0,4),rowspan=11,colspan=11)
ax1.hist2d(sgulmcmc[:,0],sgulmcmc[:,1],bins=60,range=[[-6,6],[-6,6]],cmap='Reds')
ax2=plt.subplot2grid((15,15),(0,0),rowspan=11,colspan=3)
ax2.hist(sgulmcmc[:,1],bins=60,range=[-6,6], orientation='horizontal',density=True)
ax2.plot(Ys[:,1],Xs,color='r')
plt.ylim((-6,6))
ax3=plt.subplot2grid((15,15),(12,4),rowspan=3,colspan=11)
ax3.hist(sgulmcmc[:,0],bins=60,range=[-6,6], orientation='vertical',density=True,label='SG_UL_MCMC')
ax3.plot(Xs,Ys[:,0],color='r',label='ref')
plt.xlim((-6,6))
plt.legend(bbox_to_anchor=(-0.12,0.7))




plt.show()