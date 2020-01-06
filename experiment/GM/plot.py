import numpy as np
import matplotlib.pyplot as plt
lens=500000


plt.figure()
sgld=np.load('./result_samples/SGLD/samples.npy')
sgld=sgld[len(sgld)-lens:len(sgld)]
#plt.hist2d(sgld[:,0],sgld[:,1],bins=40,range=[[-6,6],[-6,6]])
plt.hist(sgld[:,0],bins=40,range=[-6,6])

plt.figure()
sgulmcmc=np.load('./result_samples/SG_UL_MCMC/samples.npy')
sgulmcmc=sgulmcmc[len(sgulmcmc)-lens:len(sgulmcmc)]
#plt.hist2d(sgulmcmc[:,0],sgulmcmc[:,1],bins=40,range=[[-6,6],[-6,6]])
plt.hist(sgulmcmc[:,0],bins=40,range=[-6,6])

plt.show()