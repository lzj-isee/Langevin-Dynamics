import torch
import numpy as np
import matplotlib.pyplot as plt

a=np.load('./SG_UL_MCMC_result/seed[2020,2020]setting[0.10,0,0.05].npy')[500000:1000000,0]
plt.hist(a,bins=120)
plt.show()






pause=1