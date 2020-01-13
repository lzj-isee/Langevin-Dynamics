import torch
import numpy as np

cov=np.ones((4,4))*1
v=np.ones(2)*0.3
x=np.ones(2)*0.4
std=np.diag(np.concatenate((x,v),axis=0))
m=cov-np.diag(np.diag(cov))+std
noise=np.random.multivariate_normal(np.zeros(4),m)




pause=1