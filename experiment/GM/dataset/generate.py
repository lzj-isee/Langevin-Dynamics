import numpy as np

num=500

a=np.random.multivariate_normal([2,2],np.diag([1,1]),num)
np.save('./dataset/a.npy',a)