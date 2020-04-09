import numpy as np
from tqdm import tqdm

def pdf(x,y,sample):
    loc=np.array([x,y])
    result=2*np.exp(-np.linalg.norm(loc-sample)**2/2)+np.exp(-np.linalg.norm(loc+sample)**2/2)
    return result


samples=np.load('./dataset/points.npy')
lens=len(samples)

Xx=np.linspace(-6,6,121)
Xy=np.linspace(-6,6,121)
delta=0.1
X=np.zeros((121,121))
Ys=np.zeros((121,2))

for i in tqdm(range(len(Xx))):
    for j in range(len(Xy)):
        for sample in samples:
            X[i,j]+=np.log(pdf(Xx[i],Xy[j],sample))
        X[i,j]=np.exp(X[i,j]/lens)

X=X/np.sum(X)

mean=np.zeros(2)

for i in tqdm(range(len(Xx))):
    for j in range(len(Xy)):
        mean+=X[i,j]*np.array([Xx[i],Xy[j]])

print(mean)
np.save('./ref/ref_MSE.npy',mean)