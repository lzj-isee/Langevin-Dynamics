import numpy as np
from tqdm import tqdm
'''
def pdf(x,sample):
    result0=(2*np.exp(-(x-sample[0])**2/2)+np.exp(-(x+sample[0])**2/2))/(3*np.sqrt(2*np.pi))
    result1=(2*np.exp(-(x-sample[1])**2/2)+np.exp(-(x+sample[1])**2/2))/(3*np.sqrt(2*np.pi))
    return np.array([result0,result1])
'''
def pdf(x,y,sample):
    loc=np.array([x,y])
    result=2*np.exp(-np.linalg.norm(loc-sample)**2/2)+np.exp(-np.linalg.norm(loc+sample)**2/2)
    return result



#samples=np.ones((500,2))*2
samples=np.load('./dataset/a.npy')
lens=len(samples)

Xx=np.linspace(-6,6,121)
Xy=np.linspace(-6,6,121)
delta=0.1
X=np.zeros((121,121))
Ys=np.zeros((121,2))

'''
for x in tqdm(Xs):
    temp=np.zeros(2)
    for sample in samples:
        temp+=np.log(pdf(x,sample))
    temp=np.exp(temp/lens)
    Ys.append(temp)
Ys=np.array(Ys)
np.save('./ref.npy',Ys)
'''
for i in tqdm(range(len(Xx))):
    for j in range(len(Xy)):
        for sample in samples:
            X[i,j]+=np.log(pdf(Xx[i],Xy[j],sample))
        X[i,j]=np.exp(X[i,j]/lens)
Ys[:,0]=np.sum(X,axis=1)/np.sum(X)/delta
Ys[:,1]=np.sum(X,axis=0)/np.sum(X)/delta
np.save('./ref.npy',Ys)


    



