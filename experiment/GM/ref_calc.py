import numpy as np
from tqdm import tqdm
def pdf(x,sample):
    result0=(2*np.exp(-(x-sample[0])**2/2)+np.exp(-(x+sample[0])**2/2))/(3*np.sqrt(2*np.pi))
    result1=(2*np.exp(-(x-sample[1])**2/2)+np.exp(-(x+sample[1])**2/2))/(3*np.sqrt(2*np.pi))
    return np.array([result0,result1])

samples=np.ones((500,2))*2
lens=len(samples)

Xs=np.linspace(-6,6,120)
Ys=[]

for x in tqdm(Xs):
    temp=np.zeros(2)
    for sample in samples:
        temp+=np.log(pdf(x,sample))
    temp=np.exp(temp/lens)
    Ys.append(temp)
Ys=np.array(Ys)
np.save('./ref.npy',Ys)

    



