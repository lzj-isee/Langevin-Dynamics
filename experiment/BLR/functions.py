import numpy as np
import scipy.sparse as sp
import torch

def dataLoad(source):
    DataSet=sp.load_npz(source)
    coor=torch.LongTensor([DataSet.row,DataSet.col])
    data=torch.FloatTensor(DataSet.data)
    shape=DataSet.shape
    TDatas=torch.sparse.FloatTensor(coor,data,torch.Size(shape))
    return TDatas
 

def grad_Calc(data,param):
    data=data.to_dense().numpy()
    labels=data[:,0]
    features=data[:,1:]
    Z=-np.dot(features,param)*labels
    A=1/(1+np.exp(-Z))
    B=np.dot(np.diag(labels),features)
    result=np.dot(np.diag(A),B)
    #----test-----
    print(-np.log((1/(1+np.exp(Z))).mean()))
    return -result