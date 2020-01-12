import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset

def dataLoad(f_source,l_source):
    features=sp.load_npz(f_source)
    labels=np.load(l_source)
    coor=torch.LongTensor([features.row,features.col])
    data=torch.FloatTensor(features.data)
    shape=features.shape
    sp_features=torch.sparse.FloatTensor(coor,data,torch.Size(shape))
    DataSet=TensorDataset(sp_features,torch.Tensor(labels))
    return DataSet
 

def grad_Calc(data,param):
    labels=(data[1]).numpy().reshape(-1)
    features=(data[0].to_dense()).numpy()
    Z=-np.dot(features,param)*labels
    A=1/(1+np.exp(-Z))
    B=np.dot(np.diag(labels),features)
    result=np.dot(np.diag(A),B)
    return -result

def nll_Cala(data,param):
    labels=(data.tensors[1]).numpy().reshape(-1)
    features=(data.tensors[0].to_dense()).numpy()
    Z=np.dot(features,param)*labels
    nll=-np.log(1/(1+np.exp(-Z)))
    nll=nll.mean()
    return nll