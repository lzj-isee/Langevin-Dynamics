import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import scipy.sparse as sp


dim=123
train_labels=[]
train_features=[]
test_labels=[]
test_features=[]
dataNum=0
source='./dataset/a9a-test.txt'
'''
读取数据
'''
with open(source,'r') as f:
    datas=f.readlines()
    dataNum=len(datas)
'''
转换格式
'''
for i in range(dataNum):
    line=datas[i]
    position=line.find(' ')
    label=eval(line[:position])
    feature=eval('{'+line[position+1:len(line)-2].replace(' ',',')+'}')
    temp=np.zeros(dim+1)
    temp[0]=1
    for key in feature.keys():
        temp[key]=1
    train_labels.append([label])
    train_features.append(temp)
train_labels=np.array(train_labels)
train_features=np.array(train_features)
trainDatas=np.concatenate((train_labels,train_features),axis=1)
trainDatas=sp.coo_matrix(trainDatas)
sp.save_npz('./dataset/testDatas.npz',trainDatas)

