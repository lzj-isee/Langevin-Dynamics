import pandas
import numpy as np
import scipy.sparse as sp

path='./DataSet/superconduct_train.csv'
np.random.seed(1)

raw_datas=pandas.read_csv(path).values
num=raw_datas.shape[0]
dim=raw_datas.shape[1]-1
test_num=1263
train_num=num-test_num
choice=np.random.choice(np.arange(num),size=test_num,replace=False)

#train_datas=np.delete(raw_datas,choice,axis=0)
train_datas=raw_datas
test_datas=raw_datas[choice]

train_features=sp.csr_matrix(train_datas[:,0:dim])
train_labels=train_datas[:,dim]
test_features=sp.csr_matrix(test_datas[:,0:dim])
test_labels=test_datas[:,dim]

np.save('./DataSet/train_labels.npy',train_labels)
np.save('./DataSet/test_labels.npy',test_labels)
sp.save_npz('./DataSet/train_features.npz',train_features)
sp.save_npz('./DataSet/test_features.npz',test_features)