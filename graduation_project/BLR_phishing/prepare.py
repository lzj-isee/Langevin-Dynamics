import scipy.sparse as sparse
import numpy as np
from libsvm.python.svmutil import *


np.random.seed(1)
path='./DataSet/phishing.txt'
raw_labels,raw_features = svm_read_problem(path, return_scipy=True)
raw_features=raw_features.toarray()
num=len(raw_labels)
test_num=2055
train_num=num-test_num
choice=np.random.choice(np.arange(num),size=test_num,replace=False)

test_labels=raw_labels[choice]
train_labels=np.delete(raw_labels,choice,axis=0)

test_features=sparse.csr_matrix(raw_features[choice])
train_features=sparse.csr_matrix(np.delete(raw_features,choice,axis=0))

np.save('./DataSet/train_labels.npy',train_labels)
np.save('./DataSet/test_labels.npy',test_labels)
sparse.save_npz('./DataSet/train_features.npz',train_features)
sparse.save_npz('./DataSet/test_features.npz',test_features)