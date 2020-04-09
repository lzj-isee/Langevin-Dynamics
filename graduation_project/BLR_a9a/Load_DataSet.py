from libsvm.python.svmutil import *
from scipy.sparse import csr_matrix
import torch
import numpy as np



def load_dataset(
    device,
    train_path='./DataSet/a9a-train.txt',
    test_path='./DataSet/a9a-test.txt'):
    trainLabels, trainFeatures = svm_read_problem(train_path, return_scipy=True)
    testLabels, testFeatures = svm_read_problem(test_path, return_scipy=True)
    train_num=len(trainLabels)
    test_num=len(testLabels)
    #第一个维度补偏置
    trainFeatures=np.concatenate((np.ones([train_num,1]),trainFeatures.toarray()),axis=1)
    testFeatures=np.concatenate((np.ones([test_num,1]),testFeatures.toarray()),axis=1)
    train_Datas={'labels':torch.Tensor(trainLabels).to(device),\
        'features':torch.Tensor(trainFeatures).to(device)}
    test_Datas={'labels':torch.Tensor(testLabels).to(device),\
        'features':torch.Tensor(testFeatures).to(device)}
    train_dim=trainFeatures.shape[1]
    test_dim=testFeatures.shape[1]
    assert(train_dim>=test_dim,"ERROR: train_dim < test_dim")
    return train_Datas,test_Datas