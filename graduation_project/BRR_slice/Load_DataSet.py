import scipy.sparse as sp
import torch
import numpy as np



def load_dataset(
    device,
    train_labels_path='./DataSet/train_labels.npy',
    train_features_path='./DataSet/train_features.npz',
    test_labels_path='./DataSet/test_labels.npy',
    test_features_path='./DataSet/test_features.npz'):
    trainLabels=np.load(train_labels_path)
    trainFeatures = sp.load_npz(train_features_path)
    testLabels=np.load(test_labels_path)
    testFeatures = sp.load_npz(test_features_path)
    train_num=len(trainLabels)
    test_num=len(testLabels)
    #第一个维度补偏置
    trainFeatures=np.concatenate((np.ones([train_num,1]),trainFeatures.toarray()),axis=1)
    testFeatures=np.concatenate((np.ones([test_num,1]),testFeatures.toarray()),axis=1)
    train_Datas={'labels':torch.Tensor(trainLabels).to(device),\
        'features':torch.Tensor(trainFeatures).to(device)}
    test_Datas={'labels':torch.Tensor(testLabels).to(device),\
        'features':torch.Tensor(testFeatures).to(device)}
    return train_Datas,test_Datas
