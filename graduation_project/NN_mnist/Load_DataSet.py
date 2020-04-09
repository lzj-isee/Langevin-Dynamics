import torch
import numpy as np

def load_dataset(
    device,
    train_labels_path='./DataSet/train_labels.npy',
    train_images_path='./DataSet/train_images.npy',
    test_labels_path='./DataSet/test_labels.npy',
    test_images_path='./DataSet/test_images.npy'):
    trainLabels=np.load(train_labels_path)
    trainImages = np.load(train_images_path)
    testLabels=np.load(test_labels_path)
    testImages = np.load(test_images_path)
    train_num=len(trainLabels)
    test_num=len(testLabels)
    train_Datas={'labels':torch.Tensor(trainLabels).view(train_num).long().to(device),\
        'images':torch.Tensor(trainImages).to(device)}
    test_Datas={'labels':torch.Tensor(testLabels).view(test_num).long().to(device),\
        'images':torch.Tensor(testImages).to(device)}
    return train_Datas,test_Datas