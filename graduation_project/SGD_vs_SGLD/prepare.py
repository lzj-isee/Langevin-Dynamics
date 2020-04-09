import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import scipy.sparse as sp
import numpy as np

train_num=60000
test_num=10000


transform=transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_datas=torchvision.datasets.FashionMNIST(root='./DataSet',train=True,download=True,transform=transform)
test_datas=torchvision.datasets.FashionMNIST(root='./DataSet',train=False,download=True,transform=transform)
train_loader=DataLoader(dataset=train_datas,batch_size=train_num)
test_loader=DataLoader(dataset=test_datas,batch_size=test_num)
i,(train_images,train_labels)=list(enumerate(train_loader))[0]
i,(test_images,test_labels)=list(enumerate(test_loader))[0]
train_images=train_images.view(-1,28*28).numpy().astype('float32')
train_labels=train_labels.view(-1,1).numpy().astype('float32')
test_images=test_images.view(-1,28*28).numpy().astype('float32')
test_labels=test_labels.view(-1,1).numpy().astype('float32')
np.save('./DataSet/train_images.npy',train_images)
np.save('./DataSet/train_labels.npy',train_labels)
np.save('./DataSet/test_images.npy',test_images)
np.save('./DataSet/test_labels.npy',test_labels)
a=1
