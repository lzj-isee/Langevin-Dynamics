import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, RandomSampler


def Load_MNIST(batchSize):
    transform=transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    train_set=torchvision.datasets.MNIST(root='./DataSet',train=True,download=True,transform=transform)
    test_set=torchvision.datasets.MNIST(root='./DataSet',train=False,download=True,transform=transform)
    train_loader=DataLoader(
        train_set,
        num_workers=4,
        batch_sampler=BatchSampler(RandomSampler(train_set,replacement=True),batch_size=batchSize,drop_last=True))
    full_train_loader=DataLoader(
        train_set,
        num_workers=4,
        shuffle=True,
        batch_size=1000,
        drop_last=False)
    test_loader=DataLoader(
        test_set,
        shuffle=False,
        batch_size=1000,
        num_workers=4)
    
    return train_set,train_loader,full_train_loader,test_set,test_loader
    