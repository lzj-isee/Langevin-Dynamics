import numpy as np

def grad_Calc(features,labels,param):
    Z= -np.dot(features,param)*labels
    A=1/(1+np.exp(-Z))
    B=np.dot(np.diag(labels),features)
    result=np.dot(np.diag(A),B)
    return -result

def forward(features,labels,param):
    Z=np.dot(features,param)*labels
    result=1/(1+np.exp(-Z))
    return result


def loss_and_Corr_eval(full_train_loader,test_loader,param,train_num,test_num):
    #train_eval
    train_loss=0
    train_corr=0
    for _,(features,labels) in enumerate(full_train_loader):
        features=features.numpy()
        labels=labels.numpy()
        outputs=forward(features,labels,param)
        train_loss+=(-np.log(outputs)).sum()
        train_corr+=(np.round(outputs)).sum()
    #test_eval
    test_loss=0
    test_corr=0
    for _,(features,labels) in enumerate(test_loader):
        features=features.numpy()
        labels=labels.numpy()
        outputs=forward(features,labels,param)
        test_loss+=(-np.log(outputs)).sum()
        test_corr+=(np.round(outputs)).sum()
    train_loss/=train_num
    train_corr/=train_num
    test_loss/=test_num
    test_corr/=test_num
    return  train_loss, train_corr, test_loss, test_corr


        



