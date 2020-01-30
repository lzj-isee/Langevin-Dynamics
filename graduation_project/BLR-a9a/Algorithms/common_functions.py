import torch 
import numpy as np
import scipy 

def sample_datas(trainDatas,train_num,batchSize,weights,device):
    # Sample datas from dataset and preprocess
    indices=np.random.choice(
        a=np.arange(train_num),size=batchSize,replace=True,p=weights)
    Labels=(trainDatas['labels'])[indices]
    Features=(trainDatas['features'])[indices].toarray()
    Features=np.concatenate((np.ones((batchSize,1)),Features),axis=1)
    Labels=torch.Tensor(Labels).to(device)
    Features=torch.Tensor(Features).to(device)
    return Labels,Features,indices

def grad_Calc(x,features,labels):
    # Calculate the gradients
    Z= -torch.matmul(features,x)*labels
    A= 1/(1+torch.exp(-Z))
    result = torch.matmul(torch.diag(A*labels),features)
    return -result

def forward(x,features,labels):
    Z= torch.matmul(features,x)*labels
    result= 1/(1+torch.exp(-Z))
    return result

def loss_acc_eval(trainDatas,testDatas,train_num,test_num,x,device):
    # Evaluate the loss and acc on trainSet and testSet
    trainLabels=trainDatas['labels']
    trainFeatures=trainDatas['features'].toarray()
    trainFeatures=np.concatenate((np.ones((train_num,1)),trainFeatures),axis=1)
    trainLabels=torch.Tensor(trainLabels).to(device)
    trainFeatures=torch.Tensor(trainFeatures).to(device)

    testLabels=testDatas['labels']
    testFeatures=testDatas['features'].toarray()
    testFeatures=np.concatenate((np.ones((test_num,1)),testFeatures),axis=1)
    # a9a test的feature缺少了第123维，手动补上了
    testFeatures=np.concatenate((testFeatures,np.zeros((test_num,1))),axis=1)
    testLabels=torch.Tensor(testLabels).to(device)
    testFeatures=torch.Tensor(testFeatures).to(device)

    train_outputs=forward(x,trainFeatures,trainLabels)
    test_outputs=forward(x,testFeatures,testLabels)
    train_loss=(-torch.log(train_outputs)).mean()
    train_acc=(torch.round(train_outputs)).sum()/train_num
    test_loss=(-torch.log(test_outputs)).mean()
    test_acc=(torch.round(test_outputs)).sum()/test_num
    return train_loss.to('cpu').numpy(),train_acc.to('cpu').numpy()\
        ,test_loss.to('cpu').numpy(),test_acc.to('cpu').numpy()



