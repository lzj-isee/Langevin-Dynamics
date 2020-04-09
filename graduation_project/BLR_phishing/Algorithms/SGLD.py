import torch
import numpy as np
import scipy.sparse
 
class Alg_SGLD(object):
    def __init__(self,curr_x,device):
        # The param of BLR model
        self.curr_x=curr_x.to(device)
        # The dimension of Data
        self.dim=len(curr_x)
        self.device=device
        # Already burn in?
        self.burn_in=False
        # The sum of the step size of past
        self.lr_sum=0
        # The step size in this iteration
        self.lr_new=0
        # Outputs
        self.train_outputs_avg=None
        self.test_outputs_avg=None

    def Datas_Transform(self,Datas):
        # Transfrom the data from sparse to array 
        Labels=Datas['labels']
        batchSize=(Datas['features']).shape[0]
        Features=(Datas['features'])
        return Labels,Features
    
    def Sample_Datas(self,trainSet,train_num,batchSize,weights):
        # Sample Datas from T=trainSet and preprocess
        self.indices=np.random.choice(
            a=np.arange(train_num),
            size=batchSize,
            replace=True,
            p=weights)
        self.Labels=(trainSet['labels'])[self.indices]
        self.Features=(trainSet['features'])[self.indices]
    
    def Grads_Calc(self):
        # Calculate the gradients
        Z= -torch.matmul(self.Features,self.curr_x)*self.Labels
        A= torch.sigmoid(Z)
        self.grads= -(A*self.Labels).view(-1,1)*self.Features

    def Grads_Calc_r(self,features,labels):
        # Calculate the gradients
        Z= -torch.matmul(features,self.curr_x)*labels
        A= torch.sigmoid(Z)
        return -(A*labels).view(-1,1)*features

    
    def forward(self,features,labels):
        Z= torch.matmul(features,self.curr_x)*labels
        result= torch.sigmoid(Z)
        return result
    
    def variance_eval(self):
        batch_size=len(self.indices)
        mean=self.grads.mean(0).view(1,-1)
        variance=(torch.norm(self.grads-mean,dim=1)**2).sum()/(batch_size-1)
        return variance

    def loss_acc_eval(self,trainSet,testSet,train_num,test_num):
        # Evaluate the loss and acc on trainSet and testSet
        train_dim=trainSet['features'].shape[1]
        trainLabels=trainSet['labels']
        trainFeatures=trainSet['features']

        test_dim=testSet['features'].shape[1]
        testLabels=testSet['labels']
        testFeatures=testSet['features']
        if (train_dim-test_dim)!=0:
            testFeatures=torch.cat(\
                (testFeatures,torch.ones(test_num,train_dim-test_dim).to(self.device)),dim=1)


        train_outputs=self.forward(trainFeatures,trainLabels)
        test_outputs=self.forward(testFeatures,testLabels)
        if self.burn_in==False:  # Not burn in, no average 
            train_loss=(-torch.log(train_outputs)).mean()
            train_acc=(torch.round(train_outputs)).sum()/train_num
            test_loss=(-torch.log(test_outputs)).mean()
            test_acc=(torch.round(test_outputs)).sum()/test_num
            return train_loss.to('cpu').item(),train_acc.to('cpu').item()\
                ,test_loss.to('cpu').item(),test_acc.to('cpu').item()
        else: # Already burn in, consider average
            if self.train_outputs_avg==None:
                # the first time to record outputs
                self.train_outputs_avg=train_outputs
                self.test_outputs_avg=test_outputs
                self.lr_sum=self.lr_new
            else:
                # consider the average of samples(different self.curr_x)
                self.train_outputs_avg=(self.lr_sum*self.train_outputs_avg+\
                    self.lr_new*train_outputs)/(self.lr_new+self.lr_sum)
                self.test_outputs_avg=(self.lr_sum*self.test_outputs_avg+\
                    self.lr_new*test_outputs)/(self.lr_new+self.lr_sum)
                self.lr_sum=self.lr_sum+self.lr_new
            train_loss=(-torch.log(self.train_outputs_avg)).mean()
            train_acc=(torch.round(self.train_outputs_avg)).sum()/train_num
            test_loss=(-torch.log(self.test_outputs_avg)).mean()
            test_acc=(torch.round(self.test_outputs_avg)).sum()/test_num
            return train_loss.to('cpu').item(),train_acc.to('cpu').item()\
                ,test_loss.to('cpu').item(),test_acc.to('cpu').item()

        
