import torch
import torch.nn as nn
import torch.nn.functional as F


class NET(nn.Module):
    def __init__(self):
        self.burn_in=False
        self.lr_new=0
        self.lr_sum=0
        self.train_outputs_avg=None
        self.test_outputs_avg=None
        super(NET,self).__init__()
        self.fc1=nn.Linear(28*28,100)
        self.fc2=nn.Linear(100,10)

    def forward(self,x):
        x1=F.relu(self.fc1(x))
        x2=self.fc2(x1)
        return x1,x2

    @torch.no_grad()
    def loss_acc_eval(self,trainSet,testSet,train_num,test_num):
        # Evaluate the loss and acc on trainSet and testSet
        trainLabels=trainSet['labels']
        trainImages=trainSet['images']
        testLabels=testSet['labels']
        testImages=testSet['images']


        _,train_outputs=self.forward(trainImages)
        _,test_outputs=self.forward(testImages)
        if self.burn_in==False:  # Not burn in, no average 
            train_loss=F.cross_entropy(train_outputs,trainLabels,reduction='mean').cpu().item()
            test_loss=F.cross_entropy(test_outputs,testLabels,reduction='mean').cpu().item()
            _,train_predicted=torch.max(train_outputs,1)
            _,test_predicted=torch.max(test_outputs,1)
            train_acc=(train_predicted == trainLabels).sum().cpu().item()/train_num
            test_acc=(test_predicted == testLabels).sum().cpu().item()/test_num
            return train_acc,train_loss,test_acc,test_loss
        else: # Already burn in, consider average
            if self.train_outputs_avg==None:
                # the first time to record outputs
                self.train_outputs_avg=F.softmax(train_outputs,dim=1)
                self.test_outputs_avg=F.softmax(test_outputs,dim=1)
                self.lr_sum=self.lr_new
            else:
                # consider the average of samples(different self.curr_x)
                self.train_outputs_avg=(self.lr_sum*self.train_outputs_avg+\
                    self.lr_new*F.softmax(train_outputs,dim=1))/(self.lr_new+self.lr_sum)
                self.test_outputs_avg=(self.lr_sum*self.test_outputs_avg+\
                    self.lr_new*F.softmax(test_outputs,dim=1))/(self.lr_new+self.lr_sum)
                self.lr_sum=self.lr_sum+self.lr_new
            train_loss=F.nll_loss(torch.log(self.train_outputs_avg),trainLabels,reduction='mean').cpu().item()
            test_loss=F.nll_loss(torch.log(self.test_outputs_avg),testLabels,reduction='mean').cpu().item()
            _,train_predicted=torch.max(self.train_outputs_avg,1)
            _,test_predicted=torch.max(self.test_outputs_avg,1)
            train_acc=(train_predicted == trainLabels).sum().cpu().item()/train_num
            test_acc=(test_predicted == testLabels).sum().cpu().item()/test_num
            return train_acc,train_loss,test_acc,test_loss