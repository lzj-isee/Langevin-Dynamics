import torch
import numpy as np
import scipy.sparse
 
class Alg_SGLD(object):
    def __init__(self,curr_x,device):
        # The param of BRR model
        self.curr_x=curr_x.to(device)
        self.lambda_likeli=30
        self.lambda_prior=1
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
        A= -(self.Labels-torch.matmul(self.Features,self.curr_x)).view(-1,1)
        self.grads= A*self.Features/self.lambda_likeli

    def Grads_Calc_r(self,features,labels):
        # Calculate the gradients
        A= -(labels-torch.matmul(features,self.curr_x)).view(-1,1)
        return A*features/self.lambda_likeli

    
    def forward(self,features,labels):
        # Data forward
        A= torch.matmul(features,self.curr_x)
        return A

    def negative_log_prob(self,inputs,labels):
        # 很容易 -inf， 不建议用
        prob=torch.exp(-(labels-inputs)**2/2/self.lambda_likeli)/np.sqrt(2*np.pi*self.lambda_likeli)
        return -torch.log(prob)

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
            train_mse=((train_outputs-trainLabels)**2).mean()
            test_mse=((test_outputs-testLabels)**2).mean()
            return train_mse.cpu().item(), test_mse.cpu().item()
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
            train_mse=((train_outputs-trainLabels)**2).mean()
            test_mse=((test_outputs-testLabels)**2).mean()
            return train_mse.cpu().item(), test_mse.cpu().item()

        
