from Model import NET
import torch
import torch.nn.functional as F

class NET_rais(NET):
    def __init__(self,class_num,train_num,d,device):
        super(NET_rais,self).__init__(class_num,device)
        self.device=device
        self.class_num=class_num
        self.train_num=train_num
        self.d=d    # Param to Calculate v
        self.grad_norm=torch.zeros(self.train_num).to(device) # gradient norm
        self.v=torch.zeros(self.train_num).to(device)  # param to update p
        self.p=torch.ones(self.train_num).to(device)/train_num # Init uniform sample
    
    @torch.no_grad()
    def approximate_grad_norm(self,last_outputs,final_outputs,labels):
        batch_size=len(labels)
        z=F.softmax(final_outputs,dim=1) # matrix: batch_size * class_num
        labels_onehot=torch.zeros(batch_size, self.class_num).to(self.device).scatter_(1, labels.view(batch_size,1), 1)
        loss_partial_finalout=(z-labels_onehot).view(batch_size,1,self.class_num)
        finalout_partial_lastout=torch.cat((last_outputs,torch.ones(batch_size,1).to(self.device)),1).view(batch_size,last_outputs.shape[1]+1,1)
        grads=torch.matmul(finalout_partial_lastout,loss_partial_finalout).view(batch_size,-1)
        grads_norm=torch.norm(grads,dim=1)
        return grads_norm

    @torch.no_grad()
    def initialize(self,trainSet):
        labels=trainSet['labels']
        images=trainSet['images']
        last_outputs,final_outputs=self.forward(images)
        self.grad_norm = self.approximate_grad_norm(last_outputs,final_outputs,labels)

    @torch.no_grad()
    def update(self,last_outputs,final_outputs,labels,indices):
        # update grad_norm
        self.grad_norm[indices] = self.approximate_grad_norm(last_outputs,final_outputs,labels)
        # update v
        k=torch.sqrt(self.train_num/torch.sum(self.grad_norm))
        self.v=(1+k*self.d)*self.grad_norm
        # update p
        self.p=self.v/torch.sum(self.v)

    @torch.no_grad()
    def variance_eval(self,last_outputs,final_outputs,labels,indices):
        batch_size=len(labels)
        w=(1/self.p/self.train_num)
        z=F.softmax(final_outputs,dim=1) # matrix: batch_size * class_num
        labels_onehot=torch.zeros(batch_size, self.class_num).to(self.device).scatter_(1, labels.view(batch_size,1), 1)
        loss_partial_finalout=(z-labels_onehot).view(batch_size,1,self.class_num)
        finalout_partial_lastout=torch.cat((last_outputs,torch.ones(batch_size,1).to(self.device)),1).view(batch_size,last_outputs.shape[1]+1,1)
        grads=torch.matmul(finalout_partial_lastout,loss_partial_finalout).view(batch_size,-1)
        mean=(w[indices].view(-1,1)*grads).mean(0)
        variance=(torch.norm(w[indices].view(-1,1)*grads-mean,dim=1)**2).sum()/(batch_size-1)
        return variance



        

        

