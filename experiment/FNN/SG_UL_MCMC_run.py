import torch
import torch.nn.functional as F
from Model.FNN import FNN
from DataLoader.DataLoader import Load_MNIST
from Optimizer.SG_UL_MCMC_op import SG_UL_MCMC_op
import numpy as np
import os
import copy
import pretty_errors


def _SG_UL_MCMC_iter(model,lr_a,lr_gamma,num_epochs,train_set,train_loader,full_train_loader,\
    test_set,test_loader,loss_fn,print_interval,device='cpu'):
    train_result_loss=[]
    train_result_corr=[]
    test_result_loss=[]
    test_result_corr=[]
    train_num=len(train_set)
    test_num=len(test_set)
    optimizer=SG_UL_MCMC_op(model.parameters(),lr_a,lr_gamma,device)
    curr_iter_count=0.0

    for epoch in range(num_epochs):
        for i,data in enumerate(train_loader):
            curr_iter_count+=1
            #get  the inputs
            images,labels=data
            images=images.view(-1,28*28)
            images=images.to(device)
            labels=labels.to(device)
            #forward + backward + step
            outputs=model(images)
            loss=loss_fn(outputs,labels,reduction='mean')*train_num
            loss.backward()
            #step
            optimizer.step(curr_iter_count=curr_iter_count)

            #print & eval
            if  (curr_iter_count-1)%print_interval==0:
                #train_eval
                train_loss=0
                train_correct=0
                with torch.no_grad():
                    for _,(images_eval,labels_eval) in enumerate(full_train_loader):
                        images_eval=images_eval.view(-1,28*28)
                        images_eval=images_eval.to(device)
                        labels_eval=labels_eval.to(device)
                        eval_batchSize=len(labels_eval)
                        outputs_eval=model(images_eval)
                        _,predicted=torch.max(outputs_eval.data,1)
                        train_correct+=(predicted == labels_eval).sum().item()
                        train_loss+=eval_batchSize*loss_fn(outputs_eval,labels_eval)
                #test_eval
                test_loss=0
                test_correct=0
                with torch.no_grad():
                    for _,(images_eval,labels_eval) in enumerate(test_loader):
                        images_eval=images_eval.view(-1,28*28)
                        images_eval=images_eval.to(device)
                        labels_eval=labels_eval.to(device)
                        eval_batchSize=len(labels_eval)
                        outputs_eval=model(images_eval)
                        _,predicted=torch.max(outputs_eval.data,1)
                        test_correct+=(predicted==labels_eval).sum().item()
                        test_loss+=eval_batchSize*loss_fn(outputs_eval,labels_eval)
                print('epoch[{}/{}],  step[{}/{}]'.format(epoch,num_epochs,i,len(train_loader)))
                print('loss_train:{:.4f}, acc_train:{:.4f}'.format(
                    train_loss/train_num,train_correct/train_num))
                print('loss_test:{:.4f}, acc_test:{:.4f}\n'.format(
                    test_loss/test_num,test_correct/test_num))
                train_result_loss.append(train_loss/train_num)
                train_result_corr.append(train_correct/train_num)
                test_result_loss.append(test_loss/test_num)
                test_result_corr.append(test_correct/test_num)
    return np.array(train_result_loss),np.array(train_result_corr),np.array(test_result_loss),np.array(test_result_corr)
                
    

def SG_UL_MCMC_train(lr_a,lr_gamma,num_epochs,batchSize,loss_fn,print_interval,random_seed,save_folder,device="cpu"):
    if torch.cuda.is_available():
        device=torch.device("cuda:0")
    else:
        print("no cuda device available, use cpu")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model=FNN()
    model.to(device)
    train_set,train_loader,full_train_loader,test_set,test_loader=Load_MNIST(batchSize)
    save_name=save_folder+\
        'SG_UL_MCMC'+' '+\
        'lr_a[{}]'.format(lr_a)+\
        'lr_gamma[{}]'.format(lr_gamma)
    print('SG_UL_MCMC: lr_a:{}, lr_gamma:{}'.format(lr_a,lr_gamma))
    train_loss,train_corr,test_loss,test_corr=_SG_UL_MCMC_iter(model,lr_a,lr_gamma,num_epochs,\
        train_set,train_loader,full_train_loader,test_set,test_loader,\
            loss_fn,print_interval,device=device)
    result=np.array([train_loss,train_corr,test_loss,test_corr])
    np.save(save_name+'.npy',result)
    

if __name__ == "__main__":
    num_epochs=10
    batchSize=500
    lr_a=1e-05
    lr_gamma=0.1
    print_interval=12
    random_seed=2020
    save_folder='./result/SG_UL_MCMC/'
    loss_fn=F.cross_entropy
    SG_UL_MCMC_train(
        lr_a,
        lr_gamma,
        num_epochs,
        batchSize,
        loss_fn,
        print_interval,
        random_seed,
        save_folder,
        device='cpu')