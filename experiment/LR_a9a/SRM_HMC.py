import torch
import pretty_errors
from common_functions.functions import grad_Calc, loss_and_Corr_eval
from DataLoader.DataLoader import Load_DataSet
import numpy as np
import os


def _SRM_HMC_iter(lr_a,lr_gamma,friction,p,num_epochs,train_set,train_loader,full_train_loader,\
    test_set,test_loader,print_interval):
    train_result_loss=[]
    train_result_corr=[]
    test_result_loss=[]
    test_result_corr=[]
    train_num=len((train_set.tensors)[1])
    test_num=len((test_set.tensors)[1])
    dim=(((train_set.tensors)[0]).shape)[1]
    gamma=friction/lr_a
    curr_iter_count=0.0
    param=np.zeros(dim)
    param_v=np.zeros(dim)
    param_last=np.zeros(dim)

    g=grad_Calc((train_set.tensors)[0].numpy(),(train_set.tensors)[1].numpy(),param).sum(0)
    for epoch in range(num_epochs):
        for i, (features,labels) in enumerate(train_loader):
            curr_iter_count+=1
            features=features.numpy()
            labels=labels.numpy()
            eta=lr_a*(curr_iter_count)**(-lr_gamma)
            noise=(np.sqrt(2*eta*gamma)*torch.randn(dim)).numpy()
            param_last=param
            param=param+eta*param_v
            param_v=param_v-eta*(gamma*param_v+g)+noise
            rho=1/(curr_iter_count)**p
            g=grad_Calc(features,labels,param).mean(0)*train_num+(1-rho)*(g-grad_Calc(features,labels,param_last).mean(0)*train_num)

            #print & eval
            if (curr_iter_count-1)%print_interval==0:
                train_loss, train_corr, test_loss, test_corr=loss_and_Corr_eval(
                    full_train_loader,test_loader,param,train_num,test_num)
                print('epoch[{}/{}], step[{}/{}]'.format(
                    epoch,num_epochs,i,len(train_loader)))
                print('train_loss:{:.4f}, train_acc:{:.4f}'.format(
                    train_loss,train_corr))
                print('test_loss:{:.4f}, test_acc{:.4f}'.format(
                    test_loss,test_corr))
                print('\n')
                train_result_loss.append(train_loss)
                train_result_corr.append(train_corr)
                test_result_loss.append(test_loss)
                test_result_corr.append(test_corr)
    return np.array(train_result_loss), np.array(train_result_corr), np.array(test_result_loss), np.array(test_result_corr)



def SRM_HMC_trian(lr_a,lr_gamma,friction,p,num_epochs,batchSize,print_interval,random_seed,save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    train_set,train_loader,full_train_loader,test_set,test_loader=Load_DataSet(batchSize)
    save_name=save_folder+\
        'SRM_HMC'+' '+\
        'lr_a[{:.2e}]'.format(lr_a)+\
        'lr_gamma[{}]'.format(lr_gamma)+\
        'friction[{:.2f}]'.format(friction)+\
        'p[{}]'.format(p)
    print('SRM_HMC: lr_a:{}, lr_gamma:{}, friction:{}, p:{}'.format(lr_a,lr_gamma,friction,p))
    train_loss,train_corr,test_loss,test_corr=_SRM_HMC_iter(
        lr_a,lr_gamma,p,friction,num_epochs,train_set,train_loader,full_train_loader,test_set,test_loader,print_interval)
    result=np.array([train_loss,train_corr,test_loss,test_corr])
    np.save(save_name+'.npy',result)



if __name__ == "__main__":
    num_epochs=50
    batchSize=500
    lr_a=0.4e-2
    lr_gamma=0.0
    friction=0.7
    p=1.0
    print_interval=10
    random_seed=2020
    save_folder='./result/SRM_HMC/'
    SRM_HMC_trian(
        lr_a,
        lr_gamma,
        friction,
        p,
        num_epochs,
        batchSize,
        print_interval,
        random_seed,
        save_folder)

