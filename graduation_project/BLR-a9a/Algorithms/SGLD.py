import torch
import numpy as np
import scipy
import os
from tqdm import tqdm
from sys import path
path.append('./')
from DataLoader.Load_DataSet import Load_a9a
from common_functions import *

def _SGLD(trainDatas,testDatas,lr_sched,num_epochs,batchSize,eval_interval,device):
    inner_loops=int(len(lr_sched)/num_epochs)
    train_num=len(trainDatas['labels'])
    test_num=len(testDatas['labels'])
    dim=trainDatas['features'].shape[1]
    train_result_loss=[]
    train_result_acc=[]
    test_result_loss=[]
    test_result_acc=[]
    curr_iter_count=0
    x=torch.zeros(dim+1).to(device)
    for epoch in tqdm(range(num_epochs)):
        for i in range(inner_loops):
            curr_iter_count+=1
            labels,features,_ =sample_datas(
                trainDatas,train_num,batchSize,np.ones(train_num)/train_num,device)
            grad_l=grad_Calc(x,features,labels).mean(0)*train_num
            grad=x+grad_l
            x=x-lr_sched[curr_iter_count-1]*grad

            # Eval and Print
            if (curr_iter_count-1)%eval_interval==0:
                train_loss, train_acc, test_loss, test_acc=loss_acc_eval(
                    trainDatas,testDatas,train_num,test_num,x,device)
                '''
                print('epoch[{}/{}], step[{}/{}]'.format(
                    epoch,num_epochs,i,inner_loops))
                print('train_loss:{:.4f}, train_acc:{:.4f}'.format(
                    train_loss,train_acc))
                print('test_loss:{:.4f}, test_acc{:.4f}'.format(
                    test_loss,test_acc))
                print('\n')
                '''
                train_result_loss.append(train_loss)
                train_result_acc.append(train_acc)
                test_result_loss.append(test_loss)
                test_result_acc.append(test_acc)
    return np.array(train_result_loss), np.array(train_result_acc), np.array(test_result_loss), np.array(test_result_acc)


def SGLD_trian(lr_a,lr_b,lr_gamma,num_epochs,batchSize,\
    eval_interval,random_seed,save_folder,use_gpu=True,\
    train_path='./DataSet/a9a-train.txt',\
    test_path='./DataSet/a9a-test.txt'):
    # Use GPU if cuda is available
    if torch.cuda.is_available() and use_gpu==True:
        device=torch.device("cuda:0")
        print(torch.cuda.get_device_name(0))
    else:
        device=torch.device('cpu')
        print("use cpu")
    # Creat the save folder and name for result
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_name=save_folder+\
        'SGLD'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}]'.format(lr_a,lr_b,lr_gamma)
    # Print information before trian
    print('SGLD lr[{:.2e},{:.2f},{:.2f}]'.format(lr_a,lr_b,lr_gamma))
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Load DataSet as sparse matrix
    trainDatas,testDatas=Load_a9a(train_path,test_path)
    # Calculate settings
    train_num=len(trainDatas['labels'])
    dim=trainDatas['features'].shape[1]
    total_loops=num_epochs*round(train_num/batchSize)
    lr_sched=lr_a*(lr_b+np.arange(total_loops)+1)**(-lr_gamma)
    _SGLD(trainDatas,testDatas,lr_sched,num_epochs,batchSize,eval_interval,device)


SGLD_trian(1e-5,0,0.5,50,500,12,2020,'./result/SGLD/',use_gpu=True)