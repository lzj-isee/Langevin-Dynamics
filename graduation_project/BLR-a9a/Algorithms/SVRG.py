import torch
import numpy as np
import scipy
import os
from tqdm import tqdm
from sys import path
path.append('./')
from DataLoader.Load_DataSet import Load_a9a
from Algorithms.common_functions import *
from tensorboardX import SummaryWriter
from Class.SVRG import model_SVRG

def _SVRG(trainDatas,testDatas,lr_sched,num_epochs,batchSize,eval_interval,device,writer):
    inner_loops=int(len(lr_sched)/num_epochs)
    train_num=len(trainDatas['labels'])
    test_num=len(testDatas['labels'])
    dim=trainDatas['features'].shape[1]
    curr_iter_count=0
    model=model_SVRG(  # Init
        torch.zeros(dim+1).to(device),
        torch.zeros(dim+1).to(device),device)
    for epoch in tqdm(range(num_epochs)):
        labels,features=transform_datas(trainDatas,device)
        model.alpha=grad_Calc(model.curr_x,features,labels).sum(0)
        model.snap_x=model.curr_x
        for i in range(inner_loops):
            curr_iter_count+=1
            labels,features,_ =sample_datas(  # Sample datas from set
                trainDatas,train_num,batchSize,np.ones(train_num)/train_num,device)
            grad_l=grad_Calc(model.curr_x,features,labels).mean(0)*train_num
            grad_s=grad_Calc(model.snap_x,features,labels).mean(0)*train_num
            grad=(model.curr_x+grad_l)-grad_s+model.alpha
            noise=torch.randn_like(model.curr_x).to(device)*np.sqrt(2*lr_sched[curr_iter_count-1])
            model.curr_x=model.curr_x-lr_sched[curr_iter_count-1]*grad+noise

            # Eval and Print
            if (curr_iter_count-1)%eval_interval==0:
                train_loss, train_acc, test_loss, test_acc=loss_acc_eval(
                    trainDatas,testDatas,train_num,test_num,model.curr_x,device)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
                writer.add_scalar('test loss',test_loss,global_step=curr_iter_count)
                writer.add_scalar('test acc',test_acc,global_step=curr_iter_count)
    writer.close()

    


def SVRG_trian(lr_a,lr_b,lr_gamma,num_epochs,batchSize,\
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
    save_name='SVRG'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}]'.format(lr_a,lr_b,lr_gamma)
    writer=SummaryWriter(log_dir=save_folder+save_name)
    # Print information before trian
    print(save_name)
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
    _SVRG(trainDatas,testDatas,lr_sched,num_epochs,batchSize,eval_interval,device,writer)
