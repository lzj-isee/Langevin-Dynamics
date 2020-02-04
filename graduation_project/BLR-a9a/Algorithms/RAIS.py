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
from Class.RAIS import model_RAIS

def _RAIS(trainDatas,testDatas,total_loops,lr_a,lr_b,lr_gamma,\
    num_epochs,batchSize,alpha,d,eval_interval,device,writer):
    inner_loops=int(total_loops/num_epochs)
    train_num=len(trainDatas['labels'])
    test_num=len(testDatas['labels'])
    dim=trainDatas['features'].shape[1]
    curr_iter_count=0
    model=model_RAIS(
        torch.zeros(dim+1).to(device),
        train_num,alpha=alpha,d=d,device=device)
    # initialize
    labels,features=transform_datas(trainDatas,device)
    grads=grad_Calc(model.curr_x,features,labels)
    model.initialize(grads)
    for epoch in tqdm(range(num_epochs)):
        for i in range(inner_loops):
            curr_iter_count+=1
            labels,features,indices=sample_datas(  # Sample datas from set
                trainDatas,train_num,batchSize,model.p.numpy(),device)
            grads=grad_Calc(model.curr_x,features,labels)
            grad_l=model.avg_grad(grads,indices)*train_num
            model.update(grads,indices)
            grad=model.curr_x+grad_l
            eta=lr_a*(round(model.t.item())+lr_b+1)**(-lr_gamma)*model.r.item()
            noise=torch.randn_like(model.curr_x).to(device)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-eta*grad+noise

            # Eval and Print
            if (curr_iter_count-1)%eval_interval==0:
                train_loss, train_acc, test_loss, test_acc=loss_acc_eval(
                    trainDatas,testDatas,train_num,test_num,model.curr_x,device)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
                writer.add_scalar('test loss',test_loss,global_step=curr_iter_count)
                writer.add_scalar('test acc',test_acc,global_step=curr_iter_count)
    writer.close()

    


def RAIS_trian(lr_a,lr_b,lr_gamma,num_epochs,batchSize,\
    alpha,d,eval_interval,random_seed,save_folder,use_gpu=True,\
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
    save_name='RAIS'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}] alpha[{:.2f}] d[{:.1f}]'.format(\
        lr_a,lr_b,lr_gamma,alpha,d)
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
    _RAIS(trainDatas,testDatas,total_loops,lr_a,lr_b,lr_gamma,\
        num_epochs,batchSize,alpha,d,eval_interval,device,writer)
