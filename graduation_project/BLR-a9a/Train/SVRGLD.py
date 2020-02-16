from sys import path
path.append('./')
from Algorithms.SVRGLD import Alg_SVRGLD
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Load_DataSet import load_dataset_sp

def _SVRGLD_it(trainSet,testSet,writer,device,**kw):
    train_num=len(trainSet['labels'])
    test_num=len(testSet['labels'])
    dim=trainSet['features'].shape[1]
    inner_loops=round(train_num/kw['batchSize'])
    curr_iter_count=0
    model=Alg_SVRGLD(
        torch.zeros(dim+1).to(device),
        torch.zeros(dim+1).to(device),
        device=device)
    for epoch in tqdm(range(kw['num_epochs'])):
        model.update(trainSet)
        for i in range(inner_loops):
            curr_iter_count+=1
            model.Sample_Datas(trainSet,train_num,kw['batchSize'],np.ones(train_num)/train_num)
            model.Grads_Calc()
            model.snap_Grads_Calc()
            grad=(model.curr_x+model.grads.mean(0)*train_num)\
                -(model.grad_snap-model.grad_alpha)*train_num
            eta=kw['lr_a']*(curr_iter_count+kw['lr_b'])**(-kw['lr_gamma'])
            noise=torch.randn_like(model.curr_x).to(model.device)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-grad*eta+noise

            # Eval and Print
            if curr_iter_count>=kw['burn_in']:
                model.burn_in=True
            if (curr_iter_count-1)%kw['eval_interval']==0:
                model.lr_new=eta
                train_loss, train_acc, test_loss, test_acc=model.loss_acc_eval(
                    trainSet,testSet,train_num,test_num)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('test loss',test_loss,global_step=curr_iter_count)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
                writer.add_scalar('test acc',test_acc,global_step=curr_iter_count)
                model.lr_sum=model.lr_sum+model.lr_new
    writer.close()



def SVRGLD_train(**kw):
    # Use GPU if cuda is available
    if torch.cuda.is_available() and kw['use_gpu']==True:
        device=torch.device('cuda:0')
        print(torch.cuda.get_device_name(0))
    else:
        device=torch.device('cpu')
        print('use cpu')
    # Creat the save folder and name for result
    save_name='SVRGLD'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}]'.format(kw['lr_a'],kw['lr_b'],kw['lr_gamma'])
    writer=SummaryWriter(log_dir=kw['save_folder']+save_name)
    # Print information before train
    print('SVRGLD lr[{:.2e},{:.2f},{:.2f}]'.format(kw['lr_a'],kw['lr_b'],kw['lr_gamma']))
    # Set the random seed
    torch.manual_seed(kw['random_seed'])
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet,testSet=load_dataset_sp()
    # Main function
    _SVRGLD_it(trainSet,testSet,writer,device,**kw)