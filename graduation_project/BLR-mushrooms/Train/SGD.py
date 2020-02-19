from sys import path
path.append('./')
from Algorithms.SGLD import Alg_SGLD
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Load_DataSet import load_dataset_sp



def _SGD_it(trainSet,writer,device,**kw):
    train_num=len(trainSet['labels'])
    dim=trainSet['features'].shape[1]
    inner_loops=round(train_num/kw['batchSize'])
    curr_iter_count=0
    model=Alg_SGLD(
        torch.zeros(dim+1).to(device),
        device=device)
    for epoch in tqdm(range(kw['num_epochs'])):
        for i in range(inner_loops):
            curr_iter_count+=1
            model.Sample_Datas(trainSet,train_num,kw['batchSize'],np.ones(train_num)/train_num)
            model.Grads_Calc()
            grad=model.curr_x+model.grads.mean(0)*train_num
            eta=kw['lr_a']*(curr_iter_count+kw['lr_b'])**(-kw['lr_gamma'])
            #noise=torch.randn_like(model.curr_x).to(model.device)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-grad*eta

            # Eval and Print
            if (curr_iter_count-1)%kw['eval_interval']==0:
                train_loss, train_acc=model.loss_acc_eval(
                    trainSet,train_num)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
    writer.close()



def SGD_train(**kw):
    # Use GPU if cuda is available
    if torch.cuda.is_available() and kw['use_gpu']==True:
        device=torch.device('cuda:0')
        print(torch.cuda.get_device_name(0))
    else:
        device=torch.device('cpu')
        print('use cpu')
    # Creat the save folder and name for result
    save_name='SGD'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}]'.format(kw['lr_a'],kw['lr_b'],kw['lr_gamma'])
    writer=SummaryWriter(log_dir=kw['save_folder']+save_name)
    # Print information before train
    print('SGD lr[{:.2e},{:.2f},{:.2f}]'.format(kw['lr_a'],kw['lr_b'],kw['lr_gamma']))
    # Set the random seed
    torch.manual_seed(kw['random_seed'])
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet=load_dataset_sp()
    # Main function
    _SGD_it(trainSet,writer,device,**kw)

