from sys import path
path.append('./')
from Algorithms.RAISLD import Alg_RAISLD
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Load_DataSet import load_dataset_sp



def _RAISLDe_it(trainSet,writer,device,**kw):
    train_num=len(trainSet['labels'])
    dim=trainSet['features'].shape[1]
    inner_loops=round(train_num/kw['batchSize'])
    curr_iter_count=0
    model=Alg_RAISLD(
        torch.zeros(dim+1).to(device),
        train_num,
        kw['alpha'],
        kw['d'],
        device=device)
    model.initialize(trainSet)
    for epoch in tqdm(range(kw['num_epochs'])):
        for i in range(inner_loops):
            curr_iter_count+=1
            model.Sample_Datas(trainSet,train_num,kw['batchSize'],model.p.numpy())
            model.Grads_Calc()
            model.average_grads()
            grad=model.curr_x+model.grad_avg*train_num
            model.update()
            #eta=kw['lr_a']*(round(model.t.item())+kw['lr_b'])**(-kw['lr_gamma'])*model.r.item()
            eta=kw['lr_a']*(curr_iter_count+kw['lr_b'])**(-kw['lr_gamma'])
            noise=torch.randn_like(model.curr_x).to(model.device)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-grad*eta+noise

            # Eval and Print
            if curr_iter_count>=kw['burn_in'] and kw['burn_in']!=False:
                model.burn_in=True
            if (curr_iter_count-1)%kw['eval_interval']==0:
                model.lr_new=eta
                train_loss, train_acc,=model.loss_acc_eval(
                    trainSet,train_num)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
                model.lr_sum=model.lr_sum+model.lr_new
    writer.close()



def RAISLDe_train(**kw):
    # Use GPU if cuda is available
    if torch.cuda.is_available() and kw['use_gpu']==True:
        device=torch.device('cuda:0')
        print(torch.cuda.get_device_name(0))
    else:
        device=torch.device('cpu')
        print('use cpu')
    # Creat the save folder and name for result
    save_name='RAISLDe'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}] alpha[{:.2f}] d[{:.1f}]'.format(\
            kw['lr_a'],kw['lr_b'],kw['lr_gamma'],kw['alpha'],kw['d'])
    writer=SummaryWriter(log_dir=kw['save_folder']+save_name)
    # Print information before train
    print(save_name)
    # Set the random seed
    torch.manual_seed(kw['random_seed'])
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet=load_dataset_sp()
    # Main function
    _RAISLDe_it(trainSet,writer,device,**kw)

