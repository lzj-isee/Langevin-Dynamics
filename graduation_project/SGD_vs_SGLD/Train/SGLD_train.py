from sys import path
path.append('./')
from Optimizer.SGLD_op import SGLD_op
from Model import NET
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Load_DataSet import load_dataset



def _SGLD_it(trainSet,testSet,writer,device,**kw):
    train_num=len(trainSet['labels'])
    test_num=len(testSet['labels'])
    inner_loops=round(train_num/kw['batchSize'])
    curr_iter_count=0
    # Prepare model and optimizer
    model=NET()
    model.to(device)
    optimizer=SGLD_op(\
        model.parameters(),
        lr_a=kw['lr_a'],
        lr_gamma=kw['lr_gamma'],
        device=device)
    for epoch in tqdm(range(kw['num_epochs'])):
        for i in range(inner_loops):
            curr_iter_count+=1
            # Sample datas
            indices=np.random.choice(\
                a=np.arange(train_num),
                size=kw['batchSize'],
                replace=True,
                p=np.ones(train_num)/train_num)
            labels=(trainSet['labels'])[indices]
            images=(trainSet['images'])[indices]
            # Forward
            _,outputs=model(images)
            loss=torch.nn.functional.cross_entropy(outputs,labels,reduction='mean')*train_num
            # Zero grad
            optimizer.zero_grad()
            # Backward
            loss.backward()
            # Calculate eta
            eta=kw['lr_a']*(curr_iter_count)**(-kw['lr_gamma'])
            if eta < kw['lr_a']*kw['lr_threshold'] : eta=kw['lr_a']*kw['lr_threshold']
            # Update (add eta*grad and noise)
            optimizer.step(eta=eta)


            # Eval and Print
            if curr_iter_count>=kw['burn_in'] and kw['burn_in']!=False:
                model.burn_in=True
            if (curr_iter_count-1)%kw['eval_interval']==0:
                model.lr_new=eta
                train_acc,train_loss,test_acc,test_loss=model.loss_acc_eval(
                    trainSet,testSet,train_num,test_num)
                writer.add_scalar('train acc',train_acc,global_step=curr_iter_count)
                writer.add_scalar('train loss',train_loss,global_step=curr_iter_count)
                writer.add_scalar('test acc',test_acc,global_step=curr_iter_count)
                writer.add_scalar('test loss',test_loss,global_step=curr_iter_count)
                
    writer.close()



def SGLD_train(**kw):
    # Use GPU if cuda is available
    if torch.cuda.is_available() and kw['use_gpu'][0]==True:
        device=torch.device(kw['use_gpu'][1])
        print(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(kw['random_seed'])
    else:
        device=torch.device('cpu')
        print('use cpu')
    # Creat the save folder and name for result
    save_name='SGLD'+' '+\
        'lr[{:.2e},{:.2f},{:.3f}]'.format(kw['lr_a'],kw['lr_gamma'],kw['lr_threshold'])
    writer=SummaryWriter(log_dir=kw['save_folder']+save_name)
    # Print information before train
    print(save_name)
    # Set the random seed
    torch.manual_seed(kw['random_seed'])
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet,testSet=load_dataset(device=device)
    # Main function
    _SGLD_it(trainSet,testSet,writer,device,**kw)

