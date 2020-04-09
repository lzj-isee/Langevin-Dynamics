from Algorithm.SVRGLD import Alg_SVRGLD
from Load_Dataset import load_dataset,load_ref
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def _SVRGLD_it(trainSet,ref,save_path,**kw):
    x_list=[]
    train_num=trainSet.shape[0]
    dim=trainSet.shape[1]
    inner_loops=round(train_num/kw['batchSize'])
    model=Alg_SVRGLD(
        np.zeros(dim),
        np.zeros(dim))
    curr_iter_count=0
    for epoch in tqdm(range((int)(kw['num_epochs']))):
        model.update(trainSet)
        for i in range(inner_loops):
            curr_iter_count+=1
            model.Sample_Datas(
                trainSet,
                train_num,
                kw['batchSize'],
                np.ones(train_num)/train_num)
            model.Grads_Calc()
            model.snap_Grads_Calc()
            grad=model.grads.mean(0)-model.grad_snap+model.grad_alpha
            eta=kw['lr_a']*(curr_iter_count+kw['lr_b'])**(-kw['lr_gamma'])
            noise=np.random.normal(0,1,dim)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-grad*eta+noise
            x_list.append(model.curr_x)
    x_list=np.array(x_list).astype(np.float32)
    model.save_figure(x_list,ref,save_path,'SVRGLD')
    np.save(save_path,x_list)

def SVRGLD_sample(**kw):
    # Creat the save folder and name for result
    save_name='SVRGLD'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}]'.format(kw['lr_a'],kw['lr_b'],kw['lr_gamma'])
    if not os.path.exists(kw['save_folder']+'SVRGLD'):
        os.makedirs(kw['save_folder']+'SVRGLD')
    save_path=kw['save_folder']+'SVRGLD/'+save_name
    # Print information before train
    print(save_name)
    # Set the random seed
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet=load_dataset()
    ref=load_ref()
    # Main function
    _SVRGLD_it(trainSet,ref,save_path,**kw)
