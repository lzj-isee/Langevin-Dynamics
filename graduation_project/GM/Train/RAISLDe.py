from Algorithm.RAISLD import Alg_RAISLD
from Load_Dataset import load_dataset,load_ref
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def _RAISLDe_it(trainSet,ref,save_path,**kw):
    x_list=[]
    train_num=trainSet.shape[0]
    dim=trainSet.shape[1]
    inner_loops=round(train_num/kw['batchSize'])
    model=Alg_RAISLD(
        np.zeros(dim),
        train_num,kw['alpha'],kw['d'])
    curr_iter_count=0
    model.initialize(trainSet)
    for epoch in tqdm(range((int)(kw['num_epochs']))):
        for i in range(inner_loops):
            curr_iter_count+=1
            model.Sample_Datas(
                trainSet,
                train_num,
                kw['batchSize'],
                model.p)
            model.Grads_Calc()
            model.average_grads()
            grad=model.grad_avg
            model.update()
            #eta=kw['lr_a']*(round(model.t.item())+kw['lr_b'])**(-kw['lr_gamma'])*model.r.item()
            eta=kw['lr_a']*(curr_iter_count+kw['lr_b'])**(-kw['lr_gamma'])
            noise=np.random.normal(0,1,dim)*np.sqrt(2*eta)
            model.curr_x=model.curr_x-grad*eta+noise
            x_list.append(model.curr_x)
    x_list=np.array(x_list).astype(np.float32)
    model.save_figure(x_list,ref,save_path,'RAISLDe')
    np.save(save_path,x_list)

def RAISLDe_sample(**kw):
    # Creat the save folder and name for result
    save_name='RAISLDe'+' '+\
        'lr[{:.2e},{:.2f},{:.2f}] alpha[{:.2f}] d[{:.1f}]'.format(\
            kw['lr_a'],kw['lr_b'],kw['lr_gamma'],kw['alpha'],kw['d'])
    if not os.path.exists(kw['save_folder']+'RAISLDe'):
        os.makedirs(kw['save_folder']+'RAISLDe')
    save_path=kw['save_folder']+'RAISLDe/'+save_name
    # Print information before train
    print(save_name)
    # Set the random seed
    np.random.seed(kw['random_seed'])
    # Load DataSet as sparse matrix
    trainSet=load_dataset()
    ref=load_ref()
    # Main function
    _RAISLDe_it(trainSet,ref,save_path,**kw)
