import os
import numpy as np
from tqdm import tqdm


def MSE_Calc(datas):
    mean=np.load('./ref_mean.npy')
    lens=len(datas)
    temp=0
    mse=np.zeros(lens)
    for i in range(lens):
        temp=(temp*i+datas[i])/(i+1)
        mse[i]=np.linalg.norm(temp-mean)**2
    return mse


path=[
    './SGLD_result_it',
    './SGHMC_result_it',
    './SVGR_LD_result',
    './SG_UL_MCMC_result_it',
    './SRVR_HMC_result_it',
    './SRM_HMC_result_it',
    './NSRM_HMC_result_it3']

name=[
    'SGLD',
    'SGHMC',
    'SVRG_LD',
    'SG_UL_MCMC',
    'SRVR_HMC',
    'SRM_HMC',
    'NSRM_HMC']

for i in range(2,3):
    print(path[i])
    for files in tqdm(os.listdir(path[i])):
        if files[len(files)-3:len(files)]=='npy':
            datas=np.load(path[i]+'/'+files)
            file_name=name[i]+'_'+files
            mse=MSE_Calc(datas)
            np.save('./MSE_ALL/'+file_name,mse)
