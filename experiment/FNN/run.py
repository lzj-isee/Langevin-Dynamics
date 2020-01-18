from SRVR_HMC_run import SRVR_HMC_train
from SRM_HMC_run import SRM_HMC_train
from NSRM_HMC_run import NSRM_HMC_train
from SGLD_run import SGLD_train
from SVRG_LD_run import SVRG_LD_train
import torch.nn.functional as F

import numpy as np

'''
lr_as=np.array([0.1,0.15,0.2,0.25,0.3,0.35])*1e-4
lr_gammas=np.array([0.0])
for lr_a in lr_as:
    for lr_gamma in lr_gammas:
        SRVR_HMC_train(
            lr_a,
            lr_gamma,
            friction=0.5,
            num_epochs=10,
            batchSize=500,
            loss_fn=F.cross_entropy,
            print_interval=12,
            random_seed=2020,
            save_folder='./result/SRVR_HMC/',
            device='cpu')
'''
'''
lr_as=np.array([1.2,1.0,0.8,0.6,0.4,0.2,0.1])*1e-3
lr_gammas=np.array([0.0,0.05,0.15,0.2,0.25])
for lr_a in lr_as:
    for lr_gamma in lr_gammas:
        if lr_gamma==0.0:
            friction=0.3
        else:
            friction=1.0
        SRM_HMC_train(
            lr_a,
            lr_gamma,
            p=0.5,
            friction=friction,
            num_epochs=10,
            batchSize=500,
            loss_fn=F.cross_entropy,
            print_interval=12,
            random_seed=2020,
            save_folder='./result/SRM_HMC/',
            device='cpu')
'''
lr_as=np.array([1.4,1.2,1.0,0.8,0.6])*1e-3
lr_gammas=np.array([0.0,0.05,0.15,0.2])
for lr_a in lr_as:
    for lr_gamma in lr_gammas:
        if lr_gamma==0.0:
            friction=0.3
        else:
            friction=1.0
        NSRM_HMC_train(
            lr_a,
            lr_gamma,
            reboot=1,
            friction=friction,
            num_epochs=10,
            batchSize=500,
            loss_fn=F.cross_entropy,
            print_interval=12,
            random_seed=2020,
            save_folder='./result/NSRM_HMC/',
            device='cpu')