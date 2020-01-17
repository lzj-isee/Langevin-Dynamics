from SRVR_HMC_run import SRVR_HMC_train
from SRM_HMC_run import SRM_HMC_train
import torch.nn.functional as F

import numpy as np
'''
SRM_HMC_train(
    lr_a=0.75e-3,
    lr_gamma=0.1,
    p=1,
    num_epochs=10,
    batchSize=500,
    loss_fn=F.cross_entropy,
    print_interval=12,
    random_seed=2020,
    save_folder='./result/SRM_HMC/',
    device='cpu')
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