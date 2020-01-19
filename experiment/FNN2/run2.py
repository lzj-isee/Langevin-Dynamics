from SGLD_run import SGLD_train
from SVRG_LD_run import SVRG_LD_train
from SGHMC_run import SGHMC_train
from SRVR_HMC_run import SRVR_HMC_train
from SRM_HMC_run import SRM_HMC_train
from NSRM_HMC_run import NSRM_HMC_train
from SRVRt import SRVRt_HMC_train
import torch.nn.functional as F
import numpy as np

num_epochs=50
batchSize=500
print_interval=24
random_seed=2020


res=np.array([2,3,4,6,7,8,9])
for re in res:
    SRVRt_HMC_train(
        lr_a=6e-5,
        lr_gamma=0.05,
        friction=1.0,
        re=re,
        num_epochs=num_epochs,
        batchSize=batchSize,
        loss_fn=F.cross_entropy,
        print_interval=print_interval,
        random_seed=random_seed,
        save_folder='./result/SRVRt/')

lr_as=np.array([1.0,1.1,1.2,1.3,1.4])*1e-3
lr_gammas=np.array([0.15,0.2,0.25])
for lr_a in lr_as:
    for lr_gamma in lr_gammas:
        SGHMC_train(
            lr_a,
            lr_gamma,
            num_epochs=num_epochs,
            batchSize=batchSize,
            loss_fn=F.cross_entropy,
            print_interval=print_interval,
            random_seed=random_seed,
            save_folder='./result/SGHMC/',
            device='cpu')

