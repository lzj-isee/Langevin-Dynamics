from SGLD_run import SGLD_train
from SVRG_LD_run import SVRG_LD_train
from SGHMC_run import SGHMC_train
from SRVR_HMC_run import SRVR_HMC_train
from SRM_HMC_run import SRM_HMC_train
from NSRM_HMC_run import NSRM_HMC_train
import torch.nn.functional as F

num_epochs=50
batchSize=500
print_interval=24
random_seed=2020
'''
SGLD_train(
    lr_a=1e-5,
    lr_gamma=0.5,
    num_epochs=num_epochs,
    batchSize=batchSize,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=2020,
    save_folder='./result/SGLD/')
'''
'''
SVRG_LD_train(
    lr_a=1e-05,
    lr_gamma=0.5,
    num_epochs=num_epochs,
    batchSize=batchSize,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/SVRG/')
'''

SGHMC_train(
    lr_a=1.1e-3,
    lr_gamma=0.2,
    num_epochs=num_epochs,
    batchSize=500,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/SGHMC/')

'''
SGHMC_train(
    lr_a=1.5e-3,
    lr_gamma=0.25,
    num_epochs=num_epochs,
    batchSize=2000,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/SGHMC2000/')
'''

SRVR_HMC_train(
    lr_a=1.2e-4,
    lr_gamma=0.05,
    friction=1.0,
    num_epochs=200,
    batchSize=6000,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/SRVR6000/')

'''
SRM_HMC_train(
    lr_a=1.2e-3,
    lr_gamma=0.05,
    p=0.5,
    friction=1.0,
    num_epochs=num_epochs,
    batchSize=batchSize,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/SRM/')
'''
'''
NSRM_HMC_train(
    lr_a=1.0e-3,
    lr_gamma=0.05,
    reboot=1,
    friction=1.0,
    num_epochs=num_epochs,
    batchSize=batchSize,
    loss_fn=F.cross_entropy,
    print_interval=print_interval,
    random_seed=random_seed,
    save_folder='./result/NSRM/')
'''

