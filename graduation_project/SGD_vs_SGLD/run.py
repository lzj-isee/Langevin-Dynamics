import pretty_errors
from Train.SGD_train import SGD_train
from Train.SGLD_train import SGLD_train
from Train.RAIS_trian import RAIS_train
from Train.SGD_LD import SGDLD_train


settings={
    'lr_a':2e-7,    # common setting
    'lr_gamma':0.5, # common setting
    'lr_threshold':0.000, # common settings
    'num_epochs':100,   # common setting
    'num_epochs_2':100,
    'batchSize':50,    # common setting
    'eval_interval':100, # common setting
    'burn_in':False,  # abandon
    'random_seed':1,    # common setting
    'alpha':0.1,    # RAIS only
    'd':0.0,          # RAIS only
    'save_folder':'./result/result1/', 
    'use_gpu':[True,'cuda:0']}
print(settings)
SGD_train(**settings)
SGDLD_train(**settings)



with open(settings['save_folder']+'settings.md',mode='w') as f:
    for key in settings:
        f.write(key+": "+'{}'.format(settings[key])+'  \n')