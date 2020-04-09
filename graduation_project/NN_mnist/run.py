import pretty_errors
from Train.SGD_train import SGD_train
from Train.SGLD_train import SGLD_train
from Train.RAIS_trian import RAIS_train


settings={
    'lr_a':7e-6,    # common setting
    'lr_gamma':0.5, # common setting
    'lr_threshold':0.000, # common settings
    'num_epochs':1600,   # common setting
    'batchSize':500,    # common setting
    'eval_interval':100, # common setting
    'burn_in':3000,  # common setting, the threshold of burn in
    'random_seed':1,    # common setting
    'alpha':0.1,    # RAIS only
    'd':0.0,          # RAIS only
    'save_folder':'./result/', 
    'use_gpu':[True,'cuda:0']}
print(settings)
#SGD_train(**settings)
RAIS_train(**settings)
SGLD_train(**settings)




with open(settings['save_folder']+'settings.md',mode='w') as f:
    for key in settings:
        f.write(key+": "+'{}'.format(settings[key])+'  \n')