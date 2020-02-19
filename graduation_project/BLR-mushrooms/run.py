import pretty_errors
from Train.SGLD import SGLD_train
from Train.SGD import SGD_train
from Train.SVRGLD import SVRGLD_train
from Train.RAISLD import RAISLD_train
from Train.RAISLDe import RAISLDe_train

settings={
    'lr_a':4e-4,    # common setting
    'lr_b':0,       # common setting
    'lr_gamma':0.5, # common setting
    'num_epochs':100,   # common setting
    'batchSize':32,    # common setting
    'eval_interval':100, # common setting
    'burn_in':6000,  # common setting, the threshold of burn in
    'random_seed':1,    # common setting
    'alpha':0.1,    # RAIS only
    'd':1,          # RAIS only
    'save_folder':'./result/result3b/', 
    'use_gpu':False}

SGLD_train(**settings)
#SGD_train(**settings)
SVRGLD_train(**settings)
RAISLD_train(**settings)
RAISLDe_train(**settings)


with open(settings['save_folder']+'settings.md',mode='w') as f:
    for key in settings:
        f.write(key+": "+'{}'.format(settings[key])+'  \n')
    