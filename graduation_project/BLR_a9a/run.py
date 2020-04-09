import pretty_errors
from Train.SGLD import SGLD_train
from Train.RAISLDe import RAISLDe_train

settings={
    'lr_a':3e-5,    # common setting
    'lr_b':0,       # common setting
    'lr_gamma':0.5, # common setting
    'num_epochs':50,   # common setting
    'batchSize':50,    # common setting
    'eval_interval':20, # common setting
    'burn_in':500,  # common setting, the threshold of burn in
    'random_seed':1,    # common setting
    'alpha':0.1,    # RAIS only
    'd':3.0,          # RAIS only
    'save_folder':'./result/', 
    'use_gpu':False}
print(settings)
SGLD_train(**settings)
#SVRGLD_train(**settings)
#RAISLD_train(**settings)
RAISLDe_train(**settings)


with open(settings['save_folder']+'settings.md',mode='w') as f:
    for key in settings:
        f.write(key+": "+'{}'.format(settings[key])+'  \n')
    