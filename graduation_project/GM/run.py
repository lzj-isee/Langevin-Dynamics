import pretty_errors
from Train.SGLD import SGLD_sample
from Train.SVRGLD import SVRGLD_sample
from Train.RAISLD import RAISLD_sample

settings={
    'lr_a':1,
    'lr_b':0,
    'lr_gamma':0.5,
    'batchSize':10,
    'num_epochs':2e4,
    'alpha':0.1,
    'd':1,
    'save_folder':'./result/',
    'random_seed':1}
#SGLD_sample(**settings)
#SVRGLD_sample(**settings)
#RAISLD_sample(**settings)