import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

path='./MSE_ALL'

color={
    'SGLD':'grey',
    'SGHMC':'olive',
    'SVRG_LD':'yellow',
    'SG_UL_MCMC':'greenyellow',
    'SRVR_HMC':'cyan',
    'SRM_HMC':'deeppink',
    'NSRM_HMC':'red'}

for files in tqdm(os.listdir(path)):
    position=files.rfind('_')
    alg_name=files[0:position]
    if alg_name=='SVRG_LD':
        datas=np.load(path+'/'+files)
        plt.plot(datas,color=color[alg_name])

plt.ylim([-0.05,3])
plt.show()