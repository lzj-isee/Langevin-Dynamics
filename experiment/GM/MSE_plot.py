import numpy as np
import matplotlib.pyplot as plt

path='./MSE/'

sgld_mse=np.load(path+'SGLD.npy')
sg_ul_mcmc_mse=np.load(path+'SG_UL_MCMC.npy')
sghmc_mse=np.load(path+'SGHMC.npy')
svrg_ld_mse=np.load(path+'SVRG_LD.npy')
srvr_hmc_mse=np.load(path+'SRVR_HMC.npy')
srm_hmc_mse=np.load(path+'SRM_HMC.npy')

plt.plot(sgld_mse,label='SGLD')
plt.plot(sg_ul_mcmc_mse,label='SG_UL_MCMC')
plt.plot(sghmc_mse,label='SGHMC')
plt.plot(svrg_ld_mse,label='SVRG_LD')
plt.plot(srvr_hmc_mse,label='SRVR_HMC')
plt.plot(srm_hmc_mse,label='SRM_HMC')
plt.legend()
plt.ylim([0,3])
plt.show()