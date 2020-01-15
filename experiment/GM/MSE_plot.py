import numpy as np
import matplotlib.pyplot as plt



sgld_mse=np.load('./MSE_ALL/SGLD_seed[2020]setting[0.8,0,0.08].npy')[0:500000:10000]
sg_ul_mcmc_mse=np.load('./MSE_ALL/SG_UL_MCMC_seed[2020,2020]setting[0.10,0,0.05].npy')[0:500000:10000]
sghmc_mse=np.load('./MSE_ALL/SGHMC_seed[2020]setting[1.2,0,0.05].npy')[0:500000:10000]
svrg_ld_mse=np.load('./MSE_ALL/SVRG_LD_seed[2020,2020]setting[0.7,0,0.01].npy')[0:500000:10000]
srvr_hmc_mse=np.load('./MSE_ALL/SRVR_HMC_seed[2020,2020]setting[0.07,0,0.00].npy')[0:500000:10000]
srm_hmc_mse=np.load('./MSE_ALL/SRM_HMC_seed[2020,2020]setting[0.08,0,0.05].npy')[0:500000:10000]
nsrm_hmc_mse=np.load('./MSE_ALL/NSRM_HMC_seed[2020,2020]setting[0.06,0,0.01]interval[100].npy')[0:500000:10000]
x=np.linspace(0,500000,50)
plt.plot(x,sgld_mse,label='SGLD',color='blue')
plt.plot(x,sghmc_mse,label='SGHMC',color='cyan')
plt.plot(x,svrg_ld_mse,label='SVRG_LD',color='gold')
plt.plot(x,sg_ul_mcmc_mse,label='SG_UL_MCMC',color='deeppink')
plt.plot(x,srvr_hmc_mse,label='SRVR_HMC',color='lime')
plt.plot(x,srm_hmc_mse,label='SRM_HMC',color='dodgerblue')
plt.plot(x,nsrm_hmc_mse,label='NSRM_HMC',color='red')
plt.legend()
plt.ylim([-0.01,3])
plt.xlabel('Number of iterations')
plt.ylabel('Mean square error')
plt.show()