import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
mean=np.load('./ref_mean.npy')

sgld=np.load('./SGLD_result/seed[2020]setting[1,0,0.1].npy')
sg_ul_mcmc=np.load('./SG_UL_MCMC_result/seed[2020,2020]setting[0.10,0,0.05].npy')
sghmc=np.load('./SGHMC_result/seed[2020]setting[1,0,0.05].npy')
svrg_ld=np.load('./SVGR_LD_result/seed[2020,2020]setting[1.80,0,0.15].npy')
srvr_hmc=np.load('./SRVR_HMC_result/seed[2020,2020]setting[0.04,0,0.00].npy')
srm_hmc=np.load('./SRM_HMC_result/seed[2020,2020]setting[0.06,0,0.00].npy')

lens=len(sgld)
sgld_mse=np.zeros(lens)
sg_ul_mcmc_mse=np.zeros(lens)
sghmc_mse=np.zeros(lens)
svrg_ld_mse=np.zeros(lens)
srvr_hmc_mse=np.zeros(lens)
srm_hmc_mse=np.zeros(lens)

sgld_temp=0
sg_ul_mcmc_temp=0
sghmc_temp=0
svrg_ld_temp=0
srvr_hmc_temp=0
srm_hmc_temp=0

for i in tqdm(range(lens)):
    sgld_temp=(sgld_temp*i+sgld[i])/(i+1)
    sgld_mse[i]=np.linalg.norm(sgld_temp-mean)**2

    sg_ul_mcmc_temp=(sg_ul_mcmc_temp*i+sg_ul_mcmc[i])/(i+1)
    sg_ul_mcmc_mse[i]=np.linalg.norm(sg_ul_mcmc_temp-mean)**2

    sghmc_temp=(sghmc_temp*i+sghmc[i])/(i+1)
    sghmc_mse[i]=np.linalg.norm(sghmc_temp-mean)**2

    svrg_ld_temp=(svrg_ld_temp*i+svrg_ld[i])/(i+1)
    svrg_ld_mse[i]=np.linalg.norm(svrg_ld_temp-mean)**2

    srvr_hmc_temp=(srvr_hmc_temp*i+srvr_hmc[i])/(i+1)
    srvr_hmc_mse[i]=np.linalg.norm(srvr_hmc_temp-mean)**2

    srm_hmc_temp=(srm_hmc_temp*i+srm_hmc[i])/(i+1)
    srm_hmc_mse[i]=np.linalg.norm(srm_hmc_temp-mean)**2

np.save('./MSE/SGLD.npy',sgld_mse)
np.save('./MSE/SG_UL_MCMC.npy',sg_ul_mcmc_mse)
np.save('./MSE/SGHMC.npy',sghmc_mse)
np.save('./MSE/SVRG_LD.npy',svrg_ld_mse)
np.save('./MSE/SRVR_HMC.npy',srvr_hmc_mse)
np.save('./MSE/SRM_HMC.npy',srm_hmc_mse)