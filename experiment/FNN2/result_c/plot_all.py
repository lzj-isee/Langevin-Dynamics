import numpy as np
import matplotlib.pyplot as plt


choice=3

sgld_data=np.load('./result_c/SGLD/SGLD lr_a[1.000e-05]lr_gamma[0.5].npy')
svrg_data=np.load('./result_c/SVRG/SVRG_LD lr_a[1.000e-05]lr_gamma[0.5].npy')
sghmc_data_15_20=np.load('./result_c/SGHMC/SGHMC lr_a[1.500e-03]lr_gamma[0.2].npy')
sghmc_data_15_25=np.load('./result_c/SGHMC/SGHMC lr_a[1.500e-03]lr_gamma[0.25].npy')
sghmc_data_12_20=np.load('./result_c/SGHMC/SGHMC lr_a[1.200e-03]lr_gamma[0.2].npy')
sghmc_data_11_20=np.load('./result_c/SGHMC/SGHMC lr_a[1.100e-03]lr_gamma[0.2].npy')
sghmc_data_10_25=np.load('./result_c/SGHMC/SGHMC lr_a[1.000e-03]lr_gamma[0.25].npy')
sghmc_data_10_20=np.load('./result_c/SGHMC/SGHMC lr_a[1.000e-03]lr_gamma[0.2].npy')
sghmc_data_10_15=np.load('./result_c/SGHMC/SGHMC lr_a[1.000e-03]lr_gamma[0.15].npy')
srvr_data_re5=np.load('./result_c/SRVRt/SRVRt lr_a[6.000e-05]lr_gamma[0.05]friction[1.0]re[5].npy')
srm_data=np.load('./result_c/SRM/SRM_HMC lr_a[1.200e-03]lr_gamma[0.05]p[0.5]friction[1.0].npy')
nsrm_data=np.load('./result_c/NSRM/NSRM_HMC lr_a[1.000e-03]lr_gamma[0.05]reboot[1]friction[1.0].npy')
plt.plot(sgld_data[choice],label='SGLD')
plt.plot(svrg_data[choice],label='SVRG')
plt.plot(sghmc_data_15_25[choice],label='SGHMC-1.5-2.5')
plt.plot(sghmc_data_15_20[choice],label='SGHMC-1.5-2.0')
plt.plot(sghmc_data_12_20[choice],label='SGHMC-1.2-2.0')
plt.plot(sghmc_data_11_20[choice],label='SGHMC-1.1-2.0')
plt.plot(sghmc_data_10_25[choice],label='SGHMC-1.0-2.5')
plt.plot(sghmc_data_10_20[choice],label='SGHMC-1.0-2.0')
plt.plot(sghmc_data_10_15[choice],label='SGHMC-1.0-1.5')
plt.plot(srvr_data_re5[choice],label='SRVR-re5')
plt.plot(srm_data[choice],label='SRM')
plt.plot(nsrm_data[choice],label='NSRM')
plt.legend()
if choice==1 or choice==3:
    plt.ylim([0.9,0.98])
else :
    plt.ylim([0.0,1.0])

plt.grid()
plt.show()