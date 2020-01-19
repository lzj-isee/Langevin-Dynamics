import numpy as np
import matplotlib.pyplot as plt


choice=1

sgld_data=np.load('./result/SGLD/SGLD lr_a[1.000e-05]lr_gamma[0.5].npy')
svrg_data=np.load('./result/SVRG/SVRG_LD lr_a[1.000e-05]lr_gamma[0.5].npy')
sghmc_data=np.load('./result/SGHMC/SGHMC lr_a[1.500e-03]lr_gamma[0.2].npy')
srvr_data_in10_500=np.load('./result/SRVR_in10_HMC/SRVR_HMC lr_a[6.000e-05]lr_gamma[0.05]friction[1.0]500.npy')
srvr_data_in10_2000=np.load('./result/SRVR_in10_HMC/SRVR_HMC lr_a[6.000e-05]lr_gamma[0.05]friction[1.0]2000.npy')
srvr_data_in5_500=np.load('./result/SRVR_in5_HMC/SRVR_HMC lr_a[6.000e-05]lr_gamma[0.05]friction[1.0]500.npy')
srvr_data_in5_500=srvr_data_in5_500[:,0:len(srvr_data_in5_500[0]):2]
srvr_data_in5_2000=np.load('./result/SRVR_in5_HMC/SRVR_HMC lr_a[6.000e-05]lr_gamma[0.05]friction[1.0]2000.npy')
srm_data=np.load('./result/SRM/SRM_HMC lr_a[1.200e-03]lr_gamma[0.05]p[0.5]friction[1.0].npy')
nsrm_data=np.load('./result/NSRM/NSRM_HMC lr_a[1.000e-03]lr_gamma[0.05]reboot[1]friction[1.0].npy')
plt.plot(sgld_data[choice],label='SGLD')
plt.plot(svrg_data[choice],label='SVRG')
plt.plot(sghmc_data[choice],label='SGHMC')
#plt.plot(srvr_data[choice],label='SRVR')
plt.plot(srvr_data_in10_500[choice],label='SRVR-10-500')
plt.plot(srvr_data_in10_2000[choice],label='SRVR-10-2000')
plt.plot(srvr_data_in5_500[choice],label='SRVR-5-500')
plt.plot(srvr_data_in5_2000[choice],label='SRVR-5-2000')
plt.plot(srm_data[choice],label='SRM')
plt.plot(nsrm_data[choice],label='NSRM')
plt.legend()

if choice==1 or choice==3:
    plt.ylim([0.5,1.0])
    pass
else :
    pass

plt.grid()
plt.show()
