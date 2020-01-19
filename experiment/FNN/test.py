import numpy as np
import matplotlib.pyplot as plt


choice=0

sgld_data=np.load('./result/SGLD/SGLD lr_a[1e-05]lr_gamma[0.5].npy')
svrg_data=np.load('./result/SVRG_LD/SVRG_LD lr_a[1e-05]lr_gamma[0.5].npy')
sghmc_data=np.load('result/SGHMC/SGHMC lr_a[0.0015]lr_gamma[0.2].npy')
srvr_data=np.load('./result/SRVR_HMC/SRVR_HMC lr_a[6e-05]lr_gamma[0.05].npy')
srm_data=np.load('./result/SRM_HMC/SRM_HMC lr_a[1.200e-03]lr_gamma[0.05]p[0.5]friction[1.0].npy')
nsrm_data=np.load('./result/NSRM_HMC/NSRM_HMC lr_a[1.000e-03]lr_gamma[0.05]reboot[1]friction[1.0].npy')
plt.plot(sgld_data[choice],label='SGLD')
plt.plot(svrg_data[choice],label='SVRG')
plt.plot(sghmc_data[choice],label='SGHMC')
plt.plot(srvr_data[choice],label='SRVR')
plt.plot(srm_data[choice],label='SRM')
plt.plot(nsrm_data[choice],label='NSRM')
plt.legend()

if choice==1 or choice==3:
    plt.ylim([0.8,0.95])
    pass
else :
    pass

plt.grid()




plt.show()
