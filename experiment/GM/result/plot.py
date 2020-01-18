import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
path='./result/datas/'
def plot_fig(samples,name):
    Ys=np.load('./ref.npy')
    Xs=np.linspace(-6,6,len(Ys))
    plt.figure(figsize=(6.4,6.4))
    ax1=plt.subplot2grid((15,15),(0,4),rowspan=11,colspan=11)
    ax1.hist2d(samples[:,0],samples[:,1],bins=120,range=[[-6,6],[-6,6]],cmap='Blues')
    ax2=plt.subplot2grid((15,15),(0,0),rowspan=11,colspan=3)
    ax2.hist(samples[:,1],bins=120,range=[-6,6], orientation='horizontal',density=True)
    ax2.plot(Ys[:,1],Xs,color='r')
    plt.ylim((-6,6))
    ax3=plt.subplot2grid((15,15),(12,4),rowspan=3,colspan=11)
    ax3.hist(samples[:,0],bins=120,range=[-6,6], orientation='vertical',density=True,label=name)
    ax3.plot(Xs,Ys[:,0],color='r',label='ref')
    plt.xlim((-6,6))
    plt.legend(bbox_to_anchor=(-0.12,0.7))

sgld_path=path+'SGLD/'
svrg_ld_path=path+'SVRG_LD/'
sghmc_path=path+'SGHMC/'
sg_ul_mcmc_path=path+'SG_UL_MCMC/'
srvr_hmc_path=path+'SRVR_HMC/'
srm_hmc_path=path+'SRM_HMC/'
nsrm_hmc_path=path+'NSRM_HMC/'

path_and_name=[
    [sgld_path,'SGLD'],
    [svrg_ld_path,'SVRG_LD'],
    [sghmc_path,'SGHMC'],
    [sg_ul_mcmc_path,'SG_UL_MCMC'],
    [srvr_hmc_path,'SRVR_HMC'],
    [srm_hmc_path,'SRM_HMC'],
    [nsrm_hmc_path,'NSRM_HMC']]

save_folder='./result/figure_eps/'
for i in tqdm(range(7)):
    for file_name in os.listdir(path_and_name[i][0]):
        datas=np.load(path_and_name[i][0]+file_name)
        file_name=file_name[:len(file_name)-4]
        plot_fig(datas,path_and_name[i][1])
        if not os.path.exists(save_folder+path_and_name[i][1]+'/'):
            os.makedirs(save_folder+path_and_name[i][1]+'/')
        plt.savefig(save_folder+path_and_name[i][1]+'/'+file_name+'.eps', format='eps', dpi=2000)
        #plt.savefig(save_folder+path_and_name[i][1]+'/'+file_name+'.png', format='png')
        plt.close()