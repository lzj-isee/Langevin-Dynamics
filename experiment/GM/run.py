from SGLD import SGLD_sample
from SVRG_LD import SVRG_LD_sample
from SGHMC import SGHMC_sample
from SG_UL_MCMC import SG_UL_MCMC_sample
from SRVR_HMC import SRVR_HMC_sample
from SRM_HMC import SRM_HMC_sample
import numpy as np

#SGLD
factor_a_s=np.array([0.6,0.8,1.0,1.2,1.4])
factor_gamma_s=np.array([0,0.05,0.08,0.1])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma
            }
        save_folder='./SGLD_result_it'
        SGLD_sample(random_seed,train_setting,save_folder)

#SVRG_LD
factor_a_s=np.array([0.7,1.0,1.2,1.5,1.8])
factor_gamma_s=np.array([0,0.05,0.10,0.15])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma
            }
        save_folder='./SVRG_LD_result_it'
        SVRG_LD_sample(random_seed,train_setting,save_folder)

#SGHMC
factor_a_s=np.array([0.6,0.8,1.0,1.2,1.4])
factor_gamma_s=np.array([0,0.05,0.08,0.1])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma
            }
        save_folder='./SGHMC_result_it'
        SGHMC_sample(random_seed,train_setting,save_folder)

#SG_UL_MCMC
factor_a_s=np.array([0.06,0.08,0.1,0.15,0.2])
factor_gamma_s=np.array([0,0.05,0.08,0.10])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020
            }
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1
            }
        save_folder='./SG_UL_MCMC_result_it'
        SG_UL_MCMC_sample(random_seed,train_setting,save_folder)

#SRVR_HMC
factor_a_s=np.array([0.03,0.04,0.05,0.06,0.07])
factor_gamma_s=np.array([0,0.05,0.08,0.10])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSizeB0':500,
            'batchSizeB':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1
            }
        save_folder='./SRVR_HMC_result_it'
        SRVR_HMC_sample(random_seed,train_setting,save_folder)

#SRM_HMC
factor_a_s=np.array([0.02,0.04,0.06,0.08,0.10])
factor_gamma_s=np.array([0,0.05,0.08,0.10])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1
            }
        save_folder='./SRM_HMC_result_it'
        SRM_HMC_sample(random_seed,train_setting,save_folder)